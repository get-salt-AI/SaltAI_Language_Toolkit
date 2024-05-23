import os
import warnings
import re
import random
import pandas as pd
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import json
import yaml
import mimetypes

warnings.filterwarnings("ignore", category=FutureWarning)

from .. import logger

CLEAN_CHARS = {
    "[": "",
    "]": "",
    "(": "",
    ")": "",
    ":": "",
    "-": " ",
    "_": " ",
    "~": "",
    "@": "",
    "\"": "",
    "'": "",
    ";": "",
    "/": " or ",
    "\\": " ",
    "&": " and ",
    "*": " asterisk ",
    "#": " hashtag ",
    "%": " percent",
    "^": "",
    "`": "",
    "|": ", ",
    "+": " plus ",
    "=": " equals "
}

class InvalidFileTypeError(Exception):
    def __init__(self, message="Invalid file type provided"):
        self.message = message
        super().__init__(self.message)

class ParquetReader1D:
    def __init__(self, file_path=None):
        self.df = pd.read_parquet(file_path) if file_path else None
        pd.set_option('display.max_colwidth', None)

    @staticmethod
    def remove_urls(text):
        pattern = r'https?://\S+|www\.\S+'
        return re.sub(pattern, '', text)

    def list_columns(self):
        return self.df.columns.tolist() if self.df is not None else []

    def get_random_word(self, columns, seed=None):
        if seed is not None:
            random.seed(seed)
        random_column = random.choice(columns)
        valid_rows = self.df.dropna(subset=[random_column])
        if not valid_rows.empty:
            random_row = valid_rows.sample(n=1, random_state=seed).iloc[0]
            words = str(random_row[random_column]).split()
            return random.choice(words) if words else None
        return None

    def parse_content(self, content):
        pattern = re.compile("|".join(re.escape(k) for k in CLEAN_CHARS.keys()))
        def replace_chars(match):
            return CLEAN_CHARS[match.group(0)]
        
        if isinstance(content, str):
            content = pattern.sub(replace_chars, content)
            return self.remove_urls(content)
        elif isinstance(content, dict):
            return {k: self.parse_content(v) for k, v in content.items()}
        elif isinstance(content, list):
            return [self.parse_content(item) for item in content]
        return content

    def search(self, search_term, columns, exclude_terms=None, max_results=10, case_sensitive=False,
               term_relevancy_score=None, num_threads=None, min_length=0, max_length=-1, max_dynamic_retries=0,
               parse_content=False, seed=None, use_relevancy=True):
        if '*' in columns:
            columns = self.df.columns.tolist()
        else:
            columns = [col for col in columns if col in self.df.columns]

        if search_term:
            keywords = search_term.split(',')
        else:
            keywords = [self.get_random_word(columns, seed), self.get_random_word(columns, seed+1)]
        
        keywords = [kw.strip() for kw in keywords if kw.strip()]
        exclude_terms = exclude_terms.split(',') if isinstance(exclude_terms, str) and exclude_terms else []
        exclude_terms = [et.strip() for et in exclude_terms if et.strip()]

        logger.info("Search terms:", keywords)
        logger.info("Exclude terms:", exclude_terms)

        def calculate_relevance(row):
            relevance = 0
            for column in columns:
                text = self.parse_content(row[column]) if parse_content else row[column]
                if text is not None:
                    text = text if case_sensitive else text.lower()
                    text_length = len(text)

                    if (min_length > 0 and text_length < min_length) or (max_length > 0 and text_length > max_length):
                        continue

                    column_relevance = sum(text.count(keyword) * len(keyword) / text_length for keyword in keywords)
                    if exclude_terms:
                        column_relevance -= sum(text.count(term) * 0.05 for term in exclude_terms)

                    relevance += column_relevance

            return max(0, min(relevance, 1))

        def search_segment(data_segment):
            if not case_sensitive:
                data_segment = data_segment.applymap(lambda x: x.lower() if isinstance(x, str) else x)
            
            conditions = [data_segment[col].str.contains(keyword, na=False, regex=False) for col in columns for keyword in keywords]
            if not conditions:
                return pd.DataFrame()  # Return empty DataFrame if no conditions
            
            condition = pd.concat(conditions, axis=1).any(axis=1)
            filtered_segment = data_segment[condition]

            if exclude_terms:
                exclude_conditions = [filtered_segment[col].str.contains(term, na=False, regex=False) for col in columns for term in exclude_terms]
                if exclude_conditions:
                    exclude_condition = pd.concat(exclude_conditions, axis=1).any(axis=1)
                    filtered_segment = filtered_segment[~exclude_condition]

            # Enforce length constraints on filtered_segment
            filtered_segment = filtered_segment.applymap(lambda x: x if (pd.isna(x) or (min_length <= len(str(x)) <= (max_length if max_length > 0 else len(str(x))))) else None)
            filtered_segment.dropna(how='all', inplace=True)

            return filtered_segment

        num_threads = max(2, os.cpu_count() // 2) if not num_threads else num_threads
        chunks = np.array_split(self.df, num_threads)
        logger.info(f"Running search with {num_threads} threads and {len(chunks)} chunks")

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            results = list(executor.map(search_segment, chunks))

        combined_results = pd.concat(results)
        logger.info(f"Combined results size: {combined_results.shape}")

        if not combined_results.empty and use_relevancy:
            combined_results['relevance'] = combined_results.apply(lambda row: calculate_relevance(row), axis=1)
            if term_relevancy_score is not None:
                combined_results['relevance_diff'] = combined_results['relevance'].apply(lambda x: abs(x - term_relevancy_score))
                sorted_results = combined_results.sort_values(by='relevance_diff').head(max_results)
            else:
                sorted_results = combined_results.sort_values(by='relevance', ascending=False).head(max_results)
        else:
            sorted_results = combined_results.head(max_results)

        if sorted_results.empty and max_dynamic_retries > 0:
            logger.warning("No results found, retrying with random search terms.")
            new_seed = random.randint(1,99999999)
            return self.search(
                self.get_random_word(columns, new_seed), columns, exclude_terms, max_results, case_sensitive,
                term_relevancy_score, num_threads, min_length, max_length, max_dynamic_retries - 1, parse_content, new_seed, use_relevancy
            )
        
        results_list = list(sorted_results.apply(
            lambda row: {column: row[column] for column in columns}, axis=1
        ))

        return results_list

    def from_parquet(self, path):
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")
            
        self.df = pd.read_parquet(path)
        return self

    def from_text(self, path, recache=False):
        path = os.path.normpath(path)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is not None and not mime_type.startswith('text'):
            raise InvalidFileTypeError(f"Invalid text file. Please make sure the file is plain text: {path}")

        logger.info("Reading text file:", path)
        parquet_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.parquet')
        if not os.path.exists(parquet_file) or not os.path.isfile(parquet_file) or recache:        
            with open(path, 'r', encoding='utf-8') as file:
                lines = file.readlines()
            df = pd.DataFrame(lines, columns=['Prompt'])
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_file)
            logger.info(f"Text file '{path}' converted to parquet file '{parquet_file}'")
        else:
            logger.info("Found existing parquet file for this text file, loading parquet file instead.")
            logger.warning("To force re-caching pass `recache=True` param to the `from_text()` method")
            
        self.df = pd.read_parquet(parquet_file)
        return self

    def from_json(self, path, recache=False):
        path = os.path.normpath(path)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is not None and not mime_type == 'application/json':
            raise InvalidFileTypeError(f"Invalid JSON file. Please make sure the file is a valid JSON: {path}")

        logger.info("Reading JSON file:", path)
        parquet_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.parquet')
        if not os.path.exists(parquet_file) or not os.path.isfile(parquet_file) or recache:
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
            data_dict = {k: [item for item in v if item is not None] for k, v in data.items() if isinstance(v, list)}
            flat_data = {'Prompt': [item for sublist in data_dict.values() for item in sublist if item]}
            df = pd.DataFrame(flat_data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_file)
            logger.info(f"JSON file '{path}' converted to parquet file '{parquet_file}'")
        else:
            logger.info("Found existing parquet file for this JSON file, loading parquet file instead.")
            logger.warning("To force re-caching pass `recache=True` param to the `from_json()` method")
        
        self.df = pd.read_parquet(parquet_file)
        return self

    def from_yaml(self, path, recache=False):
        path = os.path.normpath(path)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")

        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is not None and not mime_type in ['application/x-yaml', 'text/yaml']:
            raise InvalidFileTypeError(f"Invalid YAML file. Please make sure the file is a valid YAML: {path}")

        logger.info("Reading YAML file:", path)
        parquet_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.parquet')
        if not os.path.exists(parquet_file) or not os.path.isfile(parquet_file) or recache:
            with open(path, 'r', encoding='utf-8') as file:
                data = yaml.safe_load(file)
            data_dict = {k: [item for item in v if item is not None] for k, v in data.items() if isinstance(v, list)}
            flat_data = {'Prompt': [item for sublist in data_dict.values() for item in sublist if item]}
            df = pd.DataFrame(flat_data)
            table = pa.Table.from_pandas(df)
            pq.write_table(table, parquet_file)
            logger.info(f"YAML file '{path}' converted to parquet file '{parquet_file}'")
        else:
            logger.info("Found existing parquet file for this YAML file, loading parquet file instead.")
            logger.warning("To force re-caching pass `recache=True` param to the `from_yaml()` method")
        
        self.df = pd.read_parquet(parquet_file)
        return self

    def from_csv(self, path, recache=False):
        path = os.path.normpath(path)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")
    
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is not None and not mime_type == 'text/csv':
            raise InvalidFileTypeError(f"Invalid CSV file. Please make sure the file is a valid CSV: {path}")
    
        logger.info("Reading CSV file:", path)
        parquet_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.parquet')
        if not os.path.exists(parquet_file) or not os.path.isfile(parquet_file) or recache:
            df = pd.read_csv(path)
            flat_data = {'Prompt': []}
            for _, row in df.iterrows():
                for item in row.dropna():
                    flat_data['Prompt'].append(str(item))
            df_flat = pd.DataFrame(flat_data)
            table = pa.Table.from_pandas(df_flat)
            pq.write_table(table, parquet_file)
            logger.info(f"CSV file '{path}' converted to parquet file '{parquet_file}'")
        else:
            logger.info("Found existing parquet file for this CSV file, loading parquet file instead.")
            logger.warning("To force re-caching pass `recache=True` param to the `from_csv()` method")
        
        self.df = pd.read_parquet(parquet_file)
        return self

    def from_excel(self, path, sheet_name=0, recache=False):
        path = os.path.normpath(path)
        if not os.path.exists(path) or not os.path.isfile(path):
            raise ValueError(f"The supplied path is invalid, and not a valid file: {path}")
    
        mime_type, _ = mimetypes.guess_type(path)
        if mime_type is not None and not mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            raise InvalidFileTypeError(f"Invalid Excel file. Please make sure the file is a valid Excel file: {path}")
    
        logger.info("Reading Excel file:", path)
        parquet_file = os.path.join(os.path.dirname(path), os.path.splitext(os.path.basename(path))[0] + '.parquet')
        if not os.path.exists(parquet_file) or not os.path.isfile(parquet_file) or recache:
            df = pd.read_excel(path, sheet_name=sheet_name)
            flat_data = {'Prompt': []}
            for _, row in df.iterrows():
                for item in row.dropna():
                    flat_data['Prompt'].append(str(item))
            df_flat = pd.DataFrame(flat_data)
            table = pa.Table.from_pandas(df_flat)
            pq.write_table(table, parquet_file)
            logger.info(f"Excel file '{path}' converted to parquet file '{parquet_file}'")
        else:
            logger.info("Found existing parquet file for this Excel file, loading parquet file instead.")
            logger.warning("To force re-caching pass `recache=True` param to the `from_excel()` method")
        
        self.df = pd.read_parquet(parquet_file)
        return self 
