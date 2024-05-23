import requests
import re
import logging
import urllib3
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import xml.etree.ElementTree as ET
from llama_index.core import Document

urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.ERROR)

class WebCrawler:
    def __init__(self, urls, exclude_domains=None, keywords=None, max_links=None, evaluate_links=False, evaluate_page_content=False, max_threads=None, jina_scrape=False):
        self.urls = urls
        self.visited = set()
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) saltai-agent'
        self.robots_parsers = {}
        self.errors = []
        self.exclude_domains = set(exclude_domains.split(',')) if exclude_domains else set()
        self.keywords = set(map(str.strip, keywords.lower().split(','))) if keywords else set()
        self.max_links = max_links if max_links is not None else float('inf')
        self.evaluate_links = evaluate_links
        self.evaluate_page_content = evaluate_page_content
        self.jina_scrape = jina_scrape

        self.max_threads = max_threads
        if not self.max_threads or self.max_threads > os.cpu_count():
            self.max_threads = max(1, os.cpu_count() // 2)

        if '*' in self.exclude_domains:
            self.allowed_domains = {urlparse(url).netloc for url in self.urls}
            self.exclude_domains.remove('*')
        else:
            self.allowed_domains = None

    def normalize_url(self, url):
        parsed_url = urlparse(url)
        normalized = urlunparse(parsed_url._replace(fragment=""))
        return normalized.rstrip('/')

    def fetch_robots_txt(self, base_url):
        robots_url = urljoin(base_url, 'robots.txt')
        try:
            response = requests.get(robots_url, headers={'User-Agent': self.user_agent}, verify=False)
            response.raise_for_status()
            parser = RobotFileParser()
            parser.parse(response.text.splitlines())
            return parser
        except requests.RequestException as e:
            error = f"Error fetching {robots_url}: {str(e)}"
            if error not in self.errors:
                self.errors.append(error)
            return None

    def can_fetch(self, url):
        parsed_url = urlparse(url)
        base_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        if self.allowed_domains is not None:
            if parsed_url.netloc not in self.allowed_domains:
                logging.info(f"Skipping {url} due to allowed domains restriction.")
                return False
        elif base_url in self.exclude_domains:
            logging.info(f"Skipping {url} due to exclusion.")
            return False
        if base_url not in self.robots_parsers:
            parser = self.fetch_robots_txt(base_url)
            if parser:
                self.robots_parsers[base_url] = parser
            else:
                return True
        return self.robots_parsers[base_url].can_fetch(self.user_agent, url) if self.robots_parsers[base_url] else True

    def fetch_content(self, url, verify_ssl=True):
        normalized_url = self.normalize_url(url)
        if normalized_url in self.visited or not self.can_fetch(normalized_url):
            return None, None
        self.visited.add(normalized_url)
        try:
            headers = {'User-Agent': self.user_agent}
            if self.jina_scrape:
                jina_url = f"https://r.jina.ai/{url}"
                response = requests.get(jina_url)
                if response.status_code == 200:
                    html_content = f"<div>{response.text}</div>"
                    return html_content, None
                else:
                    self.errors.append(f"Error fetching {jina_url}: {response.status_code}")
                    return None, None
            else:
                response = requests.get(url, headers=headers, verify=verify_ssl)
                response.raise_for_status()
                content_type = response.headers.get('Content-Type', '').lower()
                try:
                    if 'text/html' in content_type or 'text/plain' in content_type:
                        return response.text, response.headers
                    elif 'application/json' in content_type:
                        return json.dumps(response.json(), indent=2), response.headers
                    elif 'application/xml' in content_type or 'text/xml' in content_type:
                        return response.text, response.headers
                    else:
                        response.text
                        return response.text, response.headers
                except Exception as e:
                    self.errors.append(f"Unsupported content type: {content_type} for URL: {url}. Error: {str(e)}")
                    return None, None
        except requests.RequestException as e:
            self.errors.append(f"Error fetching {url}: {str(e)}")
            return None, None

    def parse_html(self, html, response_headers, trim_line_breaks=False):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No title'

        for script in soup(["script", "style"]):
            script.decompose()

        text = soup.get_text(separator="\n")
        
        if trim_line_breaks:
            text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'[\r\n]+', '\n', text)

        text = text.strip()

        return {'title': title, 'text': text}

    def parse_json(self, json_data):
        try:
            json_str = json.dumps(json_data, indent=2)
            return {'title': 'JSON Data', 'text': json_str}
        except Exception as e:
            self.errors.append(f"Error parsing JSON data: {str(e)}")
            return {'title': 'JSON Data', 'text': ''}

    def parse_xml(self, xml_data):
        try:
            root = ET.fromstring(xml_data)
            xml_str = ET.tostring(root, encoding='unicode', method='xml')
            return {'title': root.tag, 'text': xml_str}
        except ET.ParseError as e:
            self.errors.append(f"Error parsing XML data: {str(e)}")
            return {'title': 'XML Data', 'text': ''}

    def evaluate_link(self, link_text, link_url):
        if not self.keywords:
            return 1 
        score = 0
        if any(keyword in link_text.lower() for keyword in self.keywords):
            score += 1
        if any(keyword in link_url.lower() for keyword in self.keywords):
            score += 1
        return score

    def crawl(self, url, depth=0, max_depth=2, trim_line_breaks=False, verify_ssl=True):
        normalized_url = self.normalize_url(url)
        if depth > max_depth or normalized_url in self.visited or not self.can_fetch(url):
            return []
        content, headers = self.fetch_content(url, verify_ssl=verify_ssl)
        if content is None:
            return []
        if isinstance(content, dict):
            page_details = self.parse_json(content)
        elif '<' in content:
            if content.strip().startswith('<'):
                page_details = self.parse_xml(content) if 'xml' in headers.get('Content-Type', '').lower() else self.parse_html(content, headers, trim_line_breaks)
            else:
                page_details = self.parse_html(content, headers, trim_line_breaks)
        else:
            page_details = {'title': 'Text Data', 'text': content}
        
        if self.evaluate_page_content and self.keywords:
            if not any(keyword in page_details['text'].lower() for keyword in self.keywords):
                return []

        soup = BeautifulSoup(content, 'html.parser') if '<' in content else None
        found_urls = []
        links_processed = 0
        if soup:
            for link in soup.find_all('a', href=True):
                if links_processed >= self.max_links:
                    break
                link_text = link.get_text()
                link_url = link.get('href')
                full_url = urljoin(url, link_url)
                normalized_full_url = self.normalize_url(full_url)
                if urlparse(normalized_full_url).scheme in ['http', 'https'] and normalized_full_url not in self.visited:
                    score = self.evaluate_link(link_text, normalized_full_url)
                    if not self.evaluate_links or score > 0:
                        found_urls.append((score, normalized_full_url))
                    links_processed += 1

        found_urls.sort(reverse=True, key=lambda x: x[0])
        sorted_urls = [url for _, url in found_urls]
        results = [{'url': normalized_url, 'title': page_details['title'], 'text': page_details['text'], 'page_urls': sorted_urls}]
        logging.info(f"Crawling: {normalized_url} | Links found: {len(found_urls)}")
        return results

    def crawl_thread(self, url, depth=0, max_depth=2, trim_line_breaks=False, verify_ssl=True):
        results = self.crawl(url, depth, max_depth, trim_line_breaks, verify_ssl)
        found_urls = [result['page_urls'] for result in results if 'page_urls' in result]
        found_urls = [url for sublist in found_urls for url in sublist]

        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            futures = {executor.submit(self.crawl, url, depth + 1, max_depth, trim_line_breaks, verify_ssl): url for url in found_urls}
            for future in as_completed(futures):
                try:
                    results.extend(future.result())
                except Exception as e:
                    logging.error(f"Error during threaded crawl: {e}")
        return results

    def parse_sites(self, crawl=False, max_depth=2, trim_line_breaks=False, verify_ssl=False):
        results = []
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_url = {executor.submit(self.crawl_thread if crawl else self.fetch_content, url, 0, max_depth, trim_line_breaks, verify_ssl): url for url in self.urls}
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    if crawl:
                        results.extend(future.result())
                    else:
                        content, headers = future.result()
                        if content:
                            if isinstance(content, dict):
                                page_details = self.parse_json(content)
                            elif '<' in content:
                                if content.strip().startswith('<'):
                                    page_details = self.parse_xml(content) if 'xml' in headers.get('Content-Type', '').lower() else self.parse_html(content, headers, trim_line_breaks)
                                else:
                                    page_details = self.parse_html(content, headers, trim_line_breaks)
                            else:
                                page_details = {'title': 'Text Data', 'text': content}
                            results.append({'url': url, 'title': page_details['title'], 'text': page_details['text']})
                except Exception as e:
                    logging.error(f"Error fetching {url}: {e}")
        if self.errors:
            logging.error("Errors encountered:")
            for error in self.errors:
                logging.error(error)
        return CrawlResults(results)

class CrawlResults:
    def __init__(self, results):
        self.results = results

    def to_documents(self):
        documents = []
        for result in self.results:
            content = result.pop("text", "")
            documents.append(Document(text=content, extra_info=result))
        return documents

    def to_dict(self):
        return self.results

    def add_metadata(self, metadata):
        for result in self.results:
            result.update(metadata)
        return self

    def filter_by_keyword(self, keyword):
        self.results = [result for result in self.results if keyword in result['text']]
        return self

    def __getitem__(self, key):
        return self.results[key]

    def __len__(self):
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __repr__(self):
        return str(self.results)
