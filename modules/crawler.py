import requests
import re
import logging
import urllib3

from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

from llama_index.core import Document

urllib3.disable_warnings()
logging.getLogger("urllib3").setLevel(logging.ERROR)

class WebCrawler:
    def __init__(self, urls, exclude_domains=None, relevancy_keywords=None, max_links=None):
        self.urls = urls
        self.visited = set()
        self.user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) saltai-agent'
        self.robots_parsers = {}
        self.errors = []
        self.exclude_domains = set(exclude_domains.split(',')) if exclude_domains else set()
        self.relevancy_keywords = set(map(str.strip, relevancy_keywords.split(','))) if relevancy_keywords else set()
        self.max_links = max_links

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
            self.errors.append(f"Error fetching {robots_url}: {str(e)}")
            return None

    def can_fetch(self, url):
        parsed_url = urlparse(url)
        base_url = f'{parsed_url.scheme}://{parsed_url.netloc}'
        if base_url in self.exclude_domains:
            print(f"Skipping {url} due to exclusion.")
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
            response = requests.get(url, headers=headers, verify=verify_ssl)
            response.raise_for_status()
            return response.text, response.headers
        except requests.RequestException as e:
            self.errors.append(f"Error fetching {url}: {str(e)}")
            return None, None

    def parse_html(self, html, response_headers, trim_line_breaks=False):
        soup = BeautifulSoup(html, 'html.parser')
        title = soup.find('title').text if soup.find('title') else 'No title'
        content_blocks = soup.find_all(['article', 'main', 'section'])
        text = '\n'.join(block.get_text(separator='\n') for block in content_blocks if block)
        if trim_line_breaks:
            text = re.sub(r'\n{5,}', '\n\n\n\n', text)
        return {'title': title, 'text': text}

    def crawl(self, url, depth=0, max_depth=2, trim_line_breaks=False, verify_ssl=True):
        normalized_url = self.normalize_url(url)
        if depth > max_depth or normalized_url in self.visited or not self.can_fetch(url):
            return []
        html, headers = self.fetch_content(url, verify_ssl=verify_ssl)
        if html is None:
            return []
        page_details = self.parse_html(html, headers, trim_line_breaks)
        if self.relevancy_keywords:
            if not any(keyword.lower() in page_details['text'].lower() for keyword in self.relevancy_keywords):
                return []
        soup = BeautifulSoup(html, 'html.parser')
        found_urls = set()
        links_processed = 0
        for link in soup.find_all('a', href=True):
            if links_processed >= self.max_links:
                break
            link_url = link.get('href')
            full_url = urljoin(url, link_url)
            normalized_full_url = self.normalize_url(full_url)
            if urlparse(normalized_full_url).scheme in ['http', 'https'] and normalized_full_url not in self.visited:
                found_urls.add(normalized_full_url)
                links_processed += 1
        results = [{'url': normalized_url, 'title': page_details['title'], 'text': page_details['text'], 'page_urls': list(found_urls)}]
        print(f"Crawling: {normalized_url} | Links found: {len(found_urls)}")
        for link_url in found_urls:
            results.extend(self.crawl(link_url, depth + 1, max_depth, trim_line_breaks, verify_ssl))
        return results

    def parse_sites(self, crawl=False, max_depth=2, trim_line_breaks=False, verify_ssl=False):
        results = []
        for url in self.urls:
            print(f"Starting crawl of: {url}")
            if crawl:
                results.extend(self.crawl(url, 0, max_depth, trim_line_breaks, verify_ssl))
            else:
                page_details = self.fetch_content(url)
                if page_details:
                    results.append({'url': url, 'title': page_details['title'], 'text': page_details['text']})
        if self.errors:
            print("Errors encountered:")
            for error in self.errors:
                print(error)
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