from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup, NavigableString
import json
import time
import os
import re


from ..utils.prompt import create_prompt_from_file
from ..utils.llm import get_lm_response

# convert full-width to half-width
def full_to_half(s):
    punct_map = {
        '–': '-',   # En dash
        '—': '-',   # Em dash
        '‘': "'",   # 左单弯引号
        '’': "'",   # 右单弯引号
        '“': '"',   # 左双弯引号
        '”': '"',   # 右双弯引号
    }
    buf = []
    for ch in s:
        if ch in punct_map:                 # 先处理特殊标点
            buf.append(punct_map[ch])
        elif ord(ch) == 0x3000:             # 全角空格
            buf.append(' ')
        elif ch == ' ':
            buf.append(' ')
        elif 0xFF01 <= ord(ch) <= 0xFF5E:   # 全角 ASCII 区
            buf.append(chr(ord(ch) - 0xFEE0))
        else:
            buf.append(ch)
    return ''.join(buf)

class LitflCrawler:
    def __init__(self):
        self.driver = self._setup_driver()
        self.main_url = "https://litfl.com/ecg-library/diagnosis/"
        self.diagnosis_links_path = "diagnosis_links_path.jsonl"
        self.liftfl_data_path = "litfl_data.jsonl"
        self.refined_data_path = "litfl.jsonl"
        
    def _setup_driver(self):
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")

        driver = webdriver.Chrome(options=chrome_options)
        return driver
    
    def _get_diagnosis_links(self, force_refresh=False):
        if not force_refresh and os.path.exists(self.diagnosis_links_path):
            print(os.path.abspath(self.diagnosis_links_path))
            return self._load_from_jsonl(self.diagnosis_links_path)
        
        try:
            print(f"get diagnosis links from page: {self.main_url}...")
            self.driver.get(self.main_url)

            # make sure image rendering
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".entry-content.is-layout-flow a"))
            )
            time.sleep(5)

            html = self.driver.page_source
            soup = BeautifulSoup(html, 'html.parser')

            # select the second .entry-content.is-layout-flow block(which contains urls)
            outer_block = soup.select('.entry-content.is-layout-flow')[1]

            results = []
            for letter in "abcdefghijklmnopqrstuvwxyz":
                h2 = outer_block.find("h2", {"id": f"h-{letter}"})
                if not h2:
                    continue

                # next <ul class="wp-block-list"> after this h2
                ul = h2.find_next_sibling("ul", class_="wp-block-list")
                if not ul:
                    continue

                for a in ul.select("a[href]"):
                    href = a["href"].strip()
                    title = full_to_half(a.get_text(strip=True))
                    if href.startswith("http") and title:
                        results.append({title: href})

            self._save_to_jsonl(self.diagnosis_links_path, results)
            
            return results

        except Exception as e:
            print("Error:", e)
            return []
        
    def _extract_one_diagnosis_info(self, url: str) -> str:
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".entry-content.is-layout-flow"))
            )
            time.sleep(3)

            def _format_node(node) -> str:
                if node.name in {"h4", "h5", "h6"}:
                    return node.get_text(" ", strip=True)
                if node.name == "p":
                    return node.get_text(" ", strip=True)
                if node.name == "ul":
                    return "\n".join(li.get_text(" ", strip=True) for li in node.find_all("li", recursive=False))
                if node.name == "div":
                    ul = node.find("ul")
                    if ul:
                        return "\n".join(li.get_text(" ", strip=True) for li in ul.find_all("li", recursive=False))
                    p = node.find("p")
                    if p:
                        return p.get_text(" ", strip=True)
                    txt = node.get_text(" ", strip=True)
                    return txt if txt else ""

                return ""
            
            soup = BeautifulSoup(self.driver.page_source, "html.parser")
            main_element = soup.find('main', id='main')
            if not main_element:
                print(f"no main element found in url: {url}")
                return ""

            entry_content = main_element.find('div', class_='entry-content is-layout-flow')
            if not entry_content:
                print(f"no entry content found in url: {url}")
                return ""

            STOP_PAT = re.compile(r"example|reference|advanced|further", re.I)
            lines = []

            for node in entry_content.children:
                if isinstance(node, NavigableString):
                    continue

                if node.name in {"h4", "h5", "h6"}:
                    text_id = (node.get("id") or "").lower()
                    text_val = node.get_text(" ", strip=True).lower()
                    if STOP_PAT.search(text_id) or STOP_PAT.search(text_val):
                        break
            
                lines.append(_format_node(node))

            return "\n".join(lines).strip()

        except Exception as e:
            print(f"while extract from url: {url}, Error: {e}")
            return ""
    
    def _extract_diagnosis_infos(self, links, force_refresh=False):
        if not force_refresh and os.path.exists(self.liftfl_data_path):
            return self._load_from_jsonl(self.liftfl_data_path)
        
        results = []
        total_links = len(links)
        for i, link_dict in enumerate(links):
            for title, url in link_dict.items():
                print(f"now extract info from ({i+1}/{total_links}): {title}")
                diag_info = self._extract_one_diagnosis_info(url)
                results.append({
                    'title': title,
                    'url': url,
                    'diag_info': full_to_half(diag_info)
                })
        
        self._save_to_jsonl(self.liftfl_data_path, results)            
        return results
    
    def _save_to_jsonl(self, path, data):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                for item in data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
        except Exception as e:
            print(f"while save to jsonl {e}")
            
    def _load_from_jsonl(self, path):
        with open(path, 'r') as file:
            buffer = ""
            for line in file:
                buffer += line.strip()
                if buffer:
                    try:
                        obj = json.loads(buffer)
                        yield obj
                        buffer = ""
                    except json.JSONDecodeError:
                        continue
            if buffer:
                try:
                    yield json.loads(buffer)
                except json.JSONDecodeError:
                    raise ValueError("Trailing data is not valid JSON")
        
    def _refine_result(self, raw_results):
        for item in raw_results:
            title = item.get('title', '')
            print("refining item:", title)
            diag_info = item.get('diag_info', '')
            
            prompt = create_prompt_from_file("./dataset_preparation/prompt_templates/litfl_refine.txt", 
                                             title=title,
                                             diag_info=diag_info)

            _, answer_str = get_lm_response(prompt)

            diagnosis_match = re.search(r'"diagnosis_validity":\s*"([^"]*)"', answer_str)
            diagnosis_validity = diagnosis_match.group(1) if diagnosis_match else None
            if diagnosis_validity is None:
                print(f"Could not find diagnosis_validity in LLM response for title: {title}")
                continue

            ecg_features = re.findall(r'"ecg_features":\s*\[(.*?)\]', answer_str, re.DOTALL)
            if ecg_features:
                feature_items = re.findall(r'"([^"]*)"', ecg_features[0])
            else:
                feature_items = []

            print(f"diagnosis_validity: {diagnosis_validity}, features: {'\n'.join(feature_items)}")
            
            with open(self.refined_data_path, 'a', encoding='utf-8') as f:
                f.write(json.dumps({diagnosis_validity: "\n".join(feature_items)}, ensure_ascii=False) + '\n')

            time.sleep(3)                
                                
    def run_crawler(self):
        # 1. get diagnosis links    
        links = self._get_diagnosis_links(force_refresh=False)
        
        if not links:
            print("diagnosis links are empty")
            return []
        
        # 2. extract diagnosis infos
        results = self._extract_diagnosis_infos(links, force_refresh=False)
        
        if not results:
            print("extracted diagnosis infos are empty")
            return []
        
        if self.driver:
            self.driver.quit()
        
        # 3. refine and save data
        self._refine_result(results)

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    crawler = LitflCrawler()
    
    crawler.run_crawler()