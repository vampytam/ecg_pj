from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import json
import time
import os

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
            # print(os.path.abspath(self.diagnosis_links_path))
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
        
    def _extract_one_diagnosis_info(self, url):
        try:
            self.driver.get(url)
            WebDriverWait(self.driver, 20).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, ".entry-content.is-layout-flow h5"))
            )
            time.sleep(3)
            
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            
            main_element = soup.find('main', id='main')
            
            if not main_element:
                print("main part is empty")
                return []
            
            entry_content = main_element.find('div', class_='entry-content')
            if not entry_content:
                entry_content = main_element.find('div', class_=lambda x: x and 'entry-content' in x)
            if not entry_content:
                print("entry content is empty")
                return []
            
            # get all of entry_content's children
            all_elements = [el for el in entry_content.children if el.name]
            
            skip_keywords = {'example', 'reference', 'advanced', 'further'}
            filtered = []
            for el in all_elements:
                if el.name in ('h4', 'h5', 'h6'):
                    el_id = (el.get('id') or '').lower()
                    el_text = el.get_text(strip=True).lower()
                    if any(kw in el_id or kw in el_text for kw in skip_keywords):
                        break
                filtered.append(el)
            
            diag_info = {}
            current_key = None
            current_value_parts = []
            
            def extract_text(elem):
                text = elem.get_text(separator=' ', strip=True)
                # convert full-width to half-width
                return full_to_half(text)
                        
            def format_content(elem, level=0):
                if elem.name in ('h4', 'h5', 'h6'):
                    return extract_text(elem)
                elif elem.name == 'p':
                    return '\n' * (level + 1) + extract_text(elem)
                elif elem.name == 'ul':
                    lis = [extract_text(li) for li in elem.find_all('li', recursive=False)]
                    return '\n' * (level + 1) + '\n'.join(lis)
                elif elem.name == 'div':
                    ul = elem.find('ul')
                    if ul:
                        lis = [extract_text(li) for li in ul.find_all('li', recursive=False)]
                        return '\n' * (level + 1) + '\n'.join(lis)
                    else:
                        text = extract_text(elem)
                        if text:
                            return '\n' * (level + 1) + text
                return ''
                        
            for el in filtered:
                if el.name in ('h4', 'h5', 'h6'):
                    if current_key is not None:
                        diag_info[current_key] = ('\n'.join(current_value_parts))
                        current_value_parts = []
                    current_key = extract_text(el)
                else:
                    formatted_content = format_content(el, level=1)
                    if formatted_content:
                        current_value_parts.append(formatted_content)
            if current_key is not None:
                diag_info[current_key] = ('\n'.join(current_value_parts))
                current_value_parts = []
            if current_value_parts:
                diag_info['Potential Feature'] = ('\n'.join(current_value_parts))
        
        except Exception as e:
            print(f"while extract from url: {url}, Error:", e)
            return {}
        
        return diag_info 
    
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
                    'diag_info': diag_info
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
        try:
            if not os.path.exists(path):
                return []
            
            data = []
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data.append(json.loads(line))
            return data
        except Exception as e:
            print(f"while load from jsonl {e}")
            return []
        
    def _refine_result(self, raw_results):
        results = {}
        for item in raw_results:
            title = item.get('title', '')
            infos = item.get('diag_info', {})
            
            feature_descs = [infos[key] for key in infos if "feature" in key.lower() or "morphology" in key.lower()]
            if feature_descs:
                feature_desc = '\n'.join(feature_descs)
                results[title] = feature_desc
                continue
            
            definition_desc = [infos[key] for key in infos if "definition" in key.lower() or "overview" in key.lower()]
            if definition_desc:
                definition_desc_combined = '\n'.join(definition_desc)
                results[title] = definition_desc_combined
                continue
            
            criteria_desc = [infos[key] for key in infos if "criteria" in key.lower()]
            if criteria_desc:
                criteria_desc_combined = '\n'.join(criteria_desc)
                results[title] = criteria_desc_combined
                continue
            
            effect_desc = [infos[key] for key in infos if "effect" in key.lower()]
            if effect_desc:
                effect_desc_combined = '\n'.join(effect_desc)
                results[title] = effect_desc_combined
                continue
        
        with open(self.refined_data_path, 'w', encoding='utf-8') as f:
            f.write(json.dumps(results, ensure_ascii=False))
            
        return results
                
                                
    def run_crawler(self):
        # 1. get diagnosis links    
        links = self._get_diagnosis_links(force_refresh=True)
        
        if not links:
            print("diagnosis links are empty")
            return []
        
        # 2. extract diagnosis infos
        results = self._extract_diagnosis_infos(links, force_refresh=True)
        
        if not results:
            print("extracted diagnosis infos are empty")
            return []
        
        if self.driver:
            self.driver.quit()
        
        # 3. refine and save data
        self._refine_result(results)
        
        return results

# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    crawler = LitflCrawler()
    
    results = crawler.run_crawler()
    
    # print(results)