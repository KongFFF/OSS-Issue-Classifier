import requests
import pandas as pd
import time
from tqdm import tqdm
import os
#  配置
token = os.environ.get('GITHUB_TOKEN', '')
HEADERS = {'Authorization': f'token {token}'}
REPO = 'microsoft/vscode'

# 定义目标采集量：每个类别爬取多少条
# 建议设置为 1500 (即 Bug 1500 + Feature 1500 = 3000条）
TARGET_PER_CLASS = 1500

# 定义需要的类别和对应的搜索关键词
# 格式: (内部类别ID, 搜索用的标签名)
TARGETS = [
    (0, 'bug'),              # 类别 0: Bug
    (1, 'feature-request')   # 类别 1: Feature
]
# ===========================================

def search_issues(label, limit):

    dataset = []
    print(f"Starting collection for Label: [{label}] (Target: {limit})")
    
    # 分两波抓取
    # round 1: sort='created', order='desc' (获取最新的)
    # round 2: sort='created', order='asc'  (获取最旧的)
    sort_orders = ['desc', 'asc']
    
    pbar = tqdm(total=limit)
    
    for order in sort_orders:
        if len(dataset) >= limit:
            break
            
        page = 1
        while len(dataset) < limit:
            # 搜索语法: repo:microsoft/vscode label:bug state:closed
            query = f"repo:{REPO} label:{label} state:closed"
            url = "https://api.github.com/search/issues"
            params = {
                'q': query,
                'per_page': 100,  # 单页最大 100
                'page': page,
                'sort': 'created',
                'order': order
            }
            
            try:
                response = requests.get(url, headers=HEADERS, params=params)
                
                # 处理限流 (403 Forbidden 或 429 Too Many Requests)
                if response.status_code in [403, 429]:
                    print("\nRate limit hit. Sleeping for 60s...")
                    time.sleep(60)
                    continue
                
                if response.status_code != 200:
                    print(f"\nAPI Error {response.status_code}: {response.text}")
                    break
                    
                items = response.json().get('items', [])
                if not items:
                    break # 这一波结束
                
                # 数据提取
                new_count = 0
                for item in items:
                    # 过滤掉 Pull Request，只保留纯 Issue
                    if 'pull_request' in item:
                        continue
                        
                    # 查重：防止正序和倒序抓到了重复的数据
                    if any(d['id'] == item['id'] for d in dataset):
                        continue
                        
                    # 提取训练所需的字段
                    dataset.append({
                        'id': item['id'],
                        'title': item['title'],
                        'body': item['body'],
                        'label_name': label,
                        'url': item['html_url']
                    })
                    new_count += 1
                    pbar.update(1)
                    
                    if len(dataset) >= limit:
                        break
                
                if new_count == 0:
                    print("\nNo new unique items found on this page. Moving to next sort order.")
                    break
                    
                page += 1
                time.sleep(2) # 防止 422
                
            except Exception as e:
                print(f"\nException: {e}")
                break
                
    pbar.close()
    return dataset

if __name__ == "__main__":
    all_data = []
    
    for class_id, label_name in TARGETS:
        # 采集数据
        data = search_issues(label_name, TARGET_PER_CLASS)
        
        # 打上数字标签 (0 或 1)
        for item in data:
            item['label'] = class_id
        
        all_data.extend(data)
        print(f"Finished {label_name}: Collected {len(data)} items.")
        time.sleep(2)

    # 修正
    # 直接使用 DataFrame 构造函数
    df = pd.DataFrame(all_data)
    
    # 清洗：去除 body 为空的数据
    initial_len = len(df)
    df = df.dropna(subset=['body'])
    print(f"\nData Cleaning: Dropped {initial_len - len(df)} empty rows.")
    
    filename = 'vscode_dataset_clean.csv'
    # escapechar 防止奇怪字符破坏 CSV 格式
    df.to_csv(filename, index=False, encoding='utf-8', escapechar='\\')
    
    print(f"\nSuccess! Total data: {len(df)}")
    print("Class Distribution:")
    print(df['label_name'].value_counts())
# TODO: 这里还没写完，明天继续
