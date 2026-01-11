import os
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# 指向之前的 CSV 文件 
CSV_PATH = os.path.join(BASE_DIR, '../../vscode_dataset_clean.csv') 
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model')
INDEX_OUTPUT_PATH = os.path.join(BASE_DIR, 'issue_index.pt')


def build_index():
    print(f"Loading data from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    # 只要 Title 和 Body，URL 用于展示
    # 填充空值防止报错
    df['text'] = df['title'].fillna('') + " " + df['body'].fillna('')
    documents = df[['text', 'url', 'title']].to_dict('records')
    
    print(f"Loading model from {MODEL_PATH}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    embeddings = []
    metadata = []
    
    print("Generating embeddings (this may take 1-2 mins)...")
    batch_size = 32
    
    with torch.no_grad():
        for i in tqdm(range(0, len(documents), batch_size)):
            batch_docs = documents[i : i + batch_size]
            batch_texts = [d['text'][:256] for d in batch_docs] # 截断一下防止爆显存
            
            inputs = tokenizer(
                batch_texts, 
                padding=True, 
                truncation=True, 
                max_length=128, 
                return_tensors="pt"
            ).to(device)
            
            # 获取 hidden_states 需要在调用时指定
            # 对于 SequenceClassification 模型，通常取最后一层的 CLS token 作为句向量
            outputs = model(**inputs, output_hidden_states=True)
            
            # hidden_states 是一个 tuple，最后一个元素是最后一层的输出
            # shape: [batch_size, seq_len, hidden_size]
            last_hidden_state = outputs.hidden_states[-1]
            
            # 取 CLS token (第一个 token) 的向量: [batch, 0, :]
            cls_embeddings = last_hidden_state[:, 0, :]
            
            # 归一化 (方便后续直接算 Cosine Similarity)
            cls_embeddings = F.normalize(cls_embeddings, p=2, dim=1)
            
            embeddings.append(cls_embeddings.cpu())
            
            # 保存对应的 metadata
            for doc in batch_docs:
                metadata.append({
                    'title': doc['title'],
                    'url': doc['url']
                })
    
    # 合并所有 batch
    all_embeddings = torch.cat(embeddings, dim=0)
    
    print(f"Saving index to {INDEX_OUTPUT_PATH}...")
    torch.save({
        'embeddings': all_embeddings,
        'metadata': metadata
    }, INDEX_OUTPUT_PATH)
    print(" Index built successfully!")

if __name__ == '__main__':
    build_index()