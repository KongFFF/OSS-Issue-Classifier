import os
import torch
import requests
import torch.nn.functional as F
from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#全局加载 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'saved_model')
INDEX_PATH = os.path.join(BASE_DIR, 'issue_index.pt')

print(" Initializing System...")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
tokenizer = None
issue_index = None
issue_metadata = None

try:
    print(f"1. Loading Model from {MODEL_PATH}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.to(device)
    model.eval()
    
    print(f"2. Loading Index from {INDEX_PATH}...")
    if os.path.exists(INDEX_PATH):
        data = torch.load(INDEX_PATH, map_location=device)
        issue_index = data['embeddings'].to(device) # [N, 768]
        issue_metadata = data['metadata']           # List[Dict]
        print(f"   - Index size: {len(issue_metadata)} documents")
    else:
        print("⚠️ Warning: Index file not found. Similarity search disabled.")

    print(" System Ready!")
except Exception as e:
    print(f" Error during initialization: {e}")



def index(request):
    return render(request, 'index.html')

def predict(request):
    if request.method == 'POST':
        text = request.POST.get('text', '')
        if not text:
            return JsonResponse({'error': 'No text provided'}, status=400)
        
        if model is None:
            return JsonResponse({'error': 'Model not loaded'}, status=500)

        # 1. 预处理
        inputs = tokenizer(
            text, 
            return_tensors="pt", 
            truncation=True, 
            padding=True, 
            max_length=128
        ).to(device)

        with torch.no_grad():
            # 2. 推理：同时获取 logits (分类) 和 hidden_states (向量)
            outputs = model(**inputs, output_hidden_states=True)
            
            # --- 分类逻辑 ---
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            pred_idx = torch.argmax(probs).item()
            conf = probs[0][pred_idx].item()
            
            labels = {0: 'Bug 🐛', 1: 'Feature ✨'}
            pred_label = labels.get(pred_idx, 'Unknown')

            # --- 相似检索逻辑 ---
            similar_issues = []
            if issue_index is not None:
                # 提取当前输入的向量 (CLS token)
                last_hidden_state = outputs.hidden_states[-1]
                query_embedding = last_hidden_state[:, 0, :]
                query_embedding = F.normalize(query_embedding, p=2, dim=1)
                
                # 计算相似度 (Matrix Multiplication)
                # [1, 768] x [768, N] = [1, N]
                scores = torch.mm(query_embedding, issue_index.t())
                scores = scores.squeeze(0)
                
                # 取 Top 3
                topk_scores, topk_indices = torch.topk(scores, k=3)
                
                for score, idx in zip(topk_scores, topk_indices):
                    meta = issue_metadata[idx.item()]
                    similar_issues.append({
                        'title': meta['title'],
                        'url': meta['url'],
                        'score': f"{score.item()*100:.1f}%"
                    })

        return JsonResponse({
            'prediction': pred_label,
            'confidence': f"{conf*100:.2f}%",
            'similar_issues': similar_issues
        })
    
    return JsonResponse({'error': 'GET not allowed'}, status=405)



#新增：批量扫描逻辑

# 填入 GitHub Token
GITHUB_TOKEN = os.environ.get('GITHUB_TOKEN', '')
HEADERS = {'Authorization': f'token {GITHUB_TOKEN}'}

def scan_repo(request):
    """一键扫描仓库最新 Issue (智能过滤 PR 版)"""
    if request.method == 'POST':
        repo_name = request.POST.get('repo_name', '').strip()
        if 'github.com/' in repo_name:
            repo_name = repo_name.split('github.com/')[-1]
        
        if not repo_name:
            return JsonResponse({'error': 'Please provide a repository name'}, status=400)

        # 核心修改 1: 将 per_page 设为 100，确保能“捞”到被 PR 淹没的 Issue
        # 即使只有 12 个 Issue，这样也能把它们全包进来
        api_url = f"https://api.github.com/repos/{repo_name}/issues?state=open&per_page=100"
        
        try:
            resp = requests.get(api_url, headers=HEADERS)
            if resp.status_code != 200:
                return JsonResponse({'error': f'GitHub API Error: {resp.status_code}'}, status=resp.status_code)
            
            raw_issues = resp.json()
        except Exception as e:
            return JsonResponse({'error': str(e)}, status=500)

        results = []
        target_count = 10  # 修改 2: 展示 10 条有效数据
        
        # 3. 遍历筛选
        for item in raw_issues:
            # 如果已经凑够了 10 条，就直接停止，节省时间
            if len(results) >= target_count:
                break

            # 跳过 PR
            if 'pull_request' in item:
                continue
                
            # 拼接文本
            text = item['title'] + " " + (item['body'] or "")
            
            # --- AI 分析逻辑 (保持不变) ---
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1)
                pred_idx = torch.argmax(probs).item()
                conf = probs[0][pred_idx].item()
                
                # 相似度检测
                is_duplicate = False
                duplicate_info = ""
                if issue_index is not None:
                    last_hidden_state = outputs.hidden_states[-1]
                    query_embedding = F.normalize(last_hidden_state[:, 0, :], p=2, dim=1)
                    scores = torch.mm(query_embedding, issue_index.t()).squeeze(0)
                    best_score, best_idx = torch.topk(scores, k=1)
                    if best_score.item() > 0.85:
                        is_duplicate = True
                        meta = issue_metadata[best_idx.item()]
                        duplicate_info = f"{meta['title']} ({best_score.item()*100:.1f}%)"

            results.append({
                'number': item['number'],
                'title': item['title'],
                'url': item['html_url'],
                'type': 'Bug 🐛' if pred_idx == 0 else 'Feature ✨',
                'confidence': f"{conf*100:.0f}%",
                'is_duplicate': is_duplicate,
                'duplicate_info': duplicate_info
            })
            
        return JsonResponse({'results': results, 'repo': repo_name})

    return JsonResponse({'error': 'Method not allowed'}, status=405)
# 检查一下逻辑
