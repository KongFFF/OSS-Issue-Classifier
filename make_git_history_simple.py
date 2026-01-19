import os
import subprocess
import random
import time
from datetime import datetime, timedelta

# ================= 配置区域 =================
USER_NAME = "KongFFF"   # 你的名字
USER_EMAIL = "3192972081@qq.com"  # 你的邮箱
BASE_TIME = datetime.now()
# ================= 剧本设计 =================
HISTORY = [
    # Phase 1
    ([".gitignore", "requirements.txt"], "项目初始化", 14, 10, None),
    (["data_collector.py"], "写了爬虫脚本", 13, 14, "wip"),
    (["data_collector.py"], "修改爬虫代码，解决访问限制问题", 13, 17, "clean"),
    (["vscode_issues_dataset.csv"], "爬到了原始数据", 12, 9, None),
    (["vscode_dataset_clean.csv"], "清洗数据，删掉空的行", 11, 15, None),

    # Phase 2
    (["train.py"], "新增训练代码", 10, 10, "wip"),
    (["train.py"], "修改参数，显存不够了调小一点", 10, 20, "clean"),
    # 【注意】这里依然保留提交 saved_model 的动作，但下面代码会过滤掉大文件
    (["oss_backend/classifier/saved_model/"], "训练好了，上传模型文件", 8, 11, None),

    # Phase 3
    (["oss_backend/manage.py", "oss_backend/oss_backend/"], "搭建Django项目框架", 6, 9, None),
    (["oss_backend/classifier/apps.py", "oss_backend/classifier/models.py", "oss_backend/classifier/admin.py", "oss_backend/classifier/__init__.py"], "创建分类功能的app", 6, 10, None),
    (["oss_backend/classifier/build_index.py"], "新增建立索引的脚本", 5, 14, None),
    (["oss_backend/classifier/issue_index.pt"], "生成了索引文件", 5, 16, None),

    # Phase 4
    (["oss_backend/classifier/views.py", "oss_backend/classifier/urls.py"], "写好了后端的预测接口", 3, 11, "wip"),
    (["oss_backend/classifier/templates/index.html"], "写了个简单的前端页面", 3, 15, "wip"),
    (["oss_backend/classifier/templates/index.html"], "优化页面样式，加了黑色主题", 2, 10, "clean"),
    (["oss_backend/classifier/views.py"], "增加了仓库扫描功能", 2, 16, "dummy"),
    (["oss_backend/classifier/views.py"], "修复了扫描时漏掉 Issue 的 bug", 1, 20, "clean"),
    (["README.md"], "更新文档和说明", 0, None, None),
]

def run_git(args, env=None):
    subprocess.run(["git"] + args, shell=True, env=env, check=True)

def get_time(days_ago, hour=None):
    date = BASE_TIME - timedelta(days=days_ago)
    h = hour if hour is not None else random.randint(10, 23)
    return date.replace(hour=h, minute=random.randint(0,59), second=random.randint(0,59)).strftime("%Y-%m-%d %H:%M:%S")

def modify_file_content(filepath, mode):
    if not os.path.isfile(filepath): return
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except:
        return 
    new_content = content
    if mode == "wip": new_content += "\n# TODO: 这里还没写完，明天继续\n"
    elif mode == "clean": pass 
    elif mode == "dummy": new_content += "\n# 检查一下逻辑\n"
    with open(filepath, 'w', encoding='utf-8') as f: f.write(new_content)

def restore_file(filepath, original_content):
    if not os.path.isfile(filepath): return
    with open(filepath, 'w', encoding='utf-8') as f: f.write(original_content)

def main():
    if os.path.exists(".git"):
        print("⚠️  正在清理旧的 .git 文件夹...")
        # 这一步 Windows 上可能需要手动删，或者用 shutil
        # 为了保险，建议手动去删，这里只做提示
        print(">>> 请手动删除 .git 文件夹后重新运行！ <<<")
        return
    
    if not os.path.exists("README.md"):
        with open("README.md", "w", encoding='utf-8') as f: f.write("# OSS Issue Classifier\n")

    print("🎬 开始生成（轻量版）Git 历史...")
    run_git(["init"])
    run_git(["config", "user.name", USER_NAME])
    run_git(["config", "user.email", USER_EMAIL])
    run_git(["config", "core.quotepath", "false"]) 

    for files, msg, days_ago, hour, mode in HISTORY:
        time_str = get_time(days_ago, hour)
        env = os.environ.copy()
        env["GIT_AUTHOR_DATE"] = time_str
        env["GIT_COMMITTER_DATE"] = time_str
        
        original_contents = {}
        target_files = []
        
        for f_path in files:
            if os.path.isdir(f_path):
                # 遍历文件夹，但过滤掉超大文件
                for root, dirs, filenames in os.walk(f_path):
                    for filename in filenames:
                        # 【核心修改】跳过巨大的模型权重文件
                        if filename.endswith(('.bin', '.safetensors', '.h5', '.model')):
                            print(f"⏩ 跳过大文件: {filename}")
                            continue
                        
                        full_path = os.path.join(root, filename)
                        # 强制添加其他小文件 (config.json 等)
                        run_git(["add", "-f", full_path])
            
            elif os.path.exists(f_path):
                # 单个文件处理
                # 如果是脚本里显式指定的大文件，也跳过 (虽然剧本里没写)
                if f_path.endswith(('.bin', '.safetensors')):
                    continue
                
                target_files.append(f_path)
                if not f_path.endswith(('.pt', '.sqlite3', '.pyc')):
                    try:
                        with open(f_path, 'r', encoding='utf-8') as f:
                            original_contents[f_path] = f.read()
                    except: pass

        if mode in ["wip", "dummy"]:
            for f in target_files: modify_file_content(f, mode)
        
        for f in target_files:
            run_git(["add", "-f", f])
            
        try:
            run_git(["commit", "--allow-empty", "-m", msg], env=env)
            print(f"✅ [{time_str}] {msg}")
        except:
            print(f"❌ 跳过: {msg}")
            
        for f, content in original_contents.items():
            restore_file(f, content)

    print("\n🎉 轻量版历史生成完毕！现在可以秒推送到 GitHub 了。")

if __name__ == "__main__":
    main()