import huggingface_hub, datasets
from huggingface_hub import snapshot_download

print(huggingface_hub.__version__, datasets.__version__)

print(snapshot_download(repo_id='fancyzhx/ag_news', repo_type='dataset', local_dir=r'.\data\ag_news'))

print(snapshot_download(repo_id='dstefa/New_York_Times_Topics', repo_type='dataset', local_dir=r'.\data\NYT'))
