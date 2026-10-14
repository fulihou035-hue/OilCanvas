"""Download fp16 weights once; inference then runs fully offline."""
from pathlib import Path
import os
import requests

root = Path(__file__).resolve().parent
os.environ['HF_HOME'] = str(root / 'models' / 'cache')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['HF_HUB_DISABLE_XET'] = '1'
os.environ['HF_HUB_ETAG_TIMEOUT'] = '30'
os.environ['HF_HUB_DOWNLOAD_TIMEOUT'] = '60'
from huggingface_hub import snapshot_download, configure_http_backend

def backend():
    client = requests.Session()
    client.trust_env = False
    return client

if __name__ == '__main__':
    configure_http_backend(backend_factory=backend)
    snapshot_download('stable-diffusion-v1-5/stable-diffusion-v1-5',
        local_dir=str(root / 'models' / 'sd15'), max_workers=3,
        allow_patterns=['model_index.json', 'scheduler/*.json', 'unet/config.json',
            'unet/diffusion_pytorch_model.fp16.safetensors', 'vae/config.json',
            'vae/diffusion_pytorch_model.fp16.safetensors', 'text_encoder/config.json',
            'text_encoder/model.fp16.safetensors', 'tokenizer/*', 'README.md'])
    print('模型已就绪。可运行 python app.py --browser。')

