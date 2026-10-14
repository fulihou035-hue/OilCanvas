"""Prepare public IP-Adapter and Canny ControlNet components; resume downloads."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import requests

ROOT = Path(__file__).resolve().parent
config = ROOT/'local_config.json'
settings = json.loads(config.read_text(encoding='utf-8')) if config.exists() else {}
TARGET = Path(os.getenv('OILCANVAS_ARTISTIC_DIR',settings.get('artistic_dir',ROOT/'models'/'artistic')))
FILES = [
 ('h94/IP-Adapter','models/image_encoder/config.json','ip_adapter/models/image_encoder/config.json'),
 ('h94/IP-Adapter','models/image_encoder/model.safetensors','ip_adapter/models/image_encoder/model.safetensors'),
 ('h94/IP-Adapter','models/ip-adapter_sd15.safetensors','ip_adapter/models/ip-adapter_sd15.safetensors'),
 ('h94/IP-Adapter','models/ip-adapter-plus_sd15.safetensors','ip_adapter/models/ip-adapter-plus_sd15.safetensors'),
 ('h94/IP-Adapter','README.md','ip_adapter/README.md'),
 ('lllyasviel/control_v11p_sd15_canny','config.json','controlnet/config.json'),
 ('lllyasviel/control_v11p_sd15_canny','diffusion_pytorch_model.fp16.safetensors','controlnet/diffusion_pytorch_model.fp16.safetensors'),
 ('lllyasviel/control_v11p_sd15_canny','README.md','controlnet/README.md'),
]
def fetch(item):
    repo,remote,local = item
    dest = TARGET/local
    if dest.exists():
        return
    dest.parent.mkdir(parents=True,exist_ok=True)
    partial = dest.with_suffix(dest.suffix+'.part')
    offset = partial.stat().st_size if partial.exists() else 0
    client = requests.Session()
    client.trust_env = False
    headers = {'Range':f'bytes={offset}-'} if offset else {}
    with client.get(f'https://hf-mirror.com/{repo}/resolve/main/{remote}',headers=headers,stream=True,timeout=(15,45)) as response:
        response.raise_for_status()
        if offset and response.status_code != 206:
            offset = 0
        written = offset
        next_log = written+128*1024**2
        with partial.open('ab' if offset else 'wb') as handle:
            for block in response.iter_content(1024*1024):
                handle.write(block)
                written += len(block)
                if written >= next_log:
                    print(f'{local}: {written//1024**2} MiB',flush=True)
                    next_log += 128*1024**2
    partial.replace(dest)
    print('READY '+local,flush=True)
if __name__ == '__main__':
    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(fetch,FILES))
    print('Artistic models ready: '+str(TARGET),flush=True)
