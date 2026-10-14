"""Double-click launcher, including reuse of an already running local APP."""
from pathlib import Path
import json
import subprocess
import sys
import webbrowser
from urllib.request import build_opener, ProxyHandler

def app_is_running(url='http://127.0.0.1:7860'):
    try:
        with build_opener(ProxyHandler({})).open(url,timeout=1.5) as response:
            return response.status == 200 and b'OilCanvas' in response.read(200000)
    except OSError:
        return False

def main():
    root = Path(__file__).resolve().parent
    if app_is_running():
        webbrowser.open('http://127.0.0.1:7860')
        return 0
    python = root/'.venv'/'Scripts'/'python.exe'
    config = root/'local_config.json'
    if not python.exists() and config.exists():
        python = Path(json.loads(config.read_text(encoding='utf-8'))['python'])
    if not python.exists():
        if sys.version_info[:2] not in ((3,10),(3,11)):
            print('尚未找到项目环境。请按README使用Python 3.10或3.11创建.venv并安装依赖。')
            return 1
        python = Path(sys.executable)
    return subprocess.call([str(python),str(root/'app.py'),'--browser'],cwd=root)

if __name__ == '__main__':
    raise SystemExit(main())

