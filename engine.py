"""Thin local application layer around the MIT-licensed StyleID implementation."""
from pathlib import Path
import json
import os
import subprocess
import sys
import threading
import time
import uuid
import math
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()

def model_directory():
    override = os.getenv('OILCANVAS_MODEL_DIR')
    if override:
        return Path(override).resolve()
    config = ROOT / 'local_config.json'
    if config.exists():
        return Path(json.loads(config.read_text(encoding='utf-8'))['model_dir']).resolve()
    return ROOT / 'models' / 'sd15'

def prepare_images(photo, style, size=512, framing='保持完整构图'):
    if not isinstance(photo, Image.Image) or not isinstance(style, Image.Image):
        raise ValueError('请先上传照片，并选择或上传参考画作。')
    if size not in (512, 768):
        raise ValueError('输出长边只支持 512 或 768 像素。')
    photo = ImageOps.exif_transpose(photo).convert('RGB')
    style = ImageOps.exif_transpose(style).convert('RGB')
    if min(photo.size) < 64 or min(style.size) < 64:
        raise ValueError('图片过小，请使用边长至少 64 像素的图片。')
    if framing == '中心裁剪正方形':
        dimensions = (size, size)
        photo = ImageOps.fit(photo, dimensions, method=Image.Resampling.LANCZOS)
    else:
        ratio = size / max(photo.size)
        resized = tuple(max(1, round(value * ratio)) for value in photo.size)
        dimensions = tuple(math.ceil(value / 64) * 64 for value in resized)
        photo = photo.resize(resized, Image.Resampling.LANCZOS)
        left, top = (dimensions[0]-resized[0])//2, (dimensions[1]-resized[1])//2
        array = np.asarray(photo)
        photo = Image.fromarray(np.pad(array,((top,dimensions[1]-resized[1]-top),(left,dimensions[0]-resized[0]-left),(0,0)),mode='edge'))
    style = ImageOps.fit(style, dimensions, method=Image.Resampling.LANCZOS)
    return photo, style

def content_box(photo, size, framing):
    if not isinstance(photo, Image.Image):
        raise ValueError('请先上传照片。')
    if framing == '中心裁剪正方形':
        return (0,0,size,size)
    photo = ImageOps.exif_transpose(photo)
    ratio = size/max(photo.size)
    w,h = (max(1,round(value*ratio)) for value in photo.size)
    pw,ph = math.ceil(w/64)*64,math.ceil(h/64)*64
    x,y = (pw-w)//2,(ph-h)//2
    return (x,y,x+w,y+h)

def stylize(photo, style, gamma=0.95, size=512, steps=20,
            preserve_tones=True, framing='保持完整构图', progress=None):
    if not 0.5 <= float(gamma) <= 0.99 or int(steps) not in (20, 30):
        raise ValueError('参数超出支持范围。')
    from artistic_backend import release_models
    release_models()
    models = model_directory()
    if not (models / 'unet' / 'diffusion_pytorch_model.fp16.safetensors').exists():
        raise ValueError('尚未找到模型。请先运行模型准备脚本，或配置 local_config.json。')
    if not LOCK.acquire(blocking=False):
        raise ValueError('另一张图片正在生成，请稍后再试。')
    folder = ROOT / 'results' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    try:
        box = content_box(photo,int(size),framing)
        photo, style = prepare_images(photo, style, int(size), framing)
        folder.mkdir(parents=True)
        photo.save(folder / 'photo.png')
        style.save(folder / 'style.png')
        if progress:
            progress(0.15, desc='图片已准备，正在生成油画…')
        command = [sys.executable, str(ROOT / 'vendor' / 'styleid_runtime' / 'run_styleid_diffusers.py'),
            '--cnt_fn', str(folder / 'photo.png'), '--sty_fn', str(folder / 'style.png'),
            '--save_dir', str(folder), '--sd_version', '1.5', '--ddim_steps', str(int(steps)),
            '--gamma', str(float(gamma)), '--T', '1.5']
        if preserve_tones:
            command.append('--without_init_adain')
        environment = os.environ.copy()
        environment.update(OILCANVAS_MODEL_DIR=str(models), HF_HUB_OFFLINE='1',
                           TRANSFORMERS_OFFLINE='1', PYTHONUTF8='1')
        start = time.perf_counter()
        completed = subprocess.run(command, cwd=ROOT, env=environment, capture_output=True,
            encoding='utf-8', errors='replace', timeout=300)
        (folder / 'inference.log').write_text(completed.stdout + completed.stderr, encoding='utf-8')
        if completed.returncode:
            if 'out of memory' in completed.stderr.lower():
                raise ValueError('显存不足，请关闭其他 GPU 应用，或将长边改为 512 像素。')
            raise ValueError('生成失败，详细记录保存在结果文件夹的 inference.log。')
        stats = json.loads((folder / 'run_stats.json').read_text(encoding='utf-8'))
        model_size = photo.size
        photo = photo.crop(box)
        image = Image.open(folder / 'stylized_image.png').convert('RGB').crop(box)
        stats.update(app_seconds=round(time.perf_counter() - start, 2),
                     width=photo.width, height=photo.height, model_width=model_size[0],model_height=model_size[1],framing=framing,
                     preserve_tones=bool(preserve_tones), backend='StyleID / SD1.5')
        (folder / 'parameters.json').write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding='utf-8')
        image.save(folder / 'oil_painting.png')
        photo.save(folder / 'display_photo.png')
        if progress:
            progress(1, desc='生成完成')
        summary = f"完成：{photo.width}×{photo.height}，用时 {stats['app_seconds']} 秒。"
        summary += f"\n显存分配峰值 {stats['peak_gpu_allocated_mib']} MiB。结果已保存在本机。"
        return photo, image, str(folder / 'oil_painting.png'), str(folder / 'parameters.json'), summary
    except subprocess.TimeoutExpired:
        raise ValueError('生成超过 5 分钟，请改用 512 像素再试。')
    finally:
        LOCK.release()

def stylize_artistic(photo, style, size=512, steps=20, framing='保持完整构图', strength=0.55, style_scale=0.45, structure_scale=0.65, lighting='明亮日光', progress=None, texture='柔和'):
    from artistic_backend import render,artistic_directory
    assets = artistic_directory(ROOT)
    if not (model_directory()/'unet'/'diffusion_pytorch_model.fp16.safetensors').exists():
        raise ValueError('请先运行 prepare_model.py 准备基础模型。')
    if not (assets/'ip_adapter'/'models'/'image_encoder'/'model.safetensors').exists() or not (assets/'controlnet'/'diffusion_pytorch_model.fp16.safetensors').exists():
        raise ValueError('请先运行 prepare_artistic.py 准备新方案组件。')
    if texture == '厚涂' and not (assets/'ip_adapter'/'models'/'ip-adapter-plus_sd15.safetensors').exists():
        raise ValueError('请运行 prepare_artistic.py 准备细节画风组件。')
    if int(steps) not in (20,30):
        raise ValueError('处理步数只支持20或30。')
    if not LOCK.acquire(blocking=False):
        raise ValueError('另一张图片正在生成，请稍后再试。')
    start = time.perf_counter()
    folder = ROOT/'results'/(time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
    try:
        box = content_box(photo,int(size),framing)
        padded,_ = prepare_images(photo,style,int(size),framing)
        folder.mkdir(parents=True)
        padded.save(folder/'photo.png')
        style.convert('RGB').save(folder/'style.png')
        if progress:
            progress(0.15,desc='正在绘制油画，首次加载稍慢…')
        result,stats = render(padded,style,model_directory(),assets,steps=int(steps),strength=strength,style_scale=style_scale,structure_scale=structure_scale,retain=True,cache_reference=True,lighting=lighting,texture=texture,variant='detail' if texture=='厚涂' else 'standard')
        result.save(folder/'stylized_image.png')
        result = result.crop(box)
        original = padded.crop(box)
        original.save(folder/'display_photo.png')
        result.save(folder/'oil_painting.png')
        stats.update(app_seconds=round(time.perf_counter()-start,2),width=result.width,height=result.height,model_width=padded.width,model_height=padded.height,framing=framing)
        (folder/'parameters.json').write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding='utf-8')
        if progress:
            progress(1,desc='生成完成')
        status = f"完成：{result.width}×{result.height}，用时{stats['app_seconds']}秒。\n绘画重绘可能改变原图细节。"
        return original,result,str(folder/'oil_painting.png'),str(folder/'parameters.json'),status
    except RuntimeError as error:
        if 'out of memory' in str(error).lower():
            from artistic_backend import release_models
            release_models()
            raise ValueError('显存不足，请关闭其他GPU应用，并改用512像素。') from error
        raise
    finally:
        LOCK.release()

