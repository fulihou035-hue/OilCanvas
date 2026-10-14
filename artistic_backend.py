"""Image-guided painting with SD1.5, IP-Adapter and Canny ControlNet.

Model cache and reference embeddings are opt-in until visual validation.
Official API reference: https://huggingface.co/docs/diffusers/v0.35.1/using-diffusers/ip_adapter
"""
from pathlib import Path
import hashlib
import json
import os
import threading
import time
import numpy as np
from PIL import Image

_PIPE = None
_MODEL_KEY = None
_EMBED_CACHE = {}
_LOCK = threading.Lock()

def artistic_directory(root):
    config = root/'local_config.json'
    settings = json.loads(config.read_text(encoding='utf-8')) if config.exists() else {}
    return Path(os.getenv('OILCANVAS_ARTISTIC_DIR',settings.get('artistic_dir',root/'models'/'artistic')))

def load_pipeline(base, assets, retain=False, variant='standard'):
    global _PIPE, _MODEL_KEY
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel, StableDiffusionControlNetImg2ImgPipeline, DPMSolverMultistepScheduler
    from transformers import CLIPTextModel, CLIPTokenizer, CLIPVisionModelWithProjection, CLIPImageProcessor
    weight = 'ip-adapter-plus_sd15.safetensors' if variant == 'detail' else 'ip-adapter_sd15.safetensors'
    key = (str(base.resolve()),str(assets.resolve()),weight)
    if retain and _PIPE is not None and _MODEL_KEY == key:
        return _PIPE,True
    if _PIPE is not None and _MODEL_KEY != key:
        _PIPE,_MODEL_KEY = None,None
        _EMBED_CACHE.clear()
        torch.cuda.empty_cache()
    opts = dict(torch_dtype=torch.float16,variant='fp16',use_safetensors=True,local_files_only=True)
    control = ControlNetModel.from_pretrained(str(assets/'controlnet'),**opts)
    vision = CLIPVisionModelWithProjection.from_pretrained(str(assets/'ip_adapter'/'models'/'image_encoder'),torch_dtype=torch.float16,local_files_only=True)
    pipe = StableDiffusionControlNetImg2ImgPipeline(
        vae=AutoencoderKL.from_pretrained(str(base),subfolder='vae',**opts),
        text_encoder=CLIPTextModel.from_pretrained(str(base),subfolder='text_encoder',**opts),
        tokenizer=CLIPTokenizer.from_pretrained(str(base),subfolder='tokenizer',local_files_only=True),
        unet=UNet2DConditionModel.from_pretrained(str(base),subfolder='unet',**opts),
        controlnet=control,
        scheduler=DPMSolverMultistepScheduler.from_pretrained(str(base),subfolder='scheduler',local_files_only=True),
        safety_checker=None,feature_extractor=CLIPImageProcessor(),image_encoder=vision,requires_safety_checker=False,
    )
    pipe.load_ip_adapter(str(assets/'ip_adapter'),subfolder='models',weight_name=weight,image_encoder_folder=None,local_files_only=True)
    pipe.to('cuda')
    pipe.enable_vae_slicing()
    if retain:
        _PIPE,_MODEL_KEY = pipe,key
    return pipe,False

def release_models():
    global _PIPE,_MODEL_KEY
    with _LOCK:
        if _PIPE is not None:
            import torch
            _PIPE,_MODEL_KEY = None,None
            torch.cuda.empty_cache()

def render(photo, reference, base, assets, steps=20, strength=0.55, style_scale=0.45, structure_scale=0.65, seed=42, retain=False, cache_reference=False, lighting='明亮日光', texture='柔和', variant='standard'):
    import torch
    import cv2
    if not 0.2 <= float(strength) <= 0.8 or not 0 <= float(style_scale) <= 1 or not 0 <= float(structure_scale) <= 1:
        raise ValueError('画风、结构或重绘参数超出范围。')
    with _LOCK:
        started = time.perf_counter()
        torch.cuda.reset_peak_memory_stats()
        pipe,model_hit = load_pipeline(base,assets,retain,variant)
        loaded = time.perf_counter()
        rgb = np.asarray(photo.convert('RGB'))
        edges = cv2.Canny(rgb,100,200)
        control_image = Image.fromarray(np.repeat(edges[...,None],3,axis=2))
        pipe.set_ip_adapter_scale(float(style_scale))
        reference = reference.convert('RGB')
        key = hashlib.sha256(reference.tobytes()+str(reference.size).encode()+str(assets.resolve()).encode()+variant.encode()).hexdigest()
        hit = cache_reference and key in _EMBED_CACHE
        if hit:
            embeds = [value.to('cuda') for value in _EMBED_CACHE[key]]
        else:
            pipe.image_encoder.to('cuda')
            embeds = pipe.prepare_ip_adapter_image_embeds(reference,None,torch.device('cuda'),1,True)
            if cache_reference:
                if len(_EMBED_CACHE) >= 8:
                    _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
                _EMBED_CACHE[key] = [value.detach().cpu() for value in embeds]
        pipe.image_encoder.to('cpu')
        illumination = 'bright natural daylight' if lighting == '明亮日光' else 'preserve the source time of day and lighting, balanced gentle illumination'
        prompt = ('a luminous impressionist oil painting of the scene in the input photograph, '
                  +illumination+', clean vibrant colors, soft broad painted brushstrokes, '
                  'gentle transitions, refined painterly composition, beautiful light, oil on canvas')
        negative = ('embossed, engraving, harsh outlines, muddy colors, dark gloomy lighting, '
                    'dirty texture, monochrome, distorted architecture, deformed face, blurry')
        if texture == '厚涂':
            prompt = ('an expressive hand-painted oil painting of the original scene, '+illumination+
                      ', visible directional bristle brush marks, textured impasto oil pigments, '
                      'layered strokes of thick paint, painterly shapes, rich clean colors, oil on canvas')
            negative += ', airbrushed, smooth plastic, digital render, pixelated, low resolution, noise, text, signature, watermark'
        image = pipe(prompt=prompt,negative_prompt=negative,image=photo,control_image=control_image,
                     ip_adapter_image_embeds=embeds,strength=float(strength),num_inference_steps=int(steps),
                     guidance_scale=6,controlnet_conditioning_scale=float(structure_scale),
                     generator=torch.Generator(device='cuda').manual_seed(int(seed))).images[0]
        stats = dict(backend='SD1.5 + IP-Adapter + Canny ControlNet',seconds=round(time.perf_counter()-started,2),
                     load_seconds=round(loaded-started,2),model_cache_hit=model_hit,reference_cache_hit=bool(hit),
                     peak_gpu_allocated_mib=round(torch.cuda.max_memory_allocated()/1024**2),steps=int(steps),
                     strength=float(strength),style_scale=float(style_scale),structure_scale=float(structure_scale),seed=int(seed),
                     prompt=prompt,negative_prompt=negative,lighting=lighting,texture=texture,adapter_variant=variant)
        if not retain:
            del pipe
            torch.cuda.empty_cache()
        return image,stats
