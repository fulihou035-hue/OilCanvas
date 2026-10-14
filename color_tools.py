"""Fast adjustments applied to the unchanged model output, never cumulatively."""
import numpy as np
from PIL import Image, ImageEnhance

def adjust_colors(image, exposure=0.0, shadows=0.0, saturation=1.0):
    if not -1.5 <= float(exposure) <= 1.5 or not 0 <= float(shadows) <= 1 or not 0 <= float(saturation) <= 2:
        raise ValueError('调色参数超出范围。')
    rgb = np.asarray(image.convert('RGB'),dtype=np.float32)/255
    rgb = rgb * (2.0**float(exposure))
    gray = rgb @ np.array([0.2126,0.7152,0.0722],dtype=np.float32)
    rgb += (float(shadows)*0.22*(1-np.clip(gray,0,1))**2)[...,None]
    adjusted = Image.fromarray(np.rint(np.clip(rgb,0,1)*255).astype('uint8'))
    return ImageEnhance.Color(adjusted).enhance(float(saturation))
