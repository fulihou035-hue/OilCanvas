import unittest
import numpy as np
from PIL import Image
from engine import prepare_images, content_box
from color_tools import adjust_colors

class UpgradeTests(unittest.TestCase):
    def test_portrait_padding_preserves_content_ratio(self):
        src = Image.new('RGB',(400,600),'red')
        model,_ = prepare_images(src,Image.new('RGB',(512,512)),512)
        result = model.crop(content_box(src,512,'保持完整构图'))
        self.assertEqual(result.size,(341,512))
        self.assertLess(abs(result.width/result.height-src.width/src.height),0.002)

    def test_no_op_colors_are_exact(self):
        image = Image.fromarray(np.random.default_rng(1).integers(0,256,(16,16,3),dtype=np.uint8))
        self.assertTrue(np.array_equal(np.asarray(image),np.asarray(adjust_colors(image))))

    def test_shadow_control_preserves_highlights(self):
        image = Image.new('RGB',(2,1),'white')
        image.putpixel((0,0),(0,0,0))
        adjusted = adjust_colors(image,shadows=1)
        self.assertGreater(adjusted.getpixel((0,0))[0],0)
        self.assertEqual(adjusted.getpixel((1,0)),(255,255,255))

    def test_negative_exposure_reduces_brightness(self):
        self.assertLess(adjust_colors(Image.new('RGB',(2,2),(128,128,128)),exposure=-1).getpixel((0,0))[0],128)

    def test_invalid_adjustment_rejected(self):
        with self.assertRaises(ValueError):
            adjust_colors(Image.new('RGB',(2,2)),exposure=10)
