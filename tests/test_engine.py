import unittest
from PIL import Image
from engine import prepare_images, stylize, LOCK

class ImagePreparationTests(unittest.TestCase):
    def test_landscape_keeps_complete_frame(self):
        photo, style = prepare_images(Image.new('RGB', (1824, 1368)), Image.new('RGB', (900, 700)), 512)
        self.assertEqual(photo.size, (512, 384))
        self.assertEqual(style.size, photo.size)

    def test_square_crop_and_rgb_conversion(self):
        photo, style = prepare_images(Image.new('RGBA', (1200, 800)), Image.new('L', (640, 480)), 768, '中心裁剪正方形')
        self.assertEqual(photo.size, (768, 768))
        self.assertEqual((photo.mode, style.mode), ('RGB', 'RGB'))

    def test_portrait_dimension_alignment(self):
        photo, _ = prepare_images(Image.new('RGB', (500, 1200)), Image.new('RGB', (700, 900)), 768)
        self.assertEqual(max(photo.size), 768)
        self.assertTrue(all(value % 64 == 0 for value in photo.size))

    def test_missing_and_small_inputs_rejected(self):
        for photo in (None, Image.new('RGB', (32, 32))):
            with self.assertRaises(ValueError):
                prepare_images(photo, Image.new('RGB', (512, 512)))

    def test_invalid_inference_parameters_rejected_before_gpu(self):
        with self.assertRaises(ValueError):
            stylize(None, None, gamma=1.1)

if __name__ == '__main__':
    unittest.main()
