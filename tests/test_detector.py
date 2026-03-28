import unittest
import numpy as np
from src.preprocessor import Preprocessor

class TestPreprocessor(unittest.TestCase):
    def test_shape(self):
        pp = Preprocessor((224, 224))
        img = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        self.assertEqual(pp.preprocess(img).shape, (224, 224, 3))
    def test_normalize(self):
        pp = Preprocessor()
        img = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        self.assertLessEqual(pp.preprocess(img).max(), 1.0)

if __name__ == "__main__": unittest.main()
