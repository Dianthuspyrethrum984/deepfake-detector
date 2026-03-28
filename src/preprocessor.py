import cv2
import numpy as np

class Preprocessor:
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size

    def preprocess(self, image, augment=False):
        img = cv2.resize(image, self.target_size)
        if augment:
            if np.random.random() > 0.5: img = cv2.flip(img, 1)
        return img.astype(np.float32) / 255.0

    def batch_preprocess(self, images, augment=False):
        return np.array([self.preprocess(img, augment) for img in images])
