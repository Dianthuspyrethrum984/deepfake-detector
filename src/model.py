import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.models import Model

class DeepfakeDetector:
    def __init__(self, input_shape=(224, 224, 3), lr=0.0001):
        self.input_shape = input_shape
        self.lr = lr
        self.model = None

    def build_model(self):
        base = EfficientNetB0(weights="imagenet", include_top=False, input_shape=self.input_shape)
        base.trainable = False
        x = GlobalAveragePooling2D()(base.output)
        x = Dropout(0.3)(x)
        x = Dense(256, activation="relu")(x)
        x = Dropout(0.2)(x)
        out = Dense(1, activation="sigmoid")(x)
        self.model = Model(inputs=base.input, outputs=out)
        self.model.compile(optimizer=keras.optimizers.Adam(self.lr), loss="binary_crossentropy", metrics=["accuracy"])
        return self.model

    def predict_single(self, image):
        if not self.model: raise RuntimeError("Model not loaded")
        prob = float(self.model.predict(np.expand_dims(image / 255.0, 0), verbose=0)[0][0])
        return {"label": "FAKE" if prob > 0.5 else "REAL", "fake_probability": prob, "confidence": abs(prob - 0.5) * 2}

    def grad_cam(self, image, layer_name="top_conv"):
        img = np.expand_dims(image / 255.0, 0)
        grad_model = Model(self.model.inputs, [self.model.get_layer(layer_name).output, self.model.output])
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img)
        grads = tape.gradient(preds, conv_out)
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(conv_out * weights[..., tf.newaxis, tf.newaxis, :], axis=-1)[0]
        cam = np.maximum(cam.numpy(), 0)
        return cam / (cam.max() + 1e-8)

    def save(self, path): self.model.save(path) if self.model else None
    def load(self, path): self.model = keras.models.load_model(path)
