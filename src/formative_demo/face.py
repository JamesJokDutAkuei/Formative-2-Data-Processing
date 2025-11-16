"""Face recognition helpers for the Formative demo CLI."""
from __future__ import annotations

from typing import Iterable, Optional, Tuple

import numpy as np

try:
    from tensorflow.keras.applications import MobileNetV2  # type: ignore[import]
    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input  # type: ignore[import]
    from tensorflow.keras.models import load_model  # type: ignore[import]
    from tensorflow.keras.preprocessing import image as keras_image  # type: ignore[import]
except Exception:  # pragma: no cover - tensorflow is optional at runtime
    MobileNetV2 = None
    preprocess_input = None
    keras_image = None
    load_model = None

from .paths import MODEL_DIR

AUTHENTICATION_THRESHOLD = 70.0


def extract_image_embedding(img_path: str) -> np.ndarray:
    if MobileNetV2 is None:
        raise RuntimeError("TensorFlow/Keras not available in this environment")

    base = MobileNetV2(include_top=False, weights="imagenet", pooling="avg")
    img = keras_image.load_img(img_path, target_size=(224, 224))
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    emb = base.predict(x)
    return emb.reshape(-1)


def load_face_model():
    model_path = MODEL_DIR / "face_authentication_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Face model not found: {model_path}")
    if load_model is None:
        raise RuntimeError("TensorFlow/Keras not available to load face model")

    from tensorflow.keras import Sequential  # type: ignore[import]
    from tensorflow.keras.layers import (  # type: ignore[import]
        RandomFlip,
        RandomRotation,
        RandomZoom,
        Lambda,
        GlobalAveragePooling2D,
        Dropout,
        Dense,
    )

    if preprocess_input is None:
        raise RuntimeError("TensorFlow/Keras preprocess_input unavailable for face model")

    data_augmentation = Sequential(
        [
            RandomFlip("horizontal", name="random_flip_5"),
            RandomRotation(0.1, name="random_rotation_5"),
            RandomZoom(0.1, name="random_zoom_5"),
        ],
        name="sequential_12",
    )

    base_model = MobileNetV2(
        include_top=False,
        weights="imagenet",
        pooling=None,
        input_shape=(224, 224, 3),
    )

    full_model = Sequential(
        [
            data_augmentation,
            Lambda(preprocess_input, name="lambda_7"),
            base_model,
            GlobalAveragePooling2D(name="global_average_pooling2d_7"),
            Dropout(0.5, name="dropout_7"),
            Dense(5, name="dense_7"),
        ],
        name="face_authentication_model",
    )

    full_model.build((None, 224, 224, 3))
    full_model.load_weights(str(model_path))

    dense_weights = full_model.get_layer("dense_7").get_weights()

    head_model = Sequential(
        [
            Dropout(0.5, name="dropout_7"),
            Dense(5, name="dense_7"),
        ],
        name="face_authentication_head",
    )
    head_model.build((None, 1280))
    head_model.get_layer("dense_7").set_weights(dense_weights)
    return head_model


def predict_face(face_model, img_path: str, class_names: Optional[Iterable[str]] = None) -> Tuple[bool, str, float]:
    input_shape = getattr(face_model, "input_shape", None)
    if input_shape and len(input_shape) == 4 and input_shape[-1] == 3:
        if keras_image is None:
            raise RuntimeError("TensorFlow/Keras image utilities unavailable")
        target_h = input_shape[1] or 224
        target_w = input_shape[2] or 224
        img = keras_image.load_img(img_path, target_size=(target_h, target_w))
        x = keras_image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        probs = face_model.predict(x)
    else:
        emb = extract_image_embedding(img_path)
        x = emb.reshape(1, -1)
        probs = face_model.predict(x)

    logits = np.asarray(probs).reshape(-1)
    logits = logits.astype(np.float64)
    logits -= np.max(logits)
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits)
    top_idx = int(np.argmax(probs))
    top_prob = float(probs[top_idx]) * 100.0
    if class_names:
        try:
            name = class_names[top_idx]
        except Exception:
            name = str(top_idx)
    else:
        name = str(top_idx)
    passed = top_prob >= AUTHENTICATION_THRESHOLD
    return passed, name, top_prob
