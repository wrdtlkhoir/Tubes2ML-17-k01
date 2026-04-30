import numpy as np
from PIL import Image
import os
from typing import List, Tuple
import tensorflow as tf
from tensorflow import keras


def load_image(image_path: str, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    img = Image.open(image_path).convert('RGB')
    img = img.resize(target_size, Image.BILINEAR)
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    return img_array


def load_batch(image_paths: List[str], target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    images = []
    for path in image_paths:
        img = load_image(path, target_size)
        images.append(img)
    return np.array(images)


def extract_features(image_paths: List[str], output_path: str, model_name: str = 'InceptionV3', target_size: Tuple[int, int] = (299, 299), batch_size: int = 32) -> None:    
    if model_name == 'InceptionV3':
        base_model = keras.applications.InceptionV3(
            input_shape=(*target_size, 3),
            include_top=False,
            weights='imagenet'
        )
    elif model_name == 'VGG16':
        base_model = keras.applications.VGG16(
            input_shape=(*target_size, 3),
            include_top=False,
            weights='imagenet'
        )
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    base_model.trainable = False
    
    features_dict = {}
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = load_batch(batch_paths, target_size)
        
        batch_features = base_model.predict(batch_images, verbose=0)
        batch_features = batch_features.reshape(batch_features.shape[0], -1)
        
        for path, feature in zip(batch_paths, batch_features):
            filename = os.path.basename(path)
            features_dict[filename] = feature
    
    os.makedirs(output_path, exist_ok=True)
    
    for filename, feature in features_dict.items():
        feature_file = os.path.join(output_path, filename.replace('.jpg', '.npy').replace('.png', '.npy'))
        np.save(feature_file, feature)
