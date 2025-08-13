import os
import cv2
import numpy as np

def load_images(path: str, mode: str='RGB') -> np.ndarray:
    
    conversion_attr = f"COLOR_BGR2{mode}"
    conversion_code = getattr(cv2, conversion_attr)
    
    images = []
    for file in os.listdir(path):
        if file.lower().endswith('.jpg'):
            img_path = os.path.join(path, file)
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                continue
            img = cv2.cvtColor(img_bgr, conversion_code)
            images.append(img)
            
    return np.array(images)


def convert_image(im: np.ndarray, mode: str='RGB') -> np.ndarray:
    if mode != 'RGB':
        conversion_code = getattr(cv2, f"COLOR_{mode}2RGB")
        im = cv2.cvtColor(im, conversion_code)
    return im