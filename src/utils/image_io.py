from pathlib import Path
import cv2
import numpy as np

def load_image_bgr(image_path:str | Path) -> np.ndarray:
    #loads the image using open cv 
    image_path=Path(image_path)
    img=cv2.imread(str(image_path),cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not load image : {image_path}")
    return img

def resize_for_features(img_bgr:np.ndarray,size:tuple[int,int]=(512,512))->np.ndarray:
    return cv2.resize(img_bgr,size,interpolation=cv2.INTER_AREA)

def to_gray(img_bgr:np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr,cv2.COLOR_BGR2GRAY)

def to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

def to_hsv(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)