import cv2
import numpy as np
import config as C

def enhance(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    avg_brightness = np.mean(gray)
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    if avg_brightness < 60:
        inv_gamma = 1.0 / 2.5
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
        l = clahe.apply(l)

    elif avg_brightness > 180:
        inv_gamma = 1.0 / 0.5
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

    else:
        inv_gamma = 1.0 / C.GAMMA
        table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8)
        frame = cv2.LUT(frame, table)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l = clahe.apply(l)

    frame = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    frame = cv2.bilateralFilter(frame, 5, 55, 55)
    return frame