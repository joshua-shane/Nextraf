"""Frame sources: a single image, a folder of images, or a video file."""
import os

import cv2

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def iter_frames(source, stride=1):
    """Yield (name, frame_index, t_sec, frame). t_sec is video time (None for images)."""
    if os.path.isdir(source):
        files = sorted(f for f in os.listdir(source) if os.path.splitext(f)[1].lower() in IMAGE_EXT)
        for i, f in enumerate(files):
            img = cv2.imread(os.path.join(source, f))
            if img is not None:
                yield f, i, None, img
        return
    if not os.path.exists(source):
        raise FileNotFoundError(f"Source not found: {source}")
    if os.path.splitext(source)[1].lower() in IMAGE_EXT:
        img = cv2.imread(source)
        if img is None:
            raise ValueError(f"Could not read image: {source}")
        yield os.path.basename(source), 0, None, img
        return
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if i % stride == 0:
                yield os.path.basename(source), i, i / fps, frame
    finally:
        cap.release()
