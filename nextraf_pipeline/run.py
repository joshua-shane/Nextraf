"""Command line: run a detector over a clip or a folder of images and write results/.

  python -m nextraf_pipeline.run --input clip.mp4 --model yolo11n.pt --out results/
  python -m nextraf_pipeline.run --input images/ --model yolo11n.pt --out results/

Needs: pip install -r nextraf_pipeline/requirements.txt (OpenCV + Ultralytics). Not needed by the viewer.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .core import Detection, run_pipeline

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


class UltralyticsDetector:
    """Wraps an Ultralytics model (.pt, .onnx, or an exported folder). Class names come from the model itself."""

    def __init__(self, model_path: str, imgsz: int = 640, conf_floor: float = 0.05):
        from ultralytics import YOLO  # imported lazily: heavy, and absent on the viewer
        self.model = YOLO(model_path)
        self.name = Path(model_path).stem
        self.imgsz, self.conf_floor = imgsz, conf_floor
        self.last_model_ms: float | None = None

    def detect(self, frame) -> list[Detection]:
        res = self.model.predict(frame, imgsz=self.imgsz, conf=self.conf_floor, verbose=False)[0]
        names = res.names
        self.last_model_ms = float(res.speed["inference"])
        return [Detection(names[int(c)], float(p), tuple(map(float, b)))
                for b, p, c in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(), res.boxes.cls.tolist())]


def _frames(path: Path):
    """Return (frames iterator, total, fps)."""
    import cv2
    if path.is_dir():
        files = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_EXT)
        if not files:
            raise SystemExit(f"No images found in {path}")
        return (cv2.imread(str(p)) for p in files), len(files), 1.0
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {path}")
    total, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS) or 30.0

    def gen():
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
        cap.release()
    return gen(), total, fps


def _save_annotated(frame, dets, path: Path) -> None:
    import cv2
    img = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = map(int, d.box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (173, 92, 11), 2)
        cv2.putText(img, f"{d.label} {d.confidence:.2f}", (x1, max(12, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (173, 92, 11), 1)
    cv2.imwrite(str(path), img)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="video file or folder of images")
    ap.add_argument("--model", required=True, help="model path, e.g. yolo11n.pt")
    ap.add_argument("--out", default="results/")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--samples", type=int, default=3, help="annotated images to save (the task needs >= 2)")
    ap.add_argument("--mode", choices=["validation", "deployment"], default="validation",
                    help="deployment saves no images")
    a = ap.parse_args()
    frames, total, fps = _frames(Path(a.input))
    det = UltralyticsDetector(a.model, a.imgsz)
    s = run_pipeline(frames, det, a.out, fps=fps, threshold=a.threshold, annotate_n=a.samples, total=total,
                     mode=a.mode, save_annotated=_save_annotated)
    print(f"model {s['model']} | {s['frames']} frames | {s['total_people']} people total | "
          f"{s['avg_inference_ms']} ms/frame | {s['throughput_fps']} FPS")
    print(f"wrote {Path(a.out) / 'counts.csv'}, samples/, inference_summary.json")


if __name__ == "__main__":
    main()
