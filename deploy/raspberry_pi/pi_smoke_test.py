"""NEXTRAF quick model test - run this FIRST on the Raspberry Pi 5, from the package folder.

    source .venv/bin/activate
    python pi_smoke_test.py                 # tests both models
    python pi_smoke_test.py --image x.jpg   # also runs one real image (optional)

It checks, in order:
  1. platform (expects Linux aarch64, Python >= 3.12) - warnings only
  2. imports of the runtime packages and their versions
  3. the ExecuTorch runtime and whether the XNNPACK backend is registered (the .pte files need it)
  4. each model folder: loads it and runs inference on a blank frame (expects 0 detections)
Exit code 0 = everything passed; 1 = something failed (read the FAIL lines and send them back).

Status: written and run on Windows 11 x64 only. It has NOT been run on a Raspberry Pi.
It needs no internet and writes no files.
"""
import argparse
import os
import platform
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)                 # model folders are relative to the package folder
sys.path.insert(0, HERE)

MODELS = [("yolo26n_executorch_model", 640), ("yolo26n_1280_executorch_model", 1280)]
failures, warnings = [], []


def ok(msg):
    print(f"  PASS  {msg}")


def warn(msg):
    warnings.append(msg)
    print(f"  WARN  {msg}")


def fail(msg, exc=None):
    failures.append(msg)
    print(f"  FAIL  {msg}")
    if exc is not None:
        print("        " + "".join(traceback.format_exception_only(type(exc), exc)).strip())


def main():
    ap = argparse.ArgumentParser(description="NEXTRAF Raspberry Pi quick model test")
    ap.add_argument("--image", default=None, help="optional real image to run through the 640 model")
    args = ap.parse_args()

    print("1. Platform")
    machine, pyv = platform.machine(), sys.version_info
    print(f"        {platform.system()} {platform.release()} | machine={machine} | Python {platform.python_version()}")
    if machine.lower() in ("aarch64", "arm64"):
        ok("64-bit ARM")
    else:
        warn(f"not aarch64 ({machine}) - fine for a desktop dry run, but this is not the Pi")
    if pyv < (3, 12):
        warn("Python < 3.12: numpy is unpinned on this version (untested combination)")
    else:
        ok("Python >= 3.12")

    print("2. Imports")
    mods = {}
    for name in ("numpy", "cv2", "psutil", "torch", "executorch", "ultralytics"):
        try:
            mods[name] = __import__(name)
            ok(f"{name} {getattr(mods[name], '__version__', '(version n/a)')}")
        except Exception as e:  # noqa: BLE001 - report every import problem
            fail(f"import {name}", e)
    if failures:
        return finish()

    print("3. ExecuTorch runtime")
    try:
        from executorch.runtime import Runtime
        names = list(Runtime.get().backend_registry.registered_backend_names)
        print(f"        registered backends: {names}")
        if "XnnpackBackend" in names:
            ok("XnnpackBackend is registered")
        else:
            fail("XnnpackBackend is NOT registered - the .pte models cannot run on this install")
    except Exception as e:  # noqa: BLE001
        fail("ExecuTorch runtime", e)
    if failures:
        return finish()

    print("4. Models (blank frame, expect 0 detections)")
    import numpy as np
    from nextraf.detector import Detector
    det640 = None
    for folder, size in MODELS:
        pte = os.path.join(folder, "model.pte")
        if not os.path.isfile(pte):
            fail(f"{pte} missing")
            continue
        try:
            t0 = time.perf_counter()
            d = Detector(folder, imgsz=size)
            load_ms = (time.perf_counter() - t0) * 1000
            blank = np.zeros((size * 9 // 16, size, 3), np.uint8)
            dets1, first_ms = d.detect(blank)   # includes warm-up
            dets2, second_ms = d.detect(blank)
            if dets1 or dets2:
                warn(f"{folder}: {len(dets2)} detections on a blank frame (unexpected)")
            ok(f"{folder} (imgsz {size}, {os.path.getsize(pte)/1e6:.1f} MB): load {load_ms:.0f} ms, "
               f"first inference {first_ms:.0f} ms (warm-up), second {second_ms:.0f} ms")
            if size == 640:
                det640 = d
        except Exception as e:  # noqa: BLE001
            fail(f"{folder}", e)

    if args.image and det640 is not None:
        import cv2
        img = cv2.imread(args.image)
        if img is None:
            fail(f"could not read --image {args.image}")
        else:
            dets, ms = det640.detect(img)
            counts = {}
            for x in dets:
                counts[x.label] = counts.get(x.label, 0) + 1
            ok(f"{args.image}: {counts or 'no detections'} in {ms:.0f} ms (640 model)")

    return finish()


def finish():
    print()
    if failures:
        print(f"RESULT: FAIL ({len(failures)} problem(s)). Send this whole output back.")
        return 1
    print(f"RESULT: PASS ({len(warnings)} warning(s)). Next: run_nextraf.py on a clip (DEPLOYMENT_GUIDE.md section 7).")
    print("Note: timings above are single runs; use summary.json from a full run for benchmark numbers.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
