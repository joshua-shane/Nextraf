"""NEXTRAF crowd-count evaluation: Mean Absolute Error (MAE) against manual counts.

Two steps (PowerShell, from the Nextraf folder):

1) TEMPLATE - pick 10-20 frames from a prediction CSV, save them as plain (unannotated)
   images for hand counting, and write a ground-truth CSV with an EMPTY gt_people column:

   .venv\\Scripts\\python.exe eval_crowd_mae.py template --pred out\\crowd\\crowd_counts.csv ^
       --source clip.mp4 --n 15 --out out\\eval\\clip
   (optional: --config config\\crowd_site.json draws the ROI outline on the counting images,
    --frames 0,30,60 picks exact frame numbers instead of evenly spaced ones)

2) EVALUATE - after you have typed your counts into gt_people:

   .venv\\Scripts\\python.exe eval_crowd_mae.py evaluate --pred out\\crowd\\crowd_counts.csv ^
       --gt out\\eval\\clip\\ground_truth.csv --out out\\eval\\clip

Matching: rows are matched on (source file name, frame) - `frame` is the ORIGINAL frame index
in the video, so --stride does not break matching. If the prediction CSV has no `source`
column (run_nextraf.py frames.csv), matching uses `frame` only.

Counting rule for gt_people: count every person you can see whose FEET (ground contact point)
are inside the ROI - the same rule the model output uses. With no ROI, count every visible
person, including partly visible people and motorcycle riders.

The script never fills in or guesses counts. Rows with an empty gt_people are reported as
NOT_COUNTED and excluded; MAE is only reported when at least one frame has both a valid manual
count and a prediction.
"""
import argparse
import csv
import json
import os
import sys

GT_FIELDS = ["source", "frame", "video_t_sec", "gt_people", "notes"]
RESULT_FIELDS = ["source", "frame", "video_t_sec", "gt_people", "pred_people", "error",
                 "abs_error", "status", "notes"]


def read_csv(path):
    if not os.path.isfile(path):
        sys.exit(f"ERROR: file not found: {path}")
    with open(path, newline="", encoding="utf-8-sig") as f:   # -sig: tolerate Excel's BOM
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"ERROR: no data rows in {path}")
    return rows


def parse_frame(value, where):
    try:
        return int(str(value).strip())
    except ValueError:
        sys.exit(f"ERROR: frame '{value}' in {where} is not a whole number")


def key_of(row, use_source):
    src = os.path.basename(row.get("source", "").strip()) if use_source else ""
    return (src, parse_frame(row.get("frame"), "a CSV row"))


# ---------------------------------------------------------------- template
def cmd_template(a):
    pred = read_csv(a.pred)
    if "frame" not in pred[0]:
        sys.exit("ERROR: prediction CSV has no 'frame' column")
    has_source = "source" in pred[0]
    src_name = os.path.basename(a.source)
    rows = [r for r in pred if not has_source or os.path.basename(r["source"]) == src_name]
    if not rows:
        sys.exit(f"ERROR: no prediction rows for source '{src_name}' in {a.pred}")

    by_frame = {parse_frame(r["frame"], a.pred): r for r in rows}
    available = sorted(by_frame)
    if a.frames:
        wanted = [parse_frame(x, "--frames") for x in a.frames.split(",") if x.strip()]
        missing = [f for f in wanted if f not in by_frame]
        if missing:
            sys.exit(f"ERROR: frames not in the prediction CSV: {missing}. "
                     f"Available range: {available[0]}..{available[-1]} ({len(available)} frames)")
        chosen = wanted
    else:
        n = min(a.n, len(available))
        chosen = sorted({available[round(i * (len(available) - 1) / max(n - 1, 1))] for i in range(n)})
    if len(chosen) < 10:
        print(f"WARNING: only {len(chosen)} frames selected; 10-20 are recommended for a meaningful MAE.")

    import cv2
    import numpy as np
    roi = None
    if a.config:
        with open(a.config) as f:
            roi = json.load(f).get("roi")
    img_dir = os.path.join(a.out, "frames")
    os.makedirs(img_dir, exist_ok=True)
    gt_path = os.path.join(a.out, "ground_truth.csv")
    if os.path.exists(gt_path) and not a.force:
        sys.exit(f"ERROR: {gt_path} already exists - it may hold your manual counts. "
                 f"Use another --out, or --force to overwrite it.")

    stem = os.path.splitext(src_name)[0]
    is_video = os.path.isfile(a.source) and os.path.splitext(a.source)[1].lower() not in (
        ".jpg", ".jpeg", ".png", ".bmp", ".webp")
    cap = cv2.VideoCapture(a.source) if is_video else None
    if is_video and not cap.isOpened():
        sys.exit(f"ERROR: could not open video {a.source}")
    saved = []
    for fr in chosen:
        if is_video:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
        elif os.path.isdir(a.source):
            files = sorted(x for x in os.listdir(a.source)
                           if os.path.splitext(x)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp", ".webp"))
            img = cv2.imread(os.path.join(a.source, files[fr])) if fr < len(files) else None
            ok = img is not None
        else:
            img = cv2.imread(a.source)
            ok = img is not None
        if not ok:
            print(f"WARNING: could not read frame {fr} from {a.source}; skipped")
            continue
        if roi:
            cv2.polylines(img, [np.asarray(roi, np.int32)], True, (255, 255, 0), 2)
        name = f"{stem}_f{fr:05d}.jpg"
        cv2.imwrite(os.path.join(img_dir, name), img)
        saved.append(fr)
    if cap is not None:
        cap.release()

    with open(gt_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=GT_FIELDS)
        w.writeheader()
        for fr in saved:
            w.writerow({"source": src_name if has_source else src_name, "frame": fr,
                        "video_t_sec": by_frame[fr].get("video_t_sec", ""),
                        "gt_people": "", "notes": ""})
    print(f"Saved {len(saved)} plain frames for counting to {img_dir}")
    print(f"Ground-truth template (gt_people EMPTY - fill it in by hand): {gt_path}")
    print("Counting images are NOT annotated with model boxes, so your counts stay independent.")


# ---------------------------------------------------------------- evaluate
def cmd_evaluate(a):
    pred = read_csv(a.pred)
    gt = read_csv(a.gt)
    if a.pred_col not in pred[0]:
        sys.exit(f"ERROR: prediction column '{a.pred_col}' not found. Columns: {list(pred[0])}")
    for col in ("frame", "gt_people"):
        if col not in gt[0]:
            sys.exit(f"ERROR: ground-truth CSV has no '{col}' column. Columns: {list(gt[0])}")
    use_source = "source" in pred[0] and "source" in gt[0]
    if not use_source:
        print("NOTE: matching on 'frame' only (prediction or ground-truth CSV has no 'source' column).")
        if "source" in gt[0] and len({r['source'] for r in gt}) > 1:
            sys.exit("ERROR: ground truth lists several sources but predictions have no 'source' column.")

    preds, dup_pred = {}, set()
    for r in pred:
        k = key_of(r, use_source)
        if k in preds:
            dup_pred.add(k)
        preds[k] = r
    if dup_pred:
        print(f"WARNING: duplicate prediction rows for {sorted(dup_pred)}; the last one is used.")

    results, seen = [], set()
    for r in gt:
        k = key_of(r, use_source)
        row = {"source": k[0] or r.get("source", ""), "frame": k[1],
               "video_t_sec": r.get("video_t_sec", ""), "gt_people": "", "pred_people": "",
               "error": "", "abs_error": "", "status": "", "notes": r.get("notes", "")}
        raw = (r.get("gt_people") or "").strip()
        p = preds.get(k)
        if p is not None:
            row["pred_people"] = p[a.pred_col]
            row["video_t_sec"] = row["video_t_sec"] or p.get("video_t_sec", "")
        if k in seen:
            row["status"] = "DUPLICATE_GT_ROW"
        elif raw == "":
            row["status"] = "NOT_COUNTED"
        elif not raw.isdigit():
            row["gt_people"], row["status"] = raw, "INVALID_GT (must be a whole number >= 0)"
        elif p is None:
            row["gt_people"], row["status"] = int(raw), "NO_PREDICTION_FOR_FRAME"
        else:
            g, pv = int(raw), int(float(p[a.pred_col]))
            row.update(gt_people=g, pred_people=pv, error=pv - g, abs_error=abs(pv - g), status="OK")
        seen.add(k)
        results.append(row)

    os.makedirs(a.out, exist_ok=True)
    res_path = os.path.join(a.out, "mae_results.csv")
    with open(res_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        w.writeheader()
        w.writerows(results)

    print(f"\n{'frame':>7} {'t (s)':>7} {'manual':>7} {'model':>7} {'error':>6}  status")
    for r in results:
        print(f"{r['frame']:>7} {str(r['video_t_sec']):>7} {str(r['gt_people']):>7} "
              f"{str(r['pred_people']):>7} {str(r['error']):>6}  {r['status']}")

    ok = [r for r in results if r["status"] == "OK"]
    counts = {}
    for r in results:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"\nRows: {len(results)} | " + ", ".join(f"{k}: {v}" for k, v in counts.items()))
    unmatched_pred = len(preds) - len({key_of(r, use_source) for r in gt} & set(preds))
    print(f"Prediction frames without a manual count (ignored): {unmatched_pred}")

    summary = {"prediction_csv": a.pred, "ground_truth_csv": a.gt, "prediction_column": a.pred_col,
               "rows_in_ground_truth": len(results), "status_counts": counts,
               "frames_evaluated": len(ok)}
    if not ok:
        print("\nMAE NOT COMPUTED: no frame has both a valid manual count and a prediction.")
        print("Fill in gt_people in the ground-truth CSV, then run evaluate again.")
        summary["mae"] = None
    else:
        n = len(ok)
        errs = [r["error"] for r in ok]
        summary.update(
            mae=round(sum(abs(e) for e in errs) / n, 3),
            mean_signed_error=round(sum(errs) / n, 3),
            max_abs_error=max(abs(e) for e in errs),
            pct_within_1=round(100 * sum(abs(e) <= 1 for e in errs) / n, 1),
            total_manual=sum(r["gt_people"] for r in ok),
            total_model=sum(r["pred_people"] for r in ok))
        print(f"\nFrames evaluated : {n}" + ("   (WARNING: fewer than 10)" if n < 10 else ""))
        print(f"MAE              : {summary['mae']} people/frame")
        print(f"Mean signed error: {summary['mean_signed_error']}  (negative = model undercounts)")
        print(f"Max abs error    : {summary['max_abs_error']}")
        print(f"Frames within +-1: {summary['pct_within_1']} %")
        print(f"Total people     : manual {summary['total_manual']} vs model {summary['total_model']}")
    with open(os.path.join(a.out, "mae_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {res_path} and {os.path.join(a.out, 'mae_summary.json')}")


def main():
    ap = argparse.ArgumentParser(description="NEXTRAF crowd-count MAE evaluation")
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("template", help="select frames, save plain images, write empty ground-truth CSV")
    t.add_argument("--pred", required=True, help="crowd_counts.csv (run_crowd.py) or frames.csv (run_nextraf.py)")
    t.add_argument("--source", required=True, help="the video / image / folder that produced --pred")
    t.add_argument("--n", type=int, default=15, help="number of evenly spaced frames (default 15)")
    t.add_argument("--frames", default=None, help="exact frame numbers, e.g. 0,30,60 (overrides --n)")
    t.add_argument("--config", default=None, help="crowd config; draws its ROI outline on the images")
    t.add_argument("--out", required=True, help="output folder for frames/ and ground_truth.csv")
    t.add_argument("--force", action="store_true", help="overwrite an existing ground_truth.csv")
    e = sub.add_parser("evaluate", help="compare manual counts with predictions and compute MAE")
    e.add_argument("--pred", required=True)
    e.add_argument("--gt", required=True, help="ground_truth.csv with gt_people filled in")
    e.add_argument("--pred-col", default="people_in_roi", help="prediction column (default people_in_roi)")
    e.add_argument("--out", required=True, help="folder for mae_results.csv and mae_summary.json")
    a = ap.parse_args()
    cmd_template(a) if a.cmd == "template" else cmd_evaluate(a)


if __name__ == "__main__":
    main()
