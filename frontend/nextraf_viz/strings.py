"""All UI text lives here so wording can switch by region and be translated later.

``text(key, region)`` returns the regional override if one exists, else the common text.
"""
from __future__ import annotations

COMMON: dict[str, str] = {
    "app_title": "NEXTRAF results viewer",
    "tagline": "Offline people counting on Raspberry Pi with ExecuTorch. People by default; vehicles from the same frame.",
    "sample_banner": "SAMPLE DATA: this is a synthetic demo folder bundled with the app, not a real site.",
    "bench_warning": "Do not run this viewer on the Raspberry Pi while benchmarking: it would distort timing and memory.",
    "placeholder_los": "Placeholder LOS bands: copy the real values from IRC:103-2012 Section 5 into site.json before using these.",
    "area_assumed": "Walking area is assumed, not measured.",
    "no_area": "No walking area is set in site.json, so space per person and crowding bands are hidden.",
    "privacy_ok": "Counts only: no footage stored",
    "sidebar_footer": "Runs offline. Stores counts, never footage.",
    "hosted_note": "Public demo with synthetic sample data. To load real results, run NEXTRAF on your own computer: it works fully offline and nothing is uploaded.",
    "privacy_warn": "Video files found in the results folder: {files}. NEXTRAF is designed to store counts only.",
    "stored_counts_only": "Only counts, metrics and configuration are stored. No video.",
    "stored_images": "No video is stored. The annotated images come from a validation run and can show real people; run with --mode deployment to keep counts only.",
    "nav_overview": "Overview",
    "nav_timeline": "Crowding timeline",
    "nav_envelope": "Affordability envelope",
    "nav_validation": "Validation",
    "nav_trust": "Trust and privacy",
    "nav_export": "Export and report",
    "sb_folder": "Results folder",
    "sb_class": "Count class",
    "sb_tolerance": "Recommendation tolerance (MAE, people/frame)",
    "sb_k": "Stability k (flag if beyond k x MAD)",
    "sb_window": "Rolling window (frames)",
    "sb_exclude": "Exclude not-trustworthy runs",
    "missing_section": "{note}",
    "cap_timeline": "People (or the chosen class) counted in each frame. Gaps are frames with no camera data, not zero people.",
    "cap_space": "Walking area divided by people. Lower means more crowded. Shaded bands show the crowding levels.",
    "cap_envelope": "Each dot is one exported model plus confidence threshold. Lower-left is better: faster and more accurate.",
    "cap_memmae": "Peak memory against counting error for every configuration. Lower-left is cheaper and better.",
    "cap_pred_true": "Detector count against the hand count for each evaluated frame.",
    "cap_err_size": "Counting error against crowd size. Points below zero are undercounts, which grow in dense frames.",
    "nav_demo": "Guided demo",
    "nav_replay": "Live replay",
    "nav_whatif": "What-if and compare",
    "sb_upload": "Or upload result files",
    "sb_upload_help": "Drop counts.csv (or frames.csv and summary.json from run_nextraf.py) and any of benchmark.csv, per_frame_eval.csv, site.json, device.json, or sample images. Files stay on this computer.",
    "help_overview": "Start here. The verdict says how crowded the site was, the findings explain why in plain words, and the cards below show the key numbers. The numbers come from counts only, and no video is stored.",
    "help_timeline": "Read left to right: each dot is one frame. A gap means the camera had no usable picture. A red cross means the count looked implausible compared with its neighbours. Shaded bands show how crowded the footpath was.",
    "help_envelope": "This answers: what is the cheapest setup that still counts reliably? Lower-left on the chart is better. The star is the recommended setup, the dotted line is the best trade-off frontier.",
    "help_validation": "Compares what the detector counted with what people counted by hand. Negative error means undercounting, which grows in dense crowds.",
    "help_trust": "Shows that the data can be checked: a hash chain makes edits detectable, and the file list shows only counts and metrics are stored.",
    "help_export": "Download the data, or build a one-page site report (Markdown and an offline HTML file with charts).",
    "help_replay": "Replays the recorded counts frame by frame, like watching the site, but from the file. There is no camera and no tracking of people.",
    "help_whatif": "Try a different walking area or thresholds to see how the verdict would change, or compare two model setups side by side. Nothing here changes your files.",
    "tip_peak": "Highest count in any frame, ignoring frames that look implausible.",
    "tip_worst": "Minutes the footpath spent in the most crowded band (needs a walking area).",
    "tip_board": "Cheapest Raspberry Pi 5 board whose RAM fits the recommended setup, with headroom.",
    "tip_integrity": "Whether counts.csv has been edited since the Pi wrote it (hash chain).",
    "cap_gauge": "Crowd density now, against the warning and critical levels (assumed thresholds).",
    "demo_problem_title": "The problem",
    "demo_problem": "Crowding at footpaths, school gates, bus stops and junctions is judged by eye. Cloud CCTV costs too much, needs constant internet and sends footage off-site. Manual headcounts do not scale and leave no evidence.",
    "demo_why": "In 2024 Bengaluru had 246 pedestrian deaths, the second-highest among India's million-plus cities (MoRTH, Road Accidents in India 2024, as reported by Deccan Herald). The June 2025 crowd crush outside the city's cricket stadium showed that crowd density is still judged by eye.",
    "demo_stat_value": "246",
    "demo_stat_label": "pedestrian deaths in Bengaluru in 2024, the second-highest among India's million-plus cities",
    "demo_stat_source": "MoRTH, Road Accidents in India 2024, as reported by Deccan Herald",
    "demo_stadium": "The June 2025 crowd crush outside the city's cricket stadium showed that crowd density is still judged by eye.",
    "demo_solution_title": "The solution",
    "demo_solution": "A pretrained detector counts people in each frame on a Raspberry Pi 5, fully offline. Only counts leave the device, never footage.",
    "demo_compare": (("Costly hardware and recurring cloud fees", "Raspberry Pi-class, ward-budget hardware"),
                     ("Needs a constant internet connection", "Runs fully offline on the device"),
                     ("Footage leaves the device (privacy risk)", "Only counts leave the device"),
                     ("Crowding judged by gut feeling", "Per-frame counts as hard evidence"),
                     ("Manual counts do not scale or repeat", "Automated and repeatable at any site")),
    "sb_region": "Region profile",
    "sb_switch": "Adaptive switch count (people)",
    "cap_overlay": "People and vehicles in each frame. Shaded frames are conflict moments: many people and many vehicles together.",
    "cap_encroach": "Space per person using the full area (assumed) and using the area left after vehicles on the {footpath} (assumed footprints).",
    "cap_surge": "People per m² of usable area. A warning needs high density that is still rising; critical is density alone. Counts only, no tracking.",
    "cap_adaptive": "The adaptive point is simulated from every configuration's per-frame predictions: cheap when quiet, accurate when crowded. Memory is an estimate.",
    "rickshaw_note": "Auto-rickshaws and e-rickshaws are not COCO classes, so they may be miscounted as other vehicles.",
    "assumed_footprint": "Vehicle footprints (about 1.5 m² per two-wheeler, 7 m² per car) are assumptions and can be edited in site.json.",
    "assumed_surge": "Surge thresholds are assumptions, configurable in site.json. UK reference: up to about 4 people/m² (DfT TAL 2/13, citing DCMS 2008).",
    "chain_valid": "Hash chain verified: no row has been edited, inserted or deleted.",
    "chain_broken": "Hash chain BROKEN at file line {line}: the data was changed after it was written.",
    "chain_absent": "No hash chain in counts.csv (no row_hash column), so edits cannot be detected.",
    "chain_absent_run": "frames.csv from run_nextraf.py has no hash chain, so edits cannot be detected.",
    "chain_limit": "Tamper-evident, not tamper-proof: someone who can rewrite the whole file could rebuild the chain. Record the last row_hash elsewhere to close that gap.",
    "limitations": (
        "Dense crowds: people block each other, so the model undercounts and error grows with crowd size.",
        "Very large crowds need density-map methods; detection counting suits tens of people per frame.",
        "Small or distant people are often missed at low resolution.",
        "Night, rain and steep camera angles are untested.",
        "No unique-person counting: counts measure occupancy at an instant, not flow.",
        "Auto-rickshaws are not a COCO class and may be miscounted.",
    ),
}

REGIONAL: dict[str, dict[str, str]] = {
    "IN": {
        "footpath": "footpath", "authority": "ward office",
        "crowding_ref": "Crowding reference: pedestrian Level of Service bands in IRC:103-2012 (Indian Roads Congress, Section 5).",
        "privacy_text": ("Privacy: frames are processed in memory only and overwritten by the next frame; only integer counts are "
                         "written. Under the Digital Personal Data Protection Act 2023 (Rules notified 14 Nov 2025) you should still "
                         "assess whether your deployment needs notice or consent. NEXTRAF does not claim to be exempt from the law."),
        "currency_default": "INR",
    },
    "UK": {
        "footpath": "pavement", "authority": "council",
        "crowding_ref": "Crowding reference: UK density guidance in people/m² (TfL, DfT Traffic Advisory Leaflet 2/13).",
        "privacy_text": ("Privacy: frames are processed in memory only and overwritten by the next frame; only integer counts are "
                         "written. Under UK GDPR and ICO guidance you should still carry out a DPIA and display signage where "
                         "required. NEXTRAF does not claim to be exempt from the law."),
        "currency_default": "GBP",
    },
}

CURRENCY_SYMBOLS = {"INR": "₹", "GBP": "£"}


def text(key: str, region: str = "IN", **fmt: object):
    """Look up UI text, preferring the regional wording; ``fmt`` fills ``{placeholders}``."""
    value = REGIONAL.get(region, {}).get(key, COMMON.get(key))
    if value is None:
        raise KeyError(f"Unknown UI string: {key}")
    return value.format(**fmt) if isinstance(value, str) and fmt else value
