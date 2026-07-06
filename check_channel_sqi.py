"""
check_channel_sqi.py

Quick diagnostic: for a given recording, load the preprocessed abdominal
signal (abd_proc, same one that goes into ICA1) and print each channel's
SQI using the already-existing evaluation/sqi.py functions. No pipeline
changes needed to run this -- it's read-only diagnosis.

Usage:
    python check_channel_sqi.py a16
    python check_channel_sqi.py a45 a60 a66

What to look for:
  - If one channel's SQI is much lower than the others (e.g. 0.15 vs
    0.55-0.70), that channel is likely dragging down the ICA decomposition
    for this recording -- worth testing channel exclusion/down-weighting.
  - If all channels look roughly similar (no clear outlier), this isn't
    the problem for that recording -- don't force a fix that isn't there.
"""
import sys
import numpy as np

from config_loader import get_config
from preprocessing.filters import preprocess_multichannel
from preprocessing.qrs_detector import detect_maternal_qrs
from evaluation.sqi import compute_channel_sqi

# Adjust this import/loader to however your dataset_handlers actually load
# a single recording -- this assumes something like the loader your
# pipeline.run() already uses internally. Replace with your real one if
# the name/signature differs.
from dataset_handlers.cinc2013 import CinC2013Handler  # adjust if named differently


def main(rec_ids):
    cfg = get_config("cinc2013")
    fs = cfg.FS

    for rec_id in rec_ids:
        handler = CinC2013Handler()
        from pathlib import Path

        path = Path("dataset_handlers") / "set-a" / f"{rec_id}.hea"

        recording = handler.load_single_recording(str(path))
        abd_raw = recording["abdomen"]# adjust key name if different
        abd_proc = preprocess_multichannel(abd_raw, fs, cfg=cfg)

        # Need maternal peaks for the consistency sub-metric; reuse the
        # pipeline's own maternal detection so this matches what ICA1
        # would have seen.
        maternal_peaks = None
        try:
            maternal_ic, maternal_peaks = detect_maternal_qrs(abd_proc, fs, cfg=cfg)
        except Exception as e:
            print(f"  (maternal peak detection failed for consistency metric: {e})")

        print(f"\n=== {rec_id}: per-channel SQI ===")
        scores = []
        for ch in range(abd_proc.shape[0]):
            sqi = compute_channel_sqi(abd_proc[ch], fs=fs, maternal_peaks=maternal_peaks)
            scores.append(sqi)
            print(f"  channel {ch}: SQI = {sqi:.3f}")

        scores = np.array(scores)
        spread = scores.max() - scores.min()
        print(f"  spread (max-min): {spread:.3f}",
              "-- possible bad channel" if spread > 0.25 else "-- channels look similar")


if __name__ == "__main__":
    main(sys.argv[1:])
