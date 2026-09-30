#!/usr/bin/env python3
"""
Score the Audio Layer's transient detector and sharp-onset test against
hand-labelled clips.

Label a clip in Audacity (Tracks > Add New > Label Track, mark each knock,
click or syllable onset, then File > Export > Export Labels), or write a CSV
with `time_s,label` columns. Labels containing words like knock / click / tap
/ bang count as impulsive; syllable / speech / word / vowel count as gradual;
anything else is reported as "other".

    python tools/validate_transients.py clip.wav clip_labels.txt
    python tools/validate_transients.py clip.mp3 labels.csv --channel 0 --jump-db 14

Runs the same default chain as the app: channel pick, 16 kHz, mains-hum
notches, spectral subtraction on the suggested quietest second, 300-3400 Hz
band-pass, detection, then the onset test on the de-hummed full-band audio.
Prints how each label was handled, a summary, and which jump/decay cutoffs
would have separated these labels best, so the defaults can be tuned to the
recordings at hand.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fvtr_dsp as d  # noqa: E402


def run_pipeline(x, sr, args):
    f, P = d.median_spectrum(x, sr)
    exc = d.spectral_excess(f, P)
    mains = d.detect_mains(f, P, exc)
    freqs = []
    if mains["base_hz"] and not args.no_hum:
        freqs = [h["freq_hz"] for h in d.hum_harmonics(f, P, mains["base_hz"], 4000, excess=exc)]
    y1 = d.notch_filter(x, sr, freqs, q=30) if freqs else x
    y2 = y1
    if not args.no_subtraction:
        env = d.frame_levels_db(x, sr, 0.05)
        sug = d.suggest_noise_window(env, 0.05, 1.0)
        if sug:
            y2 = d.spectral_subtraction(y1, sr, sug["start_s"], sug["end_s"], alpha=1.5, floor_db=20)
    y3 = d.bandpass_filter(y2, sr, args.band[0], args.band[1])
    ev = d.detect_transients(y3, sr, rise_db=args.rise_db, max_duration_s=args.max_dur)
    return d.classify_transients(ev, y1, sr, jump_db=args.jump_db, decay_db=args.decay_db), mains, freqs


def best_cutoffs(imp: list[tuple[float, float]], grad: list[tuple[float, float]]):
    """(jump_db, decay_db, balanced accuracy) with the best balanced accuracy on these labels."""
    if not imp or not grad:
        return None
    imp_a, grad_a = np.array(imp), np.array(grad)
    best = None
    for j in np.arange(6, 26, 1.0):
        for dcy in np.arange(-3, 19, 1.0):
            call = lambda a: (a[:, 0] >= j) & (a[:, 1] >= dcy)
            ba = (np.mean(call(imp_a)) + np.mean(~call(grad_a))) / 2
            if best is None or ba > best[2] + 1e-9:
                best = (j, dcy, ba)
    return best


def _pairs(rows):
    return [(r["jump_db"], r["decay_db"]) for r in rows
            if r["jump_db"] is not None and r["jump_db"] == r["jump_db"] and r["decay_db"] == r["decay_db"]]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio")
    ap.add_argument("labels")
    ap.add_argument("--channel", default="mix", help='"mix" or a 0-based channel index')
    ap.add_argument("--tolerance", type=float, default=0.1, help="seconds between a label and an event span")
    ap.add_argument("--jump-db", type=float, default=d.DEFAULT_JUMP_DB)
    ap.add_argument("--decay-db", type=float, default=d.DEFAULT_DECAY_DB)
    ap.add_argument("--rise-db", type=float, default=6.0)
    ap.add_argument("--max-dur", type=float, default=0.5)
    ap.add_argument("--band", type=float, nargs=2, default=(300, 3400))
    ap.add_argument("--no-hum", action="store_true", help="skip mains-hum notches")
    ap.add_argument("--no-subtraction", action="store_true", help="skip spectral subtraction")
    ap.add_argument("--unlabelled-as-gradual", action="store_true",
                    help="treat detections with no label as speech (for clips where only knocks were labelled)")
    args = ap.parse_args(argv)

    dec = d.decode_audio(Path(args.audio).read_bytes(), args.audio)
    ch = args.channel if args.channel == "mix" else int(args.channel)
    x, _ = d.select_channel(dec, ch)
    labels = d.parse_labels(Path(args.labels).read_text(encoding="utf-8", errors="replace"))
    events, mains, freqs = run_pipeline(x, dec.sr, args)
    res = d.evaluate_labels(events, labels, args.tolerance)

    print(f"{args.audio}: {dec.duration_s:.1f} s, {dec.format}")
    print(f"mains: {mains['base_hz'] or 'none found'} Hz, {len(freqs)} notches; "
          f"{len(events)} events detected, {len(labels)} labels\n")
    print(f"{'time_s':>8}  {'label':<18} {'category':<10} {'detected':<8} {'called':<10} {'jump_db':>7} {'decay_db':>8}")
    for r in res["rows"]:
        fmt = lambda v: "" if v is None or v != v else f"{v:.1f}"
        print(f"{r['time_s']:8.3f}  {r['label'][:18]:<18} {r['category']:<10} {str(r['detected']):<8} "
              f"{r['predicted'] or '':<10} {fmt(r['jump_db']):>7} {fmt(r['decay_db']):>8}")

    print("\nSummary")
    for cat, s in res["summary"].items():
        line = f"  {cat:<10} labelled {s['labelled']:3d}  detected {s['detected']:3d}"
        if cat != "other":
            right = s["called_impulsive"] if cat == d.IMPULSIVE else s["called_gradual"]
            line += f"  called correctly {right}/{s['detected']}"
        print(line)
    minutes = dec.duration_s / 60
    print(f"  detections with no label: {res['unlabelled_events']} "
          f"({res['unlabelled_impulsive']} called impulsive = {res['unlabelled_impulsive'] / minutes:.1f} per minute)")

    det = [r for r in res["rows"] if r["detected"]]
    imp = _pairs([r for r in det if r["category"] == d.IMPULSIVE])
    grad = _pairs([r for r in det if r["category"] == d.GRADUAL])
    if args.unlabelled_as_gradual:
        matched = {r["event_start_s"] for r in det}
        grad += _pairs([e for e in events if e["start_s"] not in matched])
    for name, vals in (("impulsive", imp), ("gradual", grad)):
        if vals:
            v = np.array(vals)
            q = lambda c: np.percentile(v[:, c], [10, 50, 90]).round(1).tolist()
            print(f"  {name:<9} n={len(v):3d}  jump_db p10/50/90 {q(0)}  decay_db p10/50/90 {q(1)}")
    bc = best_cutoffs(imp, grad)
    if bc:
        print(f"  best cutoffs on these labels: jump >= {bc[0]:.0f} dB and decay >= {bc[1]:.0f} dB "
              f"(balanced accuracy {bc[2]:.0%}); current {args.jump_db:.0f} / {args.decay_db:.0f} dB")
    return res, events


if __name__ == "__main__":
    main()
