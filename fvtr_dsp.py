"""
FVTR DSP core — the signal processing behind pages/2_Audio_Layer.py.

Kept free of Streamlit so every stage can be unit-tested and reused from the
command line (see tools/validate_transients.py). All functions take and
return plain NumPy arrays; float32 mono in [-1, 1] unless noted.

Everything here is standard signal processing. It can suppress noise that
is present, isolate a band that is present, or flag where a level or a
frequency does something abrupt. None of it recovers audio that was never
captured, and every flag is a lead for review, not a finding.
"""

from __future__ import annotations

import csv
import io
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from functools import lru_cache
from math import gcd
from pathlib import Path

import numpy as np
import soundfile as sf
from numpy.lib.stride_tricks import sliding_window_view
from scipy import signal
from scipy.ndimage import (
    binary_closing,
    median_filter,
    percentile_filter,
    uniform_filter1d,
)

ANALYSIS_SR = 16_000     # plenty for a 300-3400 Hz vocal band and for mains hum
SILENCE_DB = -120.0      # level reported for frames of exact digital zero
_SILENCE_CUTOFF_DB = -110.0
_DIGITAL_ZERO = 2.0 ** -16  # below half a 16-bit LSB

SOUNDFILE_TYPES = ["wav", "mp3", "flac", "ogg", "oga", "aif", "aiff"]
FFMPEG_TYPES = ["m4a", "aac", "mp4", "opus", "wma", "webm"]


class AudioDecodeError(RuntimeError):
    pass


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def accepted_types() -> list[str]:
    return SOUNDFILE_TYPES + (FFMPEG_TYPES if ffmpeg_available() else [])


# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------

@dataclass
class DecodedAudio:
    sr: int                       # analysis sample rate of `channels`
    native_sr: int
    channels: list[np.ndarray]    # one float32 array per source channel, at `sr`
    duration_s: float
    format: str
    decoder: str
    stats: list[dict] = field(default_factory=list)
    # Digital-silence runs per channel, (start_s, end_s), found at the native rate.
    silence: list[list[tuple[float, float]]] = field(default_factory=list)

    @property
    def n_channels(self) -> int:
        return len(self.channels)


class _RunTracker:
    """Collects runs of True in a boolean stream that arrives in blocks."""

    def __init__(self, min_len: int):
        self.min_len = max(1, min_len)
        self.open_start: int | None = None
        self.runs: list[tuple[int, int]] = []

    def _close(self, start: int, end: int) -> None:
        if end - start >= self.min_len:
            self.runs.append((start, end))

    def feed(self, mask: np.ndarray, offset: int) -> None:
        if len(mask) == 0:
            return
        if self.open_start is not None and not mask[0]:
            self._close(self.open_start, offset)
            self.open_start = None
        edges = np.diff(np.concatenate(([0], mask.view(np.int8), [0])))
        starts = np.flatnonzero(edges == 1)
        ends = np.flatnonzero(edges == -1)
        for s, e in zip(starts, ends):
            gs = offset + int(s)
            if s == 0 and self.open_start is not None:
                gs = self.open_start
                self.open_start = None
            if e == len(mask):
                self.open_start = gs
            else:
                self._close(gs, offset + int(e))

    def finish(self, total: int) -> list[tuple[int, int]]:
        if self.open_start is not None:
            self._close(self.open_start, total)
            self.open_start = None
        return self.runs


class _StreamResampler:
    """
    Block-wise scipy.signal.resample_poly that gives the same result as one
    call over the whole signal, without holding the native-rate file in memory.
    Each block is resampled with enough context on both sides to cover the
    anti-alias filter, and blocks start on multiples of `down` so output
    samples line up exactly. Blocks are channel-major: shape (channels, n).
    """

    def __init__(self, up: int, down: int, n_ch: int, block: int):
        self.up, self.down = up, down
        need = int(np.ceil(10 * max(up, down) / up)) + 4  # filter half-length, native samples
        self.pad = down * int(np.ceil(need / down))
        self.block = down * max(1, int(np.ceil(block / down)))
        self.buf = np.zeros((n_ch, 0), np.float32)
        self.buf_start = 0
        self.emitted = 0
        self.total = 0
        self.out: list[np.ndarray] = []

    def _emit(self, lo: int, hi: int, final: bool) -> None:
        c0 = self.buf_start
        c1 = self.total if final else hi + self.pad
        chunk = self.buf[:, c0 - self.buf_start: c1 - self.buf_start]
        y = signal.resample_poly(chunk, self.up, self.down, axis=1)
        o0 = (lo - c0) * self.up // self.down
        o1 = y.shape[1] if final else o0 + (hi - lo) * self.up // self.down
        self.out.append(y[:, o0:o1].astype(np.float32))

    def feed(self, block: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, block], axis=1)
        self.total += block.shape[1]
        while self.total - self.emitted >= self.block + self.pad:
            self._emit(self.emitted, self.emitted + self.block, final=False)
            self.emitted += self.block
            new_start = max(0, self.emitted - self.pad)
            self.buf = self.buf[:, new_start - self.buf_start:]
            self.buf_start = new_start

    def finish(self) -> np.ndarray:
        if self.total > self.emitted:
            self._emit(self.emitted, self.total, final=True)
        if not self.out:
            return np.zeros((self.buf.shape[0], 0), np.float32)
        return np.concatenate(self.out, axis=1)


def _decode_soundfile(source, target_sr: int, block_s: float, min_silence_s: float, decoder: str) -> DecodedAudio:
    with sf.SoundFile(source) as f:
        native_sr, n_ch = f.samplerate, f.channels
        fmt = f"{f.format_info} / {f.subtype_info}, {native_sr} Hz, {n_ch} ch"
        sr = min(target_sr, native_sr)
        g = gcd(sr, native_sr)
        up, down = sr // g, native_sr // g
        block = max(down, int(block_s * native_sr))
        resampler = _StreamResampler(up, down, n_ch, block) if up != down else None
        passthrough: list[np.ndarray] = []
        trackers = [_RunTracker(int(min_silence_s * native_sr)) for _ in range(n_ch)]
        sumsq = np.zeros(n_ch)
        peak = np.zeros(n_ch)
        clipped = np.zeros(n_ch, dtype=np.int64)
        offset = 0
        for blk in f.blocks(blocksize=block, dtype="float32", always_2d=True):
            if not len(blk):
                continue
            rows = np.ascontiguousarray(blk.T)  # channel-major: fast row-wise work
            a = np.abs(rows)
            sumsq += np.einsum("ij,ij->i", rows, rows, dtype=np.float64)
            peak = np.maximum(peak, a.max(axis=1))
            clipped += np.count_nonzero(a >= 0.999, axis=1)
            zero = a < _DIGITAL_ZERO
            for c in range(n_ch):
                trackers[c].feed(zero[c], offset)
            offset += rows.shape[1]
            if resampler is not None:
                resampler.feed(rows)
            else:
                passthrough.append(rows)
    if offset == 0:
        raise AudioDecodeError("The file decoded to zero samples.")
    out = resampler.finish() if resampler is not None else np.concatenate(passthrough, axis=1)
    channels = []
    for c in range(n_ch):
        ch = np.ascontiguousarray(out[c], dtype=np.float32)
        ch.setflags(write=False)
        channels.append(ch)
    stats, silence = [], []
    for c in range(n_ch):
        rms = np.sqrt(sumsq[c] / offset)
        runs = trackers[c].finish(offset)
        silence.append([(s / native_sr, e / native_sr) for s, e in runs])
        stats.append({
            "rms_dbfs": round(float(20 * np.log10(rms + 1e-12)), 1),
            "peak_dbfs": round(float(20 * np.log10(peak[c] + 1e-12)), 1),
            "clipped_samples": int(clipped[c]),
            "digital_silence_runs": len(runs),
        })
    return DecodedAudio(sr=sr, native_sr=native_sr, channels=channels,
                        duration_s=offset / native_sr, format=fmt, decoder=decoder,
                        stats=stats, silence=silence)


def decode_audio(raw: bytes, filename: str = "", target_sr: int = ANALYSIS_SR,
                 block_s: float = 10.0, min_silence_s: float = 0.005) -> DecodedAudio:
    """
    Decode WAV/MP3/FLAC/OGG with libsndfile (M4A/AAC and others through ffmpeg
    when it is installed), streaming in blocks and resampling to `target_sr`.
    Files already at or below `target_sr` keep their own rate. Digital-silence
    runs and clipping are measured on the native samples before resampling.
    """
    first_err: Exception | None = None
    try:
        return _decode_soundfile(io.BytesIO(raw), target_sr, block_s, min_silence_s, "libsndfile")
    except Exception as e:  # libsndfile raises several error types for unknown formats
        first_err = e
    if not ffmpeg_available():
        ext = Path(filename).suffix.lower() or "this format"
        raise AudioDecodeError(
            f"Could not decode {ext} with libsndfile ({first_err}). "
            "Install ffmpeg to open M4A/AAC and other container formats."
        )
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / ("input" + (Path(filename).suffix or ".bin"))
        dst = Path(td) / "decoded.wav"
        src.write_bytes(raw)
        proc = subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src),
             "-vn", "-map", "0:a:0", "-c:a", "pcm_f32le", str(dst)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise AudioDecodeError(f"ffmpeg could not decode the file: {proc.stderr.strip()[:500]}")
        return _decode_soundfile(str(dst), target_sr, block_s, min_silence_s, "ffmpeg")


def intersect_runs(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> list[tuple[float, float]]:
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if e > s:
            out.append((s, e))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def select_channel(dec: DecodedAudio, channel) -> tuple[np.ndarray, list[tuple[float, float]]]:
    """channel = "mix" or a 0-based index. Returns (signal, digital-silence runs)."""
    if channel == "mix" and dec.n_channels > 1:
        x = np.mean(np.stack(dec.channels), axis=0, dtype=np.float32)
        runs = dec.silence[0]
        for other in dec.silence[1:]:
            runs = intersect_runs(runs, other)
        x.setflags(write=False)
        return x, runs
    idx = 0 if channel == "mix" else int(channel)
    return dec.channels[idx], dec.silence[idx]


def to_wav_bytes(sr: int, data: np.ndarray) -> bytes:
    """Encode float32 [-1, 1] mono as 16-bit PCM WAV."""
    buf = io.BytesIO()
    sf.write(buf, np.clip(data, -1.0, 1.0), sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Levels and the noise sample
# ---------------------------------------------------------------------------

def frame_levels_db(x: np.ndarray, sr: int, frame_s: float) -> np.ndarray:
    """RMS level per non-overlapping frame in dBFS; exact digital zero reads SILENCE_DB."""
    n = max(1, int(round(frame_s * sr)))
    k = len(x) // n
    if k == 0:
        return np.zeros(0, np.float32)
    frames = x[: k * n].reshape(k, n)
    ms = np.mean(np.square(frames, dtype=np.float32), axis=1, dtype=np.float64)
    return (10 * np.log10(np.maximum(ms, 1e-12))).astype(np.float32)


def noise_floor_db(env_db: np.ndarray, pct: float = 10.0) -> float:
    """Room-tone estimate: a low percentile of frame levels, ignoring digital silence."""
    live = env_db[env_db > _SILENCE_CUTOFF_DB]
    return float(np.percentile(live, pct)) if live.size else SILENCE_DB


def typical_quiet_db(env_db: np.ndarray, share: float = 0.03, width_db: float = 3.0) -> float:
    """
    The lowest level at which at least `share` of the recording sits (within
    ±`width_db`). Room tone recurs in every pause, so it clears this bar; a
    mute or a brief dropout is too rare to.
    """
    live = np.sort(env_db[env_db > _SILENCE_CUTOFF_DB].astype(np.float64))
    if live.size == 0:
        return SILENCE_DB
    need = max(1, int(share * live.size))
    counts = np.searchsorted(live, live + width_db, side="right") - np.searchsorted(live, live - width_db)
    ok = np.flatnonzero(counts >= need)
    return float(live[ok[0]]) if ok.size else float(live[0])


def _mean_db(env_db: np.ndarray) -> float:
    return float(10 * np.log10(np.mean(10 ** (env_db.astype(np.float64) / 10)) + 1e-30))


def suggest_noise_window(env_db: np.ndarray, frame_s: float, length_s: float,
                         edge_s: float = 0.25, max_spread_db: float = 10.0,
                         max_below_floor_db: float = 15.0) -> dict | None:
    """
    Quietest steady stretch of `length_s`, by mean power.

    Skips digital silence and the file edges, windows whose level moves by
    more than `max_spread_db` (they straddle speech or an event), and windows
    more than `max_below_floor_db` under the recording's usual quiet level
    (see typical_quiet_db): those are mutes, where the room noise itself is
    attenuated and so no longer represents the noise to remove. If nothing
    passes, the steadiness and mute limits are relaxed in turn.
    """
    L = max(1, int(round(length_s / frame_s)))
    edge = int(np.ceil(edge_s / frame_s))
    n = len(env_db)
    if n < L + 2 * edge:
        return None
    silent = env_db <= _SILENCE_CUTOFF_DB
    p = np.where(silent, 0.0, 10 ** (env_db.astype(np.float64) / 10))
    cs = np.concatenate(([0.0], np.cumsum(p)))
    cz = np.concatenate(([0], np.cumsum(silent)))
    mean = (cs[L:] - cs[:-L]) / L
    usable = (cz[L:] - cz[:-L]) == 0
    usable[:edge] = False
    usable[n - L - edge + 1:] = False
    # Level spread (p95 - p5) inside each window. Windows holding digital
    # silence are already excluded, so no NaN handling is needed.
    if L > 2:
        e = env_db.astype(np.float64)
        c = np.arange(len(mean)) + L // 2          # centre of the window starting at i
        spread = (percentile_filter(e, 95, size=L, mode="nearest")[c]
                  - percentile_filter(e, 5, size=L, mode="nearest")[c])
    else:
        spread = np.zeros(len(mean))
    level = 10 * np.log10(mean + 1e-30)
    typical_quiet = typical_quiet_db(env_db)
    for use_spread, use_mute in ((True, True), (False, True), (False, False)):
        ok = usable.copy()
        if use_spread:
            ok &= spread <= max_spread_db
        if use_mute:
            ok &= level >= typical_quiet - max_below_floor_db
        if ok.any():
            i = int(np.flatnonzero(ok)[np.argmin(mean[ok])])
            return {"start_s": round(i * frame_s, 2), "end_s": round((i + L) * frame_s, 2),
                    "level_db": round(float(level[i]), 1), "steady": bool(use_spread)}
    return None


def assess_noise_window(env_db: np.ndarray, frame_s: float, start_s: float, end_s: float,
                        reference_db: float | None = None) -> dict:
    """
    Level of the chosen noise sample relative to `reference_db` (pass the level
    of the suggested quietest stretch; defaults to the 10th-percentile frame
    level), plus plain-language warnings.
    """
    n = len(env_db)
    i0 = int(np.clip(start_s / frame_s, 0, n))
    i1 = int(np.clip(np.ceil(end_s / frame_s), 0, n))
    floor = noise_floor_db(env_db) if reference_db is None else float(reference_db)
    res = {"floor_db": round(floor, 1), "warnings": []}
    duration = n * frame_s
    if end_s > duration + frame_s:
        res["warnings"].append(f"The window ends after the recording ({duration:.1f} s); it was cut short.")
    if i1 <= i0:
        res["warnings"].append("The window is empty.")
        res.update(level_db=None, above_floor_db=None, spread_db=None)
        return res
    seg = env_db[i0:i1]
    live = seg[seg > _SILENCE_CUTOFF_DB]
    silent_frac = 1 - live.size / seg.size
    level = _mean_db(live) if live.size else SILENCE_DB
    spread = float(np.percentile(live, 95) - np.percentile(live, 5)) if live.size > 2 else 0.0
    res.update(level_db=round(level, 1), above_floor_db=round(level - floor, 1), spread_db=round(spread, 1))
    if (i1 - i0) * frame_s < 0.3:
        res["warnings"].append("The window is shorter than 0.3 s, so the noise profile rests on very few frames.")
    if level - floor > 6:
        res["warnings"].append(
            f"This window is {level - floor:.0f} dB louder than the quietest stretch of the recording. "
            "It probably contains speech or an event, and subtraction will eat into those sounds wherever they occur."
        )
    if spread > 10:
        res["warnings"].append(
            f"The level moves by {spread:.0f} dB inside the window, so it is not steady background noise."
        )
    if silent_frac > 0.1:
        res["warnings"].append(
            f"{silent_frac:.0%} of the window is digital silence, which carries no noise to profile."
        )
    return res


# ---------------------------------------------------------------------------
# Hum and steady tones
# ---------------------------------------------------------------------------

def median_spectrum(x: np.ndarray, sr: int, nperseg: int | None = None,
                    max_segments: int = 400) -> tuple[np.ndarray, np.ndarray]:
    """
    Median power spectrum (dB) over up to `max_segments` evenly spaced ~1 s
    segments. The median keeps steady tones and ignores speech and events.
    """
    if nperseg is None:
        nperseg = 1 << int(round(np.log2(sr)))  # ~1 Hz resolution
    nperseg = int(min(nperseg, 1 << int(np.floor(np.log2(max(len(x), 256))))))
    count = max(1, min(max_segments, (len(x) - nperseg) // (nperseg // 2) + 1))
    starts = np.linspace(0, max(0, len(x) - nperseg), count).astype(int)
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    powers = []
    for b in range(0, len(starts), 32):
        seg = np.stack([x[s:s + nperseg] for s in starts[b:b + 32]]) * win
        powers.append((np.abs(np.fft.rfft(seg, axis=1)) ** 2).astype(np.float32))
    P = np.median(np.concatenate(powers), axis=0)
    f = np.fft.rfftfreq(nperseg, 1 / sr)
    return f, (10 * np.log10(P + 1e-20)).astype(np.float32)


def spectral_excess(f: np.ndarray, P_db: np.ndarray, smooth_hz: float = 20.0) -> np.ndarray:
    """dB above a running median of the spectrum: how far a bin stands out as a tone."""
    df = f[1] - f[0]
    size = max(5, int(smooth_hz / df) | 1)
    return P_db - median_filter(P_db, size=size, mode="nearest")


def _parabolic(y: np.ndarray, i: int) -> float:
    if 0 < i < len(y) - 1:
        a, b, c = float(y[i - 1]), float(y[i]), float(y[i + 1])
        den = a - 2 * b + c
        if den != 0:
            return i + 0.5 * (a - c) / den
    return float(i)


def _peak_near(f, P_db, excess, target, half_width):
    df = f[1] - f[0]
    lo = max(1, int((target - half_width) / df))
    hi = min(len(f) - 1, int(np.ceil((target + half_width) / df)) + 1)
    if hi <= lo:
        return None
    i = lo + int(np.argmax(excess[lo:hi]))
    return {"freq_hz": round(_parabolic(P_db, i) * df, 2), "prominence_db": round(float(excess[i]), 1)}


def detect_mains(f: np.ndarray, P_db: np.ndarray, excess: np.ndarray | None = None) -> dict:
    """Which mains family (50 or 60 Hz) stands out, scored on harmonics 1-10 not shared by both."""
    if excess is None:
        excess = spectral_excess(f, P_db)
    scores = {}
    for base in (50, 60):
        s = []
        for k in range(1, 11):
            fk = k * base
            if fk % 300 == 0 or fk >= f[-1]:
                continue
            pk = _peak_near(f, P_db, excess, fk, max(1.0, 0.004 * fk))
            s.append(max(0.0, pk["prominence_db"]) if pk else 0.0)
        scores[base] = round(float(np.mean(s)) if s else 0.0, 1)
    best = max(scores, key=scores.get)
    return {"base_hz": best if scores[best] >= 3.0 else None, "scores": scores}


def hum_harmonics(f: np.ndarray, P_db: np.ndarray, base_hz: float, fmax: float,
                  min_prominence_db: float = 6.0, excess: np.ndarray | None = None) -> list[dict]:
    """Harmonics of `base_hz` up to `fmax` that stand out of the median spectrum."""
    if excess is None:
        excess = spectral_excess(f, P_db)
    out = []
    k = 1
    while k * base_hz <= min(fmax, f[-1] * 0.98):
        fk = k * base_hz
        pk = _peak_near(f, P_db, excess, fk, max(1.0, 0.004 * fk))
        if pk and pk["prominence_db"] >= min_prominence_db:
            out.append({"harmonic": k, **pk})
        k += 1
    return out


def detect_tones(f: np.ndarray, P_db: np.ndarray, fmin: float = 40.0, fmax: float | None = None,
                 min_prominence_db: float = 10.0, exclude_hz: list[float] = (),
                 max_tones: int = 20, excess: np.ndarray | None = None) -> list[dict]:
    """Steady tones (whines, carrier tones) other than those listed in `exclude_hz`."""
    if excess is None:
        excess = spectral_excess(f, P_db)
    df = f[1] - f[0]
    fmax = f[-1] * 0.98 if fmax is None else fmax
    peaks, props = signal.find_peaks(excess, height=min_prominence_db, distance=max(1, int(5 / df)))
    out = []
    for i, h in zip(peaks, props["peak_heights"]):
        fr = _parabolic(P_db, int(i)) * df
        if fr < fmin or fr > fmax or any(abs(fr - e) < 2.0 for e in exclude_hz):
            continue
        out.append({"freq_hz": round(fr, 2), "prominence_db": round(float(h), 1)})
    out.sort(key=lambda d: -d["prominence_db"])
    return out[:max_tones]


def notch_filter(x: np.ndarray, sr: int, freqs, q: float = 30.0) -> np.ndarray:
    """Cascade of zero-phase IIR notches (bandwidth = f / q) at `freqs`."""
    sos = []
    for f0 in freqs:
        if 0 < f0 < 0.49 * sr:
            b, a = signal.iirnotch(f0, q, fs=sr)
            sos.append(signal.tf2sos(b, a))
    if not sos:
        return x
    return signal.sosfiltfilt(np.concatenate(sos), x).astype(np.float32)


# ---------------------------------------------------------------------------
# Spectral subtraction, band-pass, exciter
# ---------------------------------------------------------------------------

def _pow2(n: float) -> int:
    return 1 << max(6, int(round(np.log2(n))))


def spectral_subtraction(x: np.ndarray, sr: int, noise_start_s: float, noise_end_s: float,
                         alpha: float = 1.5, floor_db: float = 20.0,
                         nperseg: int | None = None, batch: int = 4096) -> np.ndarray:
    """
    Magnitude spectral subtraction (Boll, 1979) with a spectral floor to limit
    musical noise. Runs frame batches through a sqrt-Hann STFT with 50% overlap
    (perfect reconstruction at unity gain), so memory stays bounded on long files.
    """
    nperseg = nperseg or _pow2(0.032 * sr)
    hop = nperseg // 2
    win = np.sqrt(signal.windows.hann(nperseg, sym=False)).astype(np.float32)
    i0 = max(0, int(noise_start_s * sr))
    i1 = min(len(x), int(noise_end_s * sr))
    noise = np.asarray(x[i0:i1], dtype=np.float32)
    if len(noise) < nperseg:
        raise ValueError(f"The noise window must be at least {nperseg / sr * 1000:.0f} ms long.")
    nf = sliding_window_view(noise, nperseg)[::hop] * win
    noise_mag = np.mean(np.abs(np.fft.rfft(nf, axis=1)), axis=0).astype(np.float32)
    floor_ratio = np.float32(10 ** (-floor_db / 20.0))

    pad = hop
    xp = np.pad(np.asarray(x, np.float32), (pad, pad + nperseg))
    frames = sliding_window_view(xp, nperseg)[::hop]
    n_frames = len(frames)
    blocks = np.zeros((n_frames + 1, hop), np.float32)
    for b0 in range(0, n_frames, batch):
        fr = frames[b0:b0 + batch] * win
        Z = np.fft.rfft(fr, axis=1)
        mag = np.abs(Z)
        sub = np.maximum(mag - alpha * noise_mag, mag * floor_ratio)
        gain = np.divide(sub, mag, out=np.ones_like(mag), where=mag > 0)
        y = (np.fft.irfft(Z * gain, n=nperseg, axis=1) * win).astype(np.float32)
        blocks[b0:b0 + len(y)] += y[:, :hop]
        blocks[b0 + 1:b0 + 1 + len(y)] += y[:, hop:]
    return blocks.ravel()[pad: pad + len(x)].copy()


def bandpass_filter(x: np.ndarray, sr: int, low_hz: float = 300, high_hz: float = 3400, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth band-pass; the top edge is kept below Nyquist."""
    nyq = sr / 2.0
    low = max(1.0, low_hz) / nyq
    high = min(0.95 * nyq, high_hz) / nyq
    if high <= low:
        raise ValueError("Band-pass high edge must be above the low edge (and below half the analysis rate).")
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, x).astype(np.float32)


def harmonic_exciter(x: np.ndarray, sr: int, drive: float = 0.3, mix: float = 0.25, hp_hz: float = 1000.0) -> np.ndarray:
    """
    tanh waveshaping generates upper harmonics derived from the input itself;
    only the high-passed residual is blended back. Same principle as a studio
    exciter: it does not invent content.
    """
    driven = np.tanh(x * (1.0 + drive * 10.0))
    residual = driven - x
    nyq = sr / 2.0
    sos = signal.butter(2, min(nyq - 1.0, hp_hz) / nyq, btype="high", output="sos")
    harmonics = signal.sosfiltfilt(sos, residual)
    return np.clip(x + mix * harmonics, -1.0, 1.0).astype(np.float32)


# ---------------------------------------------------------------------------
# Transients and the sharp-onset test
# ---------------------------------------------------------------------------

IMPULSIVE = "impulsive"
GRADUAL = "gradual"
# Defaults for the sharp-onset test. Provisional: set from semi-synthetic
# clips (read speech with knocks and clicks mixed in); recalibrate against
# hand-labelled clips with tools/validate_transients.py.
DEFAULT_JUMP_DB = 12.0
DEFAULT_DECAY_DB = 6.0


def detect_transients(x: np.ndarray, sr: int, frame_ms: float = 10.0, rise_db: float = 6.0,
                      max_duration_s: float = 0.5, background_s: float = 1.0,
                      floor_db: float = -60.0) -> list[dict]:
    """
    Short bursts where the 10 ms RMS envelope rises at least `rise_db` above
    its local background (rolling median over `background_s`) for no longer
    than `max_duration_s`. The first and last 100 ms are skipped because
    filter edge effects cause spurious rises there.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    env_db = frame_levels_db(x, sr, frame_len / sr).astype(np.float64)
    if env_db.size == 0:
        return []
    win = max(3, int(background_s * 1000 / frame_ms) | 1)
    background = median_filter(env_db, size=win, mode="nearest")
    rise = env_db - background
    above = (rise > rise_db) & (env_db > floor_db)
    above = binary_closing(above, structure=np.ones(3, dtype=bool))
    edge = int(100 / frame_ms)
    above[:edge] = False
    above[-edge:] = False

    edges = np.diff(np.concatenate(([0], above.view(np.int8), [0])))
    events = []
    for i, j in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        duration = (j - i) * frame_len / sr
        if duration <= max_duration_s:
            events.append({
                "start_s": round(i * frame_len / sr, 3),
                "end_s": round(j * frame_len / sr, 3),
                "duration_s": round(duration, 3),
                "peak_db": round(float(env_db[i:j].max()), 1),
                "rise_db": round(float(rise[i:j].max()), 1),
            })
    return events


@lru_cache(maxsize=8)
def _highpass_sos(cut_hz: float, sr: int, order: int = 2) -> np.ndarray:
    return signal.butter(order, cut_hz / (sr / 2), btype="high", output="sos")


def _level_db(y: np.ndarray, n: int, hop: int) -> np.ndarray:
    """RMS level (dB) of the `n` samples starting at each hop."""
    return 10 * np.log10(uniform_filter1d(y * y, n, origin=-(n // 2), mode="nearest")[::hop] + 1e-12)


def onset_features(x: np.ndarray, sr: int, start_s: float, end_s: float,
                   prior_ms: float = 30.0) -> dict:
    """
    Attack shape of one event, measured on full-band audio. (The band-passed
    copy is not used: zero-phase band-pass filters smear an impulse's attack
    both ways.)

    onset_s  - the sharpest point in the event: where the 2 ms level most
               exceeds the loudest 2 ms of the preceding `prior_ms`.
    jump_db  - that excess. Knocks and clicks reach full level within a
               couple of ms, so they jump well above whatever came just before
               them; a syllable swells over tens of ms and jumps little
               at any one instant.
    decay_db - level in the 15 ms after the onset minus the level 60-120 ms
               after it. Knocks and clicks die away; a plosive such as /t/ or
               /k/ is just as abrupt but runs straight into a vowel.
    """
    hop = max(1, int(0.0005 * sr))
    settle = int(0.05 * sr)
    i0, i1 = int(start_s * sr), int(end_s * sr)
    a = max(0, i0 - int(prior_ms / 1000 * sr) - int(0.02 * sr) - settle)
    b = min(len(x), i1 + int(0.15 * sr))
    nan = {"onset_s": float("nan"), "jump_db": float("nan"), "decay_db": float("nan")}
    seg = np.asarray(x[a:b], dtype=np.float64)
    if len(seg) < int(0.05 * sr):
        return nan
    seg = signal.sosfilt(_highpass_sos(150.0, sr), seg)  # causal: keeps attacks sharp, removes rumble
    L2 = _level_db(seg, max(1, int(0.002 * sr)), hop)
    L10 = _level_db(seg, max(1, int(0.010 * sr)), hop)
    P, G = int(prior_ms / 0.5), 4  # frames of 0.5 ms; G = 2 ms gap
    prior = np.full_like(L2, np.inf)
    if len(L2) > P:
        run = sliding_window_view(L2, P - G + 1).max(axis=1)  # run[k] = max L2[k : k+P-G+1]
        prior[P:] = run[: len(L2) - P]                        # max over [t-P, t-G]
    jump = L2 - prior
    lo = max(P, (i0 - a) // hop - int(0.02 * sr) // hop)
    hi = min(len(L2) - int(0.12 * sr) // hop, max(lo + 1, (i1 - a) // hop))
    if hi <= lo:
        return nan
    t = lo + int(np.argmax(jump[lo:hi]))
    f = lambda ms: int(ms / 1000 * sr) // hop
    onset_level = float(L10[t:t + f(15) + 1].max())
    later = float(np.mean(L10[t + f(60):t + f(120) + 1]))
    return {"onset_s": round((a + t * hop) / sr, 4), "jump_db": round(float(jump[t]), 1),
            "decay_db": round(onset_level - later, 1)}


def classify_transients(events: list[dict], x_fullband: np.ndarray, sr: int,
                        jump_db: float = DEFAULT_JUMP_DB, decay_db: float = DEFAULT_DECAY_DB) -> list[dict]:
    """
    Add onset features and a call to each event, strongest first: impulsive
    (knock/click-like) when the sharpest point jumps at least `jump_db` and
    the sound then falls at least `decay_db`; otherwise gradual (speech-like).
    """
    out = []
    for ev in events:
        feat = onset_features(x_fullband, sr, ev["start_s"], ev["end_s"])
        j, dcy = feat["jump_db"], feat["decay_db"]
        sharp = j == j and dcy == dcy and j >= jump_db and dcy >= decay_db
        out.append({**ev, **feat, "onset": IMPULSIVE if sharp else GRADUAL})
    out.sort(key=lambda e: -e["rise_db"])
    return out


# Hand labels ------------------------------------------------------------------

_IMPULSIVE_WORDS = {"knock", "click", "tap", "bang", "thud", "impact", "impulse", "impulsive",
                    "clap", "snap", "slam", "door", "hit", "clack", "pop"}
_SPEECH_WORDS = {"speech", "syllable", "voice", "word", "vowel", "onset", "vocal", "talk",
                 "gradual", "breath", "laugh", "cough"}


def label_category(label: str) -> str:
    words = {w for w in "".join(c if c.isalnum() else " " for c in label.lower()).split()}
    if words & _IMPULSIVE_WORDS:
        return IMPULSIVE
    if words & _SPEECH_WORDS:
        return GRADUAL
    return "other"


def parse_labels(text: str) -> list[dict]:
    """
    Read hand labels from either an Audacity label export (tab-separated
    start, end, label; no header) or a CSV with a header containing a time
    column (time_s / time / start_s / start) and a label column (label / class).
    """
    rows = []
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("\\")]
    if not lines:
        return rows
    first = lines[0]
    if "\t" in first:
        for ln in lines:
            parts = ln.split("\t")
            try:
                s = float(parts[0])
            except ValueError:
                continue
            e = float(parts[1]) if len(parts) > 2 else s
            lab = parts[2] if len(parts) > 2 else parts[-1]
            rows.append({"time_s": s, "end_s": e, "label": lab.strip()})
        return rows
    reader = csv.DictReader(io.StringIO("\n".join(lines)))
    cols = {c.lower().strip(): c for c in (reader.fieldnames or [])}
    tcol = next((cols[c] for c in ("time_s", "time", "start_s", "start") if c in cols), None)
    lcol = next((cols[c] for c in ("label", "class", "type") if c in cols), None)
    ecol = next((cols[c] for c in ("end_s", "end") if c in cols), None)
    if tcol is None or lcol is None:
        raise ValueError("Label CSV needs a time column (time_s/time/start_s/start) and a label column (label/class).")
    for r in reader:
        s = float(r[tcol])
        rows.append({"time_s": s, "end_s": float(r[ecol]) if ecol and r.get(ecol) else s,
                     "label": (r[lcol] or "").strip()})
    return rows


def evaluate_labels(events: list[dict], labels: list[dict], tolerance_s: float = 0.1) -> dict:
    """
    Match each hand label to the nearest detected event (one-to-one, within
    `tolerance_s` of the event span) and score both detection and the
    impulsive/gradual call.
    """
    pairs = []
    for li, lab in enumerate(labels):
        t0, t1 = lab["time_s"], max(lab["time_s"], lab.get("end_s", lab["time_s"]))
        for ei, ev in enumerate(events):
            gap = max(ev["start_s"] - t1, t0 - ev["end_s"], 0.0)
            if gap <= tolerance_s:
                pairs.append((gap, abs(ev["start_s"] - t0), li, ei))
    pairs.sort()
    used_l, used_e, match = set(), set(), {}
    for _, _, li, ei in pairs:
        if li not in used_l and ei not in used_e:
            used_l.add(li)
            used_e.add(ei)
            match[li] = ei
    rows = []
    for li, lab in enumerate(labels):
        cat = label_category(lab["label"])
        ev = events[match[li]] if li in match else None
        rows.append({
            "time_s": lab["time_s"], "label": lab["label"], "category": cat,
            "detected": ev is not None,
            "event_start_s": ev["start_s"] if ev else None,
            "predicted": ev.get("onset") if ev else None,
            "jump_db": ev.get("jump_db") if ev else None,
            "decay_db": ev.get("decay_db") if ev else None,
            "correct": (ev.get("onset") == cat) if (ev and cat != "other") else None,
        })
    summary = {}
    for cat in (IMPULSIVE, GRADUAL, "other"):
        rs = [r for r in rows if r["category"] == cat]
        if not rs:
            continue
        det = [r for r in rs if r["detected"]]
        summary[cat] = {
            "labelled": len(rs), "detected": len(det),
            "called_impulsive": sum(r["predicted"] == IMPULSIVE for r in det),
            "called_gradual": sum(r["predicted"] == GRADUAL for r in det),
        }
    return {"rows": rows, "summary": summary,
            "unlabelled_events": len(events) - len(used_e),
            "unlabelled_impulsive": sum(1 for i, e in enumerate(events)
                                        if i not in used_e and e.get("onset") == IMPULSIVE)}


# ---------------------------------------------------------------------------
# Gain drops and digital silence
# ---------------------------------------------------------------------------

def detect_gain_drops(x: np.ndarray, sr: int, drop_db: float = 20.0, hold_ms: float = 300.0,
                      ref_ms: float = 300.0, frame_ms: float = 10.0,
                      floor_margin_db: float = 6.0) -> list[dict]:
    """
    Sustained level drops, meant to run on the unprocessed recording.

    A drop is flagged at frame i when the median level over the `ref_ms`
    before it exceeds the 90th percentile over the `hold_ms` after it by at
    least `drop_db` — i.e. the level falls and stays down. Drops into digital
    silence are left to the separate digital-silence report.

    `below_floor_db` says how far the quiet part (10th percentile after the
    drop) sits under the recording's room tone. A pause in speech only falls
    back to room tone; a mute or a gain change takes the room tone down with
    it. `below_room_tone` is True when that exceeds `floor_margin_db`.
    """
    env = frame_levels_db(x, sr, frame_ms / 1000).astype(np.float64)
    R = max(3, int(ref_ms / frame_ms) | 1)
    H = max(3, int(hold_ms / frame_ms) | 1)
    n = len(env)
    if n < R + H + 2:
        return []
    floor = noise_floor_db(env.astype(np.float32))
    med = median_filter(env, size=R, mode="nearest")
    p90 = percentile_filter(env, 90, size=H, mode="nearest")
    p10 = percentile_filter(env, 10, size=H, mode="nearest")
    p50 = median_filter(env, size=H, mode="nearest")
    idx = np.arange(R, n - H + 1)
    before = med[idx - 1 - R // 2]
    after90 = p90[idx + H // 2]
    cand = np.zeros(n, bool)
    cand[idx] = (before - after90) >= drop_db
    edges = np.diff(np.concatenate(([0], cand.view(np.int8), [0])))
    smooth = median_filter(env, size=5, mode="nearest")
    drops = []
    for g0, g1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        lo = max(1, g0 - 3)
        steps = env[lo - 1:g1 - 1] - env[lo:g1]
        i = lo + int(np.argmax(steps))
        i = min(max(i, R), n - H)
        b = float(med[i - 1 - R // 2])
        a50 = float(p50[i + H // 2])
        if a50 <= _SILENCE_CUTOFF_DB:
            continue  # into digital silence: reported separately
        rec = np.flatnonzero(smooth[i:] >= b - drop_db / 2)
        low_frames = int(rec[0]) if rec.size else n - i
        below_floor = floor - float(p10[i + H // 2])
        s0 = max(1, i - 5)
        drops.append({
            "time_s": round(i * frame_ms / 1000, 2),
            "before_db": round(b, 1),
            "after_db": round(a50, 1),
            "drop_db": round(b - a50, 1),
            "low_for_s": round(low_frames * frame_ms / 1000, 2),
            "sharpest_step_db": round(float(np.max(env[s0 - 1:i + 5] - env[s0:i + 6])), 1),
            "below_floor_db": round(below_floor, 1),
            "below_room_tone": bool(below_floor >= floor_margin_db),
        })
    return drops


def describe_silence(runs: list[tuple[float, float]], duration_s: float, min_s: float = 0.02) -> list[dict]:
    out = []
    for s, e in runs:
        if e - s < min_s:
            continue
        where = "file start" if s <= 0.001 else ("file end" if e >= duration_s - 0.001 else "mid-file")
        out.append({"start_s": round(s, 3), "end_s": round(e, 3),
                    "length_ms": round((e - s) * 1000, 1), "position": where})
    return out


# ---------------------------------------------------------------------------
# ENF (electric network frequency)
# ---------------------------------------------------------------------------

def _local_linear_jumps(t: np.ndarray, y: np.ndarray, win: int, guard: int) -> np.ndarray:
    """
    For each sample, fit straight lines to `win` samples on each side (leaving
    `guard` samples out next to it) and return the gap between the two fits'
    values at that sample.
    """
    n = len(y)
    out = np.full(n, np.nan)
    if n < 2 * (win + guard) + 1:
        return out
    ct = np.concatenate(([0.0], np.cumsum(t)))
    cy = np.concatenate(([0.0], np.cumsum(y)))
    ctt = np.concatenate(([0.0], np.cumsum(t * t)))
    cty = np.concatenate(([0.0], np.cumsum(t * y)))

    def fit(a, b):
        m = b - a
        st, sy, stt, sty = ct[b] - ct[a], cy[b] - cy[a], ctt[b] - ctt[a], cty[b] - cty[a]
        den = m * stt - st ** 2
        slope = (m * sty - st * sy) / den
        return slope, (sy - slope * st) / m

    i = np.arange(win + guard, n - win - guard)
    sl, il = fit(i - guard - win, i - guard)
    sr_, ir = fit(i + guard + 1, i + guard + 1 + win)
    out[i] = (sl * t[i] + il) - (sr_ * t[i] + ir)
    return out


def _pick_events(t, score, thr, min_sep):
    """
    Group samples where |score| > thr (merging groups closer than `min_sep`
    seconds) and return one index per group: the sample nearest the group's
    midpoint. A step makes the score high across a plateau either side of it,
    so the midpoint is a steadier location than the noisy maximum.
    """
    over = np.flatnonzero(np.nan_to_num(np.abs(score)) > thr)
    groups: list[list[int]] = []
    for i in over:
        if groups and t[i] - t[groups[-1][-1]] < min_sep:
            groups[-1].append(i)
        else:
            groups.append([i])
    picks = []
    for g in groups:
        mid = (t[g[0]] + t[g[-1]]) / 2
        picks.append(g[int(np.argmin(np.abs(t[g] - mid)))])
    return picks


def enf_analysis(x: np.ndarray, sr: int, nominal_hz: float = 50.0, harmonic: int | None = None,
                 frame_s: float = 2.0, hop_s: float = 0.5, min_snr_db: float = 10.0,
                 freq_jump_hz: float = 0.02, phase_jump_deg: float = 30.0) -> dict:
    """
    Track the mains hum and look for discontinuities.

    The recording is decimated to 1 kHz. The tracked harmonic (auto: the
    strongest of 1-6 in the median spectrum) is measured two ways:
      * frequency per `frame_s` frame (zero-padded FFT + parabolic peak),
        reported as the equivalent fundamental;
      * phase of the hum relative to the nominal frequency, from a narrow
        band-pass and the analytic signal.
    A splice between recordings made at different times shows as a step in
    frequency; a cut within one recording usually shows as a step in phase.
    Flags need the hum to be at least `min_snr_db` above nearby noise.
    Stretches where the hum vanishes are listed as `hum_dropouts`; phase is
    not judged across them.
    """
    fs = 1000
    g = gcd(fs, sr)
    y = signal.resample_poly(np.asarray(x, np.float64), fs // g, sr // g)
    f_med, P = median_spectrum(y, fs, nperseg=4096)
    exc = spectral_excess(f_med, P, smooth_hz=10)
    cands = []
    for k in range(1, 7):
        if k * nominal_hz >= 0.45 * fs:
            break
        pk = _peak_near(f_med, P, exc, k * nominal_hz, max(1.0, 0.004 * k * nominal_hz))
        cands.append((pk["prominence_db"] if pk else -99.0, k))
    if harmonic is None:
        harmonic = max(cands)[1]
    fc = harmonic * nominal_hz
    band = max(1.0, 0.01 * fc)

    # Frequency per frame
    N = int(frame_s * fs)
    hop = int(hop_s * fs)
    nfft = 1 << 16
    win = signal.windows.hann(N, sym=False)
    frames = sliding_window_view(y, N)[::hop] if len(y) >= N else np.zeros((0, N))
    fr = np.fft.rfftfreq(nfft, 1 / fs)
    lo, hi = np.searchsorted(fr, fc - band), np.searchsorted(fr, fc + band)
    nlo, nhi = np.searchsorted(fr, fc - 5 * band), np.searchsorted(fr, fc + 5 * band)
    t_f, f_enf, snr = [], [], []
    for b0 in range(0, len(frames), 256):
        F = np.abs(np.fft.rfft(frames[b0:b0 + 256] * win, n=nfft, axis=1)) ** 2
        for j, row in enumerate(F):
            seg = 10 * np.log10(row[lo:hi] + 1e-30)
            i = int(np.argmax(seg))
            f_hat = fr[lo] + _parabolic(seg, i) * (fr[1] - fr[0])
            ring = np.concatenate((row[nlo:lo], row[hi:nhi]))
            snr.append(10 * np.log10(row[lo + i] / (np.median(ring) + 1e-30) + 1e-30))
            f_enf.append(f_hat / harmonic)
            t_f.append(((b0 + j) * hop + N / 2) / fs)
    t_f, f_enf, snr = np.array(t_f), np.array(f_enf), np.array(snr)

    # Phase relative to the nominal frequency. The nominal rotation is removed
    # at the full 1 kHz rate before averaging down to 20 samples/s, so the
    # remaining phase moves slowly and unwraps without aliasing.
    sos = signal.butter(2, [(fc - 2 * band) / (fs / 2), (fc + 2 * band) / (fs / 2)], btype="band", output="sos")
    nb = signal.sosfiltfilt(sos, y)
    step = 50
    z = signal.hilbert(nb) * np.exp(-2j * np.pi * fc * np.arange(len(nb)) / fs)
    m = len(z) // step
    zd = z[: m * step].reshape(m, step).mean(axis=1)
    t_p = (np.arange(m) + 0.5) * step / fs
    ph = np.unwrap(np.angle(zd))
    if len(ph) > 1:
        ph = ph - np.polyval(np.polyfit(t_p, ph, 1), t_p)
    ph_deg = np.degrees(ph)

    good_f = snr >= min_snr_db
    freq_events = []
    # Median frequency just before vs just after each point, leaving out the
    # frames that straddle it (a phase break inside a frame distorts that
    # frame's estimate without the mains frequency having changed).
    L = max(1, int(round(frame_s / hop_s)))
    n_f = len(f_enf)
    if n_f > 6 * L:
        fg = np.where(good_f, f_enf, np.nan)
        diffs = np.full(n_f, np.nan)
        for i in range(3 * L, n_f - 3 * L):
            b, a_ = fg[i - 3 * L:i - L], fg[i + L:i + 3 * L]
            if np.count_nonzero(~np.isnan(b)) >= L and np.count_nonzero(~np.isnan(a_)) >= L:
                diffs[i] = np.nanmedian(a_) - np.nanmedian(b)
        fin = diffs[np.isfinite(diffs)]
        if fin.size:
            mad = np.median(np.abs(fin - np.median(fin)))
            thr = max(freq_jump_hz, 8 * 1.4826 * mad)
            for i in _pick_events(t_f, diffs, thr, frame_s * 2):
                w = np.abs(t_f - t_f[i]) <= frame_s
                j = np.flatnonzero(w)[int(np.nanargmax(np.abs(np.nan_to_num(diffs[w]))))]
                freq_events.append({"time_s": round(float(t_f[i]), 2), "jump_hz": round(float(diffs[j]), 4),
                                    "threshold_hz": round(float(thr), 4)})

    # Where the hum itself disappears (a mute, digital silence, the hum
    # switching off) its phase is undefined; list those stretches separately.
    rate = fs / step
    amp = median_filter(np.abs(zd), size=max(1, int(0.2 * rate)) | 1, mode="nearest")
    weak = amp < 0.25 * (np.median(amp) if amp.size else 0.0)
    dropouts = []
    edges = np.diff(np.concatenate(([0], weak.view(np.int8), [0])))
    for a0, a1 in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)):
        if (a1 - a0) / rate >= 0.2:
            dropouts.append({"start_s": round(float(a0 / rate), 2), "end_s": round(float(a1 / rate), 2)})

    phase_events = []
    guard = int(0.6 * rate)
    jumps = _local_linear_jumps(t_p, ph_deg, win=int(1.0 * rate), guard=guard)
    jumps = (jumps + 180.0) % 360.0 - 180.0  # phase is only defined modulo 360°
    near_weak = np.convolve(weak.astype(float), np.ones(2 * (guard + int(rate)) + 1), mode="same") > 0
    jumps[near_weak] = np.nan
    if len(t_f):
        snr_at_p = np.interp(t_p, t_f, snr)
        jumps = np.where(snr_at_p >= min_snr_db, jumps, np.nan)
    finite = jumps[np.isfinite(jumps)]
    if finite.size:
        sigma = 1.4826 * np.median(np.abs(finite - np.median(finite)))
        thr = max(phase_jump_deg, 8 * sigma)
        for i in _pick_events(t_p, jumps, thr, 2.0):
            w = np.abs(t_p - t_p[i]) <= 1.0
            j = np.flatnonzero(w)[int(np.nanargmax(np.abs(np.nan_to_num(jumps[w]))))]
            phase_events.append({"time_s": round(float(t_p[i]), 2), "jump_deg": round(float(jumps[j]), 1),
                                 "threshold_deg": round(float(thr), 1)})

    usable = f_enf[good_f] if good_f.any() else f_enf
    return {
        "nominal_hz": nominal_hz, "harmonic": int(harmonic), "tracked_hz": fc,
        "harmonic_prominence_db": {int(k): round(float(p), 1) for p, k in cands},
        "t_freq": t_f, "enf_hz": f_enf, "snr_db": snr,
        "t_phase": t_p, "phase_deg": ph_deg,
        "mean_hz": round(float(np.mean(usable)), 4) if usable.size else None,
        "std_hz": round(float(np.std(usable)), 4) if usable.size else None,
        "usable_fraction": round(float(np.mean(good_f)), 3) if good_f.size else 0.0,
        "freq_events": freq_events, "phase_events": phase_events,
        "hum_dropouts": dropouts,
    }


# ---------------------------------------------------------------------------
# Display helpers (no plotting library needed)
# ---------------------------------------------------------------------------

def minmax_envelope(x: np.ndarray, sr: int, n_cols: int = 2000) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-column min and max of the waveform, for drawing long files at screen width."""
    n = len(x)
    cols = max(1, min(n_cols, n))
    step = n // cols
    body = np.asarray(x[: step * cols]).reshape(cols, step)
    t = (np.arange(cols) + 0.5) * step / sr
    return t, body.min(axis=1), body.max(axis=1)


def spectrogram_image(x: np.ndarray, sr: int, max_cols: int = 1200, fmax: float = 6000.0,
                      nperseg: int | None = None, batch: int = 8192) -> tuple[np.ndarray, tuple]:
    """
    Spectrogram in dB, mean-pooled in time down to at most `max_cols` columns
    and cut at `fmax`, so a long file draws as a screen-sized image.
    Returns (img [freq x time], extent (t0, t1, f0, f1)).
    """
    nperseg = nperseg or _pow2(0.032 * sr)
    x = np.asarray(x, np.float32)
    if len(x) < nperseg:
        x = np.pad(x, (0, nperseg - len(x)))
    # Short sections get dense frames for detail; long files use 50 % overlap,
    # which still covers every sample and is pooled into columns anyway.
    hop = nperseg // 4 if len(x) < 4 * max_cols * nperseg // 4 else nperseg // 2
    frames = sliding_window_view(x, nperseg)[::hop]
    n_frames = len(frames)
    f = np.fft.rfftfreq(nperseg, 1 / sr)
    nb = int(np.searchsorted(f, min(fmax, sr / 2), side="right"))
    group = int(np.ceil(n_frames / min(max_cols, n_frames)))
    batch = max(group, batch // group * group)       # batches hold whole columns
    win = signal.windows.hann(nperseg, sym=False).astype(np.float32)
    cols = []
    for b0 in range(0, n_frames, batch):
        fr = frames[b0:b0 + batch] * win
        P = np.abs(np.fft.rfft(fr, axis=1)[:, :nb]) ** 2
        starts = np.arange(0, len(P), group)
        sums = np.add.reduceat(P, starts, axis=0)
        counts = np.diff(np.append(starts, len(P)))[:, None]
        cols.append(sums / counts)
    img = 10 * np.log10(np.concatenate(cols) + 1e-20)
    return img.T.astype(np.float32), (0.0, len(x) / sr, 0.0, float(f[nb - 1]))
