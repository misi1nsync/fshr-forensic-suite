#!/usr/bin/env python3
"""
FVTR Suite — Forensic Vocal & Transient Reconstruction (Layer 2: Audio)
========================================================================
Companion DSP module to fshr_app.py (Layer 1: Visual). Applies standard,
well-established audio engineering techniques to noisy speech recordings:
noise-profile spectral subtraction, vocal-band isolation, transient/onset
detection, a harmonic clarity enhancer, and gain-drop (muting) detection.

SCOPE AND LIMITS (read this before interpreting any output)
-------------------------------------------------------------------------
Every stage below is ordinary signal processing: it can suppress noise
that is present, isolate a frequency band that is present, or flag where
the amplitude envelope does something abrupt. None of these techniques
can recover information that was never captured in the original
recording — "cleaning up" a signal is not the same as "proving" what is
in it. In particular the Clarity Enhancement stage adds harmonic content
that is mathematically derived from the audio's own existing waveform
(the same principle as a studio "exciter" effect); it does not invent,
guess, or resynthesize new words or sounds. Treat every output — a
flagged transient, a gain drop, an enhanced clip — as a lead to look at
more closely, never as a standalone conclusion about what happened.

Tools:
  1. Noise Profile Subtraction — spectral subtraction against a
     user-selected noise-only anchor window, with a spectral floor to
     avoid musical-noise artifacts
  2. Vocal Band-Pass + Transient Detection — 300-3400 Hz Butterworth
     band-pass, envelope-based short-transient flagging
  3. Clarity Enhancement — harmonic exciter that boosts existing upper
     harmonics for perceived clarity (does not add new semantic content)
  4. Gain-Drop / Muting Detection — flags abrupt amplitude collapses
     consistent with a manual mute or gain change

Run:
  streamlit run fvtr_app.py
"""

import io
from pathlib import Path

import numpy as np
import streamlit as st
from scipy import signal
from scipy.io import wavfile

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FVTR Suite — Forensic Vocal & Transient Reconstruction",
    page_icon="🎙️",
    layout="wide",
)

st.title("Layer 2 — FVTR Forensic Audio Suite")
st.caption(
    "Noise Profiling  ·  Vocal Band-Pass  ·  Transient Detection  ·  "
    "Clarity Enhancement  ·  Gain-Drop Logging"
)
st.warning(
    "**Scope note:** this tool enhances and flags patterns in audio that is "
    "already present in the recording. It cannot recover audio that was "
    "never captured, and no output here is a standalone finding — it's a "
    "lead for further review, not proof of speech content or events.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Audio helpers
# ---------------------------------------------------------------------------

def load_wav_bytes(raw_bytes: bytes) -> tuple[int, np.ndarray]:
    """Read a WAV file from bytes, return (sample_rate, mono float32 in [-1, 1])."""
    sr, data = wavfile.read(io.BytesIO(raw_bytes))

    if data.ndim > 1:
        data = data.mean(axis=1)

    if np.issubdtype(data.dtype, np.integer):
        max_val = float(np.iinfo(data.dtype).max)
        data = data.astype(np.float32) / max_val
    else:
        data = data.astype(np.float32)

    return sr, data


def to_wav_bytes(sr: int, data: np.ndarray) -> bytes:
    """Encode a float32 [-1, 1] mono signal as 16-bit PCM WAV bytes."""
    clipped = np.clip(data, -1.0, 1.0)
    int16_data = (clipped * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    wavfile.write(buf, sr, int16_data)
    return buf.getvalue()


def spectral_subtraction(
    data: np.ndarray,
    sr: int,
    noise_start_s: float,
    noise_end_s: float,
    alpha: float = 1.5,
    floor_db: float = 20.0,
    nperseg: int = 1024,
) -> np.ndarray:
    """
    Classic magnitude spectral subtraction (Boll, 1979).

    A noise magnitude profile is averaged from a user-selected noise-only
    anchor window, scaled by `alpha`, and subtracted from every STFT frame.
    A spectral floor (`floor_db` below the original magnitude) is kept
    instead of subtracting to zero, which is the standard mitigation for
    "musical noise" artifacts in over-subtracted spectra.
    """
    i0 = max(0, int(noise_start_s * sr))
    i1 = min(len(data), int(noise_end_s * sr))
    if i1 <= i0:
        return data.copy()

    noise_clip = data[i0:i1]

    f, t, Zxx = signal.stft(data, fs=sr, nperseg=nperseg)
    _, _, Nxx = signal.stft(noise_clip, fs=sr, nperseg=min(nperseg, len(noise_clip)))

    noise_mag = np.mean(np.abs(Nxx), axis=1, keepdims=True)
    mag = np.abs(Zxx)
    phase = np.angle(Zxx)

    floor_ratio = 10 ** (-floor_db / 20.0)
    sub_mag = np.maximum(mag - alpha * noise_mag, mag * floor_ratio)

    cleaned = sub_mag * np.exp(1j * phase)
    _, out = signal.istft(cleaned, fs=sr, nperseg=nperseg)
    out = out[: len(data)]
    if len(out) < len(data):
        out = np.pad(out, (0, len(data) - len(out)))
    return out.astype(np.float32)


def bandpass_filter(data: np.ndarray, sr: int, low_hz: float = 300, high_hz: float = 3400, order: int = 5) -> np.ndarray:
    """Zero-phase Butterworth band-pass restricted to the human vocal-clarity band."""
    nyq = sr / 2.0
    low = max(1.0, low_hz) / nyq
    high = min(nyq - 1.0, high_hz) / nyq
    sos = signal.butter(order, [low, high], btype="band", output="sos")
    return signal.sosfiltfilt(sos, data).astype(np.float32)


def detect_transients(
    data: np.ndarray, sr: int, frame_ms: float = 10.0, thresh_db: float = -25.0, max_duration_s: float = 0.5
) -> list[dict]:
    """
    Flag short amplitude bursts (RMS envelope above `thresh_db`, lasting no
    longer than `max_duration_s`) — candidate impact / vocal-onset events.
    """
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(data) // frame_len
    if n_frames == 0:
        return []

    frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    env_db = 20 * np.log10(rms + 1e-12)
    above = env_db > thresh_db

    events = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            duration = (j - i) * frame_len / sr
            if duration <= max_duration_s:
                events.append(
                    {
                        "start_s": round(i * frame_len / sr, 3),
                        "end_s": round(j * frame_len / sr, 3),
                        "duration_s": round(duration, 3),
                        "peak_db": round(float(env_db[i:j].max()), 1),
                    }
                )
            i = j
        else:
            i += 1
    return events


def harmonic_exciter(data: np.ndarray, sr: int, drive: float = 0.3, mix: float = 0.25, hp_hz: float = 1000.0) -> np.ndarray:
    """
    Soft-clip (tanh) waveshaping generates upper harmonics that are
    mathematically derived from the input signal itself; only the
    high-passed harmonic residual is blended back in, so low-frequency
    content is untouched. This is the same principle as a studio "aural
    exciter" — it does not invent content that isn't already there.
    """
    driven = np.tanh(data * (1.0 + drive * 10.0))
    residual = driven - data
    nyq = sr / 2.0
    sos = signal.butter(2, min(nyq - 1.0, hp_hz) / nyq, btype="high", output="sos")
    harmonics = signal.sosfiltfilt(sos, residual)
    return np.clip(data + mix * harmonics, -1.0, 1.0).astype(np.float32)


def detect_gain_drops(data: np.ndarray, sr: int, frame_ms: float = 10.0, drop_db: float = 20.0) -> list[dict]:
    """Flag frame-to-frame RMS collapses of at least `drop_db` — candidate manual mute/gain-change events."""
    frame_len = max(1, int(sr * frame_ms / 1000))
    n_frames = len(data) // frame_len
    if n_frames < 2:
        return []

    frames = data[: n_frames * frame_len].reshape(n_frames, frame_len)
    rms = np.sqrt(np.mean(frames**2, axis=1) + 1e-12)
    env_db = 20 * np.log10(rms + 1e-12)

    drops = []
    for i in range(1, len(env_db)):
        delta = env_db[i - 1] - env_db[i]
        if delta >= drop_db:
            drops.append(
                {
                    "time_s": round(i * frame_len / sr, 3),
                    "drop_db": round(float(delta), 1),
                    "before_db": round(float(env_db[i - 1]), 1),
                    "after_db": round(float(env_db[i]), 1),
                }
            )
    return drops


def plot_waveform(data: np.ndarray, sr: int, title: str, highlight: list[dict] | None = None):
    if not HAS_MPL:
        st.line_chart(data[:: max(1, len(data) // 5000)])
        return
    fig, ax = plt.subplots(figsize=(10, 2.2))
    t = np.arange(len(data)) / sr
    ax.plot(t, data, linewidth=0.5, color="#2b6cb0")
    if highlight:
        for ev in highlight:
            ax.axvspan(ev.get("start_s", ev.get("time_s")), ev.get("end_s", ev.get("time_s") + 0.02), color="#e53e3e", alpha=0.4)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(0, len(data) / sr)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def plot_spectrogram(data: np.ndarray, sr: int, title: str):
    if not HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 3))
    f, t, Sxx = signal.spectrogram(data, fs=sr, nperseg=1024, noverlap=768)
    ax.pcolormesh(t, f, 10 * np.log10(Sxx + 1e-12), shading="auto", cmap="magma")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, min(sr / 2, 6000))
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def build_event_log(transients: list[dict], gain_drops: list[dict]) -> str:
    lines = ["FVTR Event Log", "=" * 40, ""]
    lines.append(f"Transient events flagged: {len(transients)}")
    for ev in transients:
        lines.append(
            f"  [TRANSIENT] {ev['start_s']:.3f}s - {ev['end_s']:.3f}s "
            f"(duration {ev['duration_s']:.3f}s, peak {ev['peak_db']:.1f} dB)"
        )
    lines.append("")
    lines.append(f"Gain-drop events flagged: {len(gain_drops)}")
    for ev in gain_drops:
        lines.append(
            f"  [GAIN-DROP] {ev['time_s']:.3f}s "
            f"({ev['before_db']:.1f} dB -> {ev['after_db']:.1f} dB, drop {ev['drop_db']:.1f} dB)"
        )
    lines.append("")
    lines.append(
        "Note: flagged events are candidates for manual review only. "
        "They are not verified findings of speech content, physical contact, "
        "or intent."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Sidebar — parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("FVTR Parameters")

    st.subheader("1. Noise Profile Subtraction")
    noise_start = st.number_input("Noise anchor start (s)", value=0.0, min_value=0.0, step=0.1)
    noise_end = st.number_input("Noise anchor end (s)", value=1.0, min_value=0.1, step=0.1)
    alpha = st.slider("Subtraction coefficient (α)", 0.5, 4.0, 1.5, 0.1)
    floor_db = st.slider("Spectral floor (dB below original)", 6.0, 40.0, 20.0, 1.0,
                          help="Higher = more aggressive subtraction but more musical-noise risk.")

    st.subheader("2. Vocal Band-Pass + Transients")
    low_hz = st.slider("Band-pass low (Hz)", 50, 1000, 300, 10)
    high_hz = st.slider("Band-pass high (Hz)", 1000, 8000, 3400, 100)
    trans_thresh = st.slider("Transient threshold (dB)", -60, 0, -25, 1)
    trans_max_dur = st.slider("Max transient duration (s)", 0.05, 2.0, 0.5, 0.05)

    st.subheader("3. Clarity Enhancement")
    exciter_drive = st.slider("Exciter drive", 0.0, 1.0, 0.3, 0.05)
    exciter_mix = st.slider("Exciter mix", 0.0, 1.0, 0.25, 0.05)

    st.subheader("4. Gain-Drop Detection")
    drop_db = st.slider("Gain-drop threshold (dB)", 6, 40, 20, 1)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload source audio (WAV, mono or stereo)",
    type=["wav"],
    help="Convert other formats to WAV first — this tool works directly with PCM samples.",
)

if uploaded is None:
    st.info("Upload a WAV file to begin.")
    st.stop()

sr, raw = load_wav_bytes(uploaded.read())
duration = len(raw) / sr
st.success(f"Loaded **{uploaded.name}** — {sr} Hz · {duration:.2f} s · {len(raw):,} samples")
st.audio(to_wav_bytes(sr, raw), format="audio/wav")

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_noise, tab_vocal, tab_clarity, tab_gain = st.tabs([
    "🔇 Noise Profile Subtraction",
    "🎚️ Vocal Band-Pass + Transients",
    "✨ Clarity Enhancement",
    "📉 Gain-Drop Detection & Export",
])

# ── Tab 1: Noise Profile Subtraction ────────────────────────────────────────

with tab_noise:
    st.header("Noise Profile Subtraction")
    st.caption(
        "Select a window that contains only background noise (no target speech). "
        "That window's average spectrum is treated as the noise fingerprint and "
        "subtracted from the full recording."
    )

    plot_waveform(raw, sr, "Original waveform (noise anchor selection above)")

    denoised = spectral_subtraction(raw, sr, noise_start, noise_end, alpha=alpha, floor_db=floor_db)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before")
        plot_spectrogram(raw, sr, "Original spectrogram")
        st.audio(to_wav_bytes(sr, raw), format="audio/wav")
    with col2:
        st.subheader("After")
        plot_spectrogram(denoised, sr, "Denoised spectrogram")
        st.audio(to_wav_bytes(sr, denoised), format="audio/wav")

    st.session_state["fvtr_denoised"] = denoised
    st.session_state["fvtr_sr"] = sr

# ── Tab 2: Vocal Band-Pass + Transients ─────────────────────────────────────

with tab_vocal:
    st.header("Vocal Band-Pass + Transient Detection")
    st.caption(
        f"Butterworth band-pass restricted to {low_hz}-{high_hz} Hz (human vocal-clarity band), "
        "followed by envelope-based detection of short amplitude bursts."
    )

    denoised = st.session_state.get("fvtr_denoised", raw)
    filtered = bandpass_filter(denoised, sr, low_hz=low_hz, high_hz=high_hz)
    transients = detect_transients(filtered, sr, thresh_db=trans_thresh, max_duration_s=trans_max_dur)

    plot_waveform(filtered, sr, "Band-passed waveform (red = flagged transients)", highlight=transients)
    st.audio(to_wav_bytes(sr, filtered), format="audio/wav")

    st.subheader(f"Flagged transients ({len(transients)})")
    if transients:
        st.dataframe(transients, use_container_width=True)
    else:
        st.caption("No transients crossed the current threshold.")

    st.session_state["fvtr_filtered"] = filtered
    st.session_state["fvtr_transients"] = transients

# ── Tab 3: Clarity Enhancement ───────────────────────────────────────────────

with tab_clarity:
    st.header("Clarity Enhancement (Harmonic Exciter)")
    st.info(
        "This adds harmonic overtones mathematically derived from the signal's "
        "**own existing waveform** to improve perceived clarity — the same "
        "principle as a studio 'exciter' effect used in music mastering. "
        "It does **not** invent, guess, or resynthesize new words or sounds, "
        "and a clearer-sounding result is not evidence of specific speech content.",
        icon="ℹ️",
    )

    filtered = st.session_state.get("fvtr_filtered", raw)
    enhanced = harmonic_exciter(filtered, sr, drive=exciter_drive, mix=exciter_mix)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Before enhancement")
        st.audio(to_wav_bytes(sr, filtered), format="audio/wav")
        plot_spectrogram(filtered, sr, "Pre-enhancement spectrogram")
    with col2:
        st.subheader("After enhancement")
        st.audio(to_wav_bytes(sr, enhanced), format="audio/wav")
        plot_spectrogram(enhanced, sr, "Post-enhancement spectrogram")

    st.session_state["fvtr_enhanced"] = enhanced

# ── Tab 4: Gain-Drop Detection & Export ─────────────────────────────────────

with tab_gain:
    st.header("Gain-Drop / Muting Detection")
    st.caption(
        "Flags frame-to-frame RMS collapses of at least the configured threshold — "
        "candidates for a manual mute, hardware gain change, or clipping event. "
        "This detects abrupt *amplitude envelope* behavior only, not cause or intent."
    )

    enhanced = st.session_state.get("fvtr_enhanced", raw)
    gain_drops = detect_gain_drops(enhanced, sr, drop_db=drop_db)
    transients = st.session_state.get("fvtr_transients", [])

    plot_waveform(enhanced, sr, "Final waveform (red = flagged gain drops)",
                  highlight=[{"time_s": d["time_s"]} for d in gain_drops])

    st.subheader(f"Flagged gain drops ({len(gain_drops)})")
    if gain_drops:
        st.dataframe(gain_drops, use_container_width=True)
    else:
        st.caption("No gain drops crossed the current threshold.")

    st.divider()
    st.subheader("Export")

    log_text = build_event_log(transients, gain_drops)
    final_wav = to_wav_bytes(sr, enhanced)

    dl1, dl2 = st.columns(2)
    with dl1:
        st.download_button(
            "⬇ Download Sync_Audio_Output.wav",
            data=final_wav,
            file_name="Sync_Audio_Output.wav",
            mime="audio/wav",
        )
    with dl2:
        st.download_button(
            "⬇ Download event log (.txt)",
            data=log_text,
            file_name="fvtr_event_log.txt",
            mime="text/plain",
        )

    with st.expander("Preview event log"):
        st.text(log_text)
