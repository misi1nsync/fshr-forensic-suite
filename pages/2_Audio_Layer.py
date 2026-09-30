#!/usr/bin/env python3
"""
FVTR Suite — Forensic Vocal & Transient Reconstruction (Layer 2: Audio)
========================================================================
Companion DSP module to fshr_app.py (Layer 1: Visual). Applies standard
audio engineering techniques to noisy speech recordings: hum and tone
notches, noise-profile spectral subtraction, vocal-band isolation,
transient detection with a sharp-onset test, a harmonic clarity enhancer,
gain-drop and digital-silence logging, and a mains-hum (ENF) consistency
check. The DSP itself lives in fvtr_dsp.py.

SCOPE AND LIMITS (read this before interpreting any output)
-------------------------------------------------------------------------
Every stage is ordinary signal processing: it can suppress noise that is
present, isolate a band that is present, or flag where a level or the hum
does something abrupt. None of it recovers information that was never
captured. The Clarity stage adds harmonics derived from the audio's own
waveform (a studio "exciter"); it does not invent words or sounds. Treat
every output as a lead to look at more closely, never as a conclusion.

Performance notes
-----------------
* Audio is decoded in blocks and analysed at 16 kHz (enough for a
  300-3400 Hz band and for mains hum), so a long file fits in memory.
* Every stage is cached on the file's hash plus that stage's own settings,
  so moving a slider only recomputes the stages downstream of it.
* Plots are reduced to screen size before drawing and cached as images.
* One player plays a chosen section of a chosen stage; the full-length
  WAV is only built when asked for.

Run:
  streamlit run fshr_app.py   (then open "Audio Layer" in the sidebar)
"""

import hashlib
import io
import sys
from pathlib import Path

import numpy as np
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import fvtr_dsp as d  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

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
    "Noise sample  ·  Hum & tone removal  ·  Spectral subtraction  ·  Vocal band-pass  ·  "
    "Transients  ·  Clarity  ·  Gain drops & digital silence  ·  ENF"
)
st.warning(
    "**Scope note:** this tool enhances and flags patterns in audio that is "
    "already present in the recording. It cannot recover audio that was "
    "never captured, and no output here is a standalone finding — it's a "
    "lead for further review, not proof of speech content or events.",
    icon="⚠️",
)

LEVEL_FRAME_S = 0.05
PLOT_W = 10.0

# ---------------------------------------------------------------------------
# Cached stages. Arguments starting with "_" are not hashed; the key
# arguments before them identify the input instead.
# ---------------------------------------------------------------------------


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


@st.cache_resource(max_entries=2, show_spinner="Decoding and resampling to 16 kHz …")
def get_decoded(digest: str, name: str, _raw: bytes) -> d.DecodedAudio:
    return d.decode_audio(_raw, name)


@st.cache_resource(max_entries=3, show_spinner=False)
def get_channel(digest: str, channel, _dec: d.DecodedAudio):
    return d.select_channel(_dec, channel)


@st.cache_data(max_entries=8, show_spinner=False)
def get_levels(key, _x, sr: int) -> np.ndarray:
    return d.frame_levels_db(_x, sr, LEVEL_FRAME_S)


@st.cache_data(max_entries=8, show_spinner=False)
def get_spectrum(key, _x, sr: int):
    return d.median_spectrum(_x, sr)


@st.cache_resource(max_entries=2, show_spinner="Removing hum and tones …")
def get_dehummed(key, freqs: tuple, q: float, _x, sr: int):
    return _readonly(d.notch_filter(_x, sr, freqs, q)) if freqs else _x


@st.cache_resource(max_entries=2, show_spinner="Spectral subtraction …")
def get_denoised(key, start: float, end: float, alpha: float, floor_db: float, _x, sr: int):
    return _readonly(d.spectral_subtraction(_x, sr, start, end, alpha=alpha, floor_db=floor_db))


@st.cache_resource(max_entries=2, show_spinner="Band-pass filtering …")
def get_band(key, low: float, high: float, _x, sr: int):
    return _readonly(d.bandpass_filter(_x, sr, low, high))


@st.cache_data(max_entries=4, show_spinner="Finding transients …")
def get_transients(key, rise_db, max_dur, jump_db, decay_db, _band, _full, sr: int):
    ev = d.detect_transients(_band, sr, rise_db=rise_db, max_duration_s=max_dur)
    return d.classify_transients(ev, _full, sr, jump_db=jump_db, decay_db=decay_db)


@st.cache_resource(max_entries=1, show_spinner="Clarity enhancement …")
def get_enhanced(key, drive: float, mix: float, _x, sr: int):
    return _readonly(d.harmonic_exciter(_x, sr, drive=drive, mix=mix))


@st.cache_data(max_entries=4, show_spinner="Checking for gain drops …")
def get_gain_drops(key, drop_db, hold_ms, margin_db, _x, sr: int):
    return d.detect_gain_drops(_x, sr, drop_db=drop_db, hold_ms=hold_ms, floor_margin_db=margin_db)


@st.cache_data(max_entries=4, show_spinner="Tracking the mains hum …")
def get_enf(key, nominal, harmonic, frame_s, min_snr, _x, sr: int):
    return d.enf_analysis(_x, sr, nominal_hz=nominal, harmonic=harmonic, frame_s=frame_s,
                          hop_s=frame_s / 4, min_snr_db=min_snr)


@st.cache_resource(max_entries=1, show_spinner="Encoding WAV …")
def get_wav(key, _x, sr: int) -> bytes:
    return d.to_wav_bytes(sr, _x)


# --- Plots, rendered once per input and cached as PNG ----------------------

def _png(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100)
    plt.close(fig)
    return buf.getvalue()


@st.cache_data(max_entries=24, show_spinner=False)
def png_spectrogram(key, title: str, fmax: float, _x, sr: int, t0: float = 0.0) -> bytes:
    img, (a, b, f0, f1) = d.spectrogram_image(_x, sr, max_cols=1000, fmax=fmax)
    fig, ax = plt.subplots(figsize=(PLOT_W, 3))
    vmax = float(np.percentile(img, 99.5))
    ax.imshow(img, origin="lower", aspect="auto", cmap="magma", extent=(t0 + a, t0 + b, f0, f1),
              vmin=vmax - 80, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.tight_layout()
    return _png(fig)


@st.cache_data(max_entries=24, show_spinner=False)
def png_waveform(key, title: str, spans: tuple, _x, sr: int, t0: float = 0.0) -> bytes:
    t, lo, hi = d.minmax_envelope(_x, sr, 2000)
    fig, ax = plt.subplots(figsize=(PLOT_W, 2.2))
    ax.fill_between(t0 + t, lo, hi, color="#2b6cb0", linewidth=0)
    for s, e, colour in spans[:300]:
        ax.axvspan(s, max(e, s + len(_x) / sr / 800), color=colour, alpha=0.45, linewidth=0)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    ax.set_xlim(t0, t0 + len(_x) / sr)
    fig.tight_layout()
    return _png(fig)


@st.cache_data(max_entries=16, show_spinner=False)
def png_levels(key, window: tuple, suggestion: tuple | None, floor_db: float, _env) -> bytes:
    env = np.maximum(_env, -100)
    t, lo, hi = d.minmax_envelope(env, 1 / LEVEL_FRAME_S, 2000)
    fig, ax = plt.subplots(figsize=(PLOT_W, 2.6))
    ax.fill_between(t, lo, hi, color="#4a5568", linewidth=0, label="level (50 ms RMS)")
    ax.axhline(floor_db, color="#718096", linestyle=":", linewidth=1, label="quietest-stretch level")
    if suggestion:
        ax.axvspan(*suggestion, facecolor="none", edgecolor="#d69e2e", hatch="///", linewidth=1,
                   label="suggested noise sample")
    ax.axvspan(*window, color="#38a169", alpha=0.35, label="chosen noise sample")
    ax.set_xlim(0, len(env) * LEVEL_FRAME_S)
    ax.set_ylim(max(-100, float(np.percentile(env, 1)) - 6), float(env.max()) + 3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("dBFS")
    ax.legend(loc="upper right", fontsize=8, ncols=4)
    fig.tight_layout()
    return _png(fig)


@st.cache_data(max_entries=8, show_spinner=False)
def png_spectrum(key, notches: tuple, f, before, after) -> bytes:
    fig, ax = plt.subplots(figsize=(PLOT_W, 3))
    ax.semilogx(f[1:], before[1:], color="#a0aec0", linewidth=0.8, label="before")
    if after is not None:
        ax.semilogx(f[1:], after[1:], color="#2b6cb0", linewidth=0.8, label="after notches")
    for fr in notches:
        ax.axvline(fr, color="#e53e3e", alpha=0.35, linewidth=0.8)
    ax.set_xlim(20, f[-1])
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Median power (dB)")
    ax.set_title("Steady spectrum (median over the recording) — red lines = notches")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return _png(fig)


@st.cache_data(max_entries=4, show_spinner=False)
def png_enf(key, _res) -> bytes:
    r = _res
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(PLOT_W, 5), sharex=True)
    good = r["snr_db"] >= 10
    a1.plot(r["t_freq"][good], r["enf_hz"][good], ".", ms=2, color="#2b6cb0")
    a1.plot(r["t_freq"][~good], r["enf_hz"][~good], ".", ms=2, color="#cbd5e0")
    if good.any():
        m, s = np.median(r["enf_hz"][good]), max(0.02, 4 * np.std(r["enf_hz"][good]))
        a1.set_ylim(m - s, m + s)
    a1.set_ylabel("Mains frequency (Hz)")
    a1.set_title(f"ENF from harmonic {r['harmonic']} ({r['tracked_hz']:.0f} Hz); grey = weak hum")
    a2.plot(r["t_phase"], r["phase_deg"], linewidth=0.7, color="#2f855a")
    a2.set_ylabel("Hum phase drift (°)")
    a2.set_xlabel("Time (s)")
    for ev in r["freq_events"]:
        a1.axvline(ev["time_s"], color="#e53e3e", alpha=0.6)
    for ev in r["phase_events"]:
        a2.axvline(ev["time_s"], color="#e53e3e", alpha=0.6)
    fig.tight_layout()
    return _png(fig)


def show_png(data: bytes):
    st.image(data, width="stretch")


def table_or(rows, empty: str, **kw):
    if rows:
        st.dataframe(rows, hide_index=True, **kw)
    else:
        st.caption(empty)


# ---------------------------------------------------------------------------
# Upload and decode
# ---------------------------------------------------------------------------

types = d.accepted_types()
uploaded = st.file_uploader(
    "Upload source audio (" + ", ".join(t.upper() for t in types) + ")",
    type=types,
    help="WAV, MP3, FLAC and OGG are decoded directly. M4A/AAC need ffmpeg installed on the server.",
)
if not d.ffmpeg_available():
    st.caption("M4A/AAC are not available here because ffmpeg is not installed.")

if uploaded is None:
    st.info("Upload a recording to begin.")
    st.stop()

digests = st.session_state.setdefault("fvtr_digests", {})
if uploaded.file_id not in digests:
    digests[uploaded.file_id] = hashlib.sha1(uploaded.getvalue()).hexdigest()
digest = digests[uploaded.file_id]

try:
    dec = get_decoded(digest, uploaded.name, uploaded.getvalue())
except d.AudioDecodeError as e:
    st.error(str(e))
    st.stop()

sr = dec.sr
duration = dec.duration_s
st.success(
    f"Loaded **{uploaded.name}** — {dec.format}, {duration / 60:.1f} min. "
    f"Analysed at {sr / 1000:g} kHz."
)

# ---------------------------------------------------------------------------
# Sidebar: channel and noise sample (needed before anything else)
# ---------------------------------------------------------------------------

CH_NAMES = {0: "Left", 1: "Right"}


def ch_label(c):
    if c == "mix":
        return "Both mixed"
    return CH_NAMES.get(c, f"Channel {c + 1}") if dec.n_channels > 1 else "Mono"


with st.sidebar:
    st.header("FVTR Parameters")
    st.subheader("Channel")
    ch_options = (["mix"] + list(range(dec.n_channels))) if dec.n_channels > 1 else [0]
    channel = st.radio(
        "Analyse", ch_options, format_func=ch_label, horizontal=True,
        help="With two microphones, mixing can smear or cancel sounds that reach them at "
             "different times. Try each side on its own; one is often cleaner.",
    )

x, silence_runs = get_channel(digest, channel, dec)
k0 = (digest, channel)
env = get_levels(k0, x, sr)

with st.sidebar:
    st.subheader("1. Noise sample")
    sug_len = st.slider("Suggested sample length (s)", 0.5, 5.0, 1.0, 0.5)
    suggestion = d.suggest_noise_window(env, LEVEL_FRAME_S, sug_len)
    ref_db = suggestion["level_db"] if suggestion else None

    # New file or channel: start from the suggested quiet stretch.
    if st.session_state.get("fvtr_noise_for") != k0:
        st.session_state["fvtr_noise_for"] = k0
        s0, s1 = (suggestion["start_s"], suggestion["end_s"]) if suggestion else (0.0, min(1.0, duration))
        st.session_state["noise_start"] = float(s0)
        st.session_state["noise_end"] = float(s1)

    def _use_suggestion(s, e):
        st.session_state["noise_start"] = float(s)
        st.session_state["noise_end"] = float(e)

    if suggestion:
        st.button(
            f"Use quietest stretch: {suggestion['start_s']:.2f}–{suggestion['end_s']:.2f} s",
            on_click=_use_suggestion, args=(suggestion["start_s"], suggestion["end_s"]),
            width="stretch",
        )
    noise_start = st.number_input("Noise sample start (s)", min_value=0.0, max_value=float(duration),
                                  step=0.1, format="%.2f", key="noise_start")
    noise_end = st.number_input("Noise sample end (s)", min_value=0.0, max_value=float(duration),
                                step=0.1, format="%.2f", key="noise_end")
    alpha = st.slider("Subtraction coefficient (α)", 0.0, 4.0, 1.5, 0.1,
                      help="0 turns spectral subtraction off.")
    floor_db = st.slider("Spectral floor (dB below original)", 6.0, 40.0, 20.0, 1.0,
                         help="Higher = more aggressive subtraction but more musical-noise risk.")

noise_check = d.assess_noise_window(env, LEVEL_FRAME_S, noise_start, noise_end, ref_db)

# ---------------------------------------------------------------------------
# Hum and tones
# ---------------------------------------------------------------------------

f_spec, P_spec = get_spectrum(k0, x, sr)
excess = d.spectral_excess(f_spec, P_spec)
mains = d.detect_mains(f_spec, P_spec, excess)

with st.sidebar:
    st.subheader("2. Hum & tone removal")
    detected = f"{mains['base_hz']} Hz found" if mains["base_hz"] else "none found"
    mains_choice = st.selectbox(f"Mains hum ({detected})", ["Auto", "50 Hz", "60 Hz", "Off"])
    hum_max = st.slider("Highest harmonic to notch (Hz)", 100, int(min(4000, sr / 2 * 0.95)),
                        int(min(4000, sr / 2 * 0.95)), 50)
    hum_prom = st.slider("Hum threshold (dB above surrounding spectrum)", 3.0, 20.0, 6.0, 0.5,
                         help="Only harmonics that stand out this much get a notch, so clean parts "
                              "of the spectrum are left alone.")
    use_tones = st.checkbox("Also notch other steady tones", value=False,
                            help="Whines, carrier tones, electronics. Chosen from peaks that persist "
                                 "through the whole recording (median spectrum), so speech is not picked.")
    tone_prom = st.slider("Tone threshold (dB)", 6.0, 30.0, 12.0, 1.0, disabled=not use_tones)
    notch_q = st.slider("Notch Q (narrowness)", 5, 100, 30, 1,
                        help="Notch width = frequency ÷ Q. 30 gives 1.7 Hz at 50 Hz and 33 Hz at 1 kHz, "
                             "which follows the normal drift of the mains frequency.")

base = {"Auto": mains["base_hz"], "50 Hz": 50, "60 Hz": 60, "Off": None}[mains_choice]
hum_list = d.hum_harmonics(f_spec, P_spec, base, hum_max, hum_prom, excess) if base else []
tone_list = (d.detect_tones(f_spec, P_spec, fmax=sr / 2 * 0.95, min_prominence_db=tone_prom,
                            exclude_hz=[h["freq_hz"] for h in hum_list], excess=excess)
             if use_tones else [])
notch_freqs = tuple(sorted([h["freq_hz"] for h in hum_list] + [t["freq_hz"] for t in tone_list]))
k1 = (k0, notch_freqs, notch_q)
y1 = get_dehummed(k1, notch_freqs, float(notch_q), x, sr)

# ---------------------------------------------------------------------------
# Remaining sidebar settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.subheader("3. Vocal band-pass + transients")
    nyq_cap = int(sr / 2 * 0.95)
    low_hz = st.slider("Band-pass low (Hz)", 50, 1000, 300, 10)
    high_hz = st.slider("Band-pass high (Hz)", 1000, nyq_cap, min(3400, nyq_cap), 100)
    trans_rise = st.slider("Transient rise above background (dB)", 3, 30, 6, 1,
                           help="How far a burst must jump above the surrounding ~1 s of audio.")
    trans_max_dur = st.slider("Max transient duration (s)", 0.05, 2.0, 0.5, 0.05)
    top_n = st.slider("Show strongest", 5, 200, 25, 5)
    jump_db = st.slider("Sharp onset: jump (dB)", 3.0, 30.0, d.DEFAULT_JUMP_DB, 0.5,
                        help="How far the loudest 2 ms must exceed the previous 30 ms. Knocks and "
                             "clicks jump a lot; syllables swell.")
    decay_db = st.slider("Sharp onset: decay (dB)", -5.0, 30.0, d.DEFAULT_DECAY_DB, 0.5,
                         help="How far the level must fall 60–120 ms after the onset. Knocks die away; "
                              "a /t/ or /k/ runs into a vowel.")

    st.subheader("4. Clarity enhancement")
    exciter_drive = st.slider("Exciter drive", 0.0, 1.0, 0.3, 0.05)
    exciter_mix = st.slider("Exciter mix", 0.0, 1.0, 0.25, 0.05)

    st.subheader("5. Gain drops & digital silence")
    drop_db = st.slider("Gain-drop threshold (dB)", 6, 40, 20, 1)
    hold_ms = st.slider("Must stay low for (ms)", 100, 2000, 300, 50)
    only_below_floor = st.checkbox(
        "Only drops that fall below room tone", value=True,
        help="A pause in speech only falls back to the room's background noise. A mute or gain "
             "change takes the background down with it.")
    margin_db = st.slider("…by at least (dB)", 2.0, 20.0, 6.0, 0.5, disabled=not only_below_floor)
    min_silence_ms = st.slider("Shortest digital-silence gap (ms)", 5, 500, 20, 5)

# ---------------------------------------------------------------------------
# Remaining pipeline (each stage cached on its inputs)
# ---------------------------------------------------------------------------

subtraction_error = None
k2 = (k1, "sub", noise_start, noise_end, alpha, floor_db)
if alpha > 0:
    try:
        y2 = get_denoised(k2, noise_start, noise_end, alpha, floor_db, y1, sr)
    except ValueError as e:
        subtraction_error = str(e)
        y2 = y1
else:
    y2 = y1

k3 = (k2, "band", low_hz, high_hz)
try:
    y3 = get_band(k3, low_hz, high_hz, y2, sr)
except ValueError as e:
    st.error(str(e))
    st.stop()

transients = get_transients(k3, trans_rise, trans_max_dur, jump_db, decay_db, y3, y1, sr)
k4 = (k3, "exc", exciter_drive, exciter_mix)
y4 = get_enhanced(k4, exciter_drive, exciter_mix, y3, sr)

all_drops = get_gain_drops(k0, drop_db, hold_ms, margin_db, x, sr)
gain_drops = [g for g in all_drops if g["below_room_tone"]] if only_below_floor else all_drops
silence_rows = d.describe_silence(silence_runs, duration, min_s=min_silence_ms / 1000)

STAGES = {
    "Original": (k0, x),
    "Hum removed": (k1, y1),
    "Noise-subtracted": (k2, y2),
    "Band-passed": (k3, y3),
    "Enhanced": (k4, y4),
}

# The listening panel sits above the tabs but is filled in last, once the
# events it can jump to are known.
listen_box = st.container(border=True)

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

(tab_over, tab_hum, tab_noise, tab_vocal, tab_clarity, tab_gain, tab_enf, tab_report) = st.tabs([
    "🗺️ Overview & noise sample",
    "〰️ Hum & tones",
    "🔇 Noise subtraction",
    "🎚️ Band-pass + transients",
    "✨ Clarity",
    "📉 Gain drops & silence",
    "⚡ Mains hum (ENF)",
    "📄 Report & export",
])

# ── Overview & noise sample ────────────────────────────────────────────────

with tab_over:
    st.header("Overview & noise sample")
    rows = []
    for c, s in enumerate(dec.stats):
        rows.append({"channel": CH_NAMES.get(c, f"Channel {c + 1}") if dec.n_channels > 1 else "Mono",
                     **s, "decoder": dec.decoder})
    st.dataframe(rows, hide_index=True)
    if any(s["clipped_samples"] for s in dec.stats):
        st.warning("Some samples are at full scale (clipped). Levels around them are not reliable.")

    st.subheader("Loudness timeline")
    st.caption(
        "Spectral subtraction learns the noise from the green window and removes that spectrum "
        "everywhere. Pick background only: no speech, no knocks. The hatched window is the quietest "
        f"{sug_len:g} s stretch in the recording (digital silence excluded)."
    )
    show_png(png_levels(k0, (noise_start, noise_end),
                        (suggestion["start_s"], suggestion["end_s"]) if suggestion else None,
                        ref_db if ref_db is not None else noise_check["floor_db"], env))
    m1, m2, m3 = st.columns(3)
    m1.metric("Chosen sample level", f"{noise_check['level_db']} dBFS" if noise_check["level_db"] is not None else "—")
    m2.metric("Above quietest stretch", f"{noise_check['above_floor_db']:+.1f} dB"
              if noise_check["above_floor_db"] is not None else "—")
    m3.metric("Level spread inside", f"{noise_check['spread_db']} dB" if noise_check["spread_db"] is not None else "—")
    for w in noise_check["warnings"]:
        st.warning(w)
    if not noise_check["warnings"]:
        st.success("The chosen sample looks like steady background noise.")

# ── Hum & tones ─────────────────────────────────────────────────────────────

with tab_hum:
    st.header("Hum & tone removal (notch filters)")
    st.caption(
        "Mains hum and steady tones sit at fixed frequencies, so narrow notches remove them with "
        "almost no effect on speech. Spectral subtraction handles broadband hiss; it does a poor job "
        "on strong hum because it also takes out everything near each harmonic."
    )
    sc = mains["scores"]
    st.write(
        f"Mains family: **{mains['base_hz'] or 'none clear'}** "
        f"(average harmonic prominence 50 Hz: {sc[50]} dB, 60 Hz: {sc[60]} dB). "
        f"Notches applied: **{len(notch_freqs)}** at Q {notch_q}."
    )
    after_spec = get_spectrum(k1, y1, sr)[1] if notch_freqs else None
    show_png(png_spectrum(k1, notch_freqs, f_spec, P_spec, after_spec))
    c1, c2 = st.columns(2)
    with c1:
        st.subheader(f"Hum harmonics ({len(hum_list)})")
        table_or(hum_list, "None notched.")
    with c2:
        st.subheader(f"Other steady tones ({len(tone_list)})")
        if use_tones:
            table_or(tone_list, "None above threshold.")
        else:
            st.caption("Tick “Also notch other steady tones” in the sidebar to look for them.")

# ── Noise subtraction ───────────────────────────────────────────────────────

with tab_noise:
    st.header("Noise Profile Subtraction")
    st.caption(
        "The noise sample's average spectrum (after hum removal) is subtracted from every frame, "
        "keeping a floor below the original to limit 'musical noise'. Before = hum removed; "
        "After = hum removed + subtraction."
    )
    if subtraction_error:
        st.error(f"Spectral subtraction skipped: {subtraction_error}")
    elif alpha == 0:
        st.info("Subtraction is off (α = 0).")
    for w in noise_check["warnings"]:
        st.warning(w)
    show_png(png_spectrogram(k1, "Before subtraction", 6000.0, y1, sr))
    show_png(png_spectrogram(k2, "After subtraction", 6000.0, y2, sr))

# ── Band-pass + transients ──────────────────────────────────────────────────

IMP_COLOUR, GRAD_COLOUR = "#e53e3e", "#dd6b20"

with tab_vocal:
    st.header("Vocal Band-Pass + Transient Detection")
    st.caption(
        f"Butterworth band-pass {low_hz}–{high_hz} Hz, then bursts that rise at least {trans_rise} dB "
        "above the surrounding second. Each burst gets a sharp-onset test on the hum-removed full-band "
        f"audio: **impulsive** (knock/click-like) if its sharpest 2 ms jumps ≥ {jump_db:g} dB above the "
        f"previous 30 ms and the sound then falls ≥ {decay_db:g} dB; otherwise **gradual** (speech-like)."
    )
    shown = transients[:top_n]
    spans = tuple((e["start_s"], e["end_s"], IMP_COLOUR if e["onset"] == d.IMPULSIVE else GRAD_COLOUR) for e in shown)
    show_png(png_waveform((k3, "tr", top_n, jump_db, decay_db, trans_rise, trans_max_dur),
                          f"Band-passed — strongest {len(shown)} transients (red = impulsive, orange = gradual)",
                          spans, y3, sr))
    n_imp = sum(e["onset"] == d.IMPULSIVE for e in transients)
    st.subheader(f"Strongest {len(shown)} of {len(transients)} transients ({n_imp} impulsive)")
    only_imp = st.checkbox("Impulsive only", value=False)
    table = [e for e in transients if e["onset"] == d.IMPULSIVE][:top_n] if only_imp else shown
    if table:
        st.dataframe(
            table, hide_index=True,
            column_order=["onset", "start_s", "onset_s", "duration_s", "rise_db", "peak_db", "jump_db", "decay_db"],
        )
        st.caption("Sorted by rise above background. Pick one in the listening panel above to hear it.")
    else:
        st.caption("No transients crossed the current threshold.")

    with st.expander("Check the onset test against hand-labelled clips"):
        st.markdown(
            "Label a clip in Audacity (Tracks → Add New → Label Track; mark each knock, click or "
            "syllable), then File → Export → Export Labels. A CSV with `time_s,label` columns also works. "
            "Labels containing *knock, click, tap, bang, thud…* count as impulsive; *syllable, speech, "
            "word, vowel…* as gradual. For batch runs use `tools/validate_transients.py`."
        )
        lab_file = st.file_uploader("Label file for this recording", type=["txt", "csv", "tsv"], key="labels")
        tol = st.slider("Match tolerance (s)", 0.02, 0.5, 0.1, 0.01)
        if lab_file is not None:
            try:
                labels = d.parse_labels(lab_file.getvalue().decode("utf-8", errors="replace"))
                res = d.evaluate_labels(transients, labels, tol)
                st.dataframe([{"category": k, **v} for k, v in res["summary"].items()],
                             hide_index=True)
                st.caption(f"{res['unlabelled_events']} detections had no label nearby "
                           f"({res['unlabelled_impulsive']} of them called impulsive).")
                st.dataframe(res["rows"], hide_index=True)
            except ValueError as e:
                st.error(str(e))

# ── Clarity ─────────────────────────────────────────────────────────────────

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
    show_png(png_spectrogram(k3, "Before enhancement (band-passed)", 6000.0, y3, sr))
    show_png(png_spectrogram(k4, "After enhancement", 6000.0, y4, sr))

# ── Gain drops & digital silence ────────────────────────────────────────────

with tab_gain:
    st.header("Gain Drops & Digital Silence")
    st.caption(
        "Run on the **original** audio for the selected channel: noise removal and filtering change "
        f"levels. A drop is flagged when the level falls at least {drop_db} dB below the preceding "
        f"300 ms and stays down for {hold_ms} ms or more. This detects level behaviour only, not its cause."
    )
    spans = tuple((g["time_s"], g["time_s"] + g["low_for_s"], "#e53e3e") for g in gain_drops) + \
        tuple((r["start_s"], r["end_s"], "#805ad5") for r in silence_rows)
    show_png(png_waveform((k0, "gd", drop_db, hold_ms, only_below_floor, margin_db, min_silence_ms),
                          "Original — red = gain drops, purple = digital silence", spans, x, sr))
    st.subheader(f"Gain drops ({len(gain_drops)})")
    if gain_drops:
        st.dataframe(gain_drops, hide_index=True)
    else:
        st.caption("No sustained drops crossed the current threshold.")
    hidden = len(all_drops) - len(gain_drops)
    if hidden:
        st.caption(f"{hidden} more drops only fell back to room tone (typical of pauses in speech). "
                   "Untick “Only drops that fall below room tone” to list them.")

    st.subheader(f"Digital silence ({len(silence_rows)})")
    st.caption(
        "Runs of exact zero samples, found in the decoded file at its original sample rate. A live "
        "microphone always picks up some noise, so digital silence in mid-file usually means the audio "
        "was muted, cut, or padded in software. Gaps at the very start or end are common and less telling."
        + (" For the mixed channel, only gaps silent on both sides are listed." if channel == "mix" else "")
    )
    if silence_rows:
        st.dataframe(silence_rows, hide_index=True)
    else:
        st.caption(f"No digital-silence gaps of {min_silence_ms} ms or longer.")

# ── ENF ─────────────────────────────────────────────────────────────────────

enf_res = None
with tab_enf:
    st.header("Mains hum (ENF) consistency")
    st.caption(
        "The mains frequency wanders slightly (around ±0.05 Hz) and hum picked up by a recording "
        "carries that wander. In an unedited recording the hum's frequency and phase change "
        "smoothly. A splice between recordings made at different times often shows as a step in "
        "frequency; a cut inside one recording shows as a step in phase. This checks internal "
        "consistency only; matching against a grid-frequency database is not included."
    )
    run_enf = st.toggle("Run ENF analysis", value=False)
    c1, c2, c3 = st.columns(3)
    nominal = c1.selectbox("Nominal mains", [50, 60], index=0 if (mains["base_hz"] or 50) == 50 else 1)
    harm_choice = c2.selectbox("Harmonic to track", ["Auto", 1, 2, 3, 4, 5, 6])
    enf_frame = c3.slider("Frame length (s)", 1.0, 8.0, 2.0, 0.5)
    if run_enf:
        harm = None if harm_choice == "Auto" else int(harm_choice)
        enf_res = get_enf(k0, nominal, harm, enf_frame, 10.0, x, sr)
        r = enf_res
        e1, e2, e3, e4 = st.columns(4)
        e1.metric("Tracked", f"{r['tracked_hz']:.0f} Hz (×{r['harmonic']})")
        e2.metric("Mean ENF", f"{r['mean_hz']} Hz" if r["mean_hz"] else "—")
        e3.metric("Std. dev.", f"{r['std_hz']} Hz" if r["std_hz"] is not None else "—")
        e4.metric("Usable frames", f"{r['usable_fraction']:.0%}")
        if r["usable_fraction"] < 0.5:
            st.warning("The hum is weak for much of the recording, so steps there cannot be checked.")
        show_png(png_enf((k0, nominal, harm, enf_frame), r))
        st.caption(
            "Harmonic prominence (dB): " + ", ".join(f"×{k}: {v}" for k, v in r["harmonic_prominence_db"].items())
        )
        st.subheader(f"Frequency steps ({len(r['freq_events'])})")
        table_or(r["freq_events"], "None.")
        st.subheader(f"Phase steps ({len(r['phase_events'])})")
        table_or(r["phase_events"], "None.")
        st.subheader(f"Hum dropouts ({len(r['hum_dropouts'])})")
        st.caption("Stretches where the hum itself disappears (a mute, digital silence, the source switching "
                   "off). Phase cannot be compared across them, so a cut hidden inside one would not show.")
        table_or(r["hum_dropouts"], "None.")
        st.caption(
            "Limits: a cut that happens to remove (close to) a whole number of hum cycles leaves no phase step; "
            "hum switched on or off, or a change of microphone position, can also shift phase. "
            "Listen to each flagged point before drawing anything from it."
        )

# ── Report & export ─────────────────────────────────────────────────────────


def build_event_log() -> str:
    L = ["FVTR Event Log", "=" * 40, ""]
    L.append(f"File: {uploaded.name}  (sha1 {digest})")
    L.append(f"Format: {dec.format}; decoded with {dec.decoder}; analysed at {sr} Hz")
    L.append(f"Channel analysed: {ch_label(channel)}")
    L.append(f"Noise sample: {noise_start:.2f}-{noise_end:.2f} s "
             f"({noise_check['above_floor_db']} dB above the quietest stretch)")
    for w in noise_check["warnings"]:
        L.append(f"  warning: {w}")
    L.append(f"Notches ({len(notch_freqs)}, Q {notch_q}): " + ", ".join(f"{f:.2f}" for f in notch_freqs))
    L.append(f"Spectral subtraction: alpha {alpha}, floor {floor_db} dB"
             + (f" (skipped: {subtraction_error})" if subtraction_error else ""))
    L.append(f"Band-pass: {low_hz}-{high_hz} Hz")
    L.append("")
    L.append(f"Transients flagged: {len(transients)} ({n_imp} impulsive); strongest {min(top_n, len(transients))}:")
    for ev in transients[:top_n]:
        L.append(f"  [TRANSIENT {ev['onset'].upper():9}] {ev['start_s']:.3f}s - {ev['end_s']:.3f}s "
                 f"rise {ev['rise_db']:.1f} dB, jump {ev['jump_db']} dB, decay {ev['decay_db']} dB")
    L.append("")
    L.append(f"Gain drops (original audio, held >= {hold_ms} ms"
             + (f", below room tone by >= {margin_db} dB" if only_below_floor else "") + f"): {len(gain_drops)}")
    for g in gain_drops:
        L.append(f"  [GAIN-DROP] {g['time_s']:.2f}s  {g['before_db']:.1f} -> {g['after_db']:.1f} dB "
                 f"(drop {g['drop_db']:.1f} dB, low for {g['low_for_s']:.2f} s, "
                 f"{g['below_floor_db']:+.1f} dB vs room tone)")
    L.append("")
    L.append(f"Digital silence >= {min_silence_ms} ms: {len(silence_rows)}")
    for r in silence_rows:
        L.append(f"  [SILENCE] {r['start_s']:.3f}s - {r['end_s']:.3f}s ({r['length_ms']:.0f} ms, {r['position']})")
    for c, s in enumerate(dec.stats):
        if s["clipped_samples"]:
            L.append(f"  [CLIPPING] {ch_label(c) if dec.n_channels > 1 else 'Mono'}: {s['clipped_samples']} samples at full scale")
    if enf_res is not None:
        L.append("")
        L.append(f"ENF: tracked {enf_res['tracked_hz']:.0f} Hz, mean {enf_res['mean_hz']} Hz, "
                 f"std {enf_res['std_hz']} Hz, usable {enf_res['usable_fraction']:.0%}")
        for e in enf_res["freq_events"]:
            L.append(f"  [ENF-FREQ-STEP] {e['time_s']:.2f}s  {e['jump_hz']:+.4f} Hz")
        for e in enf_res["phase_events"]:
            L.append(f"  [ENF-PHASE-STEP] {e['time_s']:.2f}s  {e['jump_deg']:+.1f} deg")
        for e in enf_res["hum_dropouts"]:
            L.append(f"  [ENF-HUM-DROPOUT] {e['start_s']:.2f}s - {e['end_s']:.2f}s")
    L.append("")
    L.append("Note: flagged events are candidates for manual review only. They are not verified "
             "findings of speech content, physical contact, editing, or intent.")
    return "\n".join(L)


with tab_report:
    st.header("Report & export")
    log_text = build_event_log()
    st.download_button("⬇ Download event log (.txt)", data=log_text,
                       file_name="fvtr_event_log.txt", mime="text/plain")
    st.divider()
    export_stage = st.selectbox("Stage to export as WAV", list(STAGES), index=len(STAGES) - 1)
    est_mb = len(x) * 2 / 1e6
    if st.checkbox(f"Prepare full-length WAV ({est_mb:.0f} MB, {sr / 1000:g} kHz mono)"):
        kx, arr = STAGES[export_stage]
        st.download_button(f"⬇ Download {export_stage} WAV", data=get_wav(kx, arr, sr),
                           file_name=f"FVTR_{export_stage.replace(' ', '_').replace('-', '_')}.wav",
                           mime="audio/wav")
    with st.expander("Preview event log", expanded=True):
        st.text(log_text)

# ---------------------------------------------------------------------------
# Listening panel (one player, one section)
# ---------------------------------------------------------------------------

jump_targets = {"— choose a flagged point —": None}
for e in transients[:top_n]:
    t = e["onset_s"] if e["onset_s"] == e["onset_s"] else e["start_s"]
    jump_targets[f"Transient {t:.2f} s — {e['onset']}, +{e['rise_db']:.0f} dB"] = t
for g in gain_drops:
    jump_targets[f"Gain drop {g['time_s']:.2f} s — {g['drop_db']:.0f} dB"] = g["time_s"]
for r in silence_rows:
    jump_targets[f"Digital silence {r['start_s']:.2f} s — {r['length_ms']:.0f} ms"] = r["start_s"]
if enf_res is not None:
    for e in enf_res["freq_events"]:
        jump_targets[f"ENF frequency step {e['time_s']:.2f} s"] = e["time_s"]
    for e in enf_res["phase_events"]:
        jump_targets[f"ENF phase step {e['time_s']:.2f} s"] = e["time_s"]


def _jump():
    t = jump_targets.get(st.session_state.get("listen_target"))
    if t is not None:
        st.session_state["listen_start"] = float(max(0.0, round(t - 2.0, 2)))


with listen_box:
    st.markdown("**🎧 Listen to a section**")
    c1, c2 = st.columns([2, 3])
    stage = c1.radio("Stage", list(STAGES), horizontal=True, index=1, key="listen_stage")
    c2.selectbox("Jump to", list(jump_targets), key="listen_target", on_change=_jump)
    c3, c4, c5 = st.columns([2, 2, 1])
    if "listen_start" not in st.session_state:
        st.session_state["listen_start"] = 0.0
    st.session_state["listen_start"] = float(min(st.session_state["listen_start"], max(0.0, duration - 0.5)))
    start = c3.number_input("Start (s)", min_value=0.0, max_value=float(max(0.0, duration - 0.5)),
                            step=1.0, format="%.2f", key="listen_start")
    length = c4.slider("Length (s)", 2, 60, 15, 1)
    normalise = c5.checkbox("Normalise", value=True, help="Raise the section to a comfortable level.")
    kx, arr = STAGES[stage]
    i0 = int(start * sr)
    seg = np.asarray(arr[i0:i0 + int(length * sr)], dtype=np.float32)
    if normalise and seg.size and np.max(np.abs(seg)) > 0:
        seg = seg * (0.9 / np.max(np.abs(seg)))
    st.audio(d.to_wav_bytes(sr, seg), format="audio/wav")
    with st.expander("Section spectrogram"):
        show_png(png_spectrogram((kx, "sec", round(start, 2), length), f"{stage}: {start:.2f}–{start + length:.2f} s",
                                 min(8000.0, sr / 2), seg, sr, t0=start))
