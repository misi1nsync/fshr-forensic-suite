import io

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

import fvtr_dsp as d

SR = 16_000
rng = np.random.default_rng(0)


def _encode(x, sr, fmt, subtype=None):
    buf = io.BytesIO()
    sf.write(buf, x, sr, format=fmt, subtype=subtype)
    return buf.getvalue()


def _hum(n, sr, base=50.0, amps=(0.02, 0.01, 0.006), enf=None):
    t = np.arange(n) / sr
    if enf is None:
        phase = 2 * np.pi * base * t
    else:
        phase = 2 * np.pi * np.cumsum(enf) / sr
    return sum(a * np.sin((k + 1) * phase) for k, a in enumerate(amps))


# --- decoding ---------------------------------------------------------------

@pytest.mark.parametrize("native", [44_100, 48_000, 22_050])
def test_stream_resampler_matches_one_shot(native):
    x = rng.standard_normal((native * 3 + 123, 2)).astype(np.float32)
    g = np.gcd(SR, native)
    up, down = SR // g, native // g
    r = d._StreamResampler(up, down, 2, block=int(0.37 * native))
    for i in range(0, len(x), 5000):
        r.feed(np.ascontiguousarray(x[i:i + 5000].T))
    got = r.finish().T
    want = signal.resample_poly(x, up, down, axis=0)
    assert got.shape == want.shape
    assert np.max(np.abs(got - want)) < 1e-5


@pytest.mark.parametrize("fmt,subtype", [("WAV", "PCM_16"), ("FLAC", None), ("OGG", "VORBIS"), ("MP3", None)])
def test_decode_formats(fmt, subtype):
    t = np.arange(44_100 * 2) / 44_100
    left = 0.3 * np.sin(2 * np.pi * 440 * t)
    right = 0.1 * np.sin(2 * np.pi * 1000 * t)
    raw = _encode(np.stack([left, right], 1), 44_100, fmt, subtype)
    dec = d.decode_audio(raw, "x." + fmt.lower())
    assert dec.sr == SR and dec.native_sr == 44_100 and dec.n_channels == 2
    assert abs(dec.duration_s - 2.0) < 0.1
    # channels stay separate: left is louder than right
    assert dec.stats[0]["rms_dbfs"] > dec.stats[1]["rms_dbfs"] + 5
    x, _ = d.select_channel(dec, 1)
    f, P = signal.welch(x, SR, nperseg=4096)
    assert abs(f[np.argmax(P)] - 1000) < 10


def test_low_rate_file_keeps_its_rate():
    raw = _encode(rng.standard_normal(8000).astype(np.float32) * 0.1, 8000, "WAV", "PCM_16")
    dec = d.decode_audio(raw, "p.wav")
    assert dec.sr == 8000 and len(dec.channels[0]) == 8000


def test_digital_silence_found_at_native_rate_and_mix_intersects():
    sr = 44_100
    n = sr * 4
    a = (rng.standard_normal(n) * 0.05).astype(np.float32)
    b = (rng.standard_normal(n) * 0.05).astype(np.float32)
    a[sr:sr + 4410] = 0          # 100 ms, left only
    a[2 * sr:2 * sr + 2205] = 0   # 50 ms, both
    b[2 * sr:2 * sr + 2205] = 0
    dec = d.decode_audio(_encode(np.stack([a, b], 1), sr, "WAV", "FLOAT"), "s.wav", block_s=0.3)
    assert [(round(s, 3), round(e, 3)) for s, e in dec.silence[0]] == [(1.0, 1.1), (2.0, 2.05)]
    _, runs = d.select_channel(dec, "mix")
    assert [(round(s, 3), round(e, 3)) for s, e in runs] == [(2.0, 2.05)]
    rows = d.describe_silence(dec.silence[0], dec.duration_s, min_s=0.06)
    assert len(rows) == 1 and rows[0]["position"] == "mid-file"


def test_run_tracker_across_blocks():
    m = np.zeros(100, bool)
    m[8:23] = True
    m[40:41] = True
    m[90:] = True
    tr = d._RunTracker(min_len=2)
    for i in range(0, 100, 10):
        tr.feed(m[i:i + 10], i)
    assert tr.finish(100) == [(8, 23), (90, 100)]


# --- noise sample -------------------------------------------------------------

def test_noise_window_suggestion_and_warning():
    n = SR * 20
    x = rng.standard_normal(n).astype(np.float32) * 0.1           # loud background
    x[SR * 12:SR * 14] *= 0.01                                      # 2 s quiet stretch
    env = d.frame_levels_db(x, SR, 0.05)
    sug = d.suggest_noise_window(env, 0.05, 1.0)
    assert 12.0 <= sug["start_s"] and sug["end_s"] <= 14.0
    ok = d.assess_noise_window(env, 0.05, sug["start_s"], sug["end_s"], sug["level_db"])
    assert not ok["warnings"]
    bad = d.assess_noise_window(env, 0.05, 2.0, 3.0, sug["level_db"])
    assert bad["above_floor_db"] > 30 and any("louder" in w for w in bad["warnings"])


def test_noise_window_skips_mutes_and_transitions():
    n = SR * 60
    x = rng.standard_normal(n).astype(np.float32) * 0.1           # speech-level activity
    for p in (5, 18, 33, 47):
        x[SR * p:SR * (p + 2)] *= 0.02                              # pauses: room tone, 8 s in all
    x[SR * 12:int(SR * 12.8)] *= 0.0001                              # 0.8 s mute
    x[SR * 25:int(SR * 26.5)] *= 0.00005                             # 1.5 s mute: steady but far too quiet
    x[SR * 40:int(SR * 40.3)] = 0                                    # digital silence
    env = d.frame_levels_db(x, SR, 0.05)
    sug = d.suggest_noise_window(env, 0.05, 1.0)
    assert any(p <= sug["start_s"] and sug["end_s"] <= p + 2 for p in (5, 18, 33, 47)), sug
    assert sug["steady"]


# --- hum ----------------------------------------------------------------------

def test_detect_mains_and_notch():
    n = SR * 10
    noise = rng.standard_normal(n) * 0.003
    x = (noise + _hum(n, SR, 50.0, amps=(0.05, 0.03, 0.02, 0.01))).astype(np.float32)
    x += (0.01 * np.sin(2 * np.pi * 1234.5 * np.arange(n) / SR)).astype(np.float32)
    f, P = d.median_spectrum(x, SR)
    mains = d.detect_mains(f, P)
    assert mains["base_hz"] == 50
    harm = d.hum_harmonics(f, P, 50, 1000)
    assert [h["harmonic"] for h in harm][:4] == [1, 2, 3, 4]
    tones = d.detect_tones(f, P, exclude_hz=[h["freq_hz"] for h in harm])
    assert abs(tones[0]["freq_hz"] - 1234.5) < 0.5
    y = d.notch_filter(x, SR, [h["freq_hz"] for h in harm] + [tones[0]["freq_hz"]], q=30)
    core = slice(SR, -SR)
    assert np.std(y[core] - noise[core]) < 0.25 * np.std(x[core] - noise[core])
    assert np.std(y[core]) < 1.2 * np.std(noise[core])


# --- spectral subtraction ------------------------------------------------------

def test_spectral_subtraction_is_transparent_at_zero_alpha():
    x = rng.standard_normal(SR * 3).astype(np.float32) * 0.1
    y = d.spectral_subtraction(x, SR, 0, 1, alpha=0.0, batch=37)
    assert np.max(np.abs(y - x)) < 1e-5


def test_spectral_subtraction_reduces_noise_keeps_tone():
    n = SR * 6
    noise = rng.standard_normal(n).astype(np.float32) * 0.05
    tone = np.zeros(n, np.float32)
    tone[SR * 3:] = 0.2 * np.sin(2 * np.pi * 700 * np.arange(SR * 3) / SR)
    y = d.spectral_subtraction(noise + tone, SR, 0.5, 2.5, alpha=2.0)
    assert np.std(y[SR:2 * SR]) < 0.3 * np.std(noise[SR:2 * SR])
    assert np.corrcoef(y[SR * 4:SR * 5], tone[SR * 4:SR * 5])[0, 1] > 0.98


def test_spectral_subtraction_rejects_short_window():
    with pytest.raises(ValueError):
        d.spectral_subtraction(np.zeros(SR, np.float32), SR, 0, 0.01)


def test_bandpass_top_edge_clamped_below_nyquist():
    y = d.bandpass_filter(rng.standard_normal(SR).astype(np.float32), SR, 300, 8000)
    assert np.all(np.isfinite(y))


# --- transients ---------------------------------------------------------------

def _knock(sr, f0=900, tau=0.015, amp=0.5):
    t = np.arange(int(0.12 * sr)) / sr
    return amp * np.exp(-t / tau) * np.sin(2 * np.pi * f0 * t)


def _syllable(sr, f0=140, attack=0.04, dur=0.25, amp=0.3):
    t = np.arange(int(dur * sr)) / sr
    env = np.minimum(1, t / attack) * np.exp(-np.maximum(0, t - attack) / 0.1)
    voiced = sum(np.sin(2 * np.pi * f0 * k * t) / k for k in range(1, 20))
    return amp * env * voiced / 3


def test_onset_test_separates_knocks_from_syllables():
    n = SR * 12
    x = rng.standard_normal(n) * 0.002 + _hum(n, SR)
    knocks = [1.5, 4.25, 7.0, 9.6]
    sylls = [3.0, 5.5, 8.2, 10.7]
    for tk in knocks:
        k = _knock(SR)
        x[int(tk * SR):int(tk * SR) + len(k)] += k
    for ts in sylls:
        s = _syllable(SR)
        x[int(ts * SR):int(ts * SR) + len(s)] += s
    x = x.astype(np.float32)
    band = d.bandpass_filter(x, SR)
    ev = d.classify_transients(d.detect_transients(band, SR), x, SR)
    near = lambda t: min(ev, key=lambda e: abs(e["start_s"] - t))
    for tk in knocks:
        e = near(tk)
        assert abs(e["start_s"] - tk) < 0.05 and e["onset"] == d.IMPULSIVE, e
    for ts in sylls:
        e = near(ts)
        assert abs(e["start_s"] - ts) < 0.06 and e["onset"] == d.GRADUAL, e
    assert ev == sorted(ev, key=lambda e: -e["rise_db"])


def test_parse_labels_audacity_and_csv_and_evaluate():
    aud = "1.500000\t1.520000\tknock\n3.0\t3.2\tsyllable\n\\ 100.0\t200.0\n6.0\t6.0\tbird\n"
    labs = d.parse_labels(aud)
    assert [l["label"] for l in labs] == ["knock", "syllable", "bird"]
    csv = "time_s,label\n1.5,Door knock\n3.0,speech\n"
    assert d.parse_labels(csv)[0]["label"] == "Door knock"
    events = [
        {"start_s": 1.49, "end_s": 1.6, "onset": d.IMPULSIVE},
        {"start_s": 3.02, "end_s": 3.3, "onset": d.IMPULSIVE},
        {"start_s": 9.0, "end_s": 9.1, "onset": d.IMPULSIVE},
    ]
    res = d.evaluate_labels(events, labs)
    assert res["summary"][d.IMPULSIVE] == {"labelled": 1, "detected": 1, "called_impulsive": 1, "called_gradual": 0}
    assert res["summary"][d.GRADUAL]["called_impulsive"] == 1
    assert res["summary"]["other"]["detected"] == 0
    assert res["unlabelled_events"] == 1


# --- gain drops ----------------------------------------------------------------

def test_gain_drops_need_hold_and_room_tone():
    n = SR * 16
    room = rng.standard_normal(n) * 0.001          # room tone ~ -60 dBFS
    speech = np.zeros(n)
    for a, b in [(1, 3), (4, 6), (7, 9), (10, 12), (13, 15)]:
        speech[a * SR:b * SR] = rng.standard_normal((b - a) * SR) * 0.05
    x = room + speech
    x[int(5.0 * SR):int(5.15 * SR)] *= 0.001       # 150 ms mute: too short
    x[int(8.0 * SR):int(8.6 * SR)] *= 0.001        # 600 ms mute: flag, below room tone
    x[int(11.0 * SR):int(11.5 * SR)] = 0           # digital silence: separate report
    drops = d.detect_gain_drops(x.astype(np.float32), SR)
    times = {round(g["time_s"], 1): g for g in drops}
    assert 8.0 in times and times[8.0]["below_room_tone"] and times[8.0]["low_for_s"] >= 0.5
    assert 5.0 not in times and 11.0 not in times
    # the ends of the speech bursts only fall back to room tone
    pauses = [g for g in drops if round(g["time_s"]) in (3, 6, 9, 12, 15)]
    assert pauses and not any(g["below_room_tone"] for g in pauses)


# --- ENF ------------------------------------------------------------------------

def _enf_signal(seconds, sr, seed, amps=(0.01, 0.02, 0.01)):
    r = np.random.default_rng(seed)
    n = seconds * sr
    walk = np.cumsum(r.standard_normal(seconds)) * 0.004
    enf = 50 + np.interp(np.arange(n) / sr, np.arange(seconds), walk)
    x = _hum(n, sr, amps=amps, enf=enf) + r.standard_normal(n) * 0.002
    return x.astype(np.float32), enf


# The phase is sampled every 50 ms: at 100 Hz that is a whole number of
# cycles, at 50 Hz it is not, so both are tested.
@pytest.mark.parametrize("amps,harmonic", [((0.01, 0.02, 0.01), 2), ((0.03, 0.01, 0.005), 1)])
def test_enf_tracks_and_finds_cut(amps, harmonic):
    x, enf = _enf_signal(60, SR, 1, amps)
    # delete 1.3725 s: not a whole number of cycles of either harmonic
    cut = slice(int(30 * SR), int(31.3725 * SR))
    y = np.concatenate([x[:cut.start], x[cut.stop:]])
    res = d.enf_analysis(y, SR, nominal_hz=50)
    assert res["harmonic"] == harmonic
    good = res["snr_db"] > 10
    true = np.interp(res["t_freq"], np.arange(len(enf)) / SR, enf)
    before = res["t_freq"] < 28
    assert np.median(np.abs(res["enf_hz"][before & good] - true[before & good])) < 0.01
    assert [round(e["time_s"]) for e in res["phase_events"]] == [30]
    assert res["freq_events"] == []  # same recording either side: no frequency step


@pytest.mark.parametrize("amps", [(0.01, 0.02, 0.01), (0.03, 0.01, 0.005)])
def test_enf_clean_recording_has_no_flags(amps):
    x, _ = _enf_signal(60, SR, 2, amps)
    res = d.enf_analysis(x, SR, nominal_hz=50)
    assert res["phase_events"] == [] and res["freq_events"] == []


def test_enf_hum_dropout_is_not_a_phase_step():
    x, _ = _enf_signal(60, SR, 4, (0.03, 0.01, 0.005))
    x[20 * SR:int(20.8 * SR)] *= 0.001      # mute
    x[40 * SR:int(40.4 * SR)] = 0           # digital silence
    res = d.enf_analysis(x, SR, nominal_hz=50)
    assert res["phase_events"] == []
    assert [round(h["start_s"]) for h in res["hum_dropouts"]] == [20, 40]


def test_enf_frequency_jump_between_recordings():
    a, _ = _enf_signal(30, SR, 3)
    n = 30 * SR
    b = (_hum(n, SR, amps=(0.01, 0.02, 0.01), enf=np.full(n, 50.08)) + rng.standard_normal(n) * 0.002).astype(np.float32)
    res = d.enf_analysis(np.concatenate([a, b]), SR, nominal_hz=50)
    assert len(res["freq_events"]) == 1 and abs(res["freq_events"][0]["time_s"] - 30) <= 1


# --- display helpers -------------------------------------------------------------

def test_spectrogram_image_is_screen_sized():
    x = rng.standard_normal(SR * 600).astype(np.float32)
    img, ext = d.spectrogram_image(x, SR, max_cols=1200, fmax=6000)
    assert img.shape[1] <= 1200 and img.shape[0] < 400
    assert ext[1] == pytest.approx(600, rel=1e-3)
    t, lo, hi = d.minmax_envelope(x, SR, 2000)
    assert len(t) == 2000 and np.all(lo <= hi)
