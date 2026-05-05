#!/usr/bin/env python3
"""
FSHR Suite — Forensic Scalp & Hair Reconstruction
==================================================
Digital Darkroom for un-warping textures and isolating peppered artifacts
in hair and mouth zones across compressed or distortion-affected footage.

Tools:
  1. Base Enhancement  — gamma, CLAHE, bilateral filter, Laplacian sharpening
  2. Frequency Separation — isolate texture layer from colour/tone layer
  3. Zone Zoom + Motion Heatmap — 400 % magnified ROI with JET motion accumulation
  4. Frame Difference Analysis — black = The Room, white = The Being

Run:
  streamlit run fshr_app.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FSHR Suite — Forensic Scalp & Hair",
    page_icon="🔬",
    layout="wide",
)

st.title("Visual Being — FSHR Forensic Suite")
st.caption(
    "Forensic Scalp & Hair Reconstruction  ·  Digital Darkroom  ·  "
    "Texture Isolation  ·  Motion Signature Analysis"
)

# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def apply_gamma(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """LUT-based gamma correction; gamma>1 brightens, gamma<1 darkens."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(image, table)


def forensic_enhance(
    frame: np.ndarray,
    clahe_clip: float = 3.0,
    noise_strength: int = 9,
    gamma_val: float = 1.2,
    sharpen_val: float = 0.6,
) -> np.ndarray:
    """
    Base forensic enhancement pipeline:
    1. Gamma correction
    2. CLAHE on L* channel (CIE LAB — no colour bias)
    3. Bilateral filter (preserves body/hair edges)
    4. Laplacian edge enhancement
    """
    # 1. Gamma
    out = apply_gamma(frame, gamma_val)

    # 2. CLAHE on luminance only
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=clahe_clip, tileGridSize=(8, 8)
    )
    l_ch = clahe.apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # 3. Bilateral filter — noise_strength controls sigma values
    d = max(5, noise_strength | 1)  # must be odd and ≥ 5
    out = cv2.bilateralFilter(out, d, noise_strength * 5, noise_strength * 5)

    # 4. Laplacian sharpening
    gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_norm = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
    lap_bgr = cv2.cvtColor(lap_norm, cv2.COLOR_GRAY2BGR)
    out = cv2.addWeighted(out, 1.0, lap_bgr, sharpen_val, 0)

    return out


def frequency_separation(
    frame: np.ndarray, blur_radius: int = 21
) -> tuple[np.ndarray, np.ndarray]:
    """
    Separate frame into low-frequency (colour/tone) and high-frequency (texture) layers.

    Low  = Gaussian blur — captures overall colour blobs and tone gradients.
    High = frame − low + 128 — captures fine texture detail (neutral grey at 128).

    Isolating the high-frequency layer proves whether "bodies" are structural
    texture within the hair or separate overlaid structures.
    """
    r = max(1, blur_radius | 1)  # must be odd
    low_freq = cv2.GaussianBlur(frame, (r, r), 0)

    # High-freq layer: shift to neutral grey so it can be displayed
    diff = frame.astype(np.int16) - low_freq.astype(np.int16) + 128
    high_freq = np.clip(diff, 0, 255).astype(np.uint8)

    return low_freq, high_freq


def amplify_high_freq(high_freq: np.ndarray, gain: float = 2.0) -> np.ndarray:
    """Amplify deviation from 128 to make subtle texture visible."""
    shifted = high_freq.astype(np.float32) - 128.0
    amplified = np.clip(shifted * gain + 128.0, 0, 255).astype(np.uint8)
    return amplified


@st.cache_data(show_spinner=False)
def build_motion_heatmap(
    cap_path: str,
    center_frame: int,
    roi: tuple[int, int, int, int],
    n_frames: int = 30,
) -> np.ndarray:
    """
    Accumulate frame-difference magnitude over n_frames frames centred on
    center_frame, cropped to roi (x1,y1,x2,y2).  Returns a JET-coloured
    heatmap the same size as the ROI crop.
    """
    cap = cv2.VideoCapture(cap_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    x1, y1, x2, y2 = roi

    half = n_frames // 2
    start = max(0, center_frame - half)
    end   = min(total - 1, center_frame + half)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, prev = cap.read()
    if not ret:
        cap.release()
        h, w = y2 - y1, x2 - x1
        return np.zeros((h, w, 3), dtype=np.uint8)

    prev_crop = cv2.cvtColor(prev[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    accum     = np.zeros_like(prev_crop, dtype=np.float32)

    for _ in range(end - start):
        ret, frame = cap.read()
        if not ret:
            break
        curr_crop = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
        diff = cv2.absdiff(prev_crop, curr_crop).astype(np.float32)
        accum += diff
        prev_crop = curr_crop

    cap.release()

    if accum.max() > 0:
        accum = (accum / accum.max() * 255).astype(np.uint8)
    else:
        accum = accum.astype(np.uint8)

    return cv2.applyColorMap(accum, cv2.COLORMAP_JET)


def zone_zoom(
    frame: np.ndarray,
    roi: tuple[int, int, int, int],
    zoom_factor: float = 4.0,
    heatmap: np.ndarray | None = None,
    heatmap_alpha: float = 0.45,
    pixel_grid: bool = True,
) -> np.ndarray:
    """
    Crop roi from frame, upscale by zoom_factor using Lanczos4, optionally
    blend a pre-computed motion heatmap and draw a pixel grid overlay.
    """
    x1, y1, x2, y2 = roi
    crop = frame[y1:y2, x1:x2].copy()

    new_w = int((x2 - x1) * zoom_factor)
    new_h = int((y2 - y1) * zoom_factor)
    zoomed = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)

    if heatmap is not None:
        hm_resized = cv2.resize(heatmap, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        zoomed = cv2.addWeighted(zoomed, 1.0 - heatmap_alpha, hm_resized, heatmap_alpha, 0)

    if pixel_grid:
        src_w, src_h = x2 - x1, y2 - y1
        step_x = max(1, int(new_w / src_w))
        step_y = max(1, int(new_h / src_h))
        grid_color = (40, 40, 40)
        for gx in range(0, new_w, step_x):
            cv2.line(zoomed, (gx, 0), (gx, new_h), grid_color, 1)
        for gy in range(0, new_h, step_y):
            cv2.line(zoomed, (0, gy), (new_w, gy), grid_color, 1)

    return zoomed


def frame_difference(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    threshold: int = 15,
    morph_size: int = 3,
    colorize: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute the absolute difference between two frames.

    Returns:
        diff_gray  — raw 8-bit grayscale difference
        binary     — thresholded + morphologically closed binary mask
        composite  — original frame_b with "The Being" highlighted in cyan
                     and static background ("The Room") darkened

    Convention: black = The Room (static), white = The Being (moving entity).
    """
    diff_gray = cv2.absdiff(
        cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY),
        cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY),
    )

    _, binary = cv2.threshold(diff_gray, threshold, 255, cv2.THRESH_BINARY)

    if morph_size > 0:
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_size, morph_size)
        )
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    if colorize:
        mask_3ch   = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
        background = cv2.addWeighted(frame_b, 0.25, np.zeros_like(frame_b), 0.75, 0)
        cyan_layer = np.full_like(frame_b, (180, 160, 0), dtype=np.uint8)  # BGR cyan-ish
        being      = cv2.addWeighted(frame_b, 0.6, cyan_layer, 0.4, 0)
        composite  = np.where(mask_3ch == 255, being, background).astype(np.uint8)
    else:
        composite = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    return diff_gray, binary, composite


@st.cache_data(show_spinner=False)
def stacked_frame_difference(
    cap_path: str,
    center_frame: int,
    n_frames: int = 30,
    threshold: int = 15,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Accumulate binary motion masks across n_frames centred on center_frame.
    Returns (accum_gray, accum_color) — bright pixels indicate persistent motion
    (The Being's signature).
    """
    cap   = cv2.VideoCapture(cap_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    half  = n_frames // 2
    start = max(0, center_frame - half)
    end   = min(total - 1, center_frame + half)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    ret, prev = cap.read()
    if not ret:
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        cap.release()
        blank = np.zeros((h, w), dtype=np.uint8)
        return blank, cv2.cvtColor(blank, cv2.COLOR_GRAY2BGR)

    h, w = prev.shape[:2]
    accum = np.zeros((h, w), dtype=np.float32)

    for _ in range(end - start):
        ret, curr = cap.read()
        if not ret:
            break
        diff = cv2.absdiff(
            cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY),
            cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY),
        )
        _, mask = cv2.threshold(diff, threshold, 1, cv2.THRESH_BINARY)
        accum += mask.astype(np.float32)
        prev = curr

    cap.release()

    if accum.max() > 0:
        norm = (accum / accum.max() * 255).astype(np.uint8)
    else:
        norm = accum.astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_HOT)
    return norm, colored


@st.cache_data(show_spinner=False)
def load_frame(cap_path: str, frame_idx: int) -> np.ndarray | None:
    """Read a single frame from a video file by index."""
    cap = cv2.VideoCapture(cap_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    return frame if ret else None


@st.cache_data(show_spinner=False)
def get_video_info(cap_path: str) -> dict:
    cap = cv2.VideoCapture(cap_path)
    info = {
        "fps":    cap.get(cv2.CAP_PROP_FPS) or 30.0,
        "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "total":  int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
    }
    cap.release()
    return info


# ---------------------------------------------------------------------------
# Sidebar — global parameters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("FSHR Parameters")

    st.subheader("Base Enhancement")
    gamma_val    = st.slider("Gamma",             0.3, 3.0, 1.2, 0.05)
    clahe_clip   = st.slider("CLAHE clip limit",  0.5, 8.0, 3.0, 0.5)
    noise_str    = st.slider("Bilateral d",       1,   21,  9,   2)
    sharpen_val  = st.slider("Sharpening weight", 0.0, 2.0, 0.6, 0.1)

    st.subheader("Frequency Separation")
    blur_radius  = st.slider("Gaussian blur radius (px)", 3, 101, 21, 2)
    hf_gain      = st.slider("High-freq amplification",   1.0, 8.0, 2.5, 0.5)

    st.subheader("Zone Zoom")
    zoom_factor  = st.slider("Zoom factor", 1.0, 8.0, 4.0, 0.5)
    hm_frames    = st.slider("Heatmap accumulation frames", 5, 120, 30, 5)
    hm_alpha     = st.slider("Heatmap blend alpha", 0.0, 1.0, 0.45, 0.05)
    pixel_grid   = st.checkbox("Pixel grid overlay", value=True)

    st.subheader("Frame Difference")
    diff_thresh  = st.slider("Motion threshold",    1, 80, 15)
    morph_size   = st.slider("Morphological close", 0, 15,  3)
    diff_frames  = st.slider("Stacked frames",      5, 90, 30, 5)
    colorize     = st.checkbox("Cyan composite overlay", value=True)


# ---------------------------------------------------------------------------
# Video upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload source video",
    type=["mp4", "mov", "avi", "mkv"],
    help="Compressed or lens-distorted footage for forensic examination.",
)

if uploaded is None:
    st.info("Upload a video to begin forensic examination.")
    st.stop()

# Persist temp path across reruns via session state
if "fshr_src_path" not in st.session_state or st.session_state.get("fshr_fname") != uploaded.name:
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=Path(uploaded.name).suffix
    ) as tf:
        tf.write(uploaded.read())
        st.session_state["fshr_src_path"] = tf.name
        st.session_state["fshr_fname"]    = uploaded.name

src_path = st.session_state["fshr_src_path"]
info     = get_video_info(src_path)
fps, width, height, total = info["fps"], info["width"], info["height"], info["total"]

st.success(
    f"Loaded **{uploaded.name}** — "
    f"{width}×{height} px · {fps:.1f} fps · {total} frames"
)

# Frame scrubber (shared across all tabs)
frame_idx = st.slider("Frame selector", 0, max(total - 1, 0), min(total // 2, total - 1))
raw_frame = load_frame(src_path, frame_idx)

if raw_frame is None:
    st.error("Could not read the selected frame.")
    st.stop()

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

tab_base, tab_freq, tab_zoom, tab_diff = st.tabs([
    "🔦 Base Enhancement",
    "🌊 Frequency Separation",
    "🔍 Zone Zoom + Motion Heatmap",
    "👁 Frame Difference — The Being",
])


# ── Tab 1: Base Enhancement ─────────────────────────────────────────────────

with tab_base:
    st.header("Base Enhancement")
    st.caption(
        "Gamma · CLAHE (L* channel) · Bilateral filter · Laplacian sharpening. "
        "Prepare the frame for downstream forensic analysis."
    )

    enhanced = forensic_enhance(
        raw_frame,
        clahe_clip=clahe_clip,
        noise_strength=noise_str,
        gamma_val=gamma_val,
        sharpen_val=sharpen_val,
    )

    col_l, col_r = st.columns(2)
    with col_l:
        st.subheader("Original")
        st.image(cv2.cvtColor(raw_frame, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col_r:
        st.subheader("Enhanced")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)

    # Side-by-side diff
    with st.expander("Enhancement delta (amplified ×4)"):
        delta = cv2.absdiff(raw_frame, enhanced)
        delta_amp = np.clip(delta.astype(np.int16) * 4, 0, 255).astype(np.uint8)
        st.image(cv2.cvtColor(delta_amp, cv2.COLOR_BGR2RGB), use_container_width=True)


# ── Tab 2: Frequency Separation ─────────────────────────────────────────────

with tab_freq:
    st.header("Frequency Separation")
    st.caption(
        "Low-frequency layer = colour/tone blobs. "
        "High-frequency layer = fine texture detail. "
        "If a 'body' appears ONLY in the high-freq layer, it is structural texture — "
        "part of the hair. If it persists in the low-freq layer, it is a separate structure."
    )

    src_for_freq = forensic_enhance(
        raw_frame,
        clahe_clip=clahe_clip,
        noise_strength=noise_str,
        gamma_val=gamma_val,
        sharpen_val=sharpen_val,
    )

    low_freq, high_freq = frequency_separation(src_for_freq, blur_radius=blur_radius)
    high_amp            = amplify_high_freq(high_freq, gain=hf_gain)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Low Frequency")
        st.caption("Colour / Tone layer")
        st.image(cv2.cvtColor(low_freq, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col2:
        st.subheader("High Frequency")
        st.caption("Texture layer (raw ± 128)")
        st.image(cv2.cvtColor(high_freq, cv2.COLOR_BGR2RGB), use_container_width=True)
    with col3:
        st.subheader(f"High Freq ×{hf_gain:.1f}")
        st.caption("Amplified texture detail")
        st.image(cv2.cvtColor(high_amp, cv2.COLOR_BGR2RGB), use_container_width=True)

    with st.expander("High-freq as false-colour heatmap"):
        hf_gray = cv2.cvtColor(high_amp, cv2.COLOR_BGR2GRAY)
        hf_jet  = cv2.applyColorMap(hf_gray, cv2.COLORMAP_JET)
        st.image(cv2.cvtColor(hf_jet, cv2.COLOR_BGR2RGB), use_container_width=True)
        st.caption(
            "Red = strong texture deviation (potential structure). "
            "Blue = flat / homogeneous region."
        )


# ── Tab 3: Zone Zoom + Motion Heatmap ───────────────────────────────────────

with tab_zoom:
    st.header("Zone Zoom + Motion Heatmap")
    st.caption(
        f"Define a Region of Interest below. "
        f"The selected zone is magnified {zoom_factor:.0f}× ({int(zoom_factor * 100)} %) "
        f"with Lanczos4 interpolation. "
        "The JET heatmap shows accumulated motion across the surrounding frames — "
        "brighter = more movement = more entity activity."
    )

    # ROI input
    st.subheader("Define ROI")
    zm_col1, zm_col2 = st.columns(2)
    with zm_col1:
        z_x1 = st.number_input("ROI x1", value=max(0, width  // 4), step=10, min_value=0, max_value=width  - 2)
        z_y1 = st.number_input("ROI y1", value=max(0, height // 4), step=10, min_value=0, max_value=height - 2)
    with zm_col2:
        z_x2 = st.number_input("ROI x2", value=min(width,  width  * 3 // 4), step=10, min_value=1, max_value=width)
        z_y2 = st.number_input("ROI y2", value=min(height, height * 3 // 4), step=10, min_value=1, max_value=height)

    if z_x2 <= z_x1 or z_y2 <= z_y1:
        st.warning("ROI is invalid — ensure x2 > x1 and y2 > y1.")
    else:
        roi = (int(z_x1), int(z_y1), int(z_x2), int(z_y2))

        # Preview with ROI box drawn on original
        preview = raw_frame.copy()
        cv2.rectangle(preview, (roi[0], roi[1]), (roi[2], roi[3]), (0, 255, 255), 2)
        st.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), caption="ROI selection (cyan box)", use_container_width=True)

        st.subheader("Zoomed View")
        with st.spinner("Building motion heatmap …"):
            heatmap = build_motion_heatmap(src_path, frame_idx, roi, n_frames=hm_frames)

        enhanced_for_zoom = forensic_enhance(
            raw_frame,
            clahe_clip=clahe_clip,
            noise_strength=noise_str,
            gamma_val=gamma_val,
            sharpen_val=sharpen_val,
        )

        zoomed_plain  = zone_zoom(enhanced_for_zoom, roi, zoom_factor, heatmap=None,    pixel_grid=pixel_grid)
        zoomed_heatmp = zone_zoom(enhanced_for_zoom, roi, zoom_factor, heatmap=heatmap, heatmap_alpha=hm_alpha, pixel_grid=pixel_grid)

        zc1, zc2 = st.columns(2)
        with zc1:
            st.subheader(f"{zoom_factor:.0f}× Enhanced Crop")
            st.image(cv2.cvtColor(zoomed_plain, cv2.COLOR_BGR2RGB), use_container_width=True)
        with zc2:
            st.subheader(f"{zoom_factor:.0f}× + Motion Heatmap")
            st.image(cv2.cvtColor(zoomed_heatmp, cv2.COLOR_BGR2RGB), use_container_width=True)
            st.caption(f"Heatmap accumulated over ±{hm_frames//2} frames · JET palette")

        with st.expander("Heatmap only (full ROI resolution)"):
            st.image(cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), use_container_width=True)


# ── Tab 4: Frame Difference — The Being ─────────────────────────────────────

with tab_diff:
    st.header("Frame Difference Analysis — The Being")
    st.caption(
        "**Black = The Room** (background, static pixels). "
        "**White = The Being** (moving entity, anomalous motion). "
        "Stacked accumulation across N frames reveals persistent motion signatures "
        "even when individual frames show faint movement."
    )

    fd_col1, fd_col2 = st.columns(2)
    with fd_col1:
        compare_offset = st.slider(
            "Compare frame offset",
            -30, 30, 1,
            help="Subtract frame (N + offset) from current frame. "
                 "+1 = next frame, -1 = previous frame.",
        )
    with fd_col2:
        show_stacked = st.checkbox("Show stacked accumulation", value=True)

    compare_idx = int(np.clip(frame_idx + compare_offset, 0, total - 1))
    frame_b     = load_frame(src_path, compare_idx)

    if frame_b is None:
        st.warning("Compare frame could not be read.")
    else:
        diff_gray, binary, composite = frame_difference(
            raw_frame, frame_b,
            threshold=diff_thresh,
            morph_size=morph_size,
            colorize=colorize,
        )

        dc1, dc2, dc3 = st.columns(3)
        with dc1:
            st.subheader("Raw Difference")
            st.caption("Pixel-level magnitude")
            st.image(diff_gray, use_container_width=True)
        with dc2:
            st.subheader("Binary Mask")
            st.caption("White = The Being")
            st.image(binary, use_container_width=True)
        with dc3:
            st.subheader("Composite")
            st.caption("Cyan = The Being · Dark = The Room")
            st.image(cv2.cvtColor(composite, cv2.COLOR_BGR2RGB), use_container_width=True)

        if show_stacked:
            st.divider()
            st.subheader("Stacked Frame Accumulation")
            st.caption(
                f"Accumulated binary motion across {diff_frames} frames centred on frame {frame_idx}. "
                "Hot spots reveal where The Being has been most active."
            )
            with st.spinner(f"Accumulating {diff_frames} frames …"):
                accum_gray, accum_color = stacked_frame_difference(
                    src_path, frame_idx, n_frames=diff_frames, threshold=diff_thresh
                )

            sc1, sc2 = st.columns(2)
            with sc1:
                st.subheader("Accumulated (greyscale)")
                st.image(accum_gray, use_container_width=True)
            with sc2:
                st.subheader("Accumulated (HOT palette)")
                st.caption("White/yellow = maximum motion · Black = The Room")
                st.image(cv2.cvtColor(accum_color, cv2.COLOR_BGR2RGB), use_container_width=True)

        # Motion statistics
        with st.expander("Motion statistics"):
            n_motion_px  = int(np.count_nonzero(binary))
            total_px     = binary.size
            motion_pct   = n_motion_px / total_px * 100
            mean_diff    = float(diff_gray.mean())
            max_diff     = int(diff_gray.max())

            stat1, stat2, stat3, stat4 = st.columns(4)
            stat1.metric("Motion pixels",  f"{n_motion_px:,}")
            stat2.metric("Frame coverage", f"{motion_pct:.2f} %")
            stat3.metric("Mean diff",      f"{mean_diff:.1f}")
            stat4.metric("Peak diff",      str(max_diff))