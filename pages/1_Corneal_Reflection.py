"""
Corneal Reflection Inspector
============================
Crops the cornea from a still photo, enhances it, and unwarps its mirror
reflection into a panorama using spherical-mirror geometry (Nishino & Nayar,
"Corneal Imaging System", 2004). A resolution budget computes how much scene
detail the cornea's pixels can physically carry.

No generative fill and no text recognition: every output pixel comes from the
input photo.
"""

import math

import cv2
import numpy as np
import streamlit as st

# Typical adult cornea: limbus radius ~5.5 mm on a ~7.8 mm radius of curvature,
# so the visible cap spans ~45 deg of the sphere and reflects ~90 deg of the scene.
CORNEA_CAP_DEG = 45.0

st.set_page_config(
    page_title="Corneal Reflection — FSHR Suite",
    page_icon="👁",
    layout="wide",
)

st.title("Corneal Reflection Inspector")
st.caption(
    "Catchlight detection  ·  Cornea crop & enhancement  ·  "
    "Spherical-mirror unwarp  ·  Resolution budget"
)
st.warning(
    "**Scope note:** this page only rearranges and enhances pixels that are "
    "actually in the photo. It does not generate missing detail or read text. "
    "In a typical selfie the cornea is a few dozen pixels across and reflects "
    "only ~2–4 % of incoming light over the iris texture — check the "
    "Resolution Budget before reading anything into the reflection.",
    icon="⚠️",
)

# ---------------------------------------------------------------------------
# Processing functions
# ---------------------------------------------------------------------------

def decode_image(raw: bytes) -> np.ndarray | None:
    return cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)


def find_catchlight_candidates(img: np.ndarray, max_candidates: int = 6) -> list[dict]:
    """
    Locate small, very bright blobs surrounded by a dark ring — the specular
    catchlight on a cornea over a dark pupil/iris. Returns candidates sorted
    by contrast score.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    min_dim = min(h, w)

    k = max(5, int(min_dim * 0.02) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

    thresh = max(40, int(np.percentile(tophat, 99.8)))
    _, mask = cv2.threshold(tophat, thresh, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)

    max_area = (k * k) * 0.6
    candidates = []
    for i in range(1, n):
        area = stats[i, cv2.CC_STAT_AREA]
        if area < 2 or area > max_area:
            continue
        cx, cy = centroids[i]
        ring_r = int(max(4, k * 0.8))
        y0, y1 = max(0, int(cy) - ring_r), min(h, int(cy) + ring_r + 1)
        x0, x1 = max(0, int(cx) - ring_r), min(w, int(cx) + ring_r + 1)
        patch = gray[y0:y1, x0:x1].astype(np.float32)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - cx, yy - cy)
        ring = patch[(dist > ring_r * 0.5) & (dist <= ring_r)]
        if ring.size == 0:
            continue
        peak = float(gray[int(cy), int(cx)])
        score = (peak - float(ring.mean())) * (1.0 - float(ring.mean()) / 255.0)
        candidates.append({"x": float(cx), "y": float(cy), "score": score})

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates[:max_candidates]


def refine_iris(img: np.ndarray, cx: float, cy: float) -> tuple[int, int, int]:
    """Fit the iris/limbus circle around a catchlight with a Hough transform."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    win = int(max(24, min(h, w) * 0.06))
    x0, y0 = max(0, int(cx) - win), max(0, int(cy) - win)
    x1, y1 = min(w, int(cx) + win), min(h, int(cy) + win)
    patch = cv2.medianBlur(gray[y0:y1, x0:x1], 5)

    circles = cv2.HoughCircles(
        patch, cv2.HOUGH_GRADIENT, dp=1.5, minDist=win,
        param1=100, param2=18,
        minRadius=max(3, int(win * 0.15)), maxRadius=int(win * 0.9),
    )
    if circles is not None:
        best = None
        for c in circles[0]:
            ccx, ccy, cr = c[0] + x0, c[1] + y0, c[2]
            if math.hypot(ccx - cx, ccy - cy) < cr * 0.8:
                if best is None or cr > best[2]:
                    best = (ccx, ccy, cr)
        if best is not None:
            return int(round(best[0])), int(round(best[1])), int(round(best[2]))
    return int(round(cx)), int(round(cy)), int(win * 0.35)


def crop_cornea(img: np.ndarray, cx: int, cy: int, r: int, pad: float = 1.15) -> np.ndarray:
    h, w = img.shape[:2]
    half = int(math.ceil(r * pad))
    x0, y0 = max(0, cx - half), max(0, cy - half)
    x1, y1 = min(w, cx + half + 1), min(h, cy + half + 1)
    return img[y0:y1, x0:x1].copy()


def enhance(img: np.ndarray, clahe_clip: float, gamma: float, sharpen: float) -> np.ndarray:
    table = ((np.arange(256) / 255.0) ** (1.0 / gamma) * 255).astype(np.uint8)
    out = cv2.LUT(img, table)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4)).apply(l_ch)
    out = cv2.cvtColor(cv2.merge([l_ch, a_ch, b_ch]), cv2.COLOR_LAB2BGR)
    if sharpen > 0:
        blur = cv2.GaussianBlur(out, (0, 0), 1.5)
        out = cv2.addWeighted(out, 1.0 + sharpen, blur, -sharpen, 0)
    return out


def upscale(img: np.ndarray, factor: float, true_pixels: bool) -> np.ndarray:
    interp = cv2.INTER_NEAREST if true_pixels else cv2.INTER_LANCZOS4
    return cv2.resize(img, None, fx=factor, fy=factor, interpolation=interp)


def unwarp_cornea(
    img: np.ndarray, cx: int, cy: int, r: int, out_w: int = 720, out_h: int = 180,
) -> np.ndarray:
    """
    Map the cornea's mirror reflection to a panorama, treating the cornea as a
    spherical cap viewed orthographically.

    Output columns are azimuth (0-360 deg around the camera axis); rows are the
    angle between the reflected ray and the direction back toward the camera
    (top = the camera itself, bottom = 90 deg off-axis, i.e. the edges of what
    the subject was facing). A normal at polar angle theta reflects the view
    ray to 2*theta, so theta = alpha / 2.
    """
    cap = math.radians(CORNEA_CAP_DEG)
    scale = r / math.sin(cap)

    phi = np.linspace(0, 2 * np.pi, out_w, endpoint=False, dtype=np.float32)
    alpha = np.linspace(0, 2 * cap, out_h, dtype=np.float32)
    phi_g, alpha_g = np.meshgrid(phi, alpha)
    theta = alpha_g / 2.0

    map_x = (cx + np.sin(theta) * np.cos(phi_g) * scale).astype(np.float32)
    map_y = (cy - np.sin(theta) * np.sin(phi_g) * scale).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)


def find_highlight(crop: np.ndarray) -> tuple[int, int, int] | None:
    """Largest bright blob in the crop — mostly the light source itself."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    thresh = max(180, int(np.percentile(gray, 98)))
    _, mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)
    n, _, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if n <= 1:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return int(centroids[i][0]), int(centroids[i][1]), int(stats[i, cv2.CC_STAT_AREA])


def deg_per_px_at_center(r_px: float) -> float:
    """Angular scene resolution per pixel at the cornea centre (best case)."""
    return math.degrees(2 * math.sin(math.radians(CORNEA_CAP_DEG)) / r_px)


def min_feature_mm(r_px: float, distance_m: float, px_needed: float = 2.0) -> float:
    return distance_m * 1000 * math.radians(deg_per_px_at_center(r_px)) * px_needed


def diameter_needed_for_text(char_mm: float, distance_m: float, px_per_char: float = 5.0) -> float:
    rad_per_px = (char_mm / (distance_m * 1000)) / px_per_char
    return 2 * (2 * math.sin(math.radians(CORNEA_CAP_DEG)) / rad_per_px)


def to_png(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    return buf.tobytes() if ok else b""


def rgb(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Reflection Parameters")
    zoom = st.slider("Zoom factor", 2.0, 16.0, 8.0, 0.5)
    true_pixels = st.checkbox(
        "Show true pixels (nearest-neighbour)", value=True,
        help="Off = Lanczos interpolation, which looks smoother but adds no information.",
    )
    st.subheader("Enhancement")
    gamma = st.slider("Gamma", 0.3, 3.0, 1.4, 0.05)
    clahe_clip = st.slider("CLAHE clip limit", 0.5, 8.0, 2.5, 0.5)
    sharpen = st.slider("Unsharp amount", 0.0, 2.0, 0.4, 0.1)
    st.subheader("Unwarp")
    unwarp_w = st.slider("Panorama width (px)", 360, 1440, 720, 60)


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------

uploaded = st.file_uploader(
    "Upload a still photo",
    type=["jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff"],
    help="Use the original full-resolution file — social-media re-uploads are downscaled and recompressed.",
)

if uploaded is None:
    st.info("Upload a photo with a visible eye to begin.")
    st.stop()

img = decode_image(uploaded.getvalue())
if img is None:
    st.error("Could not decode that image.")
    st.stop()

h, w = img.shape[:2]
st.success(f"Loaded **{uploaded.name}** — {w}×{h} px")

# ---------------------------------------------------------------------------
# Eye selection
# ---------------------------------------------------------------------------

candidates = find_catchlight_candidates(img)
options = [f"Catchlight {i + 1} at ({int(c['x'])}, {int(c['y'])})" for i, c in enumerate(candidates)]
options.append("Manual placement")

st.subheader("1 · Locate the cornea")
choice = st.selectbox(
    "Eye candidate", options,
    help="Candidates are bright specular dots on a dark surround. Pick one, then fine-tune the circle to the iris edge.",
)

if choice != "Manual placement":
    c = candidates[options.index(choice)]
    guess = refine_iris(img, c["x"], c["y"])
else:
    guess = (w // 2, h // 2, max(8, min(h, w) // 40))

r_max = max(4, min(h, w) // 4)
key = f"{uploaded.name}:{choice}"
if st.session_state.get("cr_key") != key:
    st.session_state["cr_key"] = key
    st.session_state["cr_cx"] = int(np.clip(guess[0], 0, w - 1))
    st.session_state["cr_cy"] = int(np.clip(guess[1], 0, h - 1))
    st.session_state["cr_r"] = int(np.clip(guess[2], 3, r_max))

s1, s2, s3 = st.columns(3)
with s1:
    cx = st.slider("Centre x", 0, w - 1, key="cr_cx")
with s2:
    cy = st.slider("Centre y", 0, h - 1, key="cr_cy")
with s3:
    r = st.slider("Limbus radius (px)", 3, r_max, key="cr_r")

overlay = img.copy()
thickness = max(1, min(h, w) // 400)
cv2.circle(overlay, (cx, cy), r, (0, 255, 255), thickness)
for cand in candidates:
    cv2.drawMarker(overlay, (int(cand["x"]), int(cand["y"])), (255, 0, 255),
                   cv2.MARKER_CROSS, max(8, r // 2), thickness)
st.image(rgb(overlay), caption="Cyan circle = cornea outline · magenta crosses = catchlight candidates",
         width=min(w, 640))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

crop = crop_cornea(img, cx, cy, r)
if crop.size == 0:
    st.error("The cornea circle is outside the image.")
    st.stop()

tab_zoom, tab_unwarp, tab_budget = st.tabs([
    "🔍 Cornea Zoom",
    "🌐 Spherical Unwarp",
    "📏 Resolution Budget",
])

with tab_zoom:
    st.header("Cornea Zoom")
    st.caption(
        f"The crop is {crop.shape[1]}×{crop.shape[0]} source pixels. "
        "With 'true pixels' on, each visible block is one real pixel from the photo."
    )

    zoomed_raw = upscale(crop, zoom, true_pixels)
    zoomed_enh = enhance(zoomed_raw, clahe_clip, gamma, sharpen)

    hl = find_highlight(crop)
    marked = zoomed_enh.copy()
    if hl is not None:
        hx, hy, _ = hl
        cv2.circle(marked, (int(hx * zoom), int(hy * zoom)), int(max(6, zoom * 2)), (0, 0, 255), 2)

    zc1, zc2 = st.columns(2)
    with zc1:
        st.subheader("Raw")
        st.image(rgb(zoomed_raw), use_container_width=True)
    with zc2:
        st.subheader("Enhanced")
        st.image(rgb(marked), use_container_width=True)
        if hl is not None:
            st.caption(
                f"Red circle = brightest blob ({hl[2]} px). This is usually the "
                "light source itself (window, lamp, phone flash), not scene detail."
            )

    st.download_button("⬇ Download enhanced crop (PNG)", to_png(zoomed_enh),
                       file_name="cornea_enhanced.png", mime="image/png")

with tab_unwarp:
    st.header("Spherical-Mirror Unwarp")
    st.caption(
        "Treats the cornea as a spherical cap (~45° half-angle) and maps each "
        "reflected direction to a panorama. Top edge = straight back toward the "
        "camera; bottom edge = ~90° off-axis. A cornea only reflects what the "
        "subject is facing — nothing behind them."
    )

    pano = unwarp_cornea(img, cx, cy, r, out_w=unwarp_w, out_h=max(90, unwarp_w // 4))
    pano_enh = enhance(pano, clahe_clip, gamma, sharpen)
    st.image(rgb(pano_enh), use_container_width=True)
    st.caption(
        f"Resampled from ~{int(math.pi * r * r):,} source pixels into "
        f"{pano.shape[1] * pano.shape[0]:,} output pixels — the extra pixels are "
        "interpolation, not new information. Assumes the eye looks straight at "
        "the camera; gaze rotation shifts the panorama."
    )
    st.download_button("⬇ Download panorama (PNG)", to_png(pano_enh),
                       file_name="cornea_unwarp.png", mime="image/png")

with tab_budget:
    st.header("Resolution Budget")
    st.caption(
        "Best-case geometry only: real results are worse because of lens blur, "
        "focus, JPEG compression, low corneal reflectance and the iris texture "
        "underneath the reflection."
    )

    dpp = deg_per_px_at_center(r)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Cornea diameter", f"{2 * r} px")
    m2.metric("Pixels on cornea", f"{int(math.pi * r * r):,}")
    m3.metric("Scene angle per pixel", f"{dpp:.2f}°")
    m4.metric("Smallest feature at 1 m", f"{min_feature_mm(r, 1.0) / 10:.1f} cm")

    st.subheader("What these pixels can resolve")
    rows = []
    for dist in (0.5, 1.0, 2.0, 3.0):
        rows.append({
            "Distance from subject": f"{dist:g} m",
            "Smallest detectable object (2 px)": f"{min_feature_mm(r, dist) / 10:.1f} cm",
        })
    st.table(rows)

    need = diameter_needed_for_text(4.0, 0.5)
    st.subheader("Could it read on-screen text?")
    st.markdown(
        f"Reading ~4 mm screen text at 50 cm needs roughly 5 pixels per character, "
        f"which requires a cornea about **{need:,.0f} px across**. This photo's cornea "
        f"is **{2 * r} px** — {'enough on geometry alone' if 2 * r >= need else f'about {need / (2 * r):,.0f}× too small'}."
    )
