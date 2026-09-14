import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")  # 작은 matvec/pinv 다수 호출 시 BLAS 스레드 스폰 오버헤드로 10배+ 느려지는 문제 방지 (numpy import 전에 설정해야 함)
os.environ.setdefault("OMP_NUM_THREADS", "4")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 200  # 모든 figure에 공통 적용, compact하게

import streamlit as st

import config as cfg

# report.ipynb의 1/2번 데모(Item Memory, Spatial Memory)가 공유하던 scaffold 설정.
# experiment_spatial_navigation.py는 이 값을 모듈 최초 import 시점에 한 번만
# 스냅샷(scaf_cfg = cfg.DEFAULT_SCAFFOLD)하므로, 그 import보다 먼저 설정해야 함.
cfg.DEFAULT_SCAFFOLD = cfg.ScaffoldConfig(
    module_periods=[3, 4, 5], Nh=400,
    connection_prob=0.6, threshold=0.5, nonlinearity="relu_threshold",
)

from PIL import Image
from experiments.experiment_item_capacity import (
    prepare_sensory_data, get_mem_for_Nh_sweep, render_node_states_panel, apply_noise,
)
from experiments.experiment_spatial_navigation import (
    build_fig4c_demo, build_novel_trajectory,
    demo_revisit_predictions, demo_unvisited_by_distance, plot_unvisited_distance_map,
    plot_grid_modules_square,
)
from experiments.experiment_memory_palace import (
    build_seq_scaffold, make_embedded_image_book_for_fig7,
    make_hairpin_path, path_to_indices, recall_sequence_once, cos_sim,
)
from grid_utils import GridCode

st.set_page_config(page_title="Vector-HaSH demo", layout="wide")

# 슬라이더(stepper_slider)들 사이 상하 간격 압축: 기본 vertical block gap이 넓어서
# 슬라이더 여러 개 쌓이면 세로로 길어짐.
st.markdown("""
<style>
div[data-testid="stVerticalBlock"] { gap: 0.4rem; }
div[data-testid="stHorizontalBlock"] { gap: 0.4rem; }
button p { font-size: 0.75rem; }
button[data-testid^="stBaseButton"] { padding-left: 0.25rem; padding-right: 0.25rem; min-width: 0; }
.st-key-item_memory_fig div[data-testid="stImage"] { max-width: 65% !important; margin-left: auto !important; margin-right: auto !important; }
.st-key-item_memory_fig div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
.st-key-spatial_memory_figs div[data-testid="stImage"] { max-width: 90% !important; margin-left: auto !important; margin-right: auto !important; }
.st-key-spatial_memory_figs div[data-testid="stImage"] img { width: 100% !important; height: auto !important; }
</style>
""", unsafe_allow_html=True)


def _render_open_figures():
    """plt.show()를 st.pyplot()으로 리다이렉트: 기존 실험 코드(plot 함수들)를
    수정 없이 그대로 재사용하기 위한 monkeypatch.
    use_container_width=False -- 기본값(True)은 컨테이너 폭에 무조건 맞춰
    늘려버려서 figsize를 줄여도 화면 출력 크기가 그대로였음. False로 두면
    figsize*dpi가 실제 출력 픽셀 크기가 되고(컨테이너보다 크면 그 폭까지만
    줄어듦), figsize가 다시 출력 크기를 제어하는 값이 된다."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)


plt.show = _render_open_figures


def _stepper_bump(key, delta, min_value, max_value):
    # on_click 콜백은 위젯 인스턴스화 전(rerun 직전)에 실행되므로 여기서 session_state를
    # 고쳐야 StreamlitWidgetAlreadyInstantiatedError 없이 안전하게 값이 바뀐다.
    st.session_state[key] = min(max(st.session_state[key] + delta, min_value), max_value)


def stepper_slider(label, min_value, max_value, value, step, key, container=None):
    """st.slider + 양옆 -/+ 버튼. key로 session_state에 값 보관.
    container를 넘기면(예: st.columns(2)의 한 칸) 그 안에 렌더링 -- 한 줄에 슬라이더
    여러 개를 나란히 배치할 때 사용."""
    container = container if container is not None else st
    if key not in st.session_state:
        st.session_state[key] = value
    st.session_state[key] = min(max(st.session_state[key], min_value), max_value)

    c1, c2, c3 = container.columns([10, 1, 1], gap="small")
    with c2:
        st.button("➖", key=f"{key}_minus", use_container_width=True,
                  on_click=_stepper_bump, args=(key, -step, min_value, max_value))
    with c3:
        st.button("➕", key=f"{key}_plus", use_container_width=True,
                  on_click=_stepper_bump, args=(key, step, min_value, max_value))
    with c1:
        st.slider(label, min_value, max_value, step=step, key=key)
    return st.session_state[key]


# =========================================================================
# 1. Item Memory (구 2a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Training item memory...")
def _get_item_mem(Nh, n_items_sub):
    sbook_flattened = prepare_sensory_data()
    return get_mem_for_Nh_sweep(sbook_flattened, Nh=Nh, n_items=n_items_sub)


def render_item_memory():
    st.header("1. Item Memory")
    col1, col2 = st.columns(2)
    Nh = stepper_slider("$N_h$", 200, 800, cfg.DEFAULT_SCAFFOLD.Nh, 10, key="item_Nh", container=col1)
    n_items_sub = stepper_slider("$N_s$", 1, 1000, cfg.DEFAULT_SCAFFOLD.Nh // 2 + 1, 10, key="item_Ns", container=col2)
    col3, col4 = st.columns(2)
    idx_label = stepper_slider("Item index", 1, n_items_sub, 1, 1, key="item_idx", container=col3)
    noise_ratio = stepper_slider("Noise ratio", 0.0, 0.9, 0.1, 0.1, key="item_noise_ratio", container=col4)
    noise_type = st.selectbox("Noise type", ["masking", "salt_and_pepper"], index=1)

    mem_sub, items_sub = _get_item_mem(Nh, n_items_sub)
    target_idx = min(idx_label - 1, n_items_sub - 1)
    with st.container(key="item_memory_fig"):
        render_node_states_panel(mem_sub, items_sub, target_idx, noise_type, noise_ratio)


# =========================================================================
# 2. Spatial Memory (구 3a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Building trained path model...")
def _get_3a_model(trained_length):
    return build_fig4c_demo(trained_length=trained_length)


_SPATIAL_N_OVERLAP = 3  # 시작점 포함 재방문 지점 수 고정


@st.cache_resource(max_entries=1, show_spinner="Building novel trajectory...")
def _get_3a_novel(_model, trained_length, novel_length):
    return build_novel_trajectory(_model, novel_length=novel_length, n_overlap=_SPATIAL_N_OVERLAP)


def render_spatial_memory():
    st.header("2. Spatial Memory")
    col1, col2 = st.columns(2)
    trained_length = stepper_slider("Original path length:", 20, 200, 100, 10, key="spatial_trained_length", container=col1)
    novel_length = stepper_slider("New path length:", 20, 200, 100, 10, key="spatial_novel_length", container=col2)

    model = _get_3a_model(trained_length)
    novel_model = _get_3a_novel(model, trained_length, novel_length)

    with st.container(key="spatial_memory_figs"):
        map_col, revisit_col, novel_col = st.columns(3)
        with map_col:
            unvisited = plot_unvisited_distance_map(model, novel_model, n_show=2)
        with revisit_col:
            demo_revisit_predictions(model, novel_model, n_revisits=_SPATIAL_N_OVERLAP)
        with novel_col:
            demo_unvisited_by_distance(model, novel_model, unvisited)


# =========================================================================
# Memory Palace가 사용하는 데이터 소스: miniimagenet(old item) / 숫자카드(new item).
# lambdas=(2,3,5), Ns=3600, seed=0으로 둘 다 동일해서 한 번만 계산해 공유한다.
# =========================================================================
_PALACE_LAMBDAS = (2, 3, 5)
_PALACE_NS = 3600
_PALACE_SEED = 0


def _load_number_card_images_grayscale(numbers):
    """number_card_60x60/ 폴더(1~999, 미리 렌더링해둔 PNG)에서 numbers에 해당하는
    카드만 그레이스케일로 불러와 (Ns, len(numbers)) 형태로 반환."""
    folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "number_card_60x60")
    imgs = [np.array(Image.open(os.path.join(folder, f"{n:03d}.png")).convert("L"), dtype=np.float64)
            for n in numbers]
    arr = np.stack(imgs, axis=0)
    return arr.reshape(len(numbers), -1).T


@st.cache_data(max_entries=1, show_spinner="Rendering number cards...")
def _make_numbered_card_book(Ns, Nstates, Npos, block_w, block_h, seed):
    """block_w*block_h개 위치마다 1..n_positions 숫자 카드를 중복 없이 랜덤 순서로
    배치한다. 카드가 전부 유일하므로 워터마크 없이도 pinv가 항상 full rank."""
    rng = np.random.default_rng(seed)
    img_h = img_w = int(round(np.sqrt(Ns)))
    assert img_h * img_w == Ns

    n_positions = block_w * block_h
    numbers = rng.permutation(n_positions) + 1
    img_flat = _load_number_card_images_grayscale(numbers).astype(np.float32)
    img_flat += rng.standard_normal(img_flat.shape).astype(np.float32) * 10.0
    img_flat -= img_flat.mean()

    smin, smax = np.amin(img_flat), np.amax(img_flat)
    scale = 1.9 / (smax - smin)
    shift = -0.95 - smin * scale
    img_flat *= scale
    img_flat += shift
    np.arctanh(img_flat, out=img_flat)

    assert n_positions == Nstates and Npos == block_h
    return img_flat


@st.cache_data(max_entries=1, show_spinner="Loading miniimagenet book...")
def _get_palace_books():
    Npos = int(np.prod(_PALACE_LAMBDAS))
    block_w = block_h = min(60, Npos)
    Nstates = Npos * Npos
    sbook_old, _, _ = make_embedded_image_book_for_fig7(
        _PALACE_NS, Nstates, Npos, 0, 0, block_w, block_h,
        seed=_PALACE_SEED, shuffle_images=False, use_tanh_inverse=True,
    )
    mbook_new = _make_numbered_card_book(_PALACE_NS, Nstates, Npos, block_w, block_h, _PALACE_SEED)
    path_all = make_hairpin_path(block_w, block_h, 0, 0)
    idxs_all = path_to_indices(path_all, Npos)
    return sbook_old, mbook_new, idxs_all, block_w, block_h


@st.cache_resource(max_entries=1, show_spinner="Training scaffold...")
def _get_palace_scaffold(Nh, gamma=0.6, thresh=0.5):
    return build_seq_scaffold(list(_PALACE_LAMBDAS), Nh, gamma=gamma, thresh=thresh, nruns=1)


# =========================================================================
# 3. Memory Palace: Cleanup Test (구 4b)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Running cleanup pipeline...")
def _get_4b_pipeline(Nh, depth):
    scaf = _get_palace_scaffold(Nh)
    sbook_old, mbook_new, idxs_all, _, _ = _get_palace_books()
    idxs_seq = idxs_all[:depth]
    P_seq = scaf["pbook_flat"][:, :, idxs_seq]
    S_seq = sbook_old[:, idxs_seq]
    M_seq = mbook_new[:, idxs_seq]

    S_clean, G_clean = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1), return_grid=True)
    S_addr = np.sign(S_clean[0])
    Wms = M_seq @ np.linalg.pinv(S_addr)          # 주소 -> new item
    Wsm_raw = S_seq @ np.linalg.pinv(M_seq)       # new item -> sensory (원본 스케일)
    G_true = scaf["gbook_flat"][:, idxs_seq]
    return scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw, G_clean, G_true


def render_memory_palace_b():
    st.header("3. Memory Palace")
    Ns = _PALACE_NS
    img_h = img_w = int(round(np.sqrt(Ns)))
    _, _, idxs_all, _, _ = _get_palace_books()
    n_cards = len(idxs_all)

    col1, col2 = st.columns(2)
    Nh = stepper_slider("$N_h$", 10, 400, 200, 5, key="palace_b_Nh", container=col1)
    depth = stepper_slider("$N_s$", 2, n_cards, min(30, n_cards), 1, key="palace_b_depth", container=col2)
    col3, col4 = st.columns(2)
    t = stepper_slider("Item index", 1, depth, 1, 1, key="palace_b_idx", container=col3) - 1
    noise_ratio_vis = stepper_slider("Noise ratio", 0.0, 0.9, 0.3, 0.1, key="palace_b_noise_ratio", container=col4)

    scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw, G_clean, G_true = _get_4b_pipeline(Nh, depth)

    def recover(noisy_item, tt):
        sensory_est_noisy = Wsm_raw @ noisy_item
        S_query = S_seq.copy()
        S_query[:, tt] = sensory_est_noisy
        S_rec, G_rec = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1),
                                             S_query=S_query, return_grid=True)
        sensory_cleaned = S_rec[0, :, tt]
        addr_clean = np.sign(sensory_cleaned)
        item_rec = Wms @ addr_clean
        return sensory_est_noisy, sensory_cleaned, item_rec, G_rec[0, :, tt]

    true_sensory = S_seq[:, t]
    sensory_baseline_rec = S_clean[0, :, t]
    true_item = M_seq[:, t]
    noisy_item = true_item if noise_ratio_vis == 0.0 else apply_noise(true_item, "salt_and_pepper", noise_ratio_vis, seed=2)
    sensory_est_noisy, sensory_cleaned, item_rec, g_cleanup = recover(noisy_item, t)

    panels = [
        (true_sensory, f"Stored item #{t + 1}", G_true[:, t]),
        (sensory_baseline_rec, f"Recalled item #{t + 1} (cos_sim={cos_sim(sensory_baseline_rec, true_sensory):.3f})", G_clean[0, :, t]),
        (true_item, "Mnemonic item", None),
        (noisy_item, "Noisy mnemonic item", None),
        (sensory_est_noisy, f"Noisy item recon (cos_sim={cos_sim(sensory_est_noisy, true_sensory):.3f})", None),
        (sensory_cleaned, f"Cleanup item recall #{t + 1} (cos_sim={cos_sim(sensory_cleaned, true_sensory):.3f})", g_cleanup),
        (item_rec, f"Recalled mnemonic item (cos_sim={cos_sim(item_rec, true_item):.3f})", None),
    ]
    grid_code = GridCode(module_periods=list(_PALACE_LAMBDAS))
    fig, axes = plt.subplots(2, len(panels), figsize=(3.1 * len(panels), 6.8))
    for col, (vec, title, g) in enumerate(panels):
        axes[0, col].imshow(vec.reshape(img_h, img_w), cmap="gray")
        axes[0, col].set_title(title, fontsize=9)
        axes[0, col].set_xticks([]); axes[0, col].set_yticks([])
        if g is None:
            axes[1, col].axis("off")
        else:
            plot_grid_modules_square(axes[1, col], grid_code, g)
            axes[1, col].set_title("Grid state", fontsize=9)
    plt.tight_layout()
    plt.show()


# =========================================================================
st.title("Vector-HaSH")

section = st.sidebar.radio(
    "Section",
    ["1. Item Memory", "2. Spatial Memory", "3. Memory Palace"],
)

if section == "1. Item Memory":
    render_item_memory()
elif section == "2. Spatial Memory":
    render_spatial_memory()
else:
    render_memory_palace_b()
