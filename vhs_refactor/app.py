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
)
from experiments.experiment_memory_palace import (
    build_seq_scaffold, make_embedded_image_book_for_fig7,
    make_hairpin_path, path_to_indices, recall_sequence_once, cos_sim,
)

st.set_page_config(page_title="Vector-HaSH demo", layout="wide")


def _render_open_figures():
    """plt.show()를 st.pyplot()으로 리다이렉트: 기존 실험 코드(plot 함수들)를
    수정 없이 그대로 재사용하기 위한 monkeypatch."""
    for num in plt.get_fignums():
        fig = plt.figure(num)
        st.pyplot(fig)
        plt.close(fig)


plt.show = _render_open_figures


def _stepper_bump(key, delta, min_value, max_value):
    # on_click 콜백은 위젯 인스턴스화 전(rerun 직전)에 실행되므로 여기서 session_state를
    # 고쳐야 StreamlitWidgetAlreadyInstantiatedError 없이 안전하게 값이 바뀐다.
    st.session_state[key] = min(max(st.session_state[key] + delta, min_value), max_value)


def stepper_slider(label, min_value, max_value, value, step, key):
    """st.slider + 양옆 -/+ 버튼. key로 session_state에 값 보관."""
    if key not in st.session_state:
        st.session_state[key] = value
    st.session_state[key] = min(max(st.session_state[key], min_value), max_value)

    c1, c2, c3 = st.columns([14, 1, 1], gap="small")
    with c2:
        st.button("-", key=f"{key}_minus", use_container_width=True,
                  on_click=_stepper_bump, args=(key, -step, min_value, max_value))
    with c3:
        st.button("+", key=f"{key}_plus", use_container_width=True,
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
    with col1:
        Nh = stepper_slider("N_h", 200, 800, cfg.DEFAULT_SCAFFOLD.Nh, 10, key="item_Nh")
        n_items_sub = stepper_slider("N_s", 1, 1000, cfg.DEFAULT_SCAFFOLD.Nh // 2 + 1, 10, key="item_Ns")
        idx_label = stepper_slider("Item index", 1, n_items_sub, 1, 1, key="item_idx")
    with col2:
        noise_type = st.selectbox("Noise type", ["masking", "salt_and_pepper"], index=1)
        noise_ratio = stepper_slider("Noise ratio", 0.0, 0.9, 0.1, 0.1, key="item_noise_ratio")

    mem_sub, items_sub = _get_item_mem(Nh, n_items_sub)
    target_idx = min(idx_label - 1, n_items_sub - 1)
    render_node_states_panel(mem_sub, items_sub, target_idx, noise_type, noise_ratio)


# =========================================================================
# 2. Spatial Memory (구 3a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Building trained path model...")
def _get_3a_model(trained_length):
    return build_fig4c_demo(trained_length=trained_length, seed=3)


@st.cache_resource(max_entries=1, show_spinner="Building novel trajectory...")
def _get_3a_novel(_model, trained_length, novel_length, n_overlap):
    return build_novel_trajectory(_model, novel_length=novel_length, n_overlap=n_overlap, seed=1)


def render_spatial_memory():
    st.header("2. Spatial Memory")
    trained_length = stepper_slider("Original path length:", 20, 300, 100, 10, key="spatial_trained_length")
    novel_length = stepper_slider("New path length:", 20, 300, 100, 10, key="spatial_novel_length")
    n_overlap = stepper_slider("# revisits:", 1, 10, 4, 1, key="spatial_n_overlap")

    model = _get_3a_model(trained_length)
    novel_model = _get_3a_novel(model, trained_length, novel_length, n_overlap)

    unvisited = plot_unvisited_distance_map(model, novel_model, n_show=4)
    demo_revisit_predictions(model, novel_model, n_revisits=n_overlap)
    demo_unvisited_by_distance(model, novel_model, unvisited)


# =========================================================================
# 3a/3b(Memory Palace)가 공유하는 데이터 소스: miniimagenet(old item) / 숫자카드(new item).
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
# 3a. Memory Palace: Reconstruction Beyond Hippocampus Capacity (구 4a)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Running recall sequence...")
def _get_4a_recall(Nh, depth):
    scaf = _get_palace_scaffold(Nh)
    sbook_old, mbook_new, idxs_all, _, _ = _get_palace_books()
    idxs_seq = idxs_all[:depth]
    P_seq = scaf["pbook_flat"][:, :, idxs_seq]
    S_seq = sbook_old[:, idxs_seq]
    M_new = mbook_new[:, idxs_seq]

    S_clean = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1))
    S_addr1 = np.sign(S_clean[0])
    Wms = M_new @ np.linalg.pinv(S_addr1)
    M_rec_clean = Wms @ S_addr1
    return S_clean, M_rec_clean


def render_memory_palace_a():
    st.header("3a. Memory Palace: Reconstruction Beyond Hippocampus Capacity ($N_h$)")
    Ns = _PALACE_NS
    img_h = img_w = int(round(np.sqrt(Ns)))
    _, _, idxs_all, _, _ = _get_palace_books()
    n_cards = len(idxs_all)

    Nh = stepper_slider("N_h", 10, 800, 200, 5, key="palace_a_Nh")
    depth = stepper_slider("N_s", 2, n_cards, min(13, n_cards), 1, key="palace_a_depth")
    t = stepper_slider("Item index", 1, depth, 1, 1, key="palace_a_idx") - 1

    sbook_old, mbook_new, idxs_all, _, _ = _get_palace_books()
    S_clean, M_rec_clean = _get_4a_recall(Nh, depth)

    true_s = sbook_old[:, idxs_all[t]]
    true_m = mbook_new[:, idxs_all[t]]
    panels = [
        (true_s, f"Stored item #{t + 1}", None),
        (S_clean[0, :, t], f"Recalled item #{t + 1} (cos_sim={cos_sim(S_clean[0, :, t], true_s):.3f})", None),
        (true_m, "Mnemonic item", None),
        (M_rec_clean[:, t], f"Recalled mnemonic item (cos_sim={cos_sim(M_rec_clean[:, t], true_m):.3f})", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4))
    for ax, (vec, title, _sim) in zip(axes, panels):
        ax.imshow(vec.reshape(img_h, img_w), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# =========================================================================
# 3b. Memory Palace: Cleanup Test (구 4b)
# =========================================================================
@st.cache_resource(max_entries=1, show_spinner="Running cleanup pipeline...")
def _get_4b_pipeline(Nh, depth):
    scaf = _get_palace_scaffold(Nh)
    sbook_old, mbook_new, idxs_all, _, _ = _get_palace_books()
    idxs_seq = idxs_all[:depth]
    P_seq = scaf["pbook_flat"][:, :, idxs_seq]
    S_seq = sbook_old[:, idxs_seq]
    M_seq = mbook_new[:, idxs_seq]

    S_clean = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1))
    S_addr = np.sign(S_clean[0])
    Wms = M_seq @ np.linalg.pinv(S_addr)          # 주소 -> new item
    Wsm_raw = S_seq @ np.linalg.pinv(M_seq)       # new item -> sensory (원본 스케일)
    return scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw


def render_memory_palace_b():
    st.header("3b. Memory Palace: Cleanup Test")
    Ns = _PALACE_NS
    img_h = img_w = int(round(np.sqrt(Ns)))
    _, _, idxs_all, _, _ = _get_palace_books()
    n_cards = len(idxs_all)

    Nh = stepper_slider("N_h", 10, 400, 200, 5, key="palace_b_Nh")
    depth = stepper_slider("N_s", 2, n_cards, min(30, n_cards), 1, key="palace_b_depth")
    t = stepper_slider("Item index", 1, depth, 1, 1, key="palace_b_idx") - 1
    noise_ratio_vis = stepper_slider("Noise ratio", 0.0, 0.9, 0.3, 0.1, key="palace_b_noise_ratio")

    scaf, S_seq, M_seq, P_seq, S_clean, Wms, Wsm_raw = _get_4b_pipeline(Nh, depth)

    def recover(noisy_item, tt):
        sensory_est_noisy = Wsm_raw @ noisy_item
        S_query = S_seq.copy()
        S_query[:, tt] = sensory_est_noisy
        S_rec = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1), S_query=S_query)
        sensory_cleaned = S_rec[0, :, tt]
        addr_clean = np.sign(sensory_cleaned)
        item_rec = Wms @ addr_clean
        return sensory_est_noisy, sensory_cleaned, item_rec

    true_sensory = S_seq[:, t]
    sensory_baseline_rec = S_clean[0, :, t]
    true_item = M_seq[:, t]
    noisy_item = true_item if noise_ratio_vis == 0.0 else apply_noise(true_item, "salt_and_pepper", noise_ratio_vis, seed=2)
    sensory_est_noisy, sensory_cleaned, item_rec = recover(noisy_item, t)

    panels = [
        (true_sensory, f"Stored item #{t + 1}", None),
        (sensory_baseline_rec, f"Recalled item #{t + 1} (cos_sim={cos_sim(sensory_baseline_rec, true_sensory):.3f})", None),
        (true_item, "Mnemonic item", None),
        (noisy_item, "Noisy item", None),
        (sensory_est_noisy, f"Noisy sensory recon (cos_sim={cos_sim(sensory_est_noisy, true_sensory):.3f})", None),
        (sensory_cleaned, f"Cleanup sensory recall (cos_sim={cos_sim(sensory_cleaned, true_sensory):.3f})", None),
        (item_rec, f"Recalled mnemonic item (cos_sim={cos_sim(item_rec, true_item):.3f})", None),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.4))
    for ax, (vec, title, _sim) in zip(axes, panels):
        ax.imshow(vec.reshape(img_h, img_w), cmap="gray")
        ax.set_title(title, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    plt.tight_layout()
    plt.show()


# =========================================================================
st.title("Vector-HaSH")

section = st.sidebar.radio(
    "Section",
    ["1. Item Memory", "2. Spatial Memory", "3a. Memory Palace", "3b. Cleanup Test"],
)

if section == "1. Item Memory":
    render_item_memory()
elif section == "2. Spatial Memory":
    render_spatial_memory()
elif section == "3a. Memory Palace":
    render_memory_palace_a()
else:
    render_memory_palace_b()
