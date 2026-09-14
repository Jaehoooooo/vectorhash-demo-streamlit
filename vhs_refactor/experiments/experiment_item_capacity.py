"""
experiment_item_capacity.py
Model 1 (Item Memory): recall accuracy as the number of stored items grows,
evaluated with MSE, Cosine Similarity, Bit Accuracy, Exact Match, and Address Accuracy.
Updated with dynamic sub-dataset Pseudoinverse re-fitting and MiniImageNet Dataset (전체 평균 평가).
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import config as cfg
from grid_utils import GridCode
from scaffold import GridHPCScaffold
from item_memory import ItemMemory
from datasets import add_noise, prepare_sensory_data
from metrics import exact_match_rate
import matplotlib.pyplot as plt
import numpy as np

# -------------------------------------------------------------------------
# 평가지표 계산 헬퍼 함수
# -------------------------------------------------------------------------
def compute_metrics(recovered: np.ndarray, target: np.ndarray):
    """Sensory pattern 복원 성능 평가지표 계산"""
    mse = np.mean((recovered - target) ** 2)
    
    norm_rec = np.linalg.norm(recovered)
    norm_tgt = np.linalg.norm(target)
    
    cos_sim = np.dot(recovered, target) / (norm_rec * norm_tgt + 1e-10)
        
    bit_acc = np.mean(recovered == target)
    exact_match = float(exact_match_rate(recovered, target))
    
    return mse, cos_sim, bit_acc, exact_match


def get_grid_address_accuracy(mem: ItemMemory, noisy_item: np.ndarray, target_grid_state: np.ndarray) -> float:
    """Scaffold-Address Accuracy 측정"""
    if hasattr(mem, "recall_with_address"):
        res = mem.recall_with_address(noisy_item)
        rec_g = res["grid_state"]
    else:
        h_init = mem.Whs @ noisy_item if hasattr(mem, "Whs") else None
        if h_init is not None:
            _, rec_g = mem.scaffold.cleanup(h_init)
        else:
            return 0.0
            
    return float(np.array_equal(rec_g, target_grid_state))


# -------------------------------------------------------------------------
# 실험 1: 패턴 수(P) 증가에 따른 용량(Capacity) 평가 (Fig 3e style)
# -------------------------------------------------------------------------
def run_P_sweep(sbook_flattened: np.ndarray):
    """Fig 3e style: Original P (item count) sweep calculating overall averages."""
    print("\n=== [Experiment 1] P sweep (Overall Average) ===")
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, 
        Nh=scaf_cfg.Nh, 
        connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold,
        nonlinearity=scaf_cfg.nonlinearity,
        seed=1
    )
    # fig2/fig3/fig7과 동일하게: Wgh는 item 개수(n_patterns)와 무관하게 전체
    # combinatorial grid state 공간(all_states)으로 한 번만 학습하고 고정한다.
    # subset(codeword)으로 재학습하는 건 Wsp/Wps(=우리 Whs/Wsh, item memory)뿐.
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])

    master_items = sbook_flattened.T
    Ns = master_items.shape[1]

    n_patterns_list = [200, 400, 600]

    pattern_to_sims = {n: [] for n in n_patterns_list}
    pattern_to_mses = {n: [] for n in n_patterns_list}

    for n_patterns in n_patterns_list:
        if n_patterns > master_items.shape[0]:
            print(f"  [경고] N={n_patterns} 가용 데이터 초과. 패스.")
            continue

        items_sub = master_items[:n_patterns]
        mem = ItemMemory(grid_code, scaffold, Ns)
        mem.learn(items_sub)

        # print(f"  [진행] N={n_patterns:5d} 학습 완료, 전체 패턴 복원 평가 중...")
        
        for test_idx, s_star in enumerate(items_sub):
            noisy = add_noise(s_star, noise_level=0, seed=100 + test_idx)
            recovered = mem.recall(noisy)
            
            mse, cos_sim, bit_acc, exact = compute_metrics(recovered, s_star)
            pattern_to_sims[n_patterns].append(cos_sim)
            pattern_to_mses[n_patterns].append(mse)

        m_sim = np.mean(pattern_to_sims[n_patterns])
        s_sim = np.std(pattern_to_sims[n_patterns])
        m_mse = np.mean(pattern_to_mses[n_patterns])
        # print(f"  -> 완료 | CosSim = {m_sim:.4f} ± {s_sim:.4f} | MSE = {m_mse:.4f}")

    print("\n--- [Experiment 1 Summary] ---")
    for n_pat in n_patterns_list:
        if len(pattern_to_sims[n_pat]) > 0:
            print(f"P = {n_pat:5d} | Cosine Similarity = {np.mean(pattern_to_sims[n_pat]):.4f} ± {np.std(pattern_to_sims[n_pat]):.4f}")
    print("------------------------------")


# -------------------------------------------------------------------------
# 실험 2: 해마 뉴런 수(Nh) 변화에 따른 성능 비교
# -------------------------------------------------------------------------
def run_Nh_sweep(sbook_flattened: np.ndarray, n_items=600, Nh_list=(400, 800)):
    """Fig 3-style: hold P fixed, vary hippocampal capacity Nh using Overall Average."""
    print(f"\n=== [Experiment 2] Nh sweep (Fixed P={n_items}) ===")
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)

    master_items = sbook_flattened.T
    Ns = master_items.shape[1]

    if n_items > master_items.shape[0]:
        n_items = master_items.shape[0]

    items = master_items[:n_items]

    for Nh in Nh_list:
        scaffold = GridHPCScaffold(
            grid_code,
            Nh=Nh,
            connection_prob=scaf_cfg.connection_prob,
            threshold=scaf_cfg.threshold,
            nonlinearity=scaf_cfg.nonlinearity,
            seed=1
        )
        # Wgh는 item 개수와 무관하게 전체 combinatorial grid state 공간으로 학습(fig2/3/7 방식)
        scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
        mem = ItemMemory(grid_code, scaffold, Ns)
        mem.learn(items)

        mses, cos_sims, bit_accs, exacts, addr_accs = [], [], [], [], []
        
        # print(f"  [진행] Nh={Nh:4d} 피팅 완료, 전체 복원 평가 중...")
        for i, s in enumerate(items):
            noisy = add_noise(s, noise_level=0, seed=100 + i)
            recovered = mem.recall(noisy)
            
            mse, cos_sim, bit_acc, exact = compute_metrics(recovered, s)
            mses.append(mse)
            cos_sims.append(cos_sim)
            bit_accs.append(bit_acc)
            exacts.append(exact)
            
            if hasattr(mem, "assigned_grid_states"):
                target_g = mem.assigned_grid_states[i]
                addr_accs.append(get_grid_address_accuracy(mem, noisy, target_g))

        addr_str = f" | AddrAcc = {np.mean(addr_accs):.3f}" if addr_accs else ""
        print(f"  -> Nh={Nh:4d} | P={n_items:4d} | CosSim = {np.mean(cos_sims):.4f} ± {np.std(cos_sims):.4f} | MSE = {np.mean(mses):.4f}{addr_str}")


# -------------------------------------------------------------------------
# 실험 3: Cleanup(오차 교정) 유무에 따른 노이즈 강인성 비교
# -------------------------------------------------------------------------
def run_cleanup_comparison(sbook_flattened: np.ndarray, n_items=100, noise_levels=(0.05, 0.1, 0.2, 0.3)):
    """Compare recall accuracy: without scaffold cleanup vs with scaffold cleanup."""
    print(f"\n=== [Experiment 3] Scaffold Cleanup Comparison (P={n_items}) ===")
    scaf_cfg = cfg.DEFAULT_SCAFFOLD

    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, 
        Nh=scaf_cfg.Nh, 
        connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold,
        nonlinearity=scaf_cfg.nonlinearity,
        seed=1
    )
    # [수정] all_states(3600개)로 Wgh를 미리 고정하지 않는다.

    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    
    if n_items > master_items.shape[0]:
        n_items = master_items.shape[0]
        
    items = master_items[:n_items]
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)

    # Wgh는 item 개수와 무관하게 전체 combinatorial grid state 공간으로 학습(fig2/3/7 방식)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])

    for noise in noise_levels:
        no_cl_mses, no_cl_cos = [], []
        with_cl_mses, with_cl_cos, with_cl_addr = [], [], []

        for i, s in enumerate(items):
            noisy = add_noise(s, noise_level=noise, seed=100 + i)

            # 1. Without cleanup (정화 과정 생략)
            try:
                s_rec_no_cleanup = mem.recall(noisy, n_cleanup_iter=0)
            except TypeError:
                s_rec_no_cleanup = mem.recall(noisy, n_iter=0)

            mse, cos_sim, _, _ = compute_metrics(s_rec_no_cleanup, s)
            no_cl_mses.append(mse)
            no_cl_cos.append(cos_sim)

            # 2. With cleanup (정화 과정 수행)
            s_rec_with_cleanup = mem.recall(noisy)
            mse_w, cos_sim_w, _, _ = compute_metrics(s_rec_with_cleanup, s)
            with_cl_mses.append(mse_w)
            with_cl_cos.append(cos_sim_w)

            if hasattr(mem, "assigned_grid_states"):
                target_g = mem.assigned_grid_states[i]
                with_cl_addr.append(get_grid_address_accuracy(mem, noisy, target_g))

        addr_str = f" | AddrAcc: {np.mean(with_cl_addr):.3f}" if with_cl_addr else ""

        print(f"\n[Noise Level = {noise:4.2f}]")
        print(f"  NO Cleanup   -> CosSim = {np.mean(no_cl_cos):.4f} ± {np.std(no_cl_cos):.4f} | MSE = {np.mean(no_cl_mses):.4f}")
        print(f"  WITH Cleanup -> CosSim = {np.mean(with_cl_cos):.4f} ± {np.std(with_cl_cos):.4f} | MSE = {np.mean(with_cl_mses):.4f}{addr_str}")


# -------------------------------------------------------------------------
# Jupyter Notebook 전용 인터랙티브 시각화 함수들
# -------------------------------------------------------------------------
import matplotlib.pyplot as plt
import numpy as np
from src.seq_utils import explicit_interpolation



def extract_matrix(obj):
    if isinstance(obj, np.ndarray): return obj
    if hasattr(obj, 'detach'): return obj.detach().cpu().numpy()
    for attr in ['W', 'weight', 'weights', 'matrix']:
        if hasattr(obj, attr):
            val = getattr(obj, attr)
            if isinstance(val, np.ndarray): return val
            if hasattr(val, 'detach'): return val.detach().cpu().numpy()
    raise TypeError(f"행렬 추출 실패. 타입: {type(obj)}")


def vector_to_2d_user_style(vec, sigma=10):
    """1D population vector를 정사각형 2D로 접은 뒤 explicit_interpolation으로 스무딩"""
    vec = np.array(vec).flatten()
    n = len(vec)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side)
    padded[:n] = vec
    reshaped = padded.reshape((side, side))
    return explicit_interpolation(reshaped, sigma=sigma)


def mask_sensory(vec, mask_ratio):
    """원본 fig3b 방식: 이미지(1D로 펼친 sensory pattern)의 마지막
    mask_ratio 비율만큼을 0으로 지운다 (bit-flip 노이즈가 아니라 마스킹)."""
    corrupted = np.array(vec, dtype=float).copy()
    mask_idx = int(len(corrupted) * (1 - mask_ratio))
    corrupted[mask_idx:] = 0
    return corrupted


def salt_and_pepper_sensory(vec, noise_ratio, seed=0):
    """무작위로 고른 noise_ratio 비율의 픽셀을 절반은 최댓값(salt), 절반은
    최솟값(pepper)으로 뒤집는다. mask_sensory와 달리 위치가 매번 무작위이고
    0이 아니라 값 범위의 양 극단으로 튄다."""
    rng = np.random.default_rng(seed)
    corrupted = np.array(vec, dtype=float).copy()
    n = len(corrupted)
    n_noisy = int(n * noise_ratio)
    idx = rng.choice(n, size=n_noisy, replace=False)
    half = n_noisy // 2
    vmin, vmax = corrupted.min(), corrupted.max()
    corrupted[idx[:half]] = vmax   # salt
    corrupted[idx[half:]] = vmin   # pepper
    return corrupted


def apply_noise(vec, noise_type, noise_ratio, seed=0):
    if noise_type == "masking":
        return mask_sensory(vec, noise_ratio)
    elif noise_type == "salt_and_pepper":
        return salt_and_pepper_sensory(vec, noise_ratio, seed=seed)
    raise ValueError(f"unknown noise_type: {noise_type}")


def reshape_sensory_to_image(vec):
    """실제 sensory(이미지) 패턴은 population state가 아니므로 스무딩 없이
    정사각형으로만 접는다 (Ns=3600=60*60이면 원본 이미지 그대로 복원됨)."""
    vec = np.array(vec).flatten()
    n = len(vec)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side)
    padded[:n] = vec
    return padded.reshape((side, side))


def get_trained_model_for_notebook(n_items=100):
    """기본 데모용: 지정한 n_items만큼 학습된 모델 반환"""
    print(f"[{n_items}개의 패턴으로 모델 학습 중...]")
    sbook_flattened = prepare_sensory_data()
    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    items = master_items[:n_items]

    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    print("학습 완료!")
    return mem, items


def _stepper_row(widgets, slider, show_slider=True, offset=0):
    """-/+ 버튼 + 숫자 직접 입력 가능한 텍스트박스. IntText는 브라우저 기본 number
    input이라 스피너 화살표 때문에 숫자가 가운데 정렬 안 되는 문제가 있어서, 스피너가
    없는 일반 Text + 수동 int 파싱으로 대체. show_slider=False면 드래그 슬라이더
    자체는 화면에 안 보이고 버튼/텍스트박스만 표시 (slider는 값 저장용으로 계속 씀).
    offset: slider.value(내부, 0-index)와 화면에 보여주는 숫자 사이의 차이 (예: 1이면
    내부 0을 화면엔 "1"로 보여줌 -- item/test index를 1번부터 세고 싶을 때 사용)."""
    from IPython.display import HTML, display as _display
    _display(HTML("<style>.widget-text input{text-align:center !important;}</style>"))
    slider.readout = False
    minus = widgets.Button(description="-", layout=widgets.Layout(width="32px"))
    plus = widgets.Button(description="+", layout=widgets.Layout(width="32px"))
    text = widgets.Text(value=str(slider.value + offset), layout=widgets.Layout(width="60px"))
    minus.on_click(lambda b: setattr(slider, "value", round(max(slider.min, slider.value - slider.step), 10)))
    plus.on_click(lambda b: setattr(slider, "value", round(min(slider.max, slider.value + slider.step), 10)))
    slider.observe(lambda change: setattr(text, "value", str(change["new"] + offset)), names="value")

    def _on_text_change(change):
        try:
            v = int(change["new"]) - offset
        except ValueError:
            return
        slider.value = max(slider.min, min(slider.max, v))

    text.observe(_on_text_change, names="value")
    children = [slider, minus, plus, text] if show_slider else [minus, plus, text]
    return widgets.HBox(children)


def render_node_states_panel(mem, items, target_idx, noise_type, noise_ratio, exp_label=""):
    """show_node_states_interactive의 실제 그림/print 로직 본체(위젯 없이 순수 함수).
    다른 곳(예: 상위 interact)에서 target_idx 등을 직접 넘겨서 재사용할 때 씀."""
    n_iter = 2  # cleanup 고정 (드래그 슬라이더 제거)
    s_original = items[target_idx]
    s_noisy = apply_noise(s_original, noise_type, noise_ratio, seed=target_idx)

    W_hs = extract_matrix(mem.Whs)
    W_sh = extract_matrix(mem.Wsh)
    grid_code = mem.scaffold.grid_code
    g_true = mem.codewords[target_idx]

    h_noisy = W_hs @ s_noisy
    logits_pre_wta = mem.scaffold.Wgh @ h_noisy  # WTA(argmax) 전 module별 실수값 logits
    if n_iter == 0:
        # cleanup 0번 = raw recall(WTA 정화 전 그대로)
        h_clean, g_clean = h_noisy, None
    else:
        h_clean, g_clean = mem.scaffold.cleanup(h_noisy, n_iter=n_iter)  # 실제 recall과 동일한 WTA cleanup
    s_recovered = W_sh @ h_clean

    s_orig_2d = reshape_sensory_to_image(s_original)
    s_noisy_2d = reshape_sensory_to_image(s_noisy)
    s_rec_2d = reshape_sensory_to_image(s_recovered)

    cos_sim = float(np.dot(s_recovered, s_original) /
                     (np.linalg.norm(s_recovered) * np.linalg.norm(s_original) + 1e-10))

    # sensory 코사인 유사도 대신 grid state로 몇 번 item인지 판단: 복원된
    # grid state(g_clean)가 mem.codewords 중 어느 item의 grid state와 정확히
    # 일치하는지 찾음. 어떤 item과도 안 맞으면(=학습 때 배정 안 된 빈 grid state) None.
    if g_clean is None:
        closest_idx = None
    else:
        codewords_arr = np.asarray(mem.codewords)
        match_indices = np.where(np.all(codewords_arr == g_clean, axis=1))[0]
        closest_idx = int(match_indices[0]) if len(match_indices) > 0 else None

    from experiments.experiment_spatial_navigation import plot_grid_modules_square

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))

    axes[0, 0].imshow(s_orig_2d, cmap="gray")
    axes[0, 0].set_title(f"Stored item #{target_idx + 1}", fontsize=10)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(s_noisy_2d, cmap="gray")
    axes[0, 1].set_title("Noisy item", fontsize=10)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(s_rec_2d, cmap="gray")
    title3_suffix = "None" if closest_idx is None else f"#{closest_idx + 1}"
    axes[0, 2].set_title(f"Recalled item {title3_suffix} (cos_sim={cos_sim:.3f})", fontsize=10)
    axes[0, 2].axis('off')

    plot_grid_modules_square(axes[1, 0], grid_code, g_true)
    axes[1, 0].set_title("Grid state (true)", fontsize=10)

    plot_grid_modules_square(axes[1, 1], grid_code, logits_pre_wta, vmin=None, vmax=None)
    axes[1, 1].set_title("Grid state (pre-WTA)", fontsize=10)

    if g_clean is not None:
        plot_grid_modules_square(axes[1, 2], grid_code, g_clean)
        axes[1, 2].set_title("Grid state (post-WTA)", fontsize=10)
    else:
        axes[1, 2].axis('off')

    plt.tight_layout()
    plt.show()


def show_node_states_interactive(mem, items, exp_label=""):
    """1. Original Sensory / 2. Masked Sensory (input) / 3. Reconstructed
    Sensory 3패널 인터랙티브 뷰 (기본적인 sensory 복원, cos_sim 표시).
    grid state는 이미지 대신 module별 index 번호를 print로 출력한다.
    어떤 실험에서 만든 mem이든 그대로 받아서 동일한 방식으로 시각화."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        return

    def update_plot(target_idx, noise_type, noise_ratio):
        render_node_states_panel(mem, items, target_idx, noise_type, noise_ratio, exp_label=exp_label)

    idx_slider = widgets.IntSlider(value=0, min=0, max=len(items) - 1, description="Item Index:",
                                    continuous_update=False)
    noise_dropdown = widgets.Dropdown(options=["masking", "salt_and_pepper"], value="masking", description="Noise Type:")
    ratio_slider = widgets.FloatSlider(value=0.1, min=0.1, max=0.9, step=0.1, description="Noise Ratio:",
                                        continuous_update=False)
    out = widgets.interactive_output(update_plot, {"target_idx": idx_slider, "noise_type": noise_dropdown,
                                                     "noise_ratio": ratio_slider})
    display(_stepper_row(widgets, idx_slider, offset=1), noise_dropdown, ratio_slider, out)


def show_cleanup_sweep_interactive(mem, items, exp_label="", max_iter=10):
    """item/noise 슬라이더는 그대로 두고, cleanup의 n_iter를 1..max_iter로 바꿔가며
    같은 noisy cue에 대해 sensory cos_sim과 grid state mismatch가 iteration마다
    어떻게 줄어드는지(몇 번 만에 수렴하는지) 확인."""
    try:
        import ipywidgets as widgets
    except ImportError:
        return

    def update_plot(target_idx, noise_type, noise_ratio):
        s_original = items[target_idx]
        s_noisy = apply_noise(s_original, noise_type, noise_ratio, seed=target_idx)

        W_hs = extract_matrix(mem.Whs)
        W_sh = extract_matrix(mem.Wsh)
        h_noisy = W_hs @ s_noisy
        grid_code = mem.scaffold.grid_code
        g_true = mem.codewords[target_idx]
        grid_idx_true = grid_code.decode_state(g_true)

        iters = list(range(1, max_iter + 1))
        cos_sims, n_mismatched_list = [], []
        for n_iter in iters:
            h_clean, g_clean = mem.scaffold.cleanup(h_noisy, n_iter=n_iter)
            s_recovered = W_sh @ h_clean
            cos_sim = float(np.dot(s_recovered, s_original) /
                             (np.linalg.norm(s_recovered) * np.linalg.norm(s_original) + 1e-10))
            grid_idx = grid_code.decode_state(g_clean)
            module_dists = [
                min(abs(x1 - x2), k - abs(x1 - x2)) + min(abs(y1 - y2), k - abs(y1 - y2))
                for (x1, y1), (x2, y2), k in zip(grid_idx_true, grid_idx, grid_code.module_periods)
            ]
            cos_sims.append(cos_sim)
            n_mismatched_list.append(sum(d > 0 for d in module_dists))

        print(f"item={target_idx} | noise={noise_type} ({noise_ratio:.2f}) | "
              f"cos_sim per n_iter={[round(c, 3) for c in cos_sims]}")
        print(f"mismatched modules per n_iter={n_mismatched_list}")

        fig, ax1 = plt.subplots(figsize=(8, 4))
        fig.suptitle(f"{exp_label} | item={target_idx} | cleanup n_iter sweep", fontsize=13)
        ax1.plot(iters, cos_sims, marker="o", color="tab:blue", label="sensory cos_sim")
        ax1.set_xlabel("cleanup n_iter")
        ax1.set_ylabel("cos_sim", color="tab:blue")
        ax1.set_ylim(-0.05, 1.05)

        ax2 = ax1.twinx()
        ax2.plot(iters, n_mismatched_list, marker="s", color="tab:red", label="mismatched modules")
        ax2.set_ylabel("mismatched modules", color="tab:red")
        ax2.set_ylim(-0.5, len(grid_code.module_periods) + 0.5)

        plt.tight_layout()
        plt.show()

    idx_slider = widgets.IntSlider(value=0, min=0, max=len(items) - 1, description="Item Index:")
    noise_dropdown = widgets.Dropdown(options=["masking", "salt_and_pepper"], value="masking", description="Noise Type:")
    ratio_slider = widgets.FloatSlider(value=0.1, min=0.0, max=0.9, step=0.1, description="Noise Ratio:")
    widgets.interact(update_plot, target_idx=idx_slider, noise_type=noise_dropdown, noise_ratio=ratio_slider)


def show_grid_to_sensory_interactive(mem, exp_label=""):
    """임의의 item에 배정된 grid state(module index로 지정)를 입력으로 주면
    거기에 저장된 sensory를 시각화. mem.codewords[idx]가 곧 item idx가 학습 시
    실제로 배정받은 grid state이므로, 슬라이더를 움직이며 "새 sensory 추가 시
    어떤 grid state에 저장됐는지"도 그대로 확인 가능하다. grid state는 이미지
    대신 module별 index 번호를 print로 출력한다."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        return

    W_sh = extract_matrix(mem.Wsh)

    def update_plot(item_idx):
        g = mem.codewords[item_idx]                 # 이 item에 배정된 grid state
        h = mem.scaffold.grid_to_hpc(g)              # grid -> hpc (scaffold 순전파)
        s_stored = W_sh @ h                          # hpc -> sensory (저장된 sensory 복원)

        grid_idx = mem.scaffold.grid_code.decode_state(g)
        print(f"item={item_idx} -> assigned grid state (module indices) = {grid_idx}")

        s_2d = reshape_sensory_to_image(s_stored)

        fig, ax = plt.subplots(1, 1, figsize=(6, 5))
        fig.suptitle(f"{exp_label} | item={item_idx}  (grid state -> stored sensory)", fontsize=12)
        ax.imshow(s_2d, cmap="gray")
        ax.set_title("Sensory bound to this grid state")
        ax.axis('off')

        plt.tight_layout()
        plt.show()

    idx_slider = widgets.IntSlider(value=0, min=0, max=len(mem.codewords) - 1, description="Item Index:")
    widgets.interact(update_plot, item_idx=idx_slider)


def demo_add_new_sensory(mem, new_sensory, exp_label=""):
    """새 sensory item 하나를 mem.add_item()으로 실제 추가하고, 어느 grid state에
    배정됐는지(module별 index 번호, print) + Original vs Reconstructed sensory를
    보여준다. 기존에 저장돼 있던 item들은 그대로 둔 채(scaffold Wgh도 안 건드림)
    Whs/Wsh만 1건 증분 학습된다."""
    g_new = mem.add_item(new_sensory)
    grid_idx = mem.scaffold.grid_code.decode_state(g_new)

    s_recovered = mem.recall(new_sensory)
    cos_sim = float(np.dot(s_recovered, new_sensory) /
                     (np.linalg.norm(s_recovered) * np.linalg.norm(new_sensory) + 1e-10))
    print(f"[새 item 추가] item_idx={len(mem.codewords) - 1} -> 배정된 grid state (module indices) = {grid_idx}")
    print(f"[새 item 추가] recall cos_sim = {cos_sim:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    fig.suptitle(f"{exp_label} | new item -> grid {grid_idx} | cos_sim={cos_sim:.3f}", fontsize=12)

    axes[0].imshow(reshape_sensory_to_image(new_sensory), cmap="gray")
    axes[0].set_title("Original Sensory (new item)")
    axes[0].axis('off')

    axes[1].imshow(reshape_sensory_to_image(s_recovered), cmap="gray")
    axes[1].set_title("Reconstructed Sensory")
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    return g_new, cos_sim


def show_capacity_sweep_interactive(sbook_flattened, Nh=400, max_items=1200, test_idx=0):
    """슬라이더 2개(n_items = 학습에 쓰는 이미지 개수, test_idx = 확인할 item
    번호)로 sub-capacity -> over-capacity 전환을 한 화면에서 확인. Nh는 슬라이더
    내내 고정(scaffold의 Wgh도 한 번만 fit)하고, n_items가 늘어날 때마다
    Whs/Wsh만 그 개수만큼 다시 학습해서 test_idx번째 이미지의 recall 결과
    (cos_sim)가 n_items<=Nh 구간에서는 거의 1.0이다가 n_items>Nh로 넘어가면서
    떨어지는 걸 보여준다. test_idx >= n_items면 아직 학습 안 된 item이라 안내만
    표시한다."""
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        return

    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    scaffold_grid = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        scaffold_grid, Nh=Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    scaffold.fit_wgh([scaffold_grid.encode_state(s) for s in scaffold_grid.all_states()])  # Nh 고정, 한 번만

    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    max_items = min(max_items, master_items.shape[0])

    def update_plot(n_items, test_idx):
        n_iter = 2  # cleanup 고정 (드래그 슬라이더 제거)
        if test_idx >= n_items:
            print(f"item={test_idx}는 아직 학습 안 됨 (n_items={n_items} 이하여야 학습됨). "
                  f"n_items를 {test_idx + 1} 이상으로 올리세요.")
            return

        items = master_items[:n_items]
        item_grid = GridCode(module_periods=scaf_cfg.module_periods, seed=0)  # 매번 같은 순서로 codeword 배정 재현
        mem = ItemMemory(item_grid, scaffold, Ns)
        mem.learn(items)

        s_original = items[test_idx]
        g_true = mem.codewords[test_idx]

        W_hs = extract_matrix(mem.Whs)
        W_sh = extract_matrix(mem.Wsh)
        h0 = W_hs @ s_original
        if n_iter == 0:
            h_clean, g_clean = h0, None
        else:
            h_clean, g_clean = scaffold.cleanup(h0, n_iter=n_iter)
        s_recovered = W_sh @ h_clean
        cos_sim = float(np.dot(s_recovered, s_original) /
                         (np.linalg.norm(s_recovered) * np.linalg.norm(s_original) + 1e-10))

        from experiments.experiment_spatial_navigation import plot_grid_modules_square

        regime = "sub-capacity (n_items<=Nh)" if n_items <= Nh else "OVER-CAPACITY (n_items>Nh)"
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        fig.suptitle(f"n_items={n_items} / Nh={Nh} | {regime} | test item={test_idx} | "
                     f"cleanup n_iter={n_iter} | cos_sim={cos_sim:.3f}", fontsize=13)
        axes[0, 0].imshow(reshape_sensory_to_image(s_original), cmap="gray")
        axes[0, 0].set_title("Original Sensory")
        axes[0, 0].axis('off')
        axes[0, 1].imshow(reshape_sensory_to_image(s_recovered), cmap="gray")
        axes[0, 1].set_title("Reconstructed Sensory")
        axes[0, 1].axis('off')

        plot_grid_modules_square(axes[1, 0], item_grid, g_true)
        axes[1, 0].set_title("Grid state (true)", fontsize=10)

        if g_clean is not None:
            plot_grid_modules_square(axes[1, 1], item_grid, g_clean)
            axes[1, 1].set_title("Grid state (recalled)", fontsize=10)
        else:
            axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    n_items_slider = widgets.IntSlider(value=test_idx + 1, min=1, max=max_items, step=1,
                                        description="n_items:", continuous_update=False)
    test_idx_slider = widgets.IntSlider(value=test_idx, min=0, max=max_items - 1, step=1,
                                         description="test_idx:", continuous_update=False)
    out = widgets.interactive_output(update_plot, {"n_items": n_items_slider, "test_idx": test_idx_slider})
    display(_stepper_row(widgets, n_items_slider), _stepper_row(widgets, test_idx_slider, offset=1), out)


# -------------------------------------------------------------------------
# 각 실험(P sweep / Nh sweep / cleanup comparison)과 동일한 파라미터로
# mem을 준비하는 함수들 -- 위 show_node_states_interactive에 그대로 연결
# -------------------------------------------------------------------------
def get_mem_for_P_sweep(sbook_flattened, n_patterns=400):
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    items = master_items[:n_patterns]
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    return mem, items


def get_mem_for_Nh_sweep(sbook_flattened, Nh=800, n_items=600):
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    items = master_items[:n_items]
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    return mem, items


def get_mem_for_cleanup(sbook_flattened, n_items=1000):
    scaf_cfg = cfg.DEFAULT_SCAFFOLD
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    master_items = sbook_flattened.T
    Ns = master_items.shape[1]
    items = master_items[:n_items]
    mem = ItemMemory(grid_code, scaffold, Ns)
    mem.learn(items)
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    return mem, items


def run():
    # 1. 공통 데이터 1회 로드 및 전처리
    sbook_flattened = prepare_sensory_data()
    
    # 2. 통합된 데이터로 전체 실험 순차 실행
    run_P_sweep(sbook_flattened)
    run_Nh_sweep(sbook_flattened, n_items=400) # 용량 비교를 명확히 보기 위해 400개 디폴트 세팅
    run_cleanup_comparison(sbook_flattened, n_items=1000)

if __name__ == "__main__":
    run()