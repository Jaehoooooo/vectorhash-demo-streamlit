"""
experiment_full_capacity_suite.py

VectorHASH_fig7.ipynb 에 실제로 정의된 함수들
(build_seq_scaffold, make_hairpin_path, path_to_indices,
 recall_sequence_once, pseudotrain_Wps/Wsp, module_wise_NN_2d, nonlin)
을 그대로 사용해서 "6 Required Experiments" 7개 항목을 모두 구현.

주의:
- build_seq_scaffold 내부는 numpy.random 전역 상태(randn/randint)를 사용하므로
  완전한 재현성이 필요하면 호출 전에 np.random.seed(...)를 직접 설정하세요.
- src.assoc_utils_np / src.assoc_utils_np_2D / src.seq_utils 등은
  노트북과 동일한 경로에 있다고 가정합니다.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import randn, randint

from src.assoc_utils_np import *
from src.assoc_utils_np_2D import gen_gbook_2d, path_integration_Wgg_2d, module_wise_NN_2d
from src.seq_utils import *
from src.sensory_utils import *
from src.senstranspose_utils import *


# =========================================================================
# 노트북에서 그대로 가져온 공통 유틸 (수정 없음)
# =========================================================================
def build_seq_scaffold(lambdas, Nh, gamma=0.6, thresh=2.5, nruns=1):
    Ng = int(np.sum(np.square(lambdas)))
    Npos = int(np.prod(lambdas))
    Nstates = Npos * Npos

    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    gbook_flat = gbook.reshape(Ng, Nstates)

    Wpg = randn(nruns, Nh, Ng)

    prune = int((1 - gamma) * Nh * Ng)
    mask = np.ones((Nh, Ng))
    mask[
        randint(low=0, high=Nh, size=prune),
        randint(low=0, high=Ng, size=prune)
    ] = 0
    Wpg = Wpg * mask

    pbook = nonlin(np.einsum("ijk,klm->ijlm", Wpg, gbook), thresh=thresh)
    pbook_flat = pbook.reshape(nruns, Nh, Nstates)

    Wgp = train_gcpc(pbook_flat, gbook_flat, Nstates)

    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]

    return {
        "lambdas": lambdas, "Ng": Ng, "Nh": Nh, "Npos": Npos, "Nstates": Nstates,
        "gbook": gbook, "gbook_flat": gbook_flat, "pbook": pbook, "pbook_flat": pbook_flat,
        "Wpg": Wpg, "Wgp": Wgp, "module_sizes": module_sizes,
        "module_gbooks": module_gbooks, "gamma": gamma, "thresh": thresh,
    }


def build_seq_scaffold_pinv(lambdas, Nh, gamma=0.6, thresh=2.5, nruns=1):
    """build_seq_scaffold와 완전히 동일하되, Wgp(h->g)만 Hebbian(train_gcpc) 대신
    pseudo-inverse로 학습한다 (scaffold.py의 rule="pinv"와 같은 컨셉: 원래 fig5
    노트북이 쓰던 "전체 Nstates에 대한 exact-recall Wgp" 방식). Crosstalk이 없어서
    depth<=Nh 근처까지는 완벽 복원되지만, Hebbian의 점진적 capacity phase-transition
    특성은 사라진다."""
    Ng = int(np.sum(np.square(lambdas)))
    Npos = int(np.prod(lambdas))
    Nstates = Npos * Npos

    gbook = gen_gbook_2d(lambdas, Ng, Npos)
    gbook_flat = gbook.reshape(Ng, Nstates)

    Wpg = randn(nruns, Nh, Ng)

    prune = int((1 - gamma) * Nh * Ng)
    mask = np.ones((Nh, Ng))
    mask[
        randint(low=0, high=Nh, size=prune),
        randint(low=0, high=Ng, size=prune)
    ] = 0
    Wpg = Wpg * mask

    pbook = nonlin(np.einsum("ijk,klm->ijlm", Wpg, gbook), thresh=thresh)
    pbook_flat = pbook.reshape(nruns, Nh, Nstates)

    Wgp = np.stack([gbook_flat @ np.linalg.pinv(pbook_flat[r]) for r in range(nruns)])

    module_sizes = np.square(lambdas)
    module_gbooks = [np.eye(i) for i in module_sizes]

    return {
        "lambdas": lambdas, "Ng": Ng, "Nh": Nh, "Npos": Npos, "Nstates": Nstates,
        "gbook": gbook, "gbook_flat": gbook_flat, "pbook": pbook, "pbook_flat": pbook_flat,
        "Wpg": Wpg, "Wgp": Wgp, "module_sizes": module_sizes,
        "module_gbooks": module_gbooks, "gamma": gamma, "thresh": thresh,
    }


def make_hairpin_path(width, height, x0=0, y0=0):
    path = []
    for y in range(height):
        xs = range(width) if y % 2 == 0 else range(width - 1, -1, -1)
        for x in xs:
            path.append((x + x0, y + y0))
    return np.array(path, dtype=int)


def path_to_indices(path_locations, Npos):
    return np.array([x * Npos + y for x, y in path_locations], dtype=int)


def binary_mi_from_l1(err_l1):
    m = 1 - 2 * err_l1
    a = np.clip((1 + m) / 2, 1e-12, 1 - 1e-12)
    b = np.clip((1 - m) / 2, 1e-12, 1 - 1e-12)
    H = -a * np.log2(a) - b * np.log2(b)
    return np.maximum(1 - H, 0)


def recall_sequence_once(scaf, S_seq, P_seq, Nseq, rng=None, noise_frac=0.0, skip_cleanup=False,
                          S_query=None, return_grid=False):
    """
    노트북의 두 버전(무노이즈 버전 / noise_frac 버전)을 하나로 통합.
    - rng=None 또는 noise_frac=0 : 완전 결정론적(무노이즈) 회상
      -> 항목 7의 "full-rank recalled sensory states" 조건에 해당
    - rng가 주어지고 noise_frac>0 : grid 표상(gin)에 소량 가우시안 노이즈를 섞음
      -> 항목 3의 "반복 회상" 실험에서 두 번의 독립 회상을 만들 때 사용
    - S_query : Wps/Wsp는 원래(깨끗한) S_seq로 학습하되, 실제 query로는 이 배열을
      사용 -> "노이즈 낀 사진을 보여줬을 때도 제대로 회상하는가"를 테스트할 때
      2a의 apply_noise(masking/salt_and_pepper)로 만든 이미지를 넣어주면 됨.
      None이면 S_seq를 그대로 query로 사용(무노이즈).
    - skip_cleanup=True : module_wise_NN_2d(모듈별 discrete cleanup)를 건너뛰고
      noise 낀 연속값 gin을 그대로 사용 -> 노이즈가 실제로 얼마나 표상을
      흐트러뜨리는지 시각화할 때 사용 (cleanup이 이걸 대부분 지워버리기 때문에,
      cleanup 이후 결과만 보면 노이즈 효과가 잘 안 보임).
    """
    Wps = pseudotrain_Wps(P_seq, S_seq, Nseq)
    Wsp = pseudotrain_Wsp(S_seq, P_seq, Nseq)

    S_query = S_seq if S_query is None else S_query

    pin = nonlin(Wps @ S_query, thresh=0)
    gin = scaf["Wgp"] @ pin

    if rng is not None and noise_frac > 0:
        noise_std = noise_frac * gin.std()
        gin = gin + rng.normal(scale=noise_std, size=gin.shape)

    if skip_cleanup:
        G_rec = gin
    else:
        G_rec = np.zeros((1, scaf["Ng"], Nseq))
        for k in range(Nseq):
            G_rec[:, :, k] = module_wise_NN_2d(
                gin[:, :, k, None], scaf["module_gbooks"], scaf["module_sizes"]
            )[0, :, 0]

    P_rec = nonlin(scaf["Wpg"] @ G_rec, scaf["thresh"])
    S_rec = Wsp @ P_rec
    return (S_rec, G_rec) if return_grid else S_rec


def cos_sim(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


# =========================================================================
# 1~5: palace 길이별 old-landmark 열화 + 반복회상 일관성 + item 결합 A/B 비교
# =========================================================================
def run_experiments_1_to_5(
    lambdas=(4, 5, 7), Nh=400, Ns=3600, gamma=0.6, thresh=0.5,
    depths=(200, 800, 2000, 3600), test_positions=5,
    noise_frac=0.025, seed=0,
):
    """
    1) depths 각각에 대해 palace(sequence) 구성
    2) old landmark 회상 정확도가 palace 길이 증가에 따라 나빠지는지 확인
    3) 같은 시퀀스를 서로 다른 노이즈로 두 번 독립 회상 -> 두 결과가 서로 얼마나
       일치하는지(consistency)로 attractor의 안정성 확인
    4) 새 item을 (a) 원본 landmark, (b) 실제로 회상된 landmark 에 결합
    5) 두 결합 방식의 새 item 회상 정확도 비교
    """
    rng_master = np.random.default_rng(seed)
    scaf = build_seq_scaffold(list(lambdas), Nh, gamma=gamma, thresh=thresh, nruns=1)
    Npos = scaf["Npos"]

    path = make_hairpin_path(Npos, Npos)
    idxs_full = path_to_indices(path, Npos)
    assert max(depths) <= len(idxs_full), "depths가 grid 상태공간(Npos^2)보다 큽니다."

    results = {d: {"old1": [], "old2": [], "consistency": [],
                    "item_true": [], "item_recalled": []} for d in depths}

    for d in depths:
        idxs_seq = idxs_full[:d]
        P_seq = scaf["pbook_flat"][:, :, idxs_seq]

        for t_rel in np.linspace(0, d - 1, test_positions, dtype=int):
            S_field = np.sign(rng_master.standard_normal((Ns, scaf["Nstates"])))
            M_field = np.sign(rng_master.standard_normal((Ns, scaf["Nstates"])))

            S_seq = S_field[:, idxs_seq]
            M_new = M_field[:, idxs_seq]

            rng1 = np.random.default_rng(rng_master.integers(1_000_000_000))
            rng2 = np.random.default_rng(rng_master.integers(1_000_000_000))

            # --- 2) old-landmark recall: 독립적으로 두 번 회상, 원본과 비교 ---
            S_rec1 = recall_sequence_once(scaf, S_seq, P_seq, d, rng1, noise_frac)
            S_rec2 = recall_sequence_once(scaf, S_seq, P_seq, d, rng2, noise_frac)

            true_vec = S_seq[:, t_rel]
            r1, r2 = S_rec1[0, :, t_rel], S_rec2[0, :, t_rel]

            results[d]["old1"].append(cos_sim(r1, true_vec))
            results[d]["old2"].append(cos_sim(r2, true_vec))

            # --- 3) 반복 회상 일관성: r1 vs r2 (원본과 무관하게 서로 비교) ---
            results[d]["consistency"].append(cos_sim(r1, r2))

            # --- 4) item 결합: (a) 원본 landmark  (b) 실제 회상된 landmark ---
            S_addr_true = S_seq
            S_addr_recalled = np.sign(S_rec1[0])

            Wms_true = M_new @ np.linalg.pinv(S_addr_true)
            Wms_recalled = M_new @ np.linalg.pinv(S_addr_recalled)

            # 테스트 시점 질의로는 "두 번째 독립 회상"을 사용 (재방문 시나리오)
            query_addr = np.sign(S_rec2[0])
            item_out_true = Wms_true @ query_addr
            item_out_recalled = Wms_recalled @ query_addr

            m_true_vec = M_new[:, t_rel]
            results[d]["item_true"].append(cos_sim(item_out_true[:, t_rel], m_true_vec))
            results[d]["item_recalled"].append(cos_sim(item_out_recalled[:, t_rel], m_true_vec))

    print("\n=== 1-5: Palace 길이별 old-landmark 열화 / 반복회상 일관성 / item 결합 A-B 비교 ===")
    for d in depths:
        r1m, r2m = np.mean(results[d]["old1"]), np.mean(results[d]["old2"])
        cons = np.mean(results[d]["consistency"])
        it_t = np.mean(results[d]["item_true"])
        it_r = np.mean(results[d]["item_recalled"])
        print(f"depth={d:5d} | old-landmark avg={(r1m + r2m) / 2:.3f} "
              f"| recall-vs-recall consistency={cons:.3f} "
              f"| item(원본 결합)={it_t:.3f} vs item(회상값 결합)={it_r:.3f}")

    ds = list(depths)
    old_means = [np.mean(results[d]["old1"] + results[d]["old2"]) for d in ds]
    cons_means = [np.mean(results[d]["consistency"]) for d in ds]
    it_true_means = [np.mean(results[d]["item_true"]) for d in ds]
    it_rec_means = [np.mean(results[d]["item_recalled"]) for d in ds]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(ds, old_means, "o-", label="old landmark vs true")
    axes[0].plot(ds, cons_means, "s--", label="recall-vs-recall consistency")
    axes[0].set_xlabel("Palace length (Nseq)")
    axes[0].set_ylabel("Cosine similarity")
    axes[0].set_title("Landmark recall degradation & consistency")
    axes[0].legend()

    axes[1].plot(ds, it_true_means, "o-", label="item bound to TRUE landmark")
    axes[1].plot(ds, it_rec_means, "s-", label="item bound to RECALLED landmark")
    axes[1].set_xlabel("Palace length (Nseq)")
    axes[1].set_ylabel("New-item recall cos sim")
    axes[1].set_title("Item binding: original vs recalled address")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("exp_1_to_5.pdf", bbox_inches="tight")
    plt.show()
    return results


# =========================================================================
# 6: Nh, Ns, Cs(=lambdas 조합), item 개수(=palace 길이) 스윕
# =========================================================================
def run_experiment_6(
    lambdas_list=((3, 4, 5), (4, 5, 7), (5, 7, 8)),
    Nh_list=(200, 400, 800),
    Ns_list=(1000, 3600, 10000),
    depth_list=(200, 800, 2000),
    seed=0,
):
    rng_master = np.random.default_rng(seed)
    print("\n=== 6: Nh / Ns / Cs / item 개수 스윕 ===")
    records = []

    for lambdas in lambdas_list:
        Cs = int(np.prod(lambdas)) ** 2  # 경로가 방문 가능한 전체 결합 상태 수
        for Nh in Nh_list:
            scaf = build_seq_scaffold(list(lambdas), Nh)
            Npos = scaf["Npos"]
            path = make_hairpin_path(Npos, Npos)
            idxs_full = path_to_indices(path, Npos)

            for Ns in Ns_list:
                for depth in depth_list:
                    if depth > len(idxs_full):
                        continue
                    idxs_seq = idxs_full[:depth]
                    P_seq = scaf["pbook_flat"][:, :, idxs_seq]

                    S_field = np.sign(rng_master.standard_normal((Ns, scaf["Nstates"])))
                    S_seq = S_field[:, idxs_seq]

                    S_rec = recall_sequence_once(scaf, S_seq, P_seq, depth, rng=None, noise_frac=0.0)
                    sims = [cos_sim(S_rec[0, :, k], S_seq[:, k]) for k in range(depth)]
                    mean_sim = float(np.mean(sims))
                    cap_bound = min(Cs, Ns)

                    records.append((tuple(lambdas), Cs, Nh, Ns, depth, cap_bound, mean_sim))
                    flag = "<= cap" if depth <= cap_bound else "> cap"
                    print(f"lambdas={lambdas} Cs={Cs:6d} Nh={Nh:4d} Ns={Ns:6d} "
                          f"depth={depth:5d} cap={cap_bound:6d} mean_cos={mean_sim:.3f} ({flag})")
    return records


# =========================================================================
# 7: full-rank 회상 조건에서 P_exact ~= min(Cs, Ns) 검증
# =========================================================================
def run_experiment_7(lambdas=(4, 5, 7), Nh=400, Ns_list=(1000, 3600, 10000),
                      depth_list=None, seed=0):
    """
    노이즈 없이(rng=None) 회상하여 회상된 sensory state가 full-rank에 가깝도록
    강제한 뒤, 이론적 예측 P_exact ~= min(Cs, Ns) 가 실제 정확도 급락 지점과
    일치하는지 확인.
    """
    scaf = build_seq_scaffold(list(lambdas), Nh)
    Npos = scaf["Npos"]
    Cs = Npos ** 2
    path = make_hairpin_path(Npos, Npos)
    idxs_full = path_to_indices(path, Npos)

    if depth_list is None:
        depth_list = np.unique(
            np.logspace(np.log10(50), np.log10(len(idxs_full)), 12).astype(int)
        )

    rng = np.random.default_rng(seed)
    print(f"\n=== 7: Full-rank 조건, P_exact ~= min(Cs, Ns) 검증 (Cs={Cs}) ===")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for Ns in Ns_list:
        cap_bound = min(Cs, Ns)
        S_field = np.sign(rng.standard_normal((Ns, scaf["Nstates"])))
        means = []
        for depth in depth_list:
            if depth > len(idxs_full):
                continue
            idxs_seq = idxs_full[:depth]
            P_seq = scaf["pbook_flat"][:, :, idxs_seq]
            S_seq = S_field[:, idxs_seq]

            S_rec = recall_sequence_once(scaf, S_seq, P_seq, depth, rng=None, noise_frac=0.0)
            sims = [cos_sim(S_rec[0, :, k], S_seq[:, k]) for k in range(depth)]
            means.append(np.mean(sims))
            flag = "<= cap" if depth <= cap_bound else "> cap"
            print(f"  Ns={Ns:6d} depth={depth:5d} cap={cap_bound:6d} "
                  f"mean_cos={np.mean(sims):.3f} ({flag})")

        ax.plot(depth_list[:len(means)], means, "o-", label=f"Ns={Ns}")
        ax.axvline(cap_bound, linestyle="--", alpha=0.4)

    ax.set_xscale("log")
    ax.set_xlabel("Palace length (Nseq) = number of bound items")
    ax.set_ylabel("Mean cosine similarity (recalled vs true)")
    ax.set_title(f"P_exact ~ min(Cs, Ns)  (Cs={Cs})")
    ax.legend()
    plt.tight_layout()
    # plt.savefig("exp_7_full_rank_capacity.pdf", bbox_inches="tight")
    plt.show()

# =========================================================================
# fig 7d: VectorHASH_fig7de.ipynb의 fig7d를 그대로 재현 (image-panel 데모)
# =========================================================================
def make_embedded_image_book_for_fig7(
    Ns, Nstates, Npos,
    block_x0=0, block_y0=0, block_w=60, block_h=60,
    seed=0,
    shuffle_images=False,
    use_tanh_inverse=True,
):
    """VectorHASH_fig7.py의 make_embedded_image_book_for_fig7과 동일.
    실제 MiniImageNet 이미지를 block_w x block_h 위치 블록에 심어 넣고,
    나머지 위치는 무작위 sensory 패턴으로 채운다."""
    rng = np.random.default_rng(seed)

    npy_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "BW_miniimagenet_3600_60_60_full_rank.npy")
    img = np.load(npy_path)
    img_flat = img.reshape(3600, 3600).T
    # ponytail: float32로 낮춰서 메모리 절반 (Render 512MB 한도, 데모용이라 정밀도 손실 무해)
    # cast를 먼저 해야 float64 원본(img)이 살아있는 동안 float64 중간 복사본까지
    # 추가로 안 생김 (- 후 cast하면 float64 중간값이 잠깐 더 떠서 메모리 튐)
    img_flat = img_flat.astype(np.float32)
    del img
    img_flat -= img_flat.mean()

    if shuffle_images:
        perm = rng.permutation(img_flat.shape[1])
        img_flat = img_flat[:, perm]

    if use_tanh_inverse:
        smin, smax = np.amin(img_flat), np.amax(img_flat)
        scale = 1.9 / (smax - smin)
        shift = -0.95 - smin * scale
        img_flat *= scale
        img_flat += shift  # np.interp((smin,smax)->(-0.95,0.95))와 동일한 선형 변환, in-place
        np.arctanh(img_flat, out=img_flat)
        img_embed = img_flat
    else:
        img_embed = np.sign(img_flat)
        smin, smax = None, None

    n_positions = block_w * block_h
    # block이 (0,0)부터 시작해서 Nstates 전체를 정확히 덮는 경우(이 앱의 실제 사용
    # 패턴) idx(=x*Npos+y)가 항상 k와 같은 순서로 증가 -> sbook_full은 img_embed의
    # 앞 n_positions열과 완전히 동일. (예전엔 랜덤 배열 만들고 한 칸씩 덮어썼는데
    # 전부 버려지는 값이라 낭비였음 + 계산도 전체 3600열에 대해 다 했음)
    assert block_x0 == 0 and block_y0 == 0 and n_positions == Nstates and Npos == block_h
    sbook_full = np.ascontiguousarray(img_embed[:, :n_positions])

    return sbook_full, smin, smax


def load_trump_card_images_grayscale():
    """trump_card_60x60/ 안 52장 카드 이미지를 그레이스케일 60x60으로 불러와
    (Ns=3600, n_cards=52) 형태로 반환. 파일명 정렬 순서 그대로 카드 순서가 됨."""
    import glob
    from PIL import Image
    folder = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                           "trump_card_60x60")
    files = sorted(glob.glob(os.path.join(folder, "*.png")))
    imgs = [np.array(Image.open(f).convert("L"), dtype=np.float64) for f in files]
    arr = np.stack(imgs, axis=0)  # (n_cards, 60, 60)
    return arr.reshape(len(files), -1).T  # (Ns=3600, n_cards=52)


def make_embedded_trump_card_book(
    Ns, Nstates, Npos,
    block_x0=0, block_y0=0, block_w=13, block_h=4,
    seed=0, shuffle_images=False, use_tanh_inverse=True,
):
    """make_embedded_image_book_for_fig7과 동일한 로직이지만, MiniImageNet 대신
    trump_card_60x60의 52장 카드 이미지를 심어 넣는다. block_w*block_h은
    카드 장수(52)와 같아야 한다 -- hairpin 경로가 이 block 전체를 정확히
    훑고 지나가야 depths가 전부 '실제 카드 이미지'에 대응되기 때문."""
    rng = np.random.default_rng(seed)

    img_flat = load_trump_card_images_grayscale()
    n_cards = img_flat.shape[1]
    assert block_w * block_h == n_cards, (
        f"block_w*block_h({block_w * block_h})은 카드 장수({n_cards})와 같아야 합니다."
    )
    assert Ns == img_flat.shape[0], f"Ns({Ns})는 카드 픽셀 수({img_flat.shape[0]})와 같아야 합니다."
    img_flat = img_flat - np.mean(img_flat)

    if shuffle_images:
        perm = rng.permutation(img_flat.shape[1])
        img_flat = img_flat[:, perm]

    if use_tanh_inverse:
        smin, smax = np.amin(img_flat), np.amax(img_flat)
        img_scaled = np.interp(img_flat, (smin, smax), (-0.95, 0.95))
        img_embed = np.arctanh(img_scaled)
    else:
        img_embed = np.sign(img_flat)
        smin, smax = None, None

    sbook_full = rng.standard_normal((Ns, Nstates))

    k = 0
    for x in range(block_x0, block_x0 + block_w):
        for y in range(block_y0, block_y0 + block_h):
            idx = x * Npos + y
            sbook_full[:, idx] = img_embed[:, k]
            k += 1

    return sbook_full, smin, smax


def plot_palace_path(block_w=13, block_h=4, block_x0=0, block_y0=0, show_labels=True, highlight_t=None):
    """3a의 plot_paths처럼, hairpin 경로(=palace에 이미지를 저장하는 순서)를
    x-y 평면에 그린다. 각 점이 t번째로 결합된 위치(=idxs_7d[t])와 대응.
    highlight_t를 주면 그 위치를 큰 별표로 강조 표시(현재 슬라이더 t 위치 등)."""
    path = make_hairpin_path(block_w, block_h, block_x0, block_y0)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(path[:, 0], path[:, 1], "-o", markersize=1.2, alpha=0.6, color="tab:blue")
    ax.plot(path[0, 0], path[0, 1], "gs", markersize=4, label="start (t=0)")
    ax.plot(path[-1, 0], path[-1, 1], "r^", markersize=4, label=f"end (t={len(path) - 1})")
    if highlight_t is not None:
        hx, hy = path[highlight_t]
        ax.plot(hx, hy, "*", color="orange", markersize=8, markeredgecolor="black",
                label=f"current (t={highlight_t})", zorder=5)
    if show_labels:
        for t, (x, y) in enumerate(path):
            ax.annotate(str(t), xy=(x, y), xytext=(2, 2), textcoords="offset points", fontsize=7)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(f"Palace hairpin trajectory ({block_w}x{block_h} = {len(path)} items)")
    ax.set_aspect("equal")
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()
    return path


def demo_fig7d(
    depths=(13, 26, 52),
    n_experiments=5,
    lambdas=(4, 5, 7), Nh=400, gamma=0.6, thresh=0.5,
    Ns=3600, block_w=13, block_h=4,
    noise_frac=1.0,
    pdf_path=None,
):
    """VectorHASH_fig7de.ipynb의 fig7d 데모를 그대로 재현.

    실제 이미지 하나를 палace(hairpin 경로) 위 t번째 위치에 결합해서 학습한 뒤,
    depths(=sequence length list)별로 (1행) 깨끗한(노이즈 없는) 회상, (2행) grid
    표상에 noise_frac만큼 노이즈를 섞은 회상, (3행) 그 노이즈 낀 회상 주소에
    결합한 새 item(new item)의 회상을 True 이미지와 나란히 비교한다.
    n_experiments회 반복(매번 새 무작위 scaffold/이미지 배치)해서
    각 depth에서의 cos-sim 평균/표준편차를 콘솔에 출력한다.
    """
    block_x0, block_y0 = 0, 0
    depths = list(depths)
    img_h = img_w = int(round(np.sqrt(Ns)))  # 카드/이미지 한 장의 픽셀 shape(60x60) -- block_w/h(팰리스 격자 크기)와는 별개

    pdf_pages = None
    if pdf_path is not None:
        from matplotlib.backends.backend_pdf import PdfPages
        pdf_pages = PdfPages(pdf_path)

    similarity_results = {Nseq: {0: [], 1: [], 2: []} for Nseq in depths}

    for exp in range(n_experiments):
        t = exp + 1

        scaf_7 = build_seq_scaffold(list(lambdas), Nh, gamma=gamma, thresh=thresh, nruns=1)
        Npos = scaf_7["Npos"]
        Nstates = scaf_7["Nstates"]

        sbook_old, _, _ = make_embedded_trump_card_book(
            Ns, Nstates, Npos, block_x0, block_y0, block_w, block_h,
            seed=exp, shuffle_images=False, use_tanh_inverse=True
        )
        mbook_new, _, _ = make_embedded_image_book_for_fig7(
            Ns, Nstates, Npos, block_x0, block_y0, block_w, block_h,
            seed=exp + 100, shuffle_images=True, use_tanh_inverse=True
        )

        path_7d = make_hairpin_path(block_w, block_h, block_x0, block_y0)
        idxs_7d = path_to_indices(path_7d, Npos)

        true_sensory_vec = sbook_old[:, idxs_7d[t]].flatten()
        true_new_item_vec = mbook_new[:, idxs_7d[t]].flatten()
        norm_true_sensory = np.linalg.norm(true_sensory_vec)
        norm_true_new_item = np.linalg.norm(true_new_item_vec)

        fig, ax = plt.subplots(3, len(depths) + 1, figsize=(9.5, 7.5))

        ax[0, 0].imshow(sbook_old[:, idxs_7d[t]].reshape(img_h, img_w), cmap="gray")
        ax[0, 0].set_ylabel("Sensory recall", fontsize=12)
        ax[0, 0].set_title(f"True (t={t})")
        for spine in ax[0, 0].spines.values():
            spine.set_color("#8bc34a")
            spine.set_linewidth(3)

        ax[1, 0].set_facecolor("none")
        ax[1, 0].set_ylabel(f"Noisy recall\n(noise_frac={noise_frac})", fontsize=12)
        for spine in ax[1, 0].spines.values():
            spine.set_visible(False)

        ax[2, 0].imshow(mbook_new[:, idxs_7d[t]].reshape(img_h, img_w), cmap="gray")
        ax[2, 0].set_ylabel("New item recall", fontsize=12)
        for spine in ax[2, 0].spines.values():
            spine.set_color("#2196f3")
            spine.set_linewidth(3)

        for col_idx, Nseq in enumerate(depths):
            col = col_idx + 1
            idxs_seq = idxs_7d[:Nseq]

            P_seq = scaf_7["pbook_flat"][:, :, idxs_seq]
            S_seq = sbook_old[:, idxs_seq]
            M_new = mbook_new[:, idxs_seq]

            rng1 = np.random.default_rng(seed=1)
            rng2 = np.random.default_rng(seed=2)

            S_rec1 = recall_sequence_once(scaf_7, S_seq, P_seq, Nseq, rng1)
            S_rec2 = recall_sequence_once(scaf_7, S_seq, P_seq, Nseq, rng2, noise_frac=noise_frac)

            S_addr1 = np.sign(S_rec1[0])
            S_addr2 = np.sign(S_rec2[0])
            Wms = M_new @ np.linalg.pinv(S_addr1)
            M_recall = Wms @ S_addr2

            rec1_vec = S_rec1[0, :, t].flatten()
            rec2_vec = S_rec2[0, :, t].flatten()
            m_rec_vec = M_recall[:, t].flatten()

            sim_row0 = np.dot(rec1_vec, true_sensory_vec) / (np.linalg.norm(rec1_vec) * norm_true_sensory + 1e-12)
            sim_row1 = np.dot(rec2_vec, true_sensory_vec) / (np.linalg.norm(rec2_vec) * norm_true_sensory + 1e-12)
            sim_row2 = np.dot(m_rec_vec, true_new_item_vec) / (np.linalg.norm(m_rec_vec) * norm_true_new_item + 1e-12)

            similarity_results[Nseq][0].append(sim_row0)
            similarity_results[Nseq][1].append(sim_row1)
            similarity_results[Nseq][2].append(sim_row2)

            ax[0, col].imshow(S_rec1[0, :, t].reshape(img_h, img_w), cmap="gray")
            ax[0, col].set_xlabel(f"Cos: {sim_row0:.4f}", fontsize=9.5, color="green")
            for spine in ax[0, col].spines.values():
                spine.set_color("#8bc34a")
                spine.set_linewidth(3)

            if col == 1:
                ax[0, col].set_title(f"Sequence Length\n{Nseq:,}", fontsize=10, pad=10)
            else:
                ax[0, col].set_title(f"\n{Nseq:,}", fontsize=10, pad=10)

            ax[1, col].imshow(S_rec2[0, :, t].reshape(img_h, img_w), cmap="gray")
            ax[1, col].set_xlabel(f"Cos: {sim_row1:.4f}", fontsize=9.5, color="green")
            for spine in ax[1, col].spines.values():
                spine.set_color("#8bc34a")
                spine.set_linewidth(3)

            ax[2, col].imshow(M_recall[:, t].reshape(img_h, img_w), cmap="gray")
            ax[2, col].set_xlabel(f"Cos: {sim_row2:.4f}", fontsize=9.5, color="blue")
            for spine in ax[2, col].spines.values():
                spine.set_color("#2196f3")
                spine.set_linewidth(3)

        for r in range(3):
            for c in range(len(depths) + 1):
                ax[r, c].set_xticks([])
                ax[r, c].set_yticks([])

        plt.tight_layout()
        if pdf_pages is not None:
            pdf_pages.savefig(fig, bbox_inches="tight", dpi=150)
        plt.show()
        print(f"  page {t}/{n_experiments} done")

    if pdf_pages is not None:
        pdf_pages.close()

    print("\n=== fig7d summary ===")
    for Nseq in depths:
        print(f"Sequence Length: {Nseq:,}")
        for row, name in zip([0, 1, 2], ["Sensory recall (clean)", "Sensory recall (noisy)", "New item recall"]):
            arr = np.array(similarity_results[Nseq][row])
            print(f"  {name}: mean={arr.mean():.4f} +- std={arr.std():.4f}  (per-run: "
                  f"{', '.join(f'{v:.4f}' for v in arr)})")
    return similarity_results


def demo_fig7d_interactive(
    lambdas=(4, 5, 7), Nh=400, gamma=0.6, thresh=0.5,
    Ns=3600, block_w=13, block_h=4, seed=0,
):
    """demo_fig7d를 depth/t 슬라이더로 직접 조작하며 보는 버전.
    맨 위에 palace trajectory(현재 t 위치 강조), 아래에 3개 패널:
    True | 깨끗한 회상 | new item 회상."""
    try:
        import ipywidgets as widgets
    except ImportError:
        print("ipywidgets 없음. pip install ipywidgets 필요.")
        return

    n_cards = block_w * block_h
    block_x0, block_y0 = 0, 0
    img_h = img_w = int(round(np.sqrt(Ns)))

    scaf = build_seq_scaffold(list(lambdas), Nh, gamma=gamma, thresh=thresh, nruns=1)
    Npos, Nstates = scaf["Npos"], scaf["Nstates"]

    sbook_old, _, _ = make_embedded_trump_card_book(
        Ns, Nstates, Npos, block_x0, block_y0, block_w, block_h,
        seed=seed, shuffle_images=False, use_tanh_inverse=True
    )
    mbook_new, _, _ = make_embedded_image_book_for_fig7(
        Ns, Nstates, Npos, block_x0, block_y0, block_w, block_h,
        seed=seed + 100, shuffle_images=True, use_tanh_inverse=True
    )

    path = make_hairpin_path(block_w, block_h, block_x0, block_y0)
    idxs = path_to_indices(path, Npos)

    def update_plot(depth, t):
        if t >= depth:
            print(f"t({t})는 depth({depth})보다 작아야 함. t를 {depth - 1} 이하로 내리세요.")
            return

        plot_palace_path(block_w, block_h, block_x0, block_y0, show_labels=False, highlight_t=t)

        idxs_seq = idxs[:depth]
        P_seq = scaf["pbook_flat"][:, :, idxs_seq]
        S_seq = sbook_old[:, idxs_seq]
        M_new = mbook_new[:, idxs_seq]

        true_s = sbook_old[:, idxs[t]]
        true_m = mbook_new[:, idxs[t]]

        S_clean = recall_sequence_once(scaf, S_seq, P_seq, depth, np.random.default_rng(1))

        S_addr1 = np.sign(S_clean[0])
        Wms = M_new @ np.linalg.pinv(S_addr1)
        M_rec_clean = Wms @ S_addr1

        def cs(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

        panels = [
            (true_s, "True", None),
            (S_clean[0, :, t], "Clean recall", cs(S_clean[0, :, t], true_s)),
            (M_rec_clean[:, t], "New item recall (clean addr)", cs(M_rec_clean[:, t], true_m)),
        ]

        fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
        for ax, (vec, title, sim) in zip(axes, panels):
            ax.imshow(vec.reshape(img_h, img_w), cmap="gray")
            ax.set_title(title if sim is None else f"{title}\ncos={sim:.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        plt.show()

    depth_slider = widgets.IntSlider(value=min(13, n_cards), min=1, max=n_cards, step=1,
                                      description="depth:", continuous_update=False)
    t_slider = widgets.IntSlider(value=0, min=0, max=n_cards - 1, step=1,
                                  description="t:", continuous_update=False)
    widgets.interact(update_plot, depth=depth_slider, t=t_slider)


# =========================================================================
# [추가] Grid(모듈별) / HPC 상태를 시퀀스 재생 중 각각 보여주는 인터랙티브 뷰
# =========================================================================
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'

def get_module_slices_from_scaf(scaf):
    """scaf['lambdas'] 기준 모듈별 (start, end, lam) 슬라이스.
    scaf['Ng'] == sum(lam*lam)이고 module_gbooks가 lam^2 크기 one-hot이므로
    이 reshape는 build_seq_scaffold 구조상 검증된 방식."""
    slices = []
    start = 0
    for lam in scaf["lambdas"]:
        size = lam * lam
        slices.append((start, start + size, lam))
        start += size
    assert start == scaf["Ng"], f"슬라이스 합({start}) != Ng({scaf['Ng']})"
    return slices


def smooth_population_vector(vec, sigma=10, mult=2):
    """HPC population vector(Nh,)를 정사각형 reshape 후 업샘플+스무딩"""
    vec = np.array(vec).flatten()
    n = len(vec)
    side = int(np.ceil(np.sqrt(n)))
    padded = np.zeros(side * side)
    padded[:n] = vec
    reshaped = padded.reshape(side, side)
    upsampled = upsample(reshaped, mult)
    return explicit_interpolation(upsampled, sigma=sigma)


def smooth_grid_chunk(chunk, sigma=2, mult=2):
    """grid 모듈 하나(lam x lam)를 업샘플+스무딩"""
    upsampled = upsample(chunk, mult)
    return explicit_interpolation(upsampled, sigma=sigma)


def run_recall_and_collect_states(lambdas=(3, 4, 5), Nh=400, Ns=3600,
                                   depth=200, seed=0):
    """path 위 depth개 위치를 학습 -> 무노이즈 회상하며
    매 위치의 grid state(G_rec) / HPC state(P_seq)를 리스트로 저장.
    + place field 시각화를 위해 path의 (x,y) 좌표도 함께 반환."""
    rng = np.random.default_rng(seed)
    scaf = build_seq_scaffold(list(lambdas), Nh)
    Npos = scaf["Npos"]

    path = make_hairpin_path(Npos, Npos)          # [(x, y), ...] 좌표 리스트
    path_xy = path[:depth]                         # place field 매핑용
    idxs_seq = path_to_indices(path, Npos)[:depth]
    P_seq = scaf["pbook_flat"][:, :, idxs_seq]          # (1, Nh, depth)
    S_field = np.sign(rng.standard_normal((Ns, scaf["Nstates"])))
    S_seq = S_field[:, idxs_seq]

    Wps = pseudotrain_Wps(P_seq, S_seq, depth)
    pin = nonlin(Wps @ S_seq, thresh=0)
    gin = scaf["Wgp"] @ pin

    G_rec = np.zeros((1, scaf["Ng"], depth))
    for k in range(depth):
        G_rec[:, :, k] = module_wise_NN_2d(
            gin[:, :, k, None], scaf["module_gbooks"], scaf["module_sizes"]
        )[0, :, 0]

    g_states = [G_rec[0, :, k] for k in range(depth)]
    h_states = [P_seq[0, :, k] for k in range(depth)]
    return scaf, g_states, h_states, path_xy


def build_place_field_map(neuron_id, h_states, path_xy, Npos):
    """뉴런 하나(neuron_id)의 place field: room(Npos x Npos) 좌표에
    path 위 각 위치에서의 발화율을 채워넣음. 방문 안 한 곳은 nan."""
    field = np.full((Npos, Npos), np.nan)
    for (x, y), h in zip(path_xy, h_states):
        field[y, x] = h[neuron_id]   # 발화율은 시점(step)에서의 population vector 중 해당 뉴런 값
    return field


def build_grid_place_field_map(grid_neuron_id, g_states, path_xy, Npos):
    """grid population 인덱스 하나(grid_neuron_id, 0..Ng-1)의 place field:
    HPC의 build_place_field_map과 완전히 동일한 방식(뉴런 하나 고정,
    실제 방 좌표 (x, y) 전체를 스캔)으로 만든다.

    HPC 뉴런은 방 안에서 단 한 곳(single bump)에서만 발화하는 게 정상인
    반면, grid 뉴런은 자기 모듈의 period(lam)만큼 간격을 두고 방 전체에
    걸쳐 '주기적으로 반복되는 여러 개의 bump'가 나와야 정상이다
    (x mod lam, y mod lam이 같은 모든 (x, y)에서 값이 반복되므로).
    방문 안 한 곳은 nan.
    """
    field = np.full((Npos, Npos), np.nan)
    for (x, y), g in zip(path_xy, g_states):
        field[y, x] = g[grid_neuron_id]
    return field


def _grid_neuron_module_info(grid_neuron_id, module_slices):
    """grid_neuron_id가 속한 모듈 인덱스와 그 모듈의 period(lam)를 반환.
    place field가 몇 칸 간격으로 반복돼야 정상인지 슬라이더 옆에 표시하는 용도."""
    for m_idx, (start, end, lam) in enumerate(module_slices):
        if start <= grid_neuron_id < end:
            return m_idx, lam
    raise ValueError(f"grid_neuron_id {grid_neuron_id} is out of range for module_slices")


# def show_recall_states_interactive(scaf, g_states, h_states, path_xy):
#     try:
#         import ipywidgets as widgets
#         from IPython.display import display
#     except ImportError:
#         return

#     module_slices = get_module_slices_from_scaf(scaf)
#     n_modules = len(module_slices)
#     n_cols = max(2, n_modules)
#     Npos = scaf["Npos"]
#     Nh = h_states[0].shape[0]
#     Ng = scaf["Ng"]

#     # [수정] 뉴런/스텝을 바꿔도 색 스케일이 흔들리지 않도록 전역 범위를 한 번만 계산
#     h_stack = np.stack(h_states)          # (depth, Nh)
#     g_stack = np.stack(g_states)          # (depth, Ng)
#     h_vmin, h_vmax = float(h_stack.min()), float(h_stack.max())
#     g_vmin, g_vmax = float(g_stack.min()), float(g_stack.max())

#     # [수정] 안 가본 곳(nan)을 활성값과 명확히 구분되는 색으로 표시
#     cmap = plt.get_cmap("jet").copy()
#     cmap.set_bad(color="lightgray")

#     def update_plot(step, neuron_id, grid_neuron_id):
#         g = g_states[step]
#         g_mod_idx, g_lam = _grid_neuron_module_info(grid_neuron_id, module_slices)

#         fig = plt.figure(figsize=(4 * n_cols, 12))
#         fig.suptitle(
#             f"HPC: Neuron {neuron_id}/{Nh-1}   |   "
#             f"Grid: Neuron {grid_neuron_id}/{Ng-1} (module {g_mod_idx}, period={g_lam})   |   "
#             f"Snapshot step {step}/{len(g_states)-1}",
#             fontsize=13
#         )
#         gs = fig.add_gridspec(3, n_cols)

#         ax_h = fig.add_subplot(gs[0, :])
#         h_field = build_place_field_map(neuron_id, h_states, path_xy, Npos)
#         im_h = ax_h.imshow(h_field, cmap=cmap, vmin=h_vmin, vmax=h_vmax)  # [수정] 고정 스케일
#         ax_h.set_title(f"HPC Place Field (neuron {neuron_id})")
#         ax_h.axis('off')
#         fig.colorbar(im_h, ax=ax_h, fraction=0.03)

#         ax_g_field = fig.add_subplot(gs[1, :])
#         g_field = build_grid_place_field_map(grid_neuron_id, g_states, path_xy, Npos)
#         im_g = ax_g_field.imshow(g_field, cmap=cmap, vmin=g_vmin, vmax=g_vmax)  # [수정] 고정 스케일
#         ax_g_field.set_title(f"Grid Place Field (neuron {grid_neuron_id})")
#         ax_g_field.axis('off')
#         fig.colorbar(im_g, ax=ax_g_field, fraction=0.03)

#         plt.tight_layout()
#         plt.show()

#     step_slider = widgets.IntSlider(value=0, min=0, max=len(g_states) - 1, description="Step:")
#     neuron_slider = widgets.IntSlider(value=0, min=0, max=Nh - 1, description="HPC neuron:")
#     grid_neuron_slider = widgets.IntSlider(value=0, min=0, max=Ng - 1, description="Grid neuron:")
#     widgets.interact(
#         update_plot,
#         step=step_slider,
#         neuron_id=neuron_slider,
#         grid_neuron_id=grid_neuron_slider,
#     )

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


# =========================================================================
# 랜덤 워크 경로 생성 (원본 그대로)
# =========================================================================
def make_random_path(Npos, n_steps, seed=0, start=None):
    """Npos x Npos 토러스(주기적 grid state space) 위를 랜덤 워크하는 경로.
    한 스텝에 상하좌우 중 하나로 1칸 이동하며, 경계는 module_periods와
    동일하게 wrap-around(모듈로) 처리한다. 길이는 정확히 n_steps."""
    rng = np.random.default_rng(seed)
    if start is None:
        x, y = int(rng.integers(0, Npos)), int(rng.integers(0, Npos))
    else:
        x, y = start

    path = [(x, y)]
    moves = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    for _ in range(n_steps - 1):
        dx, dy = moves[rng.integers(0, 4)]
        x, y = (x + dx) % Npos, (y + dy) % Npos
        path.append((x, y))
    return np.array(path, dtype=int)


# =========================================================================
# 랜덤 경로 기반 recall + state 수집 (원본 그대로)
# =========================================================================
def run_recall_and_collect_states(lambdas=(3, 4, 5), Nh=400, Ns=3600,
                                   depth=200, seed=0):
    """랜덤 경로 위 depth개 위치를 학습 -> 무노이즈 회상하며
    매 위치의 grid state(G_rec) / HPC state(P_seq)를 리스트로 저장.
    + place field 시각화를 위해 path의 (x,y) 좌표도 함께 반환."""
    rng = np.random.default_rng(seed)
    scaf = build_seq_scaffold(list(lambdas), Nh)
    Npos = scaf["Npos"]

    path_xy = make_random_path(Npos, depth, seed=seed)
    idxs_seq = path_to_indices(path_xy, Npos)

    P_seq = scaf["pbook_flat"][:, :, idxs_seq]          # (1, Nh, depth)
    S_field = np.sign(rng.standard_normal((Ns, scaf["Nstates"])))
    S_seq = S_field[:, idxs_seq]

    Wps = pseudotrain_Wps(P_seq, S_seq, depth)
    pin = nonlin(Wps @ S_seq, thresh=2.0)
    gin = scaf["Wgp"] @ pin

    G_rec = np.zeros((1, scaf["Ng"], depth))
    for k in range(depth):
        G_rec[:, :, k] = module_wise_NN_2d(
            gin[:, :, k, None], scaf["module_gbooks"], scaf["module_sizes"]
        )[0, :, 0]

    g_states = [G_rec[0, :, k] for k in range(depth)]
    h_states = [P_seq[0, :, k] for k in range(depth)]
    return scaf, g_states, h_states, path_xy


# =========================================================================
# [핵심 추가] cos 3방향 평면파 합성 -> 육각 격자(hexagonal grid) 패턴
# =========================================================================
def hex_grid_field(lam, Npos, phase=(0.0, 0.0)):
    """
    grid cell의 이상적인 공간 발화 패턴을 만듭니다.
    60도씩 떨어진 3개 방향의 평면파(cos)를 합성하면 정삼각형(hexagonal)
    격자 무늬가 나옵니다 (실제 grid cell의 삼각격자 발화 패턴과 동일한 원리).

    lam   : 이 grid module의 공간 주기(period)
    Npos  : 상태공간 한 변의 길이 (Npos x Npos 배경을 만듦)
    phase : (px, py) 위상 오프셋 -> 뉴런마다 다르게 주면 서로 다른 grid cell처럼 보임
    """
    xs = np.arange(Npos)
    ys = np.arange(Npos)
    X, Y = np.meshgrid(xs, ys)

    angles = [0, np.pi / 3, 2 * np.pi / 3]  # 60도 간격 3방향
    field = np.zeros_like(X, dtype=float)
    for a in angles:
        kx = (2 * np.pi / lam) * np.cos(a)
        ky = (2 * np.pi / lam) * np.sin(a)
        field += np.cos(kx * (X - phase[0]) + ky * (Y - phase[1]))

    return field  # 대략 [-3, 3] 범위, 육각 격자 간섭무늬


def neuron_phase(neuron_id, lam, seed=0):
    """뉴런마다 다른 위상을 결정론적으로 생성 (같은 neuron_id면 항상 같은 phase)."""
    rng = np.random.default_rng(seed + int(neuron_id))
    return rng.uniform(0, lam, size=2)


# =========================================================================
# 시각화 (Panel 2: 실제 gbook_flat 대신 hex_grid_field 사용)
# =========================================================================
def show_recall_states_interactive(scaf, g_states, h_states, path_xy):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets 라이브러리가 필요합니다. (노트북 환경 전용)")
        return

    print("전체 상태공간 배경 맵 생성 중 (grid panel: 육각 격자 cos 합성)...")

    Npos = scaf["Npos"]
    Nh = h_states[0].shape[0]
    Ng = scaf["Ng"]
    module_slices = get_module_slices_from_scaf(scaf)

    pbook_flat = scaf["pbook_flat"][0]       # (Nh, Nstates)

    def full_state_field(flat_row):
        return flat_row.reshape(Npos, Npos).T   # field[y, x]

    path_arr = np.array(path_xy, dtype=float)

    def update_plot(step, neuron_id, grid_neuron_id):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        g_mod_idx, g_lam = _grid_neuron_module_info(grid_neuron_id, module_slices)

        curr_path = path_arr[:step + 1]
        curr_h_fire = np.array([h[neuron_id] for h in h_states[:step + 1]])
        curr_g_fire = np.array([g[grid_neuron_id] for g in g_states[:step + 1]])

        # --- [Panel 1] HPC 뉴런: 기존과 동일 (전체 상태공간 배경 + 가우스 스무딩) ---
        h_heat_raw = full_state_field(pbook_flat[neuron_id])
        h_heat = gaussian_filter(h_heat_raw, sigma=1.5, mode='constant')

        axes[1].imshow(h_heat, cmap='jet', origin='lower',
                       extent=[0, Npos, 0, Npos],
                       interpolation='bicubic', alpha=1.0)
        
        # 경로 숨기기 (주석 처리)
        # axes[0].plot(path_arr[:, 0] + 0.5, path_arr[:, 1] + 0.5,
        #              '-', color="white", alpha=0.4, linewidth=1.5)
        
        # sc1 = axes[0].scatter(curr_path[:, 0] + 0.5, curr_path[:, 1] + 0.5,
                            #   c=curr_h_fire, cmap='Reds', s=80,
                            #   edgecolors="black", linewidths=0.5, zorder=3)
        # if len(curr_path) > 0:
        #     axes[0].scatter(curr_path[-1, 0] + 0.5, curr_path[-1, 1] + 0.5,
        #                     c='blue', s=250, marker='*', edgecolors="white", zorder=4)
        axes[1].set_title(f"HPC Place Cell #{neuron_id} (Step {step})", fontsize=14)
        axes[1].set_xlim(0, Npos); axes[0].set_ylim(0, Npos)
        axes[1].set_aspect('equal')
        # fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="Firing")

        # --- [Panel 2] Grid 뉴런: gbook_flat 대신 cos 3방향 합성 육각 격자 배경 ---
        phase = neuron_phase(grid_neuron_id, g_lam)
        g_heat_raw = hex_grid_field(g_lam, Npos, phase=phase)
        g_heat = gaussian_filter(g_heat_raw, sigma=0.5, mode='wrap')  # 육각 무늬가 뭉개지지 않도록 약하게만

        axes[0].imshow(g_heat, cmap='jet', origin='lower',
                       extent=[0, Npos, 0, Npos],
                       interpolation='bicubic', alpha=1.0)
        
        # 경로 숨기기 (주석 처리)
        # axes[1].plot(path_arr[:, 0] + 0.5, path_arr[:, 1] + 0.5,
        #              '-', color="white", alpha=0.4, linewidth=1.5)
        
        # sc2 = axes[1].scatter(curr_path[:, 0] + 0.5, curr_path[:, 1] + 0.5,
        #                       c=curr_g_fire, cmap='Reds', s=80,
        #                       edgecolors="black", linewidths=0.5, zorder=3)
        # if len(curr_path) > 0:
        #     axes[1].scatter(curr_path[-1, 0] + 0.5, curr_path[-1, 1] + 0.5,
        #                     c='blue', s=250, marker='*', edgecolors="white", zorder=4)
        axes[0].set_title(
            f"Grid Cell #{grid_neuron_id} (Mod {g_mod_idx}, Period {g_lam}, hex field) (Step {step})",
            fontsize=14
        )
        axes[0].set_xlim(0, Npos); axes[1].set_ylim(0, Npos)
        axes[0].set_aspect('equal')
        # fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label="Firing")

        plt.tight_layout()
        plt.show()

    step_slider = widgets.IntSlider(value=0, min=0, max=len(g_states) - 1, description="Time Step:")
    neuron_slider = widgets.IntSlider(value=0, min=0, max=Nh - 1, description="HPC neuron:")
    grid_neuron_slider = widgets.IntSlider(value=0, min=0, max=Ng - 1, description="Grid neuron:")

    widgets.interact(
        update_plot,
        step=step_slider,
        neuron_id=neuron_slider,
        grid_neuron_id=grid_neuron_slider,
    )


import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

def show_recall_states_interactive(scaf, g_states, h_states, path_xy):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets 라이브러리가 필요합니다. (노트북 환경 전용)")
        return

    print("실제 회상(recall) 상태 기반 place field 맵 생성 중...")

    Npos = scaf["Npos"]
    Nh = h_states[0].shape[0]
    Ng = scaf["Ng"]
    module_slices = get_module_slices_from_scaf(scaf)

    path_arr = np.array(path_xy, dtype=float)

    # 안 가본 칸을 회색으로 표시
    cmap = plt.get_cmap("jet").copy()
    cmap.set_bad(color="lightgray")

    def update_plot(step, neuron_id, grid_neuron_id):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))

        g_mod_idx, g_lam = _grid_neuron_module_info(grid_neuron_id, module_slices)

        curr_path = path_arr[:step + 1]
        curr_h_fire = np.array([h[neuron_id] for h in h_states[:step + 1]])
        curr_g_fire = np.array([g[grid_neuron_id] for g in g_states[:step + 1]])

        # --- [Panel 1] HPC: 실제 recall된 h_states 기반 place field (sparse, nan=안 가본 곳) ---
        h_field = build_place_field_map(neuron_id, h_states, path_xy, Npos)
        h_heat = gaussian_filter(np.nan_to_num(h_field), sigma=1.5, mode='constant')
        h_heat[np.isnan(h_field)] = np.nan   # 스무딩 후에도 안 가본 곳은 다시 nan으로

        axes[0].imshow(h_heat, cmap=cmap, origin='lower',
                       extent=[0, Npos, 0, Npos],
                       interpolation='bicubic', alpha=1.0)
        axes[0].plot(path_arr[:, 0] + 0.5, path_arr[:, 1] + 0.5,
                     '-', color="white", alpha=0.4, linewidth=1.5)
        axes[0].scatter(curr_path[:, 0] + 0.5, curr_path[:, 1] + 0.5,
                         c=curr_h_fire, cmap='Reds', s=80,
                         edgecolors="black", linewidths=0.5, zorder=3)
        if len(curr_path) > 0:
            axes[0].scatter(curr_path[-1, 0] + 0.5, curr_path[-1, 1] + 0.5,
                             c='blue', s=250, marker='*', edgecolors="white", zorder=4)
        axes[0].set_title(f"HPC Place Cell #{neuron_id} (Step {step})", fontsize=14)
        axes[0].set_xlim(0, Npos); axes[0].set_ylim(0, Npos)
        axes[0].set_aspect('equal')

        # --- [Panel 2] Grid: 실제 recall된 g_states 기반 place field (sparse, nan=안 가본 곳) ---
        g_field = build_grid_place_field_map(grid_neuron_id, g_states, path_xy, Npos)
        g_heat = gaussian_filter(np.nan_to_num(g_field), sigma=0.5, mode='wrap')
        g_heat[np.isnan(g_field)] = np.nan

        axes[1].imshow(g_heat, cmap=cmap, origin='lower',
                       extent=[0, Npos, 0, Npos],
                       interpolation='bicubic', alpha=1.0)
        axes[1].plot(path_arr[:, 0] + 0.5, path_arr[:, 1] + 0.5,
                     '-', color="white", alpha=0.4, linewidth=1.5)
        axes[1].scatter(curr_path[:, 0] + 0.5, curr_path[:, 1] + 0.5,
                         c=curr_g_fire, cmap='Reds', s=80,
                         edgecolors="black", linewidths=0.5, zorder=3)
        if len(curr_path) > 0:
            axes[1].scatter(curr_path[-1, 0] + 0.5, curr_path[-1, 1] + 0.5,
                             c='blue', s=250, marker='*', edgecolors="white", zorder=4)
        axes[1].set_title(
            f"Grid Cell #{grid_neuron_id} (Mod {g_mod_idx}, Period {g_lam}) (Step {step})",
            fontsize=14
        )
        axes[1].set_xlim(0, Npos); axes[1].set_ylim(0, Npos)
        axes[1].set_aspect('equal')

        plt.tight_layout()
        plt.show()

    step_slider = widgets.IntSlider(value=0, min=0, max=len(g_states) - 1, description="Time Step:")
    neuron_slider = widgets.IntSlider(value=0, min=0, max=Nh - 1, description="HPC neuron:")
    grid_neuron_slider = widgets.IntSlider(value=0, min=0, max=Ng - 1, description="Grid neuron:")

    widgets.interact(
        update_plot,
        step=step_slider,
        neuron_id=neuron_slider,
        grid_neuron_id=grid_neuron_slider,
    )
# =========================================================================
# 실행 예시
# =========================================================================
# scaf, g_states, h_states, path_xy = run_recall_and_collect_states(
#     lambdas=(3, 4, 5), Nh=400, Ns=3600, depth=200, seed=0
# )
# show_recall_states_interactive(scaf, g_states, h_states, path_xy)

def run():
    run_experiments_1_to_5()
    run_experiment_6()
    run_experiment_7()   

if __name__ == "__main__":
    run()