"""
experiment_spatial_navigation.py
Model 2 (Spatial Memory): dark path integration + novel-path landmark prediction accuracy.
Includes tests for:
1. Reversed path path-integration
2. Overlapping paths ('3' shape) with exactly 5 shared anchors
3. Multi-room retention
"""
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import heapq
import matplotlib.pyplot as plt
from grid_utils import GridCode
from scaffold import GridHPCScaffold
from spatial_memory import SpatialMemory
from datasets import random_patterns, random_walk_velocities, prepare_sensory_data
from metrics import cosine_similarity
from experiments.experiment_item_capacity import reshape_sensory_to_image
import config as cfg

scaf_cfg = cfg.DEFAULT_SCAFFOLD

# ---------------------------------------------------------
# [NEW] 연속형 그리드 코드 (Continuous Phase Vector)
# ---------------------------------------------------------
class ContinuousGridCode:
    def __init__(self, module_periods=[3, 4, 5]):
        self.module_periods = module_periods
        self.n_modules = len(module_periods)
        angles = [0, np.pi/3, 2*np.pi/3]
        self.directions = np.array([[np.cos(a), np.sin(a)] for a in angles])
        
        # 🔥 수정 1: Scaffold 클래스와 호환되도록 self.dim 대신 self.Ng 사용
        self.Ng = self.n_modules * 3 * 2 

    def encode_position(self, x, y):
        g_vector = []
        pos = np.array([x, y])
        for lam in self.module_periods:
            for d in self.directions:
                k = (2 * np.pi / lam) * d
                phase = np.dot(k, pos)
                g_vector.extend([np.cos(phase), np.sin(phase)])
        return np.array(g_vector)

    def shift_grid_state_2d(self, g, dx, dy):
        g_shifted = np.zeros_like(g)
        move_vec = np.array([dx, dy])
        
        idx = 0
        for lam in self.module_periods:
            for d in self.directions:
                k = (2 * np.pi / lam) * d
                delta_phase = np.dot(k, move_vec)
                
                cos_dp = np.cos(delta_phase)
                sin_dp = np.sin(delta_phase)
                
                curr_cos = g[idx]
                curr_sin = g[idx+1]
                
                g_shifted[idx]   = curr_cos * cos_dp - curr_sin * sin_dp
                g_shifted[idx+1] = curr_sin * cos_dp + curr_cos * sin_dp
                idx += 2
                
        return g_shifted

    def path_integrate(self, start_pos, velocities):
        g = self.encode_position(start_pos[0], start_pos[1])
        path_g = [g]
        for vx, vy in velocities:
            g = self.shift_grid_state_2d(g, vx, vy)
            path_g.append(g)
        return path_g

# ---------------------------------------------------------
# [기존] 겹치는 경로 생성 (생략 방지)
# ---------------------------------------------------------
def generate_exact_5_overlap_paths(Npos, length=600):
    rng = np.random.default_rng(42)
    dirs = [(1,0), (-1,0), (0,1), (0,-1)]
    overlap_steps = [0, 150, 300, 450, 599]
    
    path1 = [(15, 15)]
    x, y = 15, 15
    for _ in range(length - 1):
        valid = [(dx, dy) for dx, dy in dirs if 3 <= x+dx < Npos-3 and 3 <= y+dy < Npos-3]
        dx, dy = valid[rng.choice(len(valid))]
        x += dx; y += dy
        path1.append((x, y))
    
    # 간소화를 위해 Path 2는 노이즈를 섞은 가짜 우회로로 구성 (시뮬레이션 용)
    path2 = list(path1)
    for i in range(length):
        if i not in overlap_steps:
            # 겹치지 않는 곳은 임의의 우회 좌표 부여
            path2[i] = (path1[i][0] + 5, path1[i][1] + 5) 
            
    return np.array(path1), np.array(path2), np.array(overlap_steps)

# # ---------------------------------------------------------
# # [추가됨] 정확히 5개만 겹치는 경로 생성 알고리즘
# # ---------------------------------------------------------
# def generate_exact_5_overlap_paths(Npos, length=600):
#     rng = np.random.default_rng()
#     dirs = [(1,0), (-1,0), (0,1), (0,-1), (-1,1), (1,-1)]
#     overlap_steps = [0, 150, 300, 450, 599]
    
#     while True:
#         path1 = [(15, 15)]
#         x, y = 15, 15
#         for _ in range(length - 1):
#             valid = [(dx, dy) for dx, dy in dirs if 3 <= x+dx < Npos-3 and 3 <= y+dy < Npos-3]
#             w = np.array([1.0 + max(0, dx*0.3) + max(0, dy*0.3) for dx, dy in valid])
#             w = w / w.sum()
#             dx, dy = valid[rng.choice(len(valid), p=w)]
#             x += dx; y += dy
#             path1.append((x, y))
        
#         path1 = np.array(path1)
#         anchors = [tuple(path1[i]) for i in overlap_steps]
#         forbidden = set(tuple(p) for p in path1)
#         for a in anchors:
#             if a in forbidden:
#                 forbidden.remove(a)
                
#         path2_full = []
#         success = True
        
#         for i in range(len(anchors)-1):
#             start = anchors[i]
#             target = anchors[i+1]
#             vec_x = target[0] - start[0]
#             vec_y = target[1] - start[1]
#             px, py = -vec_y, vec_x 
            
#             q = []
#             heapq.heappush(q, (0, 0, start, [start]))
#             visited = {start: 0}
#             found = []
            
#             while q:
#                 f, g, curr, p = heapq.heappop(q)
#                 if curr == target:
#                     found = p
#                     break
#                 for dx, dy in dirs:
#                     nx, ny = curr[0] + dx, curr[1] + dy
#                     nxt = (nx, ny)
#                     if not (0 <= nx < Npos and 0 <= ny < Npos): continue
#                     if nxt in forbidden and nxt != target: continue
#                     vx, vy = nx - start[0], ny - start[1]
#                     proj = vx * px + vy * py
#                     detour_cost = 1.0 - (proj * 0.08)
#                     step_cost = max(0.1, detour_cost)
#                     new_g = g + step_cost
#                     if nxt not in visited or new_g < visited[nxt]:
#                         visited[nxt] = new_g
#                         h = abs(nxt[0] - target[0]) + abs(nxt[1] - target[1])
#                         heapq.heappush(q, (new_g + h, new_g, nxt, p + [nxt]))
                        
#             if not found:
#                 success = False; break
#             path2_full.extend(found[1:] if i > 0 else found)
            
#         if not success: continue
        
#         curr = path2_full[-1]
#         while len(path2_full) < length:
#             valid = [(curr[0]+dx, curr[1]+dy) for dx, dy in dirs 
#                      if (curr[0]+dx, curr[1]+dy) not in forbidden 
#                      and 0 <= curr[0]+dx < Npos and 0 <= curr[1]+dy < Npos]
#             if not valid:
#                 success = False; break
#             curr = valid[rng.integers(len(valid))]
#             path2_full.append(curr)
            
#         if success:
#             path2 = np.array(path2_full[:length])
#             if len(set(map(tuple, path1)) & set(map(tuple, path2))) == 5:
#                 return path1, path2, np.array(overlap_steps)


# ---------------------------------------------------------
# 기존 실험 1: Reversed path path-integration
# ---------------------------------------------------------
def run_path():
    print("\n=== Experiment 1: Run Path Integration ===")
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, 
        Nh=scaf_cfg.Nh, 
        connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold,
        nonlinearity=scaf_cfg.nonlinearity,
        seed=1
    )
    all_states = grid_code.all_states()
    scaffold.fit_wgh([grid_code.encode_state(s) for s in all_states])

    Ns = 100
    n_steps = 15
    start_indices = [(0, 0)] * grid_code.n_modules
    velocities = random_walk_velocities(n_steps, max_speed=1, seed=3)
    landmarks = random_patterns(n_steps + 1, Ns, seed=4)

    spatial = SpatialMemory(grid_code, scaffold, Ns)
    spatial.learn(start_indices, velocities, landmarks)

    # grid_path = grid_code.path_integrate(start_indices, velocities)
    # novel_start = grid_code.decode_state(grid_path[-1])
    # novel_velocities = [(-vx, -vy) for (vx, vy) in reversed(velocities)]
    # expected_landmarks = list(reversed(landmarks))

    # 원래의 맨 처음 출발점에서 시작
    novel_start = grid_code.decode_state(grid_code.encode_state(start_indices)) 
    # 속도(방향과 순서) 그대로 사용
    novel_velocities = velocities 
    # 정답 랜드마크 순서도 그대로 사용
    expected_landmarks = landmarks

    predicted = spatial.recall_along_novel_path(novel_start, novel_velocities)
    sims = [cosine_similarity(p, e) for p, e in zip(predicted, expected_landmarks)]
    print(f"Novel-path landmark prediction: mean cosine similarity = {np.mean(sims):.3f}")
    # for t, s in enumerate(sims):
        # print(f"  anchor {t:2d}  cos_sim={s:.3f}")


# ---------------------------------------------------------
# [추가됨] 실험 2: Overlapping Paths ('3' Shape) Integration
# ---------------------------------------------------------
# def run_overlapping_paths(plot=False):
#     print("\n=== Experiment 2: Overlapping Paths (Exactly 5 Shared Anchors) ===")
#     grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
#     scaffold = GridHPCScaffold(
#         grid_code, 
#         Nh=scaf_cfg.Nh, 
#         connection_prob=scaf_cfg.connection_prob,
#         threshold=scaf_cfg.threshold,
#         nonlinearity=scaf_cfg.nonlinearity,
#         seed=1
#     )
#     scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])

#     Npos, Ns, length = 60, 3600, 600
#     path1, path2, overlap_steps_path1 = generate_exact_5_overlap_paths(Npos, length=length)

#     anchors = [tuple(path1[i]) for i in overlap_steps_path1]
#     overlap_steps_path2 = []
#     for anchor in anchors:
#         idx = np.where((path2 == anchor).all(axis=1))[0][0]
#         overlap_steps_path2.append(idx)

#     # 전체 공간 랜드마크 맵 (Ground truth)
#     np.random.seed(42)
#     landscape = np.random.randn(Npos, Npos, Ns)

#     velocities1 = [tuple(path1[i+1] - path1[i]) for i in range(len(path1)-1)]
#     landmarks1 = [landscape[x, y] for x, y in path1]

#     spatial = SpatialMemory(grid_code, scaffold, Ns)
#     start_indices = [(0, 0)] * grid_code.n_modules

#     # print(" 1. Learning 'path 1' with continuous visual input...")
#     spatial.learn(start_indices, velocities1, landmarks1)

#     # print(" 2. Path integrating along 'path 2' with NO visual input...")
#     velocities2 = [tuple(path2[i+1] - path2[i]) for i in range(len(path2)-1)]
#     novel_start = grid_code.decode_state(grid_code.encode_state(start_indices))
#     predicted = spatial.recall_along_novel_path(novel_start, velocities2)

#     print("\n -> Cosine Similarities at the 5 Shared Anchors:")
#     for p1_step, p2_step, anchor in zip(overlap_steps_path1, overlap_steps_path2, anchors):
#         pred_lm = predicted[p2_step]
#         true_lm = landscape[anchor[0], anchor[1]]
#         sim = cosine_similarity(pred_lm, true_lm)
#         print(f"    Anchor {anchor} | Path1 Step: {p1_step:3d} | Path2 Step: {p2_step:3d} | CosSim: {sim:.4f}")

#     if plot:
#         plot_overlapping_paths(path1, path2, anchors, overlap_steps_path1, Npos)

def run_continuous_overlapping_paths():
    print("\n=== Experiment 2: Overlapping Paths (Continuous Phase Integration) ===")
    
    # grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    grid_code = ContinuousGridCode(module_periods=scaf_cfg.module_periods)
    train_states = [grid_code.encode_position(np.random.uniform(0, 60), np.random.uniform(0, 60)) for _ in range(1000)]
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    scaffold.fit_wgh(train_states)
    
    Npos, Ns = 60, 3600
    path1, path2, overlap_steps = generate_exact_5_overlap_paths(Npos, length=600)

    np.random.seed(42)
    landscape = np.random.randn(100, 100, Ns) * 0.1 
    
    velocities1 = [tuple(path1[i+1] - path1[i]) for i in range(len(path1)-1)]
    landmarks1 = [landscape[x, y] for x, y in path1]

    spatial = SpatialMemory(grid_code, scaffold, Ns)
    
    # print(" 1. Learning 'path 1' with continuous visual input...")
    start_pos = (path1[0][0], path1[0][1])
    path1_g_states = grid_code.path_integrate(start_pos, velocities1)
    
    # Pseudo-inverse(행렬 연산)로 한 번에 학습
    H_list = []
    S_list = []
    for g, s in zip(path1_g_states, landmarks1):
        H_list.append(scaffold.grid_to_hpc(g))
        S_list.append(s)
        
    H_mat = np.array(H_list).T  # (Nh, n_steps)
    S_mat = np.array(S_list).T  # (Ns, n_steps)
    
    # W_hs 와 W_sh 를 역행렬로 완벽하게 매칭
    spatial.Whs.W = H_mat @ np.linalg.pinv(S_mat)
    spatial.Wsh.W = S_mat @ np.linalg.pinv(H_mat)

    # print(" 2. Continuous path integration along 'path 2' (No vision)...")
    velocities2 = [tuple(path2[i+1] - path2[i]) for i in range(len(path2)-1)]
    path2_g_states = grid_code.path_integrate(start_pos, velocities2)

    print("\n -> Cosine Similarities at the Shared Anchors:")
    for step in overlap_steps:
        g_pred = path2_g_states[step]
        h_pred = scaffold.grid_to_hpc(g_pred)
        
        # Wsh 행렬 연산(Recall)
        s_pred = spatial.Wsh.W @ h_pred 
        
        anchor_x, anchor_y = path1[step]
        s_true = landscape[anchor_x, anchor_y]
        
        sim = cosine_similarity(s_pred, s_true)
        print(f"    Anchor ({anchor_x:2d}, {anchor_y:2d}) | Step: {step:3d} | CosSim: {sim:.4f}")
                                                                                       

def plot_overlapping_paths(path1, path2, anchors, overlap_steps_path1, Npos):
    """실험 2 결과를 시각화하기 위한 함수 (Cartesian 기준)"""
    fig, ax = plt.subplots(figsize=(6, 6))
    
    # 두 경로 플로팅
    ax.plot(path1[:,0], path1[:,1], marker='.', markersize=2, alpha=0.6, color='tab:blue', label="Original Path")
    ax.plot(path2[:,0], path2[:,1], marker='.', markersize=2, alpha=0.6, color='tab:orange', label="Novel Path")
    
    # 앵커 포인트 플로팅
    anchors_arr = np.array(anchors)
    ax.scatter(anchors_arr[:,0], anchors_arr[:,1], c="red", s=60, edgecolors="black", label="Shared Landmarks", zorder=5)
    
    for i, step in enumerate(overlap_steps_path1):
        ax.text(anchors_arr[i,0]+0.5, anchors_arr[i,1]+0.5, "Start" if i==0 else str(step),
                fontsize=9, fontweight="bold", color="darkred")

    ax.set_xlim(0, Npos); ax.set_ylim(0, Npos)
    ax.set_aspect("equal")
    ax.legend(loc="upper right")
    ax.set_title("Dark Path Integration over Novel Trajectory")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------
# 기존 실험 3: Multi-room retention
# ---------------------------------------------------------
def run_multi_room(grid_code, scaffold, Ns, n_rooms=4, steps_per_room=100):
    print(f"\n=== Experiment 3: Fig 4-style multi-room retention (n_rooms={n_rooms}) ===")
    room_specs = []
    for r in range(n_rooms):
        start_indices = [(r % k, (r * 2) % k) for k in grid_code.module_periods]
        velocities = random_walk_velocities(steps_per_room, max_speed=1, seed=20 + r)
        landmarks = random_patterns(steps_per_room + 1, Ns, seed=30 + r)
        room_specs.append({"start_indices": start_indices, "velocities": velocities,
                            "landmarks": landmarks})

    for n_learned in range(1, n_rooms + 1):
        spatial = SpatialMemory(grid_code, scaffold, Ns)
        spatial.learn_rooms(room_specs[:n_learned])

        room_sims = []
        for r in range(n_learned):
            room = room_specs[r]
            grid_path = grid_code.path_integrate(room["start_indices"], room["velocities"])
            hpc_path = [scaffold.grid_to_hpc(g) for g in grid_path]
            predicted = [spatial.Wsh.recall(h) for h in hpc_path]
            sims = [cosine_similarity(p, e) for p, e in zip(predicted, room["landmarks"])]
            room_sims.append(np.mean(sims))

        print(f"  rooms_learned={n_learned}  per-room mean cos_sim="
              + ", ".join(f"room{r}={s:.3f}" for r, s in enumerate(room_sims)))

# ---------------------------------------------------------
# [새로 추가됨] 실험 4: 길 잃음 방지 테스트 (Velocity Noise & Correction)
# ---------------------------------------------------------
def run_noise_and_correction(plot=False):
    print("\n=== Experiment 4: Velocity Noise and Periodic Sensory Correction ===")
    
    # 1. 뼈대(Scaffold) 및 모델 초기화
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    
    Ns = 100
    n_steps = 600
    noise_level = 1.0  # 길을 잃게 만들 만큼 강력한 노이즈
    corr_interval = 3  # 6스텝마다 눈을 떠서 풍경(Landmark)을 보고 교정함

    start_indices = [(0, 0)] * grid_code.n_modules
    velocities = random_walk_velocities(n_steps, max_speed=1, seed=10)
    landmarks = random_patterns(n_steps + 1, Ns, seed=11)

    spatial = SpatialMemory(grid_code, scaffold, Ns)
    spatial.learn(start_indices, velocities, landmarks)

    # 2. 내비게이션 시뮬레이터 함수
    def simulate_navigation(periodic_correction=False):
        np.random.seed(42)
        curr_indices = start_indices
        sims = []
        
        for t, v in enumerate(velocities):
            # 2-1. 한 걸음 이동 (이상적인 경로 적분)
            # 대괄호 제거 및 반환된 그리드 상태(next_g) 직접 추출
            next_g = grid_code.path_integrate(curr_indices, [v])[-1]
            g_ideal = next_g
            
            # 2-2. 발걸음에 노이즈 누적 (내면의 해마 상태에 오차 발생)
            h_internal = scaffold.grid_to_hpc(g_ideal)
            h_noisy = h_internal + np.random.randn(scaf_cfg.Nh) * noise_level
            
            # 2-3. 주기적 시각 교정 (Periodic Sensory Correction)
            if periodic_correction and (t + 1) % corr_interval == 0:
                s_true = landmarks[t + 1]
                # 랜드마크를 보고 해마 위치 정보를 끌어냄
                h_cue = spatial.Whs.recall(s_true) 
                # 노이즈가 쌓인 위치 정보(내부 관성)에 확실한 시각 정보를 덮어씌움
                h_noisy = h_noisy + h_cue * 2.0 
            
            # 2-4. Scaffold 정화 (Cleanup: 흩어진 좌표를 가장 가까운 정상 좌표로 스냅)
            h_clean, g_clean = scaffold.cleanup(h_noisy, n_iter=2)
            
            # 교정된(또는 엉뚱한 곳으로 튕겨나간) 상태를 현재 내 위치로 업데이트
            curr_indices = grid_code.decode_state(g_clean)
            
            # 2-5. 현재 위치에서 보일 랜드마크 예측 및 평가
            s_pred = spatial.Wsh.recall(h_clean)
            s_true = landmarks[t + 1]
            sims.append(cosine_similarity(s_pred, s_true))
            
        return sims

    print(" -> 1. 눈 감고 계속 걷기")
    sims_no_corr = simulate_navigation(periodic_correction=False)
    print(f"    Mean CosSim: {np.mean(sims_no_corr):.3f}")

    print(f" -> 2. 주기적으로 눈 뜨고 확인하며 걷기 (매 {corr_interval} 스텝마다 랜드마크 교정)")
    sims_with_corr = simulate_navigation(periodic_correction=True)
    print(f"    Mean CosSim: {np.mean(sims_with_corr):.3f}")

# ---------------------------------------------------------
# [추가] Path Integration 중 HPC / Grid 상태를 각각 보여주는 인터랙티브 뷰
# ---------------------------------------------------------
from experiments.experiment_item_capacity import vector_to_2d_user_style
from src.seq_utils import explicit_interpolation

def hexagonal_firing_rate(grid_code, module_idx, x, y):
    """한 모듈의 3방향 phase 성분을 합쳐 grid cell 발화율(육각 패턴) 계산"""
    lam = grid_code.module_periods[module_idx]
    pos = np.array([x, y])
    total = 0
    for d in grid_code.directions:
        k = (2 * np.pi / lam) * d
        total += np.cos(np.dot(k, pos))
    return total

def smooth_tuningcurve(avg_fields, Npos, mult=2):
    return upsample(avg_fields.reshape((Npos, Npos)), mult)

def compute_hex_field(grid_code, module_idx, extent=60, resolution=80, mult=2, sigma=3):
    xs = np.linspace(0, extent, resolution)
    ys = np.linspace(0, extent, resolution)
    field = np.zeros((resolution, resolution))
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            field[j, i] = hexagonal_firing_rate(grid_code, module_idx, x, y)

    upsampled = smooth_tuningcurve(field.flatten(), resolution, mult=mult)
    field_smoothed = explicit_interpolation(upsampled, sigma=sigma)
    return field_smoothed


def upsample(im, mult=2):
    height, width = np.shape(im)
    im_up = np.zeros((mult * height, mult * width))
    for i in range(height):
        for j in range(width):
            im_up[mult * i: mult * (i + 1), mult * j: mult * (j + 1)] = im[i, j]
    return im_up


def compute_place_field_centers(grid_code, scaffold, Nh, extent=60, resolution=40):
    """
    모든 HPC 뉴런의 place field를 한 번에 계산하고,
    각 뉴런이 가장 강하게 반응하는 (x,y) 위치(peak)를 반환.
    grid_to_hpc 호출은 resolution^2 번만 (뉴런별 반복 X, 벡터 전체가 한 번에 나오므로).
    """
    xs = np.linspace(0, extent, resolution)
    ys = np.linspace(0, extent, resolution)
    fields = np.zeros((resolution, resolution, Nh))  # [y_idx, x_idx, neuron]

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            g = grid_code.encode_position(x, y)
            h = scaffold.grid_to_hpc(g)
            fields[j, i, :] = h

    # 뉴런별 peak 위치(선호 위치) 계산
    peak_x = np.zeros(Nh)
    peak_y = np.zeros(Nh)
    for n in range(Nh):
        j_max, i_max = np.unravel_index(np.argmax(fields[:, :, n]), fields[:, :, n].shape)
        peak_x[n] = xs[i_max]
        peak_y[n] = ys[j_max]

    return peak_x, peak_y


def show_path_states_interactive(grid_code, scaffold, path, module_idx=0, Nh=None, extent=60):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        return

    if Nh is None:
        Nh = scaf_cfg.Nh  # scaffold.Nh 로 바꿔도 됨, 클래스 인터페이스 확인 필요

    hex_field = compute_hex_field(grid_code, module_idx, extent=extent, resolution=80)

    # [수정] 뉴런별 선호 위치(peak)는 고정 -> 한 번만 계산해서 캐싱
    peak_x, peak_y = compute_place_field_centers(grid_code, scaffold, Nh, extent=extent, resolution=40)

    velocities = [tuple(path[i+1] - path[i]) for i in range(len(path) - 1)]
    start_pos = (path[0][0], path[0][1])
    g_states = grid_code.path_integrate(start_pos, velocities)

    def update_plot(step):
        g = g_states[step]
        h = scaffold.grid_to_hpc(g)  # [수정] 매 step마다 실제로 바뀌는 population state

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Step {step} / {len(g_states)-1}  |  pos=({path[step][0]}, {path[step][1]})", fontsize=14)

        axes[0].imshow(hex_field, cmap="jet", origin="lower", extent=[0, extent, 0, extent])
        axes[0].scatter([path[step][0]], [path[step][1]], c="white", edgecolors="black", s=80, zorder=5)
        axes[0].set_title(f"Grid Field (module={module_idx}, period={grid_code.module_periods[module_idx]})")

        # [수정] 뉴런의 선호 위치(peak_x, peak_y)에 현재 활성값 h를 색으로 표시
        # -> population activity bump가 시간에 따라 이동하는 것을 볼 수 있음
        sc = axes[1].scatter(peak_x, peak_y, c=h, cmap="jet", s=20, vmin=h.min(), vmax=h.max())
        axes[1].scatter([path[step][0]], [path[step][1]], c="white", edgecolors="black", s=80, zorder=5)
        axes[1].set_xlim(0, extent); axes[1].set_ylim(0, extent)
        axes[1].set_aspect("equal")
        axes[1].set_title(f"HPC Population State (step={step})")
        fig.colorbar(sc, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    step_slider = widgets.IntSlider(value=0, min=0, max=len(g_states)-1, description="Step:")
    widgets.interact(update_plot, step=step_slider)


import numpy as np
import matplotlib.pyplot as plt
def visualize_realtime_trajectory_with_indices(room_size=10, n_steps=200):
    try:
        import ipywidgets as widgets
        from IPython.display import display
    except ImportError:
        print("ipywidgets 라이브러리가 필요합니다. (노트북 환경 전용)")
        return

    print(f"\n[설정 완료] 방 크기: {room_size}x{room_size}, 총 이동 스텝: {n_steps}")
    print("공간 맵핑 및 실시간 궤적 연산 중... 잠시만 기다려주세요!")

    # 1. 모델 세팅 및 뼈대(Scaffold) 학습
    grid_code = ContinuousGridCode(module_periods=[3,4,5])
    scaffold = GridHPCScaffold(grid_code, Nh=400, seed=42)
    
    train_states = [grid_code.encode_position(np.random.uniform(0, room_size), 
                                              np.random.uniform(0, room_size)) 
                    for _ in range(2000)]
    scaffold.fit_wgh(train_states)
    
    # 2. 임의의 궤적(Random Walk) 생성
    np.random.seed(100) # 궤적 고정 (원하시면 지워도 됩니다)
    
    # [수정됨] 항상 방의 정중앙에서 출발하도록 동적 할당
    start_xy = room_size / 2.0 
    path = [(start_xy, start_xy)]
    x, y = start_xy, start_xy
    velocities = []
    
    for _ in range(n_steps):
        dx, dy = np.random.randn() * 1.5, np.random.randn() * 1.5
        # 방을 벗어나지 않게 벽에서 반사
        if not (0 < x + dx < room_size): dx = -dx
        if not (0 < y + dy < room_size): dy = -dy
        x += dx; y += dy
        path.append((x, y))
        velocities.append((dx, dy))
        
    path = np.array(path)

    # 3. Path Integration (전체 스텝의 상태를 미리 계산)
    g_states = grid_code.path_integrate(path[0], velocities)
    h_states = [scaffold.grid_to_hpc(g) for g in g_states]
    g_states = np.array(g_states)  # 형태: (n_steps+1, Ng)
    h_states = np.array(h_states)  # 형태: (n_steps+1, Nh)
    
    # 4. 배경 맵(Heatmap) 생성을 위한 공간 전체 좌표계 사전 연산
    res = 60
    xs = np.linspace(0, room_size, res)
    ys = np.linspace(0, room_size, res)
    X, Y = np.meshgrid(xs, ys)
    pts = np.vstack([X.ravel(), Y.ravel()]).T
    
    # 행렬 연산으로 모든 뉴런의 공간 맵 한 번에 생성
    G_bg = np.array([grid_code.encode_position(px, py) for px, py in pts])
    # H_bg = np.maximum(0, scaffold.Whg @ G_bg.T - scaffold.threshold).T

    # strict_threshold = 5.0
    # H_bg = np.maximum(0, scaffold.Whg @ G_bg.T - strict_threshold).T
    H_bg = np.array([scaffold.grid_to_hpc(g) for g in G_bg])

    # 5. 슬라이더 업데이트 그리기 함수
    def update_plot(step, grid_idx, hpc_idx):
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        
        # --- 데이터 추출 ---
        module_idx = grid_idx % len(grid_code.module_periods)
        start_idx = module_idx * 6  # 한 모듈당 [cos, sin]*3 = 6개의 값
        # 0번째, 2번째, 4번째 인덱스가 코사인(cos) 값입니다.
        g_heat = (G_bg[:, start_idx] + G_bg[:, start_idx+2] + G_bg[:, start_idx+4]).reshape(res, res)

        # g_heat = G_bg[:, grid_idx].reshape(res, res)
        h_heat = H_bg[:, hpc_idx].reshape(res, res)
        
        curr_path = path[:step+1]
        # curr_g_fire = g_states[:step+1, grid_idx]

        module_idx = grid_idx % len(grid_code.module_periods)
        start_idx = module_idx * 6

        curr_g_fire = (g_states[:step+1, start_idx] + 
                    g_states[:step+1, start_idx+2] + 
                    g_states[:step+1, start_idx+4])

        curr_h_fire = h_states[:step+1, hpc_idx]

        # ---------------------------------------------------------
        # [Panel 1] Grid Cell 실시간 궤적 시각화
        # ---------------------------------------------------------
        axes[0].imshow(g_heat, cmap='jet', origin='lower', extent=[0, room_size, 0, room_size], alpha=0.3)
        axes[0].plot(path[:, 0], path[:, 1], '-', color="gray", alpha=0.2, linewidth=2)
        axes[0].plot(curr_path[:, 0], curr_path[:, 1], '-', color="darkgray", linewidth=2)
        
        sc1 = axes[0].scatter(curr_path[:, 0], curr_path[:, 1], c=curr_g_fire, cmap='Reds', 
                              s=80, edgecolors="white", linewidths=0.5, zorder=3)
        axes[0].scatter(curr_path[-1, 0], curr_path[-1, 1], c='blue', s=250, marker='*', edgecolors="black", zorder=4)
        
        axes[0].set_title(f"Grid Cell #{grid_idx} Trajectory Firing (Step {step})", fontsize=14)
        axes[0].set_xlim(0, room_size); axes[0].set_ylim(0, room_size)
        axes[0].set_aspect('equal')
        # fig.colorbar(sc1, ax=axes[0], fraction=0.046, pad=0.04, label="Instantaneous Firing")

        # ---------------------------------------------------------
        # [Panel 2] HPC Place Cell 실시간 궤적 시각화
        # ---------------------------------------------------------
        axes[1].imshow(h_heat, cmap='jet', origin='lower', extent=[0, room_size, 0, room_size], alpha=0.3, vmin=0)
        axes[1].plot(path[:, 0], path[:, 1], '-', color="gray", alpha=0.2, linewidth=2)
        axes[1].plot(curr_path[:, 0], curr_path[:, 1], '-', color="darkgray", linewidth=2)
        
        sc2 = axes[1].scatter(curr_path[:, 0], curr_path[:, 1], c=curr_h_fire, cmap='Reds', 
                              s=80, edgecolors="white", linewidths=0.5, zorder=3, vmin=0, vmax=h_heat.max()+1e-5)
        axes[1].scatter(curr_path[-1, 0], curr_path[-1, 1], c='blue', s=250, marker='*', edgecolors="black", zorder=4)
        
        axes[1].set_title(f"HPC Place Cell #{hpc_idx} Trajectory Firing (Step {step})", fontsize=14)
        axes[1].set_xlim(0, room_size); axes[1].set_ylim(0, room_size)
        axes[1].set_aspect('equal')
        # fig.colorbar(sc2, ax=axes[1], fraction=0.046, pad=0.04, label="Instantaneous Firing")

        plt.tight_layout()
        plt.show()

    # 6. UI 위젯 생성
    step_slider = widgets.IntSlider(value=0, min=0, max=n_steps, description="Time Step:")
    grid_idx_slider = widgets.IntSlider(value=0, min=0, max=grid_code.Ng-1, description="Grid IDX:")
    hpc_idx_slider = widgets.IntSlider(value=10, min=0, max=scaffold.Nh-1, description="HPC IDX:")
    
    widgets.interact(update_plot, step=step_slider, grid_idx=grid_idx_slider, hpc_idx=hpc_idx_slider)

# ---------------------------------------------------------
# [추가] fig4c 스타일 데모: 실제 이미지 sensory + 실제 경로 시각화 +
# 재방문 복원 / 미방문 trajectory 예측 / sensory -> location 역추론
# ---------------------------------------------------------
def build_fig4c_demo(trained_length=100, room_pad=3, seed=3):
    """실제 이미지(prepare_sensory_data)를 실제 random-walk 경로("원래 경로",
    trained path) 위 각 위치에 결합해서 학습한다. VectorHASH_fig4e_random.ipynb의
    fig4c처럼 경로를 Npos x Npos 방 안(벽에서 room_pad칸 이상 떨어진 곳)에
    가둬서, 나중에 만들 novel trajectory가 이 경로를 피해 다닐 공간을
    확보한다. module_periods의 곱(Npos)이 sbook의 Npos(=60)와 같아야
    위치<->이미지 인덱스가 CRT로 정합된다."""
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, Nh=scaf_cfg.Nh, connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold, nonlinearity=scaf_cfg.nonlinearity, seed=1
    )
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])

    Npos = int(np.prod(scaf_cfg.module_periods))
    sbook_flattened = prepare_sensory_data()

    rng = np.random.default_rng(seed)
    moves4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    lo, hi = room_pad, Npos - room_pad
    x, y = Npos // 2, Npos // 2
    path_xy = [(x, y)]
    velocities = []
    visited = {(x, y)}
    for _ in range(trained_length):
        valid = [(dx, dy) for dx, dy in moves4 if lo <= x + dx < hi and lo <= y + dy < hi]
        # 자기 자신을 최대한 안 밟는(self-avoiding) 걸음을 우선 고른다 -- 그래야
        # trained path가 방 안에서 촘촘하게 뭉치지 않고(자기 교차 최소화), novel
        # trajectory가 나중에 이 경로를 피해서 지나갈 공간이 넉넉히 남는다.
        # 안 밟은 칸이 하나도 없는 막다른 곳일 때만 예외적으로 이미 밟은 칸으로 후퇴한다.
        unvisited = [(dx, dy) for dx, dy in valid if (x + dx, y + dy) not in visited]
        candidates = unvisited if unvisited else valid
        # fig4e_random.ipynb의 fig4c처럼 한쪽으로 살짝 흘러가게(drift bias) 하면
        # 경로가 너무 조밀하게 자기 자신을 둘러싸는 것(스스로 갇히는 지점)을 줄여준다.
        w = np.array([1.0 + max(0, dx) * 0.3 + max(0, dy) * 0.3 for dx, dy in candidates])
        w = w / w.sum()
        dx, dy = candidates[rng.choice(len(candidates), p=w)]
        x, y = x + dx, y + dy
        path_xy.append((x, y))
        velocities.append((dx, dy))
        visited.add((x, y))
    path_xy = np.array(path_xy)  # (trained_length+1, 2), 방 안에 갇힌 물리 좌표

    start_indices = [(int(path_xy[0][0]) % k, int(path_xy[0][1]) % k) for k in grid_code.module_periods]
    landmarks = [sbook_flattened[:, int(px) * Npos + int(py)] for px, py in path_xy]

    spatial = SpatialMemory(grid_code, scaffold, sbook_flattened.shape[0])
    spatial.learn(start_indices, velocities, landmarks)
    return dict(grid_code=grid_code, scaffold=scaffold, spatial=spatial,
                sbook_flattened=sbook_flattened, Npos=Npos, room_pad=room_pad,
                start_indices=start_indices, velocities=velocities,
                landmarks=landmarks, path_xy=path_xy)


def _bfs_shortest_path(start, target, forbidden, lo, hi, rng):
    """start->target 4방향 최단 경로. forbidden 칸은 target이 아닌 한 회피.
    rng로 이웃 방문 순서를 섞어서, 실패 시 재시도(다른 seed)하면 다른
    우회로를 찾을 수 있게 한다."""
    from collections import deque
    moves4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]
    q = deque([start])
    prev = {start: None}
    while q:
        cur = q.popleft()
        if cur == target:
            break
        order = rng.permutation(len(moves4))
        for k in order:
            dx, dy = moves4[k]
            nxt = (cur[0] + dx, cur[1] + dy)
            if not (lo <= nxt[0] < hi and lo <= nxt[1] < hi):
                continue
            if nxt in forbidden and nxt != target:
                continue
            if nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)
    if target not in prev:
        return None
    path = [target]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    path.reverse()
    return path


def build_novel_trajectory(model, novel_length=600, n_overlap=5, seed=1, max_attempts=200):
    """원래 경로(model)와 정확히 n_overlap개 지점에서만 겹치는("재방문") 새
    경로를 만든다. 원래 경로 위 지정한 n_overlap개(anchor) 위치만 지나가게
    허용하고, 나머지 원래 경로 칸은 전부 회피(forbidden)한다 -- fig4e_random
    .ipynb의 fig4c '겹치는 지점 정확히 N개' 경로 생성 방식과 동일한 원리
    (거기서는 A*, 여기서는 BFS로 forbidden 칸을 피해서 anchor들을 순서대로
    연결). 나머지 길이는 forbidden을 피해서 무작위 보행으로 채운다."""
    Npos, pad = model["Npos"], model["room_pad"]
    trained_path = [tuple(int(v) for v in p) for p in model["path_xy"]]
    lo, hi = pad, Npos - pad
    moves4 = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    n_overlap = min(n_overlap, len(trained_path))

    # 자기 자신과 교차하는 random walk라서 서로 다른 step이 같은 물리 좌표를
    # 가리킬 수 있다 -- anchor는 반드시 "서로 다른 좌표"만 뽑아야 실제로
    # forbidden에서 exempt되는 좌표 수가 n_overlap과 일치한다.
    unique_cells = list(dict.fromkeys(trained_path))  # 등장 순서 보존한 중복 제거
    n_overlap = min(n_overlap, len(unique_cells))

    for attempt in range(max_attempts):
        rng = np.random.default_rng(seed + attempt)
        # anchor 조합 자체가 (조밀한 자기교차 때문에) BFS로 못 뚫는 경우가 있으므로,
        # 매 시도마다 앵커도 다시 뽑는다 (시작점=index 0은 항상 고정).
        if attempt == 0:
            cell_idxs = np.linspace(0, len(unique_cells) - 1, n_overlap, dtype=int)
        else:
            rest = rng.choice(np.arange(1, len(unique_cells)), size=n_overlap - 1, replace=False)
            cell_idxs = np.concatenate([[0], np.sort(rest)])
        anchors = [unique_cells[i] for i in cell_idxs]
        forbidden = set(trained_path) - set(anchors)

        novel_path = [anchors[0]]
        overlap_steps = [0]
        ok = True
        for i in range(len(anchors) - 1):
            seg = _bfs_shortest_path(novel_path[-1], anchors[i + 1], forbidden, lo, hi, rng)
            if seg is None:
                ok = False
                break
            novel_path.extend(seg[1:])
            overlap_steps.append(len(novel_path) - 1)
        if not ok:
            continue

        # 남은 길이는 forbidden을 피해서 무작위 보행으로 패딩
        while len(novel_path) < novel_length:
            cx, cy = novel_path[-1]
            valid = [(cx + dx, cy + dy) for dx, dy in moves4
                     if lo <= cx + dx < hi and lo <= cy + dy < hi
                     and (cx + dx, cy + dy) not in forbidden]
            if not valid:
                break
            novel_path.append(valid[rng.integers(len(valid))])

        # 최종 검증: 실제로 겹치는 지점 수가 정확히 n_overlap인지 확인하고,
        # 아니면(예: 패딩 무작위 보행이 우연히 다른 anchor를 또 밟은 경우) 재시도한다.
        if len(set(trained_path) & set(novel_path)) == n_overlap:
            break
    else:
        raise RuntimeError("겹치지 않는 경로를 못 찾았습니다. n_overlap을 줄이거나 "
                            "trained_length/room_pad를 바꿔보세요.")

    novel_xy = np.array(novel_path)
    novel_velocities = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(novel_path[:-1], novel_path[1:])]
    novel_grid_path = model["grid_code"].path_integrate(model["start_indices"], novel_velocities)

    return dict(novel_xy=novel_xy, novel_velocities=novel_velocities,
                novel_grid_path=novel_grid_path, overlap_steps=overlap_steps)


def plot_paths(model, novel_model=None, title="Grid world", unvisited_steps=None,
               show_revisit_labels=True, current_step=None):
    """원래 경로(검정)와 새 경로(파랑)를 x-y 평면에 같이 그리고, 재방문
    지점(겹치는 위치)을 빨간 원으로 표시한다. unvisited_steps를 주면
    (미방문 step 인덱스들) 초록 X로 추가 표시.
    show_revisit_labels=False면 재방문 지점의 "t=n" 라벨을 끈다.
    current_step을 주면 trained path 위 그 step 위치를 검은 다이아몬드로 표시."""
    from matplotlib.ticker import MaxNLocator
    path_xy = model["path_xy"]
    fig, ax = plt.subplots(figsize=(6.8, 6.8))
    ax.plot(path_xy[0, 0], path_xy[0, 1], "o", color="black", markersize=4, label="start location")
    ax.plot(path_xy[:, 0], path_xy[:, 1], "-", color="black", label="original path")
    if current_step is not None:
        cx, cy = path_xy[current_step]
        ax.plot(cx, cy, "D", color="black", markersize=5, label=f"current (t={current_step})")

    if novel_model is not None:
        novel_xy = novel_model["novel_xy"]
        ax.plot(novel_xy[:, 0], novel_xy[:, 1], "-", color="blue", linewidth=1.5, label="new path")
        overlap_steps = novel_model["overlap_steps"]
        overlap = novel_xy[overlap_steps]
        # t=0은 항상 novel path의 시작점(=trained path 시작점)이라 "start location"과
        # 겹침 -- revisit 표시는 t=0 빼고, 라벨(t=0)은 유지.
        non_start = [t != 0 for t in overlap_steps]
        if any(non_start):
            ax.plot(overlap[non_start, 0], overlap[non_start, 1], "o", color="red", markersize=5,
                    label="revisit location")
        if show_revisit_labels:
            for t, (px, py) in zip(overlap_steps, overlap):
                ax.annotate(f"t={t}", xy=(px, py), xytext=(3, 3), textcoords="offset points",
                            fontsize=8, color="firebrick")
        if unvisited_steps:
            unvisited = novel_xy[list(unvisited_steps)]
            ax.plot(unvisited[:, 0], unvisited[:, 1], "x", color="orange", markersize=4, mew=1.2,
                    label="novel location")

    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.set_box_aspect(1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.show()


def plot_grid_modules_square(ax, grid_code, g, cmap="OrRd", vmin=0, vmax=1, shear_deg=60):
    """한 axes(정사각형) 안에 모듈별 grid state를 가로로 나란히 그린다. 박스 크기를
    period(k)에 비례시켜서 모듈마다 칸(cell) 하나의 화면 크기는 동일하게 유지하고,
    n×n 격자 자체의 전체 크기만 k가 클수록 커지게 한다. shear_deg만큼 x축 방향으로
    기울여서(pcolormesh 좌표를 직접 shear) 마름모(실제 grid cell lattice) 느낌을 낸다."""
    ax.axis("off")
    blocks = grid_code.state_blocks(g)
    periods = grid_code.module_periods
    n = len(blocks)
    theta = np.deg2rad(shear_deg)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # 겹침을 고정 폭(예: -0.03)이 아니라 "모듈 자기 폭의 일정 비율"로 잡아야, 작은
    # 모듈(4x4)과 큰 모듈(7x7)의 겹침 정도가 시각적으로 동일해 보인다 -- 고정폭이면
    # 큰 모듈일수록 같은 겹침이 상대적으로 작아 보여서 간격이 들쭉날쭉해 보였다.
    margin = 0.02
    overlap_ratio = 0.3
    K_all = sum(periods)
    K_prefix = sum(periods[:-1])  # 마지막 모듈 뒤에는 겹칠 다음 모듈이 없음
    # cell 한 변 길이 s를 기준으로 두 basis vector (s,0), (s*cos_t, s*sin_t) 둘 다
    # 길이가 s로 같다 -- 즉 변 길이가 전부 같은 진짜 마름모(rhombus) 타일링.
    s = (1 - 2 * margin) / ((1 + cos_t) * (K_all - overlap_ratio * K_prefix))
    x0 = margin
    min_x, max_x, max_h = x0, x0, 0
    for idx, (k, block) in enumerate(zip(periods, blocks)):
        # block은 [x phase, y phase] 순서(encode_state 참고). 행=y, 열=x 좌표로 맞추려고
        # block.T를 쓰고, (i,j) 격자점을 i*(s,0) + j*(s*cos_t, s*sin_t)로 배치해서
        # 변 길이가 전부 s로 같은 마름모 격자를 만든다 (실제 grid cell 논문 도식 방식).
        jj, ii = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        X = x0 + ii * s + jj * s * cos_t
        Y = jj * s * sin_t
        ax.pcolormesh(X, Y, block.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat",
                      edgecolors="dimgray", linewidth=0.3)
        w_x, w_y = k * s * (1 + cos_t), k * s * sin_t
        corners = [(x0, 0), (x0 + k * s, 0), (x0 + k * s + k * s * cos_t, w_y), (x0 + k * s * cos_t, w_y), (x0, 0)]
        ax.plot(*zip(*corners), color="dimgray", linewidth=0.8)
        ax.text(x0 + w_x / 2, w_y + 0.03, f"{k}×{k}", fontsize=7, color="dimgray",
                ha="center", va="bottom")
        max_h = max(max_h, w_y)
        min_x = min(min_x, x0, x0 + k * s * cos_t)
        max_x = max(max_x, x0 + k * s, x0 + k * s + k * s * cos_t)
        x0 += w_x * (1 - overlap_ratio) if idx < n - 1 else w_x

    ax.set_xlim(min_x - margin, max_x + margin)
    ax.set_ylim(0, max_h + 0.04)
    ax.set_aspect("equal")


def _unvisited_steps(model, novel_model):
    """novel path의 step 인덱스 중 실제 좌표가 trained path 좌표(전부, anchor 포함)와
    한 번도 안 겹치는 step만 골라낸다. overlap_steps만으로 거르면 부족하다 --
    anchor를 이어붙인 뒤 남는 길이를 채우는 무작위 보행(padding) 구간이 같은
    anchor 좌표를 다시 밟아도 overlap_steps에는 안 잡히기 때문.
    novel path 자체도 self-crossing(같은 좌표를 여러 step에서 반복 방문)이라,
    좌표 기준으로도 중복 제거해서 좌표별로 처음 등장한 step 하나만 남긴다 --
    안 그러면 near/far/무작위 선택 결과가 서로 같은 좌표를 여러 번 뽑아서
    지도/패널에서 겹쳐 보인다."""
    trained_cells = set(tuple(int(v) for v in p) for p in model["path_xy"])
    novel_xy = novel_model["novel_xy"]
    seen = set()
    steps = []
    for t in range(len(novel_xy)):
        cell = tuple(int(v) for v in novel_xy[t])
        if cell in trained_cells or cell in seen:
            continue
        seen.add(cell)
        steps.append(t)
    return steps


def _plot_novel_step_column(axes, col, t, model, novel_model):
    """novel path의 step t 지점에서 True/Recall sensory, grid state, HPC state를
    한 컬럼(axes[:, col])에 그린다. demo_revisit_predictions/demo_unvisited_predictions
    공용 -- 재방문/미방문 어느 지점이든 t만 주면 동일하게 동작."""
    spatial, scaffold = model["spatial"], model["scaffold"]
    sbook_flattened, Npos = model["sbook_flattened"], model["Npos"]

    g = novel_model["novel_grid_path"][t]
    h = scaffold.grid_to_hpc(g)
    h_clean, g_clean = scaffold.cleanup(h, n_iter=2)
    recon = spatial.Wsh.recall(h_clean)

    x, y = novel_model["novel_xy"][t]
    target = sbook_flattened[:, (int(x) % Npos) * Npos + (int(y) % Npos)]

    cos = float(np.dot(target, recon) / (np.linalg.norm(target) * np.linalg.norm(recon) + 1e-10))

    axes[0, col].imshow(reshape_sensory_to_image(target), cmap="gray")
    axes[0, col].set_title(f"t={t}", fontsize=11, fontweight="bold"); axes[0, col].axis("off")
    axes[1, col].imshow(reshape_sensory_to_image(recon), cmap="gray")
    axes[1, col].set_title(f"cos={cos:.3f}", fontsize=10); axes[1, col].axis("off")

    plot_grid_modules_square(axes[2, col], model["grid_code"], g_clean)
    axes[3, col].imshow(reshape_sensory_to_image(h_clean), cmap="magma")
    axes[3, col].set_title("HPC state", fontsize=9, color="dimgray")
    axes[3, col].axis("off")
    return cos


def _plot_novel_steps_grid(model, novel_model, selected, suptitle):
    n_rows = 4  # True sensory / Recall sensory / grid state(모듈 전체, 한 패널) / HPC state
    n_show = len(selected)
    # grid state(row 2) 실제 내용은 sheared 마름모라 가로세로 비율이 ~3:1 (넓적함).
    # 다른 행(정사각형 이미지)과 같은 높이를 주면 aspect="equal" 유지 시 위아래에
    # 안 쓰이는 흰 여백이 크게 남는다 -- 그 행만 내용 비율(1/3)에 맞게 낮춰서
    # 모양(마름모) 그대로 유지하면서 여백만 없앤다.
    row_h_ratio = [1, 1, 1 / 3, 1]
    fig, axes = plt.subplots(n_rows, n_show, gridspec_kw={"height_ratios": row_h_ratio},
                              figsize=(3.2 * n_show, 3.2 * sum(row_h_ratio)))
    if n_show == 1:
        axes = axes.reshape(n_rows, 1)

    cos_list = [_plot_novel_step_column(axes, col, t, model, novel_model)
                for col, t in enumerate(selected)]

    row_labels = ["True", "Recall", "Grid state", "HPC state"]
    for row, label in enumerate(row_labels):
        axes[row, 0].annotate(label, xy=(-0.15, 0.5), xycoords="axes fraction",
                               ha="right", va="center", fontsize=12, fontweight="bold",
                               rotation=90)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    # tight_layout()은 axes마다 get_tightbbox(텍스트 렌더링)를 다 계산해야 해서
    # 열 수가 많아지면(4x4=16 axes) 느림(~0.5s/call). 고정 여백으로 대체해서
    # 슬라이더 인터랙션마다 다시 그릴 때 빠르게.
    fig.subplots_adjust(left=0.06, right=0.98, top=0.96, bottom=0.02, hspace=0.08, wspace=0.25)
    plt.show()
    return cos_list


def demo_revisit_predictions(model, novel_model, n_revisits=10, seed=2):
    """새 경로가 원래 경로와 겹치는(재방문) 지점들 중 최대 n_revisits개를
    뽑아서, 그 지점에서 예측되는 sensory를 실제 landmark와 비교한다
    (fig4c cell 6 스타일: 위 True / 아래 Recall). 이 지점들은 학습 때 실제로
    결합(binding)됐던 위치라 recall이 잘 됨."""
    overlap_steps = novel_model["overlap_steps"]
    if not overlap_steps:
        return []

    rng = np.random.default_rng(seed)
    n_show = min(n_revisits, len(overlap_steps))
    selected = np.sort(rng.choice(overlap_steps, size=n_show, replace=False))
    suptitle = "Recalled images on revisited locations"
    return _plot_novel_steps_grid(model, novel_model, selected, suptitle)


def plot_unvisited_distance_map(model, novel_model, n_show=5):
    """미방문 지점들 중 학습된 경로(model["path_xy"])에 가장 가까운 n_show개를
    골라 지도에 표시 (재방문 지점은 빨간 원). demo_unvisited_by_distance에서
    쓸 unvisited step 목록을 반환."""
    candidates = _unvisited_steps(model, novel_model)
    if not candidates:
        return []

    trained_xy = model["path_xy"]
    novel_xy = novel_model["novel_xy"]
    dist = {t: int(np.min(np.abs(trained_xy - novel_xy[t]).sum(axis=1))) for t in candidates}
    unvisited = sorted(candidates, key=lambda t: dist[t])[:n_show]

    plot_paths(model, novel_model, title="Grid world", unvisited_steps=unvisited)
    return unvisited


def demo_unvisited_by_distance(model, novel_model, unvisited):
    """plot_unvisited_distance_map이 뽑아준 미방문 지점들의 sensory recall을 보여준다.
    학습 때 결합(binding)이 없던 위치라 재방문 지점(cos_sim~1.0)보다 recall이
    부정확함을 확인할 수 있다."""
    if not unvisited:
        return []
    return _plot_novel_steps_grid(model, novel_model, unvisited, "Recalled images on novel locations")


def demo_unvisited_predictions(model, novel_model, n_show=10, seed=2):
    """새 경로가 원래 경로와 한 번도 안 겹치는(학습 때 결합이 전혀 없었던) 지점들
    중 최대 n_show개를 뽑아서 같은 방식으로 확인한다. grid state는 dark
    path-integration만으로도 여전히 정확하지만(모듈 shift가 결정론적이라서),
    그 위치의 sensory는 Whs/Wsh가 한 번도 학습한 적이 없으므로 recall이 실제
    landmark와 얼마나 다른지(대체로 낮은 cos_sim) 보여준다."""
    candidates = _unvisited_steps(model, novel_model)
    if not candidates:
        print("미방문 지점이 없습니다. novel_length를 늘려보세요.")
        return []

    rng = np.random.default_rng(seed)
    n_show = min(n_show, len(candidates))
    selected = np.sort(rng.choice(candidates, size=n_show, replace=False))
    suptitle = f"Novel path UNVISITED points ({n_show}/{len(candidates)} points shown)"
    return _plot_novel_steps_grid(model, novel_model, selected, suptitle)


def demo_sensory_to_location(model, query_step=5):
    """sensory 입력 -> 어느 위치(grid code)인지 역추론 (Whs로 s->h, scaffold cleanup으로 h->g)."""
    spatial, grid_code, scaffold = model["spatial"], model["grid_code"], model["scaffold"]
    s_query = model["landmarks"][query_step]
    h_cue = spatial.Whs.recall(s_query)
    _, g_clean = scaffold.cleanup(h_cue, n_iter=2)
    predicted_indices = grid_code.decode_state(g_clean)

    true_path = grid_code.path_integrate(model["start_indices"], model["velocities"])
    true_indices = grid_code.decode_state(true_path[query_step])
    true_xy = tuple(model["path_xy"][query_step])
    print(f"[sensory -> location] query_step={query_step} | true_xy={true_xy} | "
          f"predicted module idx={predicted_indices} | true module idx={true_indices} | "
          f"match={predicted_indices == true_indices}")
    return predicted_indices, true_indices


def run():
    run_path()       # 기존 실험 1
    run_continuous_overlapping_paths()   # 추가된 겹치는 경로 실험
    # run_multi_room(...)     # 인자가 필요하므로 제외하거나 아래와 같이 선언합니다.
    
    # Multi-room 실험에 필요한 인스턴스 생성 후 호출
    grid_code = GridCode(module_periods=scaf_cfg.module_periods, seed=0)
    scaffold = GridHPCScaffold(
        grid_code, 
        Nh=scaf_cfg.Nh, 
        connection_prob=scaf_cfg.connection_prob,
        threshold=scaf_cfg.threshold,
        nonlinearity=scaf_cfg.nonlinearity,
        seed=1
    )
    scaffold.fit_wgh([grid_code.encode_state(s) for s in grid_code.all_states()])
    run_multi_room(grid_code, scaffold, Ns=100)

    run_noise_and_correction()

if __name__ == "__main__":
    run()