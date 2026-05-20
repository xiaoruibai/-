# -*- coding: utf-8 -*-
"""
纯 ETHOS 复现版本，只保留：
1. ETHOS_IPW
2. ETHOS_DR

功能：
1. 自动扫描 DATA_DIR 中全部符合命名规则的环境文件
2. 依次运行全部环境
3. 不保存 CSV
4. 终端仅输出简洁结果表
5. 同时输出每个环境、每个方法所使用的参数

支持文件名格式示例：
    twins_linear_linear_K10.csv
    twins_nonlinear_switching_K100.csv
    ihdp_linear_switching_K10.csv
    ihdp_nonlinear_linear_K100.csv
"""

import os
import re
import numpy as np
import pandas as pd

# ============================================================
# 0) 基本设置
# ============================================================

DATA_DIR = "twins_ethos_streams"   # 改成你的数据目录
N_REPEATS = 1
BASE_SEED = 2026

METHODS_TO_RUN = [
    "ETHOS_IPW",
    "ETHOS_DR",
]

# ETHOS 原文常数
D = 1.0
G = 1.0
H = 1.0
B = 1.0
EPS = 0.05

# HTE 参数投影半径
HTE_RADIUS = D / 2.0

# DR 中 outcome regression 的预测截断
M_BOUND = 1.0

# outcome regression 参数投影半径
OUTCOME_RADIUS = 1.0

# nuisance model 自适应学习率参数（用于 ETHOS_DR 的 outcome model）
ADAPT_EPS = 1e-8
ADAPT_NUISANCE_BETA = 0.9
ADAPT_NUISANCE_POWER = 0.9
NUISANCE_ETA_MIN = 1e-3
NUISANCE_ETA_MAX = 1e-1

# 环境 CSV 中的非协变量列
META_COLS = {
    "env_name", "surface", "drift", "K", "segment", "segment_round",
    "t", "alpha", "source_row", "q", "w", "p", "y",
    "mu_-1", "mu_1", "tau",
}

ENV_FILE_REGEX = re.compile(
    r"^(?:twins|ihdp)_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv$"
)

# ============================================================
# 1) 基础工具函数
# ============================================================

def row_l2_normalize_leq1(X):
    norms = np.linalg.norm(X, axis=1)
    scale = np.maximum(1.0, norms)
    return X / (scale[:, None] + 1e-12)

def project_l2_ball_rows(theta_mat, radius):
    norms = np.linalg.norm(theta_mat, axis=1, keepdims=True)
    scale = np.maximum(1.0, norms / (radius + 1e-12))
    return theta_mat / scale

def project_l2_ball_vec(theta, radius):
    norm = float(np.linalg.norm(theta))
    if norm <= radius:
        return theta
    return theta * (radius / (norm + 1e-12))

def clip_scalar(v, lower, upper):
    return float(np.minimum(np.maximum(v, lower), upper))

def parse_env_file_info(path):
    name = os.path.basename(path)
    m = ENV_FILE_REGEX.match(name)
    if m is None:
        raise ValueError(f"环境文件名不符合约定格式: {name}")
    prefix = name.split("_")[0]
    surface = m.group(1)
    drift = m.group(2)
    K = int(m.group(3))
    return {
        "prefix": prefix,
        "surface": surface,
        "drift": drift,
        "K": K,
        "filename": name,
    }

def discover_environment_files(data_dir):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"找不到数据目录: {data_dir}")

    matched = []
    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue
        if ENV_FILE_REGEX.match(fname):
            info = parse_env_file_info(fname)
            info["full_path"] = fpath
            matched.append(info)

    if len(matched) == 0:
        raise FileNotFoundError(f"在目录 {data_dir} 中没有发现符合命名规则的环境文件。")

    surface_order = {"linear": 0, "nonlinear": 1}
    drift_order = {"linear": 0, "switching": 1}
    matched.sort(
        key=lambda x: (
            x["prefix"],
            surface_order.get(x["surface"], 999),
            drift_order.get(x["drift"], 999),
            x["K"],
            x["filename"],
        )
    )
    return matched

def get_feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]

def mean_and_se(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) == 0:
        return np.nan, np.nan
    if len(x) == 1:
        return float(np.mean(x)), 0.0
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(len(x))
    return mean, float(se)

def is_dr_method(method):
    return method.endswith("_DR")

def get_method_g_tilde(method):
    if method.endswith("_IPW"):
        return float((H + B / EPS) * G)
    if method.endswith("_DR"):
        return float((H + 2.0 * M_BOUND + (B + M_BOUND) / EPS) * G)
    raise ValueError(f"未知方法: {method}")

def get_q_array(df, w, p):
    if "q" in df.columns:
        q = df["q"].to_numpy(dtype=np.float64)
        return np.clip(q, EPS, 1.0 - EPS)
    q = np.where(w == 1.0, p, 1.0 - p)
    return np.clip(q, EPS, 1.0 - EPS)

# ============================================================
# 2) 固定随机特征映射
# ============================================================

def init_random_mlp(input_dim, seed):
    rng = np.random.default_rng(seed)
    dims = [input_dim, 256, 256, 256, 2048]
    weights = []
    biases = []
    for din, dout in zip(dims[:-1], dims[1:]):
        W = rng.normal(
            loc=0.0,
            scale=1.0 / np.sqrt(din),
            size=(din, dout)
        ).astype(np.float32)
        b = rng.normal(
            loc=0.0,
            scale=0.01,
            size=(dout,)
        ).astype(np.float32)
        weights.append(W)
        biases.append(b)
    return {"surface": "nonlinear", "weights": weights, "biases": biases}

def make_feature_map(surface, input_dim, seed):
    if surface == "linear":
        return {"surface": "linear"}
    if surface == "nonlinear":
        return init_random_mlp(input_dim, seed)
    raise ValueError("surface 只能是 'linear' 或 'nonlinear'。")

def apply_feature_map_batch(feature_map, X):
    if feature_map["surface"] == "linear":
        Phi = np.concatenate(
            [X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)],
            axis=1
        )
        return row_l2_normalize_leq1(Phi)

    Hx = np.asarray(X, dtype=np.float32)
    for layer_idx, (W, b) in enumerate(zip(feature_map["weights"], feature_map["biases"])):
        Hx = Hx @ W + b
        if layer_idx < len(feature_map["weights"]) - 1:
            Hx = np.maximum(Hx, 0.0)
    return row_l2_normalize_leq1(Hx.astype(np.float64))

# ============================================================
# 3) ETHOS 超参数、伪结果、meta 更新
# ============================================================

def get_ethos_hyperparams(T_run, g_tilde):
    alpha_meta = np.sqrt(8.0 / (T_run * (g_tilde ** 2) * (D ** 2)))
    N = int(np.floor(0.5 * np.log2(1.0 + 4.0 * T_run / 7.0))) + 1
    etas = []
    for i in range(1, N + 1):
        eta_i = (2.0 ** (i - 1)) * D / g_tilde * np.sqrt(7.0 / (2.0 * T_run))
        etas.append(float(eta_i))
    return float(alpha_meta), int(N), np.asarray(etas, dtype=np.float64)

def update_meta_weights(omega, alpha_meta, base_sur_losses):
    log_w = np.log(omega + 1e-300)
    log_w = log_w - alpha_meta * base_sur_losses
    log_w = log_w - np.max(log_w)
    new_w = np.exp(log_w)
    new_w = new_w / np.sum(new_w)
    return new_w

def get_ipw_pseudo_outcome(w, y, p):
    return float(w * y / p)

def get_dr_pseudo_outcome(w, y, q, m0_hat, m1_hat):
    if w == 1.0:
        return float((m1_hat - m0_hat) + (y - m1_hat) / q)
    return float((m1_hat - m0_hat) - (y - m0_hat) / (1.0 - q))

def init_alignment_adaptive_state(dim, eta_min, eta_max, beta, power):
    return {
        "m": np.zeros(dim, dtype=np.float64),
        "s": 0.0,
        "eta_min": float(eta_min),
        "eta_max": float(eta_max),
        "beta": float(beta),
        "power": float(power),
    }

def get_alignment_adaptive_eta(state, grad):
    beta = state["beta"]
    state["m"] = beta * state["m"] + (1.0 - beta) * grad
    state["s"] = beta * state["s"] + (1.0 - beta) * float(np.dot(grad, grad))
    align = float(np.dot(state["m"], state["m"]) / (state["s"] + ADAPT_EPS))
    align = clip_scalar(align, 0.0, 1.0)
    eta = state["eta_min"] + (state["eta_max"] - state["eta_min"]) * (align ** state["power"])
    return float(eta)

def init_outcome_model(dim):
    return {
        "beta": np.zeros(dim, dtype=np.float64),
        "lr_state": init_alignment_adaptive_state(
            dim=dim,
            eta_min=NUISANCE_ETA_MIN,
            eta_max=NUISANCE_ETA_MAX,
            beta=ADAPT_NUISANCE_BETA,
            power=ADAPT_NUISANCE_POWER,
        ),
    }

def predict_outcome_model(model, phi):
    pred = float(np.dot(phi, model["beta"]))
    return clip_scalar(pred, -M_BOUND, M_BOUND)

def update_outcome_model(model, phi, y):
    pred = predict_outcome_model(model, phi)
    grad = (pred - y) * phi
    eta = get_alignment_adaptive_eta(model["lr_state"], grad)
    model["beta"] = model["beta"] - eta * grad
    model["beta"] = project_l2_ball_vec(model["beta"], radius=OUTCOME_RADIUS)

# ============================================================
# 4) 按 segment 做 50% / 50% 随机划分
# ============================================================

def make_segment_train_test_split(df, rng):
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)
    train_idx_by_seg = {}
    test_idx_by_seg = {}
    seg_arr = df["segment"].to_numpy()

    for seg in sorted(df["segment"].unique().tolist()):
        seg_idx = np.where(seg_arr == seg)[0]
        if len(seg_idx) < 2:
            raise ValueError(f"segment={seg} 样本数小于 2，无法划分 train/test。")
        perm = rng.permutation(seg_idx)
        n_test = len(seg_idx) // 2
        n_test = max(1, min(n_test, len(seg_idx) - 1))
        test_idx = np.sort(perm[:n_test])
        train_idx = np.sort(perm[n_test:])
        train_idx_by_seg[int(seg)] = train_idx
        test_idx_by_seg[int(seg)] = test_idx

    return train_idx_by_seg, test_idx_by_seg

# ============================================================
# 5) 单次运行指定方法
# ============================================================

def run_one_method_once(df, surface, method, seed):
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)

    required_cols = {"w", "p", "y", "tau", "segment", "segment_round", "t"}
    missing = required_cols - set(df.columns)
    if len(missing) > 0:
        raise ValueError(f"数据缺少必要列: {sorted(missing)}")

    feature_cols = get_feature_columns(df)
    if len(feature_cols) == 0:
        raise ValueError("没有检测到协变量列，请检查 META_COLS。")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    w = df["w"].to_numpy(dtype=np.float64)
    p = np.clip(df["p"].to_numpy(dtype=np.float64), EPS, 1.0)
    y = df["y"].to_numpy(dtype=np.float64)
    tau_true = df["tau"].to_numpy(dtype=np.float64)
    q = get_q_array(df, w=w, p=p)

    rng_split = np.random.default_rng(seed)
    train_idx_by_seg, test_idx_by_seg = make_segment_train_test_split(df, rng_split)
    T_run = int(sum(len(v) for v in train_idx_by_seg.values()))

    feature_map = make_feature_map(surface=surface, input_dim=X.shape[1], seed=seed)
    Phi = apply_feature_map_batch(feature_map=feature_map, X=X)
    dim = Phi.shape[1]

    g_tilde = get_method_g_tilde(method)
    alpha_meta, N, etas = get_ethos_hyperparams(T_run=T_run, g_tilde=g_tilde)

    theta_mat = np.zeros((N, dim), dtype=np.float64)
    omega = np.ones(N, dtype=np.float64) / float(N)

    if is_dr_method(method):
        model_m0 = init_outcome_model(dim)
        model_m1 = init_outcome_model(dim)

    cum_true_half = 0.0

    for seg in sorted(train_idx_by_seg.keys()):
        train_idx = train_idx_by_seg[seg]
        test_idx = test_idx_by_seg[seg]

        Phi_test = Phi[test_idx]
        tau_test = tau_true[test_idx]

        for idx in train_idx:
            theta_meta = np.sum(theta_mat * omega[:, None], axis=0)
            test_preds = Phi_test @ theta_meta
            seg_mse = float(np.mean((test_preds - tau_test) ** 2))
            cum_true_half += 0.5 * seg_mse

            phi = Phi[idx]

            if is_dr_method(method):
                m0_hat = predict_outcome_model(model_m0, phi)
                m1_hat = predict_outcome_model(model_m1, phi)
                pseudo_tau = get_dr_pseudo_outcome(
                    w=float(w[idx]),
                    y=float(y[idx]),
                    q=float(q[idx]),
                    m0_hat=float(m0_hat),
                    m1_hat=float(m1_hat),
                )
            else:
                pseudo_tau = get_ipw_pseudo_outcome(
                    w=float(w[idx]),
                    y=float(y[idx]),
                    p=float(p[idx]),
                )

            base_preds = theta_mat @ phi
            base_sur_losses = 0.5 * (base_preds - pseudo_tau) ** 2
            omega = update_meta_weights(omega, alpha_meta, base_sur_losses)

            grad_mat = (base_preds - pseudo_tau)[:, None] * phi[None, :]
            theta_mat = theta_mat - etas[:, None] * grad_mat
            theta_mat = project_l2_ball_rows(theta_mat, radius=HTE_RADIUS)

            if is_dr_method(method):
                if float(w[idx]) == 1.0:
                    update_outcome_model(model_m1, phi, float(y[idx]))
                else:
                    update_outcome_model(model_m0, phi, float(y[idx]))

    used_params = {
        "T_run": T_run,
        "g_tilde": g_tilde,
        "alpha_meta": alpha_meta,
        "N": N,
        "eta_min": float(etas[0]),
        "eta_max": float(etas[-1]),
    }

    if is_dr_method(method):
        used_params["nuisance_beta"] = ADAPT_NUISANCE_BETA
        used_params["nuisance_power"] = ADAPT_NUISANCE_POWER
        used_params["nuisance_eta_min"] = NUISANCE_ETA_MIN
        used_params["nuisance_eta_max"] = NUISANCE_ETA_MAX

    return float(cum_true_half), used_params

# ============================================================
# 6) 输出格式化
# ============================================================

def format_metric(mean_val, se_val):
    return f"{mean_val:.6f} ± {se_val:.6f}"

def format_ipw_params(params):
    return (
        f"T={params['T_run']}, "
        f"g={params['g_tilde']:.6f}, "
        f"alpha={params['alpha_meta']:.6f}, "
        f"N={params['N']}, "
        f"eta_min={params['eta_min']:.6f}, "
        f"eta_max={params['eta_max']:.6f}"
    )

def format_dr_params(params):
    return (
        f"T={params['T_run']}, "
        f"g={params['g_tilde']:.6f}, "
        f"alpha={params['alpha_meta']:.6f}, "
        f"N={params['N']}, "
        f"eta_min={params['eta_min']:.6f}, "
        f"eta_max={params['eta_max']:.6f}, "
        f"nbeta={params['nuisance_beta']}, "
        f"npower={params['nuisance_power']}, "
        f"neta_min={params['nuisance_eta_min']}, "
        f"neta_max={params['nuisance_eta_max']}"
    )

def print_result_table(results):
    env_width = max(len("Environment"), max(len(r["env_name"]) for r in results))
    ipw_width = max(len("ETHOS_IPW"), max(len(r["ETHOS_IPW_metric"]) for r in results))
    dr_width = max(len("ETHOS_DR"), max(len(r["ETHOS_DR_metric"]) for r in results))

    line = "#" * (env_width + ipw_width + dr_width + 10)
    print(line)
    print(
        f"{'Environment':<{env_width}} | "
        f"{'ETHOS_IPW':<{ipw_width}} | "
        f"{'ETHOS_DR':<{dr_width}}"
    )
    print("-" * (env_width + ipw_width + dr_width + 6))

    for r in results:
        print(
            f"{r['env_name']:<{env_width}} | "
            f"{r['ETHOS_IPW_metric']:<{ipw_width}} | "
            f"{r['ETHOS_DR_metric']:<{dr_width}}"
        )
    print(line)

def print_param_table(results):
    print()
    print("Used Parameters")
    print("=" * 160)
    for r in results:
        print(f"Environment: {r['env_name']}")
        print(f"  ETHOS_IPW: {r['ETHOS_IPW_params']}")
        print(f"  ETHOS_DR : {r['ETHOS_DR_params']}")
        print("-" * 160)

# ============================================================
# 7) 主函数
# ============================================================

def main():
    env_files = discover_environment_files(DATA_DIR)

    final_results = []

    for env_info in env_files:
        env_name = env_info["filename"]
        surface = env_info["surface"]
        df = pd.read_csv(env_info["full_path"])

        row_result = {
            "env_name": env_name,
        }

        for method in METHODS_TO_RUN:
            metric_list = []
            used_params = None

            for r in range(N_REPEATS):
                seed = BASE_SEED + r
                metric, params = run_one_method_once(
                    df=df,
                    surface=surface,
                    method=method,
                    seed=seed,
                )
                metric_list.append(metric)
                used_params = params

            metric_mean, metric_se = mean_and_se(metric_list)

            if method == "ETHOS_IPW":
                row_result["ETHOS_IPW_metric"] = format_metric(metric_mean, metric_se)
                row_result["ETHOS_IPW_params"] = format_ipw_params(used_params)
            else:
                row_result["ETHOS_DR_metric"] = format_metric(metric_mean, metric_se)
                row_result["ETHOS_DR_params"] = format_dr_params(used_params)

        final_results.append(row_result)

    print_result_table(final_results)
    print_param_table(final_results)

if __name__ == "__main__":
    main()