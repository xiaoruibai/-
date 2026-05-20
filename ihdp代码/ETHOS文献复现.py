#ETHOS文献复现
# -*- coding: utf-8 -*-
"""
IHDP 上的 ETHOS 复现（ETHOS_IPW / ETHOS_DR）

核心要求：
1. 数据来源切换为 IHDP 原始协变量数据，而不是 twins。
2. 先按给定的 IHDP 半合成在线生成逻辑，构造 8 个 IHDP online environments。
3. ETHOS 算法核心保持和原 ETHOS 复现代码一致：
   - ETHOS_IPW
   - ETHOS_DR
   - 多学习率 base learners
   - 指数加权 meta aggregation
4. PEHE / 评估流程严格按“代码1”的口径执行：
   - 按 segment 做 50% / 50% train-test 随机划分
   - 每个训练步先在当前 segment 的 test split 上评估当前 theta
   - 累计:
       cum_true_half += 0.5 * seg_mse
       cum_squared_pehe += seg_mse
   - 最终:
       PEHE = sqrt(cum_squared_pehe)
5. 关键编码保持一致：
   - w ∈ {-1, 1}
   - p = P(W_t = w_t | X_t) = 观测到当前臂的概率
   - q = P(W_t = 1 | X_t)
"""

import os
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

# ============================================================
# 0) 基本设置
# ============================================================

DATA_PATH = "ihdp_data_1.csv"      # 原始 IHDP 文件
SAVE_STREAM_CSV = False            # 是否把生成的 8 个环境 csv 落盘
OUTDIR = "ihdp1_ethos_streams"      # 若 SAVE_STREAM_CSV=True，则保存到这里

N_REPEATS = 1
BASE_SEED = 2026

T = 10000
ENV_CONFIGS = [
    ("linear", "switching", 10),
    ("linear", "switching", 100),
    ("linear", "linear", 10),
    ("linear", "linear", 100),
    ("nonlinear", "switching", 10),
    ("nonlinear", "switching", 100),
    ("nonlinear", "linear", 10),
    ("nonlinear", "linear", 100),
]

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

# nuisance model 自适应学习率参数（ETHOS_DR 的 outcome model）
ADAPT_EPS = 1e-8
ADAPT_NUISANCE_BETA = 0.9
ADAPT_NUISANCE_POWER = 0.9
NUISANCE_ETA_MIN = 1e-3
NUISANCE_ETA_MAX = 1e-1

# IHDP 数据生成参数
NOISE_VAR = 0.1
NOISE_STD = float(np.sqrt(NOISE_VAR))
PROP_SUBSAMPLE = 2000

# 环境 DataFrame 中的非协变量列
META_COLS = {
    "env_name", "surface", "drift", "K", "segment", "segment_round",
    "t", "alpha", "source_row", "q", "w", "p", "y",
    "mu_-1", "mu_1", "tau",
}

# ============================================================
# 1) 基础工具函数
# ============================================================

def sigmoid_stable(z):
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def clip_prob_eps(q, eps):
    return np.clip(q, eps, 1.0 - eps)


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


def get_feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def get_method_g_tilde(method):
    if method.endswith("_IPW"):
        return float((H + B / EPS) * G)
    if method.endswith("_DR"):
        return float((H + 2.0 * M_BOUND + (B + M_BOUND) / EPS) * G)
    raise ValueError(f"未知方法: {method}")


# ============================================================
# 2) IHDP 协变量预处理（严格只取 x1~x25）
# ============================================================

def series_to_numeric_if_possible(s):
    s_num = pd.to_numeric(s, errors="coerce")
    if int(s_num.notna().sum()) == int(s.notna().sum()):
        return s_num
    return None


def is_binary_01_series(s):
    s_num = series_to_numeric_if_possible(s)
    if s_num is None:
        return False
    non_na = s_num.dropna().astype(float)
    if len(non_na) == 0:
        return False
    uniq = set(np.unique(non_na).tolist())
    return uniq.issubset({0.0, 1.0})


def is_binary_12_series(s):
    s_num = series_to_numeric_if_possible(s)
    if s_num is None:
        return False
    non_na = s_num.dropna().astype(float)
    if len(non_na) == 0:
        return False
    uniq = set(np.unique(non_na).tolist())
    return uniq.issubset({1.0, 2.0})


def parse_x_index(col_name):
    """
    支持:
      x1, x2, ..., x25
      x_1, x_2, ..., x_25
    """
    s = str(col_name).strip().lower()
    m = re.fullmatch(r"x_?(\d+)", s)
    if m is None:
        return None
    return int(m.group(1))


def pick_covariate_columns(df):
    index_to_raw = {}
    for c in df.columns:
        idx = parse_x_index(c)
        if idx is not None and 1 <= idx <= 25:
            if idx in index_to_raw:
                raise ValueError(
                    f"检测到重复协变量列映射到 x{idx}: "
                    f"{index_to_raw[idx]} 和 {c}"
                )
            index_to_raw[idx] = c

    missing = [i for i in range(1, 26) if i not in index_to_raw]
    if len(missing) > 0:
        raise ValueError(
            f"IHDP 数据中缺少协变量列: {missing}。"
            f"要求完整存在 x1~x25 或 x_1~x_25。"
        )

    return [index_to_raw[i] for i in range(1, 26)]


def preprocess_covariates(df):
    cov_cols = pick_covariate_columns(df)
    Xraw = df[cov_cols].copy()

    binary_01_cols = []
    binary_12_cols = []
    numeric_cols = []

    for c in Xraw.columns:
        s = Xraw[c]

        if is_binary_01_series(s):
            binary_01_cols.append(c)
            continue

        if is_binary_12_series(s):
            binary_12_cols.append(c)
            continue

        if pd.api.types.is_numeric_dtype(s) or (series_to_numeric_if_possible(s) is not None):
            numeric_cols.append(c)
            continue

        raise ValueError(
            f"协变量列 {c} 既不是数值列，也不是 0/1 或 1/2 二元列。"
            f"为保证维度固定为 25，当前实现不会做 one-hot。"
        )

    Xdf = pd.DataFrame(index=Xraw.index)

    for c in cov_cols:
        s = Xraw[c]

        if c in binary_01_cols:
            sc = pd.to_numeric(s, errors="coerce").astype(float)
            sc = sc.fillna(0.0)
            Xdf[c] = sc

        elif c in binary_12_cols:
            sc = pd.to_numeric(s, errors="coerce").astype(float)
            sc = sc.replace({1.0: 0.0, 2.0: 1.0})
            sc = sc.fillna(0.0)
            Xdf[c] = sc

        elif c in numeric_cols:
            sc = pd.to_numeric(s, errors="coerce").astype(float)
            med = sc.median()
            if pd.isna(med):
                med = 0.0
            sc = sc.fillna(med)
            Xdf[c] = sc

        else:
            raise RuntimeError(f"内部错误：列 {c} 未被分类。")

    Xdf = Xdf[cov_cols].copy()
    X = Xdf.to_numpy(dtype=float)
    X = row_l2_normalize_leq1(X)

    if X.shape[1] != 25:
        raise ValueError(f"最终协变量维度不是 25，而是 {X.shape[1]}。")

    return X, list(Xdf.columns)


# ============================================================
# 3) IHDP 半合成在线环境生成
# ============================================================

def sample_beta_linear(rng, d):
    vals = np.array([0, 1, 2, 3, 4], dtype=float)
    probs = np.array([0.5, 0.2, 0.15, 0.1, 0.05], dtype=float)
    beta = rng.choice(vals, size=d, p=probs).astype(float)
    if np.linalg.norm(beta) < 1e-12:
        beta[rng.integers(0, d)] = 1.0
    return beta


def sample_beta_nonlinear(rng, d):
    vals = np.array([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    probs = np.array([0.6, 0.1, 0.1, 0.1, 0.1], dtype=float)
    beta = rng.choice(vals, size=d, p=probs).astype(float)
    if np.linalg.norm(beta) < 1e-12:
        beta[rng.integers(0, d)] = 0.1
    return beta


def fit_propensity_lr(rng, X_all):
    n, d = X_all.shape
    m = min(PROP_SUBSAMPLE, n)
    idx = rng.choice(n, size=m, replace=False)
    X_sub = X_all[idx]

    gamma = rng.normal(loc=0.0, scale=1.0, size=d)
    prob = sigmoid_stable(X_sub @ gamma)
    y_sub = (rng.random(m) < prob).astype(int)

    if int(y_sub.min()) == int(y_sub.max()):
        y_sub[: m // 2] = 0
        y_sub[m // 2:] = 1
        rng.shuffle(y_sub)

    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    clf.fit(X_sub, y_sub)

    coef = clf.coef_.reshape(-1).astype(float)
    intercept = float(clf.intercept_.reshape(-1)[0])
    return {"coef": coef, "intercept": intercept}


def predict_propensity(model, X):
    return sigmoid_stable(X @ model["coef"] + model["intercept"])


def make_linear_env(rng, X_all):
    d = X_all.shape[1]
    beta1 = sample_beta_linear(rng, d)
    beta2 = sample_beta_linear(rng, d)
    rho = fit_propensity_lr(rng, X_all)
    return {"beta1": beta1, "beta2": beta2, "rho": rho}


def make_nonlinear_env(rng, X_all):
    d = X_all.shape[1]
    beta3 = sample_beta_nonlinear(rng, d)
    rho = fit_propensity_lr(rng, X_all)
    return {"beta3": beta3, "rho": rho}


def mu_linear(X, beta1, beta2):
    n1 = np.linalg.norm(beta1) + 1e-12
    n2 = np.linalg.norm(beta2) + 1e-12
    mu_m1 = (X @ beta1) / n1
    mu_p1 = (X @ beta2) / n2
    return mu_m1, mu_p1


def mu_nonlinear(X, beta3):
    nb = np.linalg.norm(beta3) + 1e-12
    mu_m1 = np.exp((X + 0.05) @ beta3) / np.exp(nb)
    mu_p1 = (X @ beta3) / nb
    return mu_m1, mu_p1


def alpha_schedule(L):
    pos = np.arange(L, dtype=float)
    alpha = (pos - 0.5 * L) / (0.5 * L)
    alpha = np.maximum(0.0, alpha)
    alpha = np.minimum(1.0, alpha)
    return alpha


def sample_online_indices_from_left_half(rng, n, L):
    perm = rng.permutation(n)
    n_test = n // 2
    test_idx = perm[:n_test]
    pool_idx = perm[n_test:]
    replace = len(pool_idx) < L
    online_idx = rng.choice(pool_idx, size=L, replace=replace)
    return online_idx, test_idx


def generate_one_stream(rng, X_all, feature_names, surface, drift, K):
    if T % K != 0:
        raise ValueError(f"T={T} 不能被 K={K} 整除。")

    L = T // K
    n = X_all.shape[0]
    rows = []
    t_global = 0

    if surface == "linear":
        env_count = K if drift == "switching" else (K + 1)
        envs = [make_linear_env(rng, X_all) for _ in range(env_count)]
    elif surface == "nonlinear":
        env_count = K if drift == "switching" else (K + 1)
        envs = [make_nonlinear_env(rng, X_all) for _ in range(env_count)]
    else:
        raise ValueError("surface 只能是 'linear' 或 'nonlinear'。")

    for seg in range(K):
        online_idx, _ = sample_online_indices_from_left_half(rng, n, L)
        X_seg = X_all[online_idx]
        alpha = alpha_schedule(L)

        if surface == "linear" and drift == "switching":
            env = envs[seg]
            mu_m1, mu_p1 = mu_linear(X_seg, env["beta1"], env["beta2"])
            q = predict_propensity(env["rho"], X_seg)

        elif surface == "linear" and drift == "linear":
            env_a = envs[seg]
            env_b = envs[seg + 1]

            mu_m1_a, mu_p1_a = mu_linear(X_seg, env_a["beta1"], env_a["beta2"])
            mu_m1_b, mu_p1_b = mu_linear(X_seg, env_b["beta1"], env_b["beta2"])

            q_a = predict_propensity(env_a["rho"], X_seg)
            q_b = predict_propensity(env_b["rho"], X_seg)

            mu_m1 = (1.0 - alpha) * mu_m1_a + alpha * mu_m1_b
            mu_p1 = (1.0 - alpha) * mu_p1_a + alpha * mu_p1_b
            q = (1.0 - alpha) * q_a + alpha * q_b

        elif surface == "nonlinear" and drift == "switching":
            env = envs[seg]
            mu_m1, mu_p1 = mu_nonlinear(X_seg, env["beta3"])
            q = predict_propensity(env["rho"], X_seg)

        elif surface == "nonlinear" and drift == "linear":
            env_a = envs[seg]
            env_b = envs[seg + 1]

            mu_m1_a, mu_p1_a = mu_nonlinear(X_seg, env_a["beta3"])
            mu_m1_b, mu_p1_b = mu_nonlinear(X_seg, env_b["beta3"])

            q_a = predict_propensity(env_a["rho"], X_seg)
            q_b = predict_propensity(env_b["rho"], X_seg)

            mu_m1 = (1.0 - alpha) * mu_m1_a + alpha * mu_m1_b
            mu_p1 = (1.0 - alpha) * mu_p1_a + alpha * mu_p1_b
            q = (1.0 - alpha) * q_a + alpha * q_b

        else:
            raise ValueError("drift 只能是 'switching' 或 'linear'。")

        q = clip_prob_eps(q, EPS)

        # 保持 ETHOS 论文口径：w ∈ {-1, 1}
        w = np.where(rng.random(L) < q, 1, -1)

        # p = P(W_t = w_t | X_t) = 当前被观测到这条臂的概率
        p = np.where(w == 1, q, 1.0 - q)

        y_m1 = mu_m1 + NOISE_STD * rng.normal(size=L)
        y_p1 = mu_p1 + NOISE_STD * rng.normal(size=L)
        y = np.where(w == 1, y_p1, y_m1)
        tau = mu_p1 - mu_m1

        for j in range(L):
            t_global += 1
            row = {
                "env_name": f"ihdp_{surface}_{drift}_K{K}",
                "surface": surface,
                "drift": drift,
                "K": int(K),
                "segment": int(seg),
                "segment_round": int(j),
                "t": int(t_global),
                "alpha": float(alpha[j] if drift == "linear" else 0.0),
                "source_row": int(online_idx[j]),
                "q": float(q[j]),
                "w": int(w[j]),
                "p": float(p[j]),
                "y": float(y[j]),
                "mu_-1": float(mu_m1[j]),
                "mu_1": float(mu_p1[j]),
                "tau": float(tau[j]),
            }

            xj = X_seg[j]
            for k, fn in enumerate(feature_names):
                row[fn] = float(xj[k])

            rows.append(row)

    return pd.DataFrame(rows)


def build_ihdp_environments():
    df_raw = pd.read_csv(DATA_PATH)
    X_all, feature_names = preprocess_covariates(df_raw)

    if SAVE_STREAM_CSV:
        os.makedirs(OUTDIR, exist_ok=True)

    env_list = []
    for idx, (surface, drift, K) in enumerate(ENV_CONFIGS):
        seed = BASE_SEED + 1000 * idx
        rng = np.random.default_rng(seed)

        df_stream = generate_one_stream(
            rng=rng,
            X_all=X_all,
            feature_names=feature_names,
            surface=surface,
            drift=drift,
            K=K,
        )

        env_name = f"ihdp_{surface}_{drift}_K{K}"
        env_list.append({
            "env_name": env_name,
            "surface": surface,
            "drift": drift,
            "K": K,
            "seed": seed,
            "df": df_stream,
        })

        if SAVE_STREAM_CSV:
            fpath = os.path.join(OUTDIR, f"{env_name}.csv")
            df_stream.to_csv(fpath, index=False)
            print(f"[OK] 已生成: {fpath} | seed={seed} | rows={len(df_stream)}")

    return env_list


# ============================================================
# 4) 固定随机特征映射（与原 ETHOS 复现一致）
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
# 5) ETHOS 超参数、伪结果、outcome model
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
    """
    ETHOS / bandit CATE 口径：
      w ∈ {-1, 1}
      p = P(W = w | X)
      pseudo_tau = w * y / p
    """
    return float(w * y / p)


def get_dr_pseudo_outcome(w, y, q, m0_hat, m1_hat):
    """
    q = P(W = 1 | X)
    w = 1  时: (m1 - m0) + (y - m1) / q
    w = -1 时: (m1 - m0) - (y - m0) / (1 - q)
    """
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
# 6) 按 segment 做 50% / 50% 随机划分（与代码1一致）
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
# 7) 单次运行指定方法（PEHE 流程严格对齐代码1）
# ============================================================

def run_one_method_once(df, surface, method, seed):
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)

    required_cols = {"w", "p", "q", "y", "tau", "segment", "segment_round", "t"}
    missing = required_cols - set(df.columns)
    if len(missing) > 0:
        raise ValueError(f"数据缺少必要列: {sorted(missing)}")

    feature_cols = get_feature_columns(df)
    if len(feature_cols) == 0:
        raise ValueError("没有检测到协变量列，请检查 META_COLS。")

    X = df[feature_cols].to_numpy(dtype=np.float32)
    w = df["w"].to_numpy(dtype=np.float64)
    p = np.clip(df["p"].to_numpy(dtype=np.float64), EPS, 1.0)
    q = clip_prob_eps(df["q"].to_numpy(dtype=np.float64), EPS)
    y = df["y"].to_numpy(dtype=np.float64)
    tau_true = df["tau"].to_numpy(dtype=np.float64)

    uniq_w = set(np.unique(w).tolist())
    if not uniq_w.issubset({-1.0, 1.0}):
        raise ValueError(f"w 必须编码为 {{-1,1}}，当前检测到: {sorted(uniq_w)}")

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

    # ===== 与代码1保持一致的评估累计 =====
    cum_true_half = 0.0
    cum_squared_pehe = 0.0

    for seg in sorted(train_idx_by_seg.keys()):
        train_idx = train_idx_by_seg[seg]
        test_idx = test_idx_by_seg[seg]

        Phi_test = Phi[test_idx]
        tau_test = tau_true[test_idx]

        for idx in train_idx:
            # 先评估，后更新
            theta_meta = np.sum(theta_mat * omega[:, None], axis=0)
            test_preds = Phi_test @ theta_meta
            seg_mse = float(np.mean((test_preds - tau_test) ** 2))

            cum_true_half += 0.5 * seg_mse
            cum_squared_pehe += seg_mse

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

    final_pehe = float(np.sqrt(cum_squared_pehe))

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

    metrics = {
        "half_loss": float(cum_true_half),
        "squared_pehe": float(cum_squared_pehe),
        "pehe": final_pehe,
    }
    return metrics, used_params


# ============================================================
# 8) 输出格式化
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


def print_half_loss_table(results):
    env_width = max(len("Environment"), max(len(r["env_name"]) for r in results))
    ipw_width = max(len("ETHOS_IPW"), max(len(r["ETHOS_IPW_half_loss"]) for r in results))
    dr_width = max(len("ETHOS_DR"), max(len(r["ETHOS_DR_half_loss"]) for r in results))

    line = "#" * (env_width + ipw_width + dr_width + 10)
    print("\nCumulative 0.5 * squared PEHE")
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
            f"{r['ETHOS_IPW_half_loss']:<{ipw_width}} | "
            f"{r['ETHOS_DR_half_loss']:<{dr_width}}"
        )
    print(line)


def print_pehe_table(results):
    env_width = max(len("Environment"), max(len(r["env_name"]) for r in results))
    ipw_width = max(len("ETHOS_IPW"), max(len(r["ETHOS_IPW_pehe"]) for r in results))
    dr_width = max(len("ETHOS_DR"), max(len(r["ETHOS_DR_pehe"]) for r in results))

    line = "#" * (env_width + ipw_width + dr_width + 10)
    print("\nCumulative PEHE = sqrt(sum_t mse_t)")
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
            f"{r['ETHOS_IPW_pehe']:<{ipw_width}} | "
            f"{r['ETHOS_DR_pehe']:<{dr_width}}"
        )
    print(line)


def print_param_table(results):
    print("\nUsed Parameters")
    print("=" * 160)
    for r in results:
        print(f"Environment: {r['env_name']}")
        print(f"  ETHOS_IPW: {r['ETHOS_IPW_params']}")
        print(f"  ETHOS_DR : {r['ETHOS_DR_params']}")
        print("-" * 160)


# ============================================================
# 9) 主函数
# ============================================================

def main():
    env_list = build_ihdp_environments()
    final_results = []

    for env_info in env_list:
        env_name = env_info["env_name"]
        surface = env_info["surface"]
        df = env_info["df"]

        row_result = {
            "env_name": env_name,
        }

        for method in METHODS_TO_RUN:
            half_loss_list = []
            pehe_list = []
            used_params = None

            for r in range(N_REPEATS):
                seed = BASE_SEED + r
                metrics, params = run_one_method_once(
                    df=df,
                    surface=surface,
                    method=method,
                    seed=seed,
                )
                half_loss_list.append(metrics["half_loss"])
                pehe_list.append(metrics["pehe"])
                used_params = params

            half_mean, half_se = mean_and_se(half_loss_list)
            pehe_mean, pehe_se = mean_and_se(pehe_list)

            if method == "ETHOS_IPW":
                row_result["ETHOS_IPW_half_loss"] = format_metric(half_mean, half_se)
                row_result["ETHOS_IPW_pehe"] = format_metric(pehe_mean, pehe_se)
                row_result["ETHOS_IPW_params"] = format_ipw_params(used_params)
            else:
                row_result["ETHOS_DR_half_loss"] = format_metric(half_mean, half_se)
                row_result["ETHOS_DR_pehe"] = format_metric(pehe_mean, pehe_se)
                row_result["ETHOS_DR_params"] = format_dr_params(used_params)

        final_results.append(row_result)

    print_half_loss_table(final_results)
    print_pehe_table(final_results)
    print_param_table(final_results)


if __name__ == "__main__":
    main()