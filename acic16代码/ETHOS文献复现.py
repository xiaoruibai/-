# -*- coding: utf-8 -*-

"""
ETHOS 复现代码（在现有 ACIC 环境 csv 上按 segment 内部 50%/50% 随机划分 train/test）

说明：
1. 直接读取现有环境 csv，不修改原始数据文件。
2. 对每个 segment 内部做 50% / 50% 随机划分：
   - 一半作为在线训练流
   - 一半作为该 segment 的测试集
3. 每轮先在该 segment 的测试集上评估当前模型，
   再读取一个训练样本并做 ETHOS 更新。
4. 最终只输出：
   ETHOS cumulative 0.5*squared error
"""

import os
import re
import numpy as np
import pandas as pd


# ============================================================
# 0) 路径与实验设置
# ============================================================

TARGET_ENV = "acic2016_linear_linear_K10.csv"
DATA_DIR = "acic2016_ethos_streams"

N_REPEATS = 1
BASE_SEED = 2026

D = 1.0
G = 1.0
H = 1.0
B = 1.0
EPS = 0.05

G_TILDE = (H + B / EPS) * G

META_COLS = {
    "env_name",
    "surface",
    "drift",
    "K",
    "segment",
    "segment_round",
    "t",
    "alpha",
    "source_row",
    "q",
    "w",
    "p",
    "y",
    "mu_-1",
    "mu_1",
    "tau",
}


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


def parse_env_name(path):
    name = os.path.basename(path)
    m = re.match(r"^acic2016_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv$", name)
    if m is None:
        raise ValueError("TARGET_ENV 文件名不符合约定格式。")
    surface = m.group(1)
    return surface


def get_feature_columns(df):
    return [c for c in df.columns if c not in META_COLS]


def mean_and_se(x):
    x = np.asarray(x, dtype=np.float64)
    if len(x) <= 1:
        return float(np.mean(x)), 0.0
    mean = float(np.mean(x))
    std = float(np.std(x, ddof=1))
    se = std / np.sqrt(len(x))
    return mean, float(se)


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
        b = rng.normal(loc=0.0, scale=0.01, size=(dout,)).astype(np.float32)
        weights.append(W)
        biases.append(b)

    return {"weights": weights, "biases": biases}


def mlp_transform_batch(mlp, X):
    Hx = np.asarray(X, dtype=np.float32)

    for layer_idx, (W, b) in enumerate(zip(mlp["weights"], mlp["biases"])):
        Hx = Hx @ W + b
        if layer_idx < len(mlp["weights"]) - 1:
            Hx = np.maximum(Hx, 0.0)

    Hx = Hx.astype(np.float64)
    Hx = row_l2_normalize_leq1(Hx)
    return Hx


def build_feature_matrix(X, surface, seed):
    if surface == "linear":
        Phi = np.concatenate(
            [X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)],
            axis=1,
        )
        Phi = row_l2_normalize_leq1(Phi)
        return Phi

    if surface == "nonlinear":
        mlp = init_random_mlp(input_dim=X.shape[1], seed=seed)
        Phi = mlp_transform_batch(mlp, X)
        return Phi

    raise ValueError("surface 只能是 'linear' 或 'nonlinear'。")


# ============================================================
# 3) ETHOS 超参数
# ============================================================

def get_ethos_hyperparams(T_run):
    alpha_meta = np.sqrt(8.0 / (T_run * (G_TILDE ** 2) * (D ** 2)))
    N = int(np.floor(0.5 * np.log2(1.0 + 4.0 * T_run / 7.0))) + 1

    etas = []
    for i in range(1, N + 1):
        eta_i = (2.0 ** (i - 1)) * D / G_TILDE * np.sqrt(7.0 / (2.0 * T_run))
        etas.append(float(eta_i))

    return float(alpha_meta), int(N), np.asarray(etas, dtype=np.float64)


def update_meta_weights(omega, alpha_meta, base_sur_losses):
    log_w = np.log(omega + 1e-300)
    log_w = log_w - alpha_meta * base_sur_losses
    log_w = log_w - np.max(log_w)
    new_w = np.exp(log_w)
    new_w = new_w / np.sum(new_w)
    return new_w


# ============================================================
# 4) 按 segment 做 50% / 50% 随机划分
# ============================================================

def make_segment_train_test_split(df, rng):
    """
    对每个 segment 内部做 50% / 50% 随机划分。
    返回：
      train_idx_by_seg: dict[seg] -> 训练索引（按原时间顺序排列）
      test_idx_by_seg:  dict[seg] -> 测试索引
    """
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)

    train_idx_by_seg = {}
    test_idx_by_seg = {}

    for seg in sorted(df["segment"].unique().tolist()):
        seg_idx = np.where(df["segment"].to_numpy() == seg)[0]
        perm = rng.permutation(seg_idx)

        n_test = len(seg_idx) // 2
        test_idx = perm[:n_test]
        train_idx = perm[n_test:]

        # 训练流保持该 segment 内原有时间顺序
        train_idx = np.sort(train_idx)

        train_idx_by_seg[int(seg)] = train_idx
        test_idx_by_seg[int(seg)] = test_idx

    return train_idx_by_seg, test_idx_by_seg


# ============================================================
# 5) 单次运行
# ============================================================

def run_ethos_once(df, surface, seed):
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)

    w = df["w"].to_numpy(dtype=np.float64)
    p = df["p"].to_numpy(dtype=np.float64)
    y = df["y"].to_numpy(dtype=np.float64)
    tau_true = df["tau"].to_numpy(dtype=np.float64)

    # 先做按 segment 的 50/50 划分
    rng_split = np.random.default_rng(seed)
    train_idx_by_seg, test_idx_by_seg = make_segment_train_test_split(df, rng_split)

    # 实际在线训练轮数 = 所有 segment 的 train 数量之和
    T_run = int(sum(len(v) for v in train_idx_by_seg.values()))

    # 固定特征映射一次性预计算
    Phi = build_feature_matrix(X=X, surface=surface, seed=seed)
    dim = Phi.shape[1]

    alpha_meta, N, etas = get_ethos_hyperparams(T_run)

    # 第 i 行对应第 i 个 base 的参数
    theta_mat = np.zeros((N, dim), dtype=np.float64)

    # 元权重初始化
    omega = np.ones(N, dtype=np.float64) / float(N)

    # 只累计论文口径的 0.5 * squared error
    cum_true_half_meta = 0.0

    radius = D / 2.0

    # 按 segment 依次进行在线学习
    for seg in sorted(train_idx_by_seg.keys()):
        train_idx = train_idx_by_seg[seg]
        test_idx = test_idx_by_seg[seg]

        Phi_test = Phi[test_idx]
        tau_test = tau_true[test_idx]

        # 在当前 segment 的训练流上逐轮在线更新
        for idx in train_idx:
            # 先用当前模型在该 segment 的 test set 上评估
            theta_meta = np.sum(theta_mat * omega[:, None], axis=0)
            test_preds = Phi_test @ theta_meta
            seg_mse = float(np.mean((test_preds - tau_test) ** 2))
            cum_true_half_meta += 0.5 * seg_mse

            # 再读取当前训练样本并更新
            phi = Phi[idx]
            pseudo_tau = float(w[idx] * y[idx] / p[idx])

            base_preds = theta_mat @ phi
            base_sur_losses = 0.5 * (base_preds - pseudo_tau) ** 2

            # 先更新元权重
            omega = update_meta_weights(omega, alpha_meta, base_sur_losses)

            # 再更新所有 base
            grad_mat = (base_preds - pseudo_tau)[:, None] * phi[None, :]
            theta_mat = theta_mat - etas[:, None] * grad_mat
            theta_mat = project_l2_ball_rows(theta_mat, radius=radius)

    return float(cum_true_half_meta)


# ============================================================
# 6) 主函数
# ============================================================

def main():
    env_path = os.path.join(DATA_DIR, TARGET_ENV)

    if not os.path.exists(env_path):
        raise FileNotFoundError(f"找不到环境文件: {env_path}")

    surface = parse_env_name(TARGET_ENV)
    df = pd.read_csv(env_path)

    metric_list = []

    for r in range(N_REPEATS):
        seed = BASE_SEED + r
        metric = run_ethos_once(df=df, surface=surface, seed=seed)
        metric_list.append(metric)

    metric_mean, metric_se = mean_and_se(metric_list)

    print(f"ETHOS cumulative 0.5*squared error: {metric_mean:.6f} ± {metric_se:.6f}")


if __name__ == "__main__":
    main()