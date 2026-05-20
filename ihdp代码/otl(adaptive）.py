# -*- coding: utf-8 -*-
"""
IHDP online OTL runner
(Route B: relative pseudo-loss, independent repeats, adaptive source + adaptive transfer)

核心原则
1) 只移除原有静态迁移(static transfer)；保留目标端核心算法、数据读取、特征映射、评估与重复实验框架。
2) 迁移层保留 Zhao & Hoi (ICML 2010) homogeneous OTL 的两专家指数加权骨架：
   - 固定 old/source predictor h
   - 在线学习 new/target predictor f
   - 两专家初始权重 w_old = w_new = 1/2
   - 每步使用相对 pseudo-loss 进行指数加权更新
   - 固定当前设定的 OTL_META_ETA，不做网格搜索/工程化修补
3) 因果推断中没有直接真标签，因此：
   - IPW 使用 IPW pseudo-outcome 构造瞬时 pseudo-loss
   - DR  使用 DR pseudo-outcome 构造瞬时 pseudo-loss
4) 源域 old/source predictor 不再使用 ETHOS 拟合，而改为：
   - ADAPTIVE_IPW source anchor
   - ADAPTIVE_DR source anchor
5) ETHOS 只保留在目标域上作为 target-only comparator：
   - ETHOS_IPW
   - ETHOS_DR
6) 只保留 adaptive transfer 方法：
   - OTL_ADAPTIVE_IPW_FROM_ADAPTIVE_IPW
   - OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR

推荐：
python ihdp_otl_exact_adaptive_indep_runner.py --n-repeats 10 --n-jobs 1
"""

from __future__ import annotations

import os
import re
import math
import json
import argparse
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# =========================================================
# Global configuration
# =========================================================

BASE_SEED = 2026
N_REPEATS = 10
DEFAULT_N_JOBS = 1
DEFAULT_OUTPUT_DIR = "ihdp_otl_routeb_relative_loss_outputs"
DEFAULT_SMOOTH_WINDOW = 200
DEFAULT_HIDDEN_DIM = 256
DEFAULT_BATCH_SIZE = 256

EPS = 0.05
EPS_NUM = 1e-8
D = 1.0
G = 1.0
H = 1.0
B = 1.0
M_BOUND = 1.0

HTE_RADIUS = D / 2.0
OUTCOME_RADIUS = 1.0
THEORY_C_ETA = 1.0 / math.sqrt(2.0)

# Route-B OTL meta layer
OTL_META_ETA = 0.075
OTL_INITIAL_WEIGHT = 0.5
OTL_LOSS_VARIANT = "relative_two_expert_squared"

META_COLS = {
    "env_name", "surface", "drift", "K",
    "repeat", "seed",
    "segment", "segment_round", "t",
    "alpha", "source_row", "q",
    "w", "p", "y", "mu_-1", "mu_1", "tau",
}

REPEAT_ENV_FILE_REGEX = re.compile(
    r"^(ihdp_(linear|nonlinear)_(switching|linear)_K(\d+))_repeat(\d+)\.csv$"
)
LEGACY_ENV_FILE_REGEX = re.compile(
    r"^(ihdp_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv)$"
)

COMPARISON_SPECS = [
    {
        "comparison_label": "ADAPTIVE_IPW <- ADAPTIVE_IPW (Route B OTL)",
        "target_method": "ADAPTIVE_IPW",
        "source_method": "ADAPTIVE_IPW",
        "baseline_method": "ADAPTIVE_IPW",
        "transfer_method": "OTL_ADAPTIVE_IPW_FROM_ADAPTIVE_IPW",
        "family": "IPW",
    },
    {
        "comparison_label": "ADAPTIVE_DR <- ADAPTIVE_DR (Route B OTL)",
        "target_method": "ADAPTIVE_DR",
        "source_method": "ADAPTIVE_DR",
        "baseline_method": "ADAPTIVE_DR",
        "transfer_method": "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR",
        "family": "DR",
    },
]

METHOD_ORDER = [
    "ADAPTIVE_IPW",
    "ADAPTIVE_DR",
    "ETHOS_IPW",
    "ETHOS_DR",
    "OTL_ADAPTIVE_IPW_FROM_ADAPTIVE_IPW",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR",
]


# =========================================================
# Small utilities
# =========================================================


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def resolve_data_dir(candidates: List[str]) -> str:
    for path in candidates:
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(f"Cannot find data directory from candidates: {candidates}")


def effective_n_jobs(requested_n_jobs: int) -> int:
    cpu = os.cpu_count() or 1
    return max(1, min(int(requested_n_jobs), cpu))


def safe_slug(text: str) -> str:
    text = re.sub(r"[<>:\"/\\|?*]+", "_", str(text).strip())
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_").lower()


def json_default_safe(obj: Any):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return str(obj)


def concat_nonempty(frames: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in frames if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    all_cols = sorted(set().union(*(df.columns for df in frames)))
    return pd.concat([df.reindex(columns=all_cols) for df in frames], ignore_index=True)


def row_l2_normalize_leq1(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    return (X / np.maximum(1.0, norms + 1e-12)).astype(np.float32)


def project_l2_ball_vec(theta: np.ndarray, radius: float) -> np.ndarray:
    norm = float(np.linalg.norm(theta))
    if norm <= radius:
        return theta.astype(np.float32, copy=False)
    return (theta * (radius / (norm + 1e-12))).astype(np.float32)


def project_l2_ball_rows(theta: np.ndarray, radius: float) -> np.ndarray:
    norms = np.linalg.norm(theta, axis=1, keepdims=True)
    scale = np.minimum(1.0, radius / (norms + 1e-12))
    return (theta * scale).astype(np.float32)


def family_from_method(method: str) -> str:
    return "IPW" if method.endswith("IPW") else "DR"


def theoretical_c_eta() -> float:
    return THEORY_C_ETA


def get_scale_free_eta(acc_sq: float, radius: float, c_eta: float = THEORY_C_ETA) -> float:
    return float(c_eta * radius / math.sqrt(acc_sq + EPS_NUM))


def get_tau_abs_bound(method: str) -> float:
    if method.endswith("IPW"):
        return float(B / EPS)
    return float(2.0 * M_BOUND + (B + M_BOUND) / EPS)


def get_method_g_tilde(method: str) -> float:
    return float((H + get_tau_abs_bound(method)) * G)


# =========================================================
# Data loading
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IHDP Route-B OTL runner aligned to independent-repeat data")
    parser.add_argument("--target-data-dir", type=str, default=None)
    parser.add_argument("--good-source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)
    parser.add_argument("--smooth-window", type=int, default=DEFAULT_SMOOTH_WINDOW)
    parser.add_argument("--showcase-env", type=str, default=None)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--feature-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-linear", action="store_true")
    parser.add_argument("--disable-plots", action="store_true")
    parser.add_argument("--save-all-trajectories", action="store_true")
    return parser.parse_args()


def discover_environment_files(data_dir: str) -> List[Dict[str, object]]:
    out: List[Dict[str, object]] = []
    legacy_hits: List[str] = []

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue

        m = REPEAT_ENV_FILE_REGEX.match(fname)
        if m:
            out.append({
                "dataset": m.group(1),
                "surface": m.group(2),
                "drift": m.group(3),
                "K": int(m.group(4)),
                "repeat": int(m.group(5)),
                "filename": fname,
                "env_name": m.group(1),
                "env_repeat_name": fname[:-4],
                "full_path": fpath,
            })
            continue

        if LEGACY_ENV_FILE_REGEX.match(fname):
            legacy_hits.append(fname)

    if out:
        out.sort(key=lambda x: (x["surface"], x["drift"], x["K"], x["repeat"], x["filename"]))
        return out

    if legacy_hits:
        raise FileNotFoundError(
            "Found legacy IHDP environment files without repeat suffix, but no independent-repeat files. "
            "Expected files like ihdp_linear_switching_K5_repeat01.csv. "
            f"Examples found instead: {sorted(legacy_hits)[:5]}"
        )

    raise FileNotFoundError(f"No IHDP independent-repeat environment files found in {data_dir}")


def validate_repeat_inventory(env_files: List[Dict[str, object]], expected_n_repeats: int) -> Dict[str, List[int]]:
    repeat_map: Dict[str, List[int]] = defaultdict(list)
    for info in env_files:
        repeat_map[str(info["env_name"])].append(int(info["repeat"]))

    problems: List[str] = []
    for env_name, repeats in sorted(repeat_map.items()):
        repeats_sorted = sorted(repeats)
        if len(repeats_sorted) != expected_n_repeats:
            problems.append(
                f"{env_name}: found {len(repeats_sorted)} repeats {repeats_sorted}, expected {expected_n_repeats}"
            )
        elif repeats_sorted != list(range(1, expected_n_repeats + 1)):
            problems.append(
                f"{env_name}: repeats are {repeats_sorted}, expected contiguous 1..{expected_n_repeats}"
            )

    if problems:
        raise ValueError("Independent-repeat inventory check failed.\n" + "\n".join(problems))

    return {k: sorted(v) for k, v in repeat_map.items()}


def unique_base_envs(env_files: List[Dict[str, object]]) -> List[Dict[str, object]]:
    seen = set()
    out: List[Dict[str, object]] = []
    for info in env_files:
        key = str(info["env_name"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "dataset": info["dataset"],
            "surface": info["surface"],
            "drift": info["drift"],
            "K": info["K"],
            "env_name": info["env_name"],
        })
    out.sort(key=lambda x: (x["surface"], x["drift"], x["K"], x["env_name"]))
    return out


def resolve_source_path(source_dir: str, target_env_info: Dict[str, object], tag: str = "good") -> str:
    target_filename = str(target_env_info["filename"])
    env_name = str(target_env_info["env_name"])
    repeat = int(target_env_info["repeat"])

    exact_path = os.path.join(source_dir, target_filename)
    if os.path.exists(exact_path):
        return exact_path

    candidates = [
        os.path.join(source_dir, f"{env_name}_repeat{repeat:02d}({tag}).csv"),
        os.path.join(source_dir, f"{env_name}_repeat{repeat:02d}（{tag}）.csv"),
        os.path.join(source_dir, f"{env_name}({tag}).csv"),
        os.path.join(source_dir, f"{env_name}（{tag}）.csv"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path

    raise FileNotFoundError(
        f"Cannot find source file aligned with target repeat file {target_filename} under {source_dir}"
    )


def infer_surface_from_name(path: str) -> str:
    name = os.path.basename(path).lower()
    return "nonlinear" if "nonlinear" in name else "linear"


def load_stream_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    fname = os.path.basename(path)
    m = REPEAT_ENV_FILE_REGEX.match(fname)

    if "segment_round" not in df.columns:
        df["segment_round"] = df.groupby("segment").cumcount().astype(np.int64)
    if "t" not in df.columns:
        df["t"] = np.arange(1, len(df) + 1, dtype=np.int64)
    if "env_name" not in df.columns:
        df["env_name"] = fname[:-4]
    if "repeat" not in df.columns and m is not None:
        df["repeat"] = int(m.group(5))
    df = df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def init_random_mlp(input_dim: int, seed: int, hidden_dim: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    dims = [input_dim, 256, 256, 256, hidden_dim]
    weights, biases = [], []
    for din, dout in zip(dims[:-1], dims[1:]):
        weights.append(rng.normal(0.0, 1.0 / np.sqrt(din), size=(din, dout)).astype(np.float32))
        biases.append(rng.normal(0.0, 0.01, size=(dout,)).astype(np.float32))
    return {"surface": "nonlinear", "weights": weights, "biases": biases}


def make_feature_map(surface: str, input_dim: int, seed: int, hidden_dim: int) -> Dict[str, object]:
    if surface == "linear":
        return {"surface": "linear"}
    return init_random_mlp(input_dim, seed, hidden_dim)


def apply_feature_map(feature_map: Dict[str, object], X: np.ndarray, batch_size: int) -> np.ndarray:
    if feature_map["surface"] == "linear":
        Phi = np.concatenate([X.astype(np.float32), np.ones((X.shape[0], 1), dtype=np.float32)], axis=1)
        return row_l2_normalize_leq1(Phi)

    weights = feature_map["weights"]
    biases = feature_map["biases"]
    outputs = []
    for start in range(0, len(X), batch_size):
        Hx = np.asarray(X[start:start + batch_size], dtype=np.float32)
        for i, (W, b) in enumerate(zip(weights, biases)):
            Hx = Hx @ W + b
            if i < len(weights) - 1:
                Hx = np.maximum(Hx, 0.0).astype(np.float32, copy=False)
        outputs.append(Hx.astype(np.float32, copy=False))
    return row_l2_normalize_leq1(np.vstack(outputs))


def prepare_env(df: pd.DataFrame, feature_map: Dict[str, object], batch_size: int) -> Dict[str, object]:
    feature_cols = get_feature_columns(df)
    X = df[feature_cols].to_numpy(dtype=np.float32)
    Phi = apply_feature_map(feature_map, X, batch_size)
    w = df["w"].to_numpy(dtype=np.float32)
    p = np.clip(df["p"].to_numpy(dtype=np.float32), EPS, 1.0 - EPS)
    y = df["y"].to_numpy(dtype=np.float32)
    tau = df["tau"].to_numpy(dtype=np.float32)
    if "q" in df.columns:
        q = np.clip(df["q"].to_numpy(dtype=np.float32), EPS, 1.0 - EPS)
    else:
        q = np.where(w == 1.0, p, 1.0 - p).astype(np.float32)
    return {
        "df": df,
        "Phi": Phi,
        "w": w,
        "p": p,
        "q": q,
        "y": y,
        "tau": tau,
        "segment": df["segment"].to_numpy(dtype=np.int64),
        "segment_round": df["segment_round"].to_numpy(dtype=np.int64),
        "t": df["t"].to_numpy(dtype=np.int64),
    }


def build_segment_boundaries(df: pd.DataFrame, env_name: str) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    cumulative = 0
    for seg in sorted(df["segment"].unique().tolist()):
        n_seg = int((df["segment"] == seg).sum())
        cumulative += n_seg
        rows.append({"env_name": env_name, "segment": int(seg), "n_steps": n_seg, "step_end": cumulative})
    return pd.DataFrame(rows)


def load_bundle(target_path: str, good_path: str, seed: int, hidden_dim: int, batch_size: int, force_linear: bool) -> Dict[str, object]:
    df_target = load_stream_csv(target_path)
    df_good = load_stream_csv(good_path)
    feature_cols = get_feature_columns(df_target)
    surface = "linear" if force_linear else infer_surface_from_name(target_path)
    feature_map = make_feature_map(surface, len(feature_cols), seed, hidden_dim)
    return {
        "surface": surface,
        "env_target": prepare_env(df_target, feature_map, batch_size),
        "env_good": prepare_env(df_good, feature_map, batch_size),
    }


# =========================================================
# Tracker
# =========================================================


class MetricTracker:
    def __init__(self, env_name: str, method: str, seed: int, record_trajectory: bool, repeat: Optional[int] = None):
        self.env_name = env_name
        self.method = method
        self.family = family_from_method(method)
        self.seed = seed
        self.repeat = repeat
        self.record_trajectory = record_trajectory
        self.step = 0
        self.cum_half = 0.0
        self.cum_sq = 0.0
        self.rows: List[Dict[str, object]] = []

    def add(
        self,
        pred_tau: float,
        tau_true: float,
        segment: int,
        segment_round: int,
        t: int,
        extra: Optional[Dict[str, object]] = None,
    ) -> None:
        sqerr = float((pred_tau - tau_true) ** 2)
        self.step += 1
        self.cum_half += 0.5 * sqerr
        self.cum_sq += sqerr
        running_mean_mse = self.cum_sq / self.step
        running_rmse = math.sqrt(max(running_mean_mse, 0.0))
        if self.record_trajectory:
            row = {
                "env_name": self.env_name,
                "seed": self.seed,
                "repeat": self.repeat,
                "method": self.method,
                "family": self.family,
                "step": self.step,
                "segment": int(segment),
                "segment_round": int(segment_round),
                "t": int(t),
                "pred_tau": float(pred_tau),
                "tau_true": float(tau_true),
                "instant_mse": sqerr,
                "instant_pehe": abs(float(pred_tau - tau_true)),
                "cumulative_half_mse": float(self.cum_half),
                "cumulative_squared_error": float(self.cum_sq),
                "running_mean_mse": float(running_mean_mse),
                "running_rmse": float(running_rmse),
            }
            if extra:
                row.update(extra)
            self.rows.append(row)

    def summary(self) -> Dict[str, float]:
        running_mean_mse = self.cum_sq / max(self.step, 1)
        return {
            "cumhalfPEHE": float(self.cum_half),
            "cumulative_squared_error": float(self.cum_sq),
            "running_mean_mse": float(running_mean_mse),
            "running_rmse": float(math.sqrt(max(running_mean_mse, 0.0))),
        }

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


# =========================================================
# Core math
# =========================================================


def get_ipw_pseudo_outcome(w: float, y: float, p: float) -> float:
    return float(w * y / p)


def get_dr_pseudo_outcome(w: float, y: float, q: float, m0_hat: float, m1_hat: float) -> float:
    if w == 1.0:
        return float((m1_hat - m0_hat) + (y - m1_hat) / q)
    return float((m1_hat - m0_hat) - (y - m0_hat) / (1.0 - q))


def predict_outcome(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(np.clip(np.dot(theta, phi), -M_BOUND, M_BOUND))


def predict_tau(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(np.clip(np.dot(theta, phi), -HTE_RADIUS, HTE_RADIUS))


def get_ethos_hyperparams(T_run: int, method: str) -> Dict[str, object]:
    g_tilde = get_method_g_tilde(method)
    alpha_meta = math.sqrt(8.0 / (T_run * (g_tilde ** 2) * (D ** 2)))
    N = int(math.floor(0.5 * math.log2(1.0 + 4.0 * T_run / 7.0))) + 1
    etas = [(2.0 ** (i - 1)) * D / g_tilde * math.sqrt(7.0 / (2.0 * T_run)) for i in range(1, N + 1)]
    return {"alpha_meta": float(alpha_meta), "N": int(N), "etas": np.asarray(etas, dtype=np.float32)}


def update_outcome(theta: np.ndarray, phi: np.ndarray, y: float, acc_sq: float) -> Tuple[np.ndarray, float]:
    pred = predict_outcome(theta, phi)
    grad = ((pred - y) * phi).astype(np.float32)
    acc_sq = acc_sq + float(np.dot(grad, grad))
    eta = get_scale_free_eta(acc_sq=acc_sq, radius=OUTCOME_RADIUS, c_eta=THEORY_C_ETA)
    theta = project_l2_ball_vec(theta - eta * grad, OUTCOME_RADIUS)
    return theta.astype(np.float32), float(acc_sq)


def unpack_target_stream(env: Dict[str, object]) -> Tuple[np.ndarray, ...]:
    return (
        env["Phi"], env["w"], env["p"], env["q"], env["y"], env["tau"],
        env["segment"], env["segment_round"], env["t"],
    )


def get_routeb_relative_pseudo_losses(pred_old: float, pred_new: float, pseudo_tau: float) -> Tuple[float, float, float, float, float]:
    err_old_sq = float((pred_old - pseudo_tau) ** 2)
    err_new_sq = float((pred_new - pseudo_tau) ** 2)
    denom = err_old_sq + err_new_sq + EPS_NUM
    loss_old = err_old_sq / denom
    loss_new = err_new_sq / denom
    loss_gap = loss_old - loss_new
    return float(loss_old), float(loss_new), float(err_old_sq), float(err_new_sq), float(loss_gap)


def update_otl_weights_exact(weights: np.ndarray, loss_old: float, loss_new: float) -> np.ndarray:
    s_old = math.exp(-OTL_META_ETA * float(loss_old))
    s_new = math.exp(-OTL_META_ETA * float(loss_new))
    num_old = float(weights[0]) * s_old
    num_new = float(weights[1]) * s_new
    denom = num_old + num_new + EPS_NUM
    return np.asarray([num_old / denom, num_new / denom], dtype=np.float32)


# =========================================================
# Base learners
# =========================================================


def run_adaptive_core(
    env_name: str,
    seed: int,
    env_target: Dict[str, object],
    method: str,
    record_trajectory: bool = False,
    repeat: Optional[int] = None,
) -> Dict[str, object]:
    Phi, w, p, q, y, tau, segment, segment_round, original_t = unpack_target_stream(env_target)
    d = Phi.shape[1]
    theta = np.zeros(d, dtype=np.float32)
    alpha0 = np.zeros(d, dtype=np.float32)
    alpha1 = np.zeros(d, dtype=np.float32)
    acc_sq = 0.0
    acc0_sq = 0.0
    acc1_sq = 0.0
    tracker = MetricTracker(env_name, method, seed, record_trajectory, repeat=repeat)

    for idx, phi in enumerate(Phi):
        tracker.add(
            predict_tau(theta, phi),
            float(tau[idx]),
            int(segment[idx]),
            int(segment_round[idx]),
            int(original_t[idx]),
        )

        if method.endswith("IPW"):
            pseudo_tau = get_ipw_pseudo_outcome(float(w[idx]), float(y[idx]), float(p[idx]))
        else:
            m0_hat = predict_outcome(alpha0, phi)
            m1_hat = predict_outcome(alpha1, phi)
            pseudo_tau = get_dr_pseudo_outcome(float(w[idx]), float(y[idx]), float(q[idx]), m0_hat, m1_hat)

        grad = ((float(np.dot(theta, phi)) - pseudo_tau) * phi).astype(np.float32)
        acc_sq += float(np.dot(grad, grad))
        eta_t = get_scale_free_eta(acc_sq=acc_sq, radius=HTE_RADIUS, c_eta=THEORY_C_ETA)
        theta = project_l2_ball_vec(theta - eta_t * grad, HTE_RADIUS)

        if method.endswith("DR"):
            if float(w[idx]) == 1.0:
                alpha1, acc1_sq = update_outcome(alpha1, phi, float(y[idx]), acc1_sq)
            else:
                alpha0, acc0_sq = update_outcome(alpha0, phi, float(y[idx]), acc0_sq)

    out = tracker.summary()
    out.update({"theta_final": theta, "c_eta": float(THEORY_C_ETA)})
    if record_trajectory:
        out["trajectory_df"] = tracker.frame()
    return out


def run_ethos_core(
    env_name: str,
    seed: int,
    env_target: Dict[str, object],
    method: str,
    record_trajectory: bool = False,
    repeat: Optional[int] = None,
) -> Dict[str, object]:
    Phi, w, p, q, y, tau, segment, segment_round, original_t = unpack_target_stream(env_target)
    hp = get_ethos_hyperparams(Phi.shape[0], method)
    alpha_meta = hp["alpha_meta"]
    etas = hp["etas"]
    N = hp["N"]

    Theta = np.zeros((N, Phi.shape[1]), dtype=np.float32)
    omega = np.ones(N, dtype=np.float32) / N
    alpha0 = np.zeros(Phi.shape[1], dtype=np.float32)
    alpha1 = np.zeros(Phi.shape[1], dtype=np.float32)
    acc0_sq = 0.0
    acc1_sq = 0.0
    tracker = MetricTracker(env_name, method, seed, record_trajectory, repeat=repeat)

    for idx, phi in enumerate(Phi):
        theta_mix = (omega @ Theta).astype(np.float32)
        tracker.add(
            predict_tau(theta_mix, phi),
            float(tau[idx]),
            int(segment[idx]),
            int(segment_round[idx]),
            int(original_t[idx]),
        )

        if method.endswith("IPW"):
            pseudo_tau = get_ipw_pseudo_outcome(float(w[idx]), float(y[idx]), float(p[idx]))
        else:
            m0_hat = predict_outcome(alpha0, phi)
            m1_hat = predict_outcome(alpha1, phi)
            pseudo_tau = get_dr_pseudo_outcome(float(w[idx]), float(y[idx]), float(q[idx]), m0_hat, m1_hat)

        preds = Theta @ phi
        losses = 0.5 * (preds - pseudo_tau) ** 2
        omega = omega * np.exp(-alpha_meta * (losses - np.min(losses))).astype(np.float32)
        omega = (omega / (np.sum(omega) + EPS_NUM)).astype(np.float32)
        Theta = project_l2_ball_rows(
            Theta - etas[:, None] * ((preds - pseudo_tau)[:, None] * phi[None, :]),
            HTE_RADIUS,
        )

        if method.endswith("DR"):
            if float(w[idx]) == 1.0:
                alpha1, acc1_sq = update_outcome(alpha1, phi, float(y[idx]), acc1_sq)
            else:
                alpha0, acc0_sq = update_outcome(alpha0, phi, float(y[idx]), acc0_sq)

    out = tracker.summary()
    out.update({
        "theta_final": (omega @ Theta).astype(np.float32),
        "alpha_meta": float(alpha_meta),
        "N": int(N),
        "eta_min": float(etas[0]),
        "eta_max": float(etas[-1]),
    })
    if record_trajectory:
        out["trajectory_df"] = tracker.frame()
    return out


def fit_source_old_classifier_adaptive(env_source: Dict[str, object], method: str) -> np.ndarray:
    source_env = {
        **env_source,
        "tau": np.zeros_like(env_source["y"]),
        "segment": np.zeros_like(env_source["y"], dtype=np.int64),
        "segment_round": np.arange(len(env_source["y"]), dtype=np.int64),
        "t": np.arange(len(env_source["y"]), dtype=np.int64),
    }
    return run_adaptive_core("__source__", 0, source_env, method, record_trajectory=False)["theta_final"].astype(np.float32)


# =========================================================
# Route B OTL transfer learners (adaptive source + adaptive target)
# =========================================================


def run_exact_otl_transfer_adaptive_core(
    env_name: str,
    seed: int,
    env_target: Dict[str, object],
    transfer_method: str,
    target_method: str,
    source_old_model: np.ndarray,
    record_trajectory: bool = False,
    repeat: Optional[int] = None,
) -> Dict[str, object]:
    Phi, w, p, q, y, tau, segment, segment_round, original_t = unpack_target_stream(env_target)
    d = Phi.shape[1]
    theta = np.zeros(d, dtype=np.float32)  # f_1 = 0
    alpha0 = np.zeros(d, dtype=np.float32)
    alpha1 = np.zeros(d, dtype=np.float32)
    acc_sq = 0.0
    acc0_sq = 0.0
    acc1_sq = 0.0

    # [old/source h, new/target f]
    expert_weights = np.asarray([OTL_INITIAL_WEIGHT, OTL_INITIAL_WEIGHT], dtype=np.float32)

    tracker = MetricTracker(env_name, transfer_method, seed, record_trajectory, repeat=repeat)

    for idx, phi in enumerate(Phi):
        pred_old = predict_tau(source_old_model, phi)
        pred_new = predict_tau(theta, phi)

        pred_mix = float(np.clip(
            expert_weights[0] * pred_old + expert_weights[1] * pred_new,
            -HTE_RADIUS,
            HTE_RADIUS,
        ))

        extra = None
        if record_trajectory:
            extra = {
                "pred_old_component": float(pred_old),
                "pred_new_component": float(pred_new),
                "pred_mix_component": float(pred_mix),
                "otl_weight_old": float(expert_weights[0]),
                "otl_weight_new": float(expert_weights[1]),
                "otl_meta_eta": float(OTL_META_ETA),
                "otl_loss_variant": OTL_LOSS_VARIANT,
            }

        tracker.add(
            pred_mix,
            float(tau[idx]),
            int(segment[idx]),
            int(segment_round[idx]),
            int(original_t[idx]),
            extra=extra,
        )

        if target_method.endswith("IPW"):
            pseudo_tau = get_ipw_pseudo_outcome(float(w[idx]), float(y[idx]), float(p[idx]))
        else:
            m0_hat = predict_outcome(alpha0, phi)
            m1_hat = predict_outcome(alpha1, phi)
            pseudo_tau = get_dr_pseudo_outcome(float(w[idx]), float(y[idx]), float(q[idx]), m0_hat, m1_hat)

        loss_old, loss_new, err_old_sq, err_new_sq, loss_gap = get_routeb_relative_pseudo_losses(
            pred_old=float(pred_old),
            pred_new=float(pred_new),
            pseudo_tau=float(pseudo_tau),
        )
        expert_weights = update_otl_weights_exact(expert_weights, float(loss_old), float(loss_new))

        if record_trajectory:
            tracker.rows[-1].update({
                "pseudo_tau_raw": float(pseudo_tau),
                "otl_loss_old": float(loss_old),
                "otl_loss_new": float(loss_new),
                "otl_loss_gap": float(loss_gap),
                "otl_err_old_sq": float(err_old_sq),
                "otl_err_new_sq": float(err_new_sq),
                "otl_weight_old_next": float(expert_weights[0]),
                "otl_weight_new_next": float(expert_weights[1]),
            })

        # target/new predictor 仍按原 ADAPTIVE 核心更新
        grad = ((float(np.dot(theta, phi)) - pseudo_tau) * phi).astype(np.float32)
        acc_sq += float(np.dot(grad, grad))
        eta_t = get_scale_free_eta(acc_sq=acc_sq, radius=HTE_RADIUS, c_eta=THEORY_C_ETA)
        theta = project_l2_ball_vec(theta - eta_t * grad, HTE_RADIUS)

        if target_method.endswith("DR"):
            if float(w[idx]) == 1.0:
                alpha1, acc1_sq = update_outcome(alpha1, phi, float(y[idx]), acc1_sq)
            else:
                alpha0, acc0_sq = update_outcome(alpha0, phi, float(y[idx]), acc0_sq)

    out = tracker.summary()
    out.update({
        "theta_target_final": theta.astype(np.float32),
        "theta_source_fixed": source_old_model.astype(np.float32),
        "c_eta": float(THEORY_C_ETA),
        "otl_meta_eta": float(OTL_META_ETA),
        "otl_loss_variant": OTL_LOSS_VARIANT,
        "otl_weight_old_final": float(expert_weights[0]),
        "otl_weight_new_final": float(expert_weights[1]),
        "transfer_strategy": "route_b_relative_pseudoloss_otl",
    })
    if record_trajectory:
        out["trajectory_df"] = tracker.frame()
    return out


# =========================================================
# Task helpers
# =========================================================


def attach_env_meta(row: Dict[str, object], env_info: Dict[str, object], seed: Optional[int]) -> Dict[str, object]:
    out = dict(row)
    out.update({
        "env_name": env_info["env_name"],
        "env_repeat_name": env_info.get("env_repeat_name"),
        "dataset": env_info["dataset"],
        "surface": env_info["surface"],
        "drift": env_info["drift"],
        "K": env_info["K"],
        "repeat": env_info.get("repeat"),
        "seed": seed,
        "family": family_from_method(str(out["method"])),
    })
    return out


def aggregate_metric_df(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str], repeat_col_name: str = "n_repeats") -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    metric_cols = [c for c in metric_cols if c in df.columns]
    grouped = df.groupby(group_cols, as_index=False, dropna=False).agg({c: ["mean", "std"] for c in metric_cols})
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns]
    counts = df.groupby(group_cols, as_index=False, dropna=False).size().rename(columns={"size": repeat_col_name})
    grouped = pd.merge(grouped, counts, on=group_cols, how="left")
    for c in metric_cols:
        std_col = f"{c}_std"
        if std_col in grouped.columns:
            grouped[std_col] = grouped[std_col].fillna(0.0)
    return grouped


def run_stage1_repeat(task: Tuple[Dict[str, object], str, str, int, int, int, bool]) -> Dict[str, object]:
    env_info, target_path, good_path, seed, hidden_dim, batch_size, force_linear = task
    env_name = str(env_info["env_name"])
    bundle = load_bundle(target_path, good_path, seed, hidden_dim, batch_size, force_linear)
    env_target = bundle["env_target"]
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None

    adaptive_ipw = run_adaptive_core(env_name, seed, env_target, "ADAPTIVE_IPW", repeat=repeat)
    adaptive_dr = run_adaptive_core(env_name, seed, env_target, "ADAPTIVE_DR", repeat=repeat)
    ethos_ipw = run_ethos_core(env_name, seed, env_target, "ETHOS_IPW", repeat=repeat)
    ethos_dr = run_ethos_core(env_name, seed, env_target, "ETHOS_DR", repeat=repeat)

    rows = [
        attach_env_meta({
            "method": "ADAPTIVE_IPW",
            "c_eta": float(THEORY_C_ETA),
            "selection_basis": "theory_c_eta",
            "cumhalfPEHE": float(adaptive_ipw["cumhalfPEHE"]),
            "running_mean_mse": float(adaptive_ipw["running_mean_mse"]),
            "running_rmse": float(adaptive_ipw["running_rmse"]),
        }, env_info, seed),
        attach_env_meta({
            "method": "ADAPTIVE_DR",
            "c_eta": float(THEORY_C_ETA),
            "selection_basis": "theory_c_eta",
            "cumhalfPEHE": float(adaptive_dr["cumhalfPEHE"]),
            "running_mean_mse": float(adaptive_dr["running_mean_mse"]),
            "running_rmse": float(adaptive_dr["running_rmse"]),
        }, env_info, seed),
        attach_env_meta({
            "method": "ETHOS_IPW",
            "c_eta": None,
            "selection_basis": "fixed_method",
            "cumhalfPEHE": float(ethos_ipw["cumhalfPEHE"]),
            "running_mean_mse": float(ethos_ipw["running_mean_mse"]),
            "running_rmse": float(ethos_ipw["running_rmse"]),
            "alpha_meta": float(ethos_ipw["alpha_meta"]),
            "N": int(ethos_ipw["N"]),
            "eta_min": float(ethos_ipw["eta_min"]),
            "eta_max": float(ethos_ipw["eta_max"]),
        }, env_info, seed),
        attach_env_meta({
            "method": "ETHOS_DR",
            "c_eta": None,
            "selection_basis": "fixed_method",
            "cumhalfPEHE": float(ethos_dr["cumhalfPEHE"]),
            "running_mean_mse": float(ethos_dr["running_mean_mse"]),
            "running_rmse": float(ethos_dr["running_rmse"]),
            "alpha_meta": float(ethos_dr["alpha_meta"]),
            "N": int(ethos_dr["N"]),
            "eta_min": float(ethos_dr["eta_min"]),
            "eta_max": float(ethos_dr["eta_max"]),
        }, env_info, seed),
    ]
    return {"env_name": env_name, "seed": int(seed), "repeat": repeat, "rows": rows}


def build_stage1_outputs(stage1_results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, float]]]:
    raw_df = pd.DataFrame([row for res in stage1_results for row in res["rows"]])
    group_cols = ["env_name", "dataset", "surface", "drift", "K", "method", "family", "selection_basis", "c_eta"]
    metric_cols = ["cumhalfPEHE", "running_mean_mse", "running_rmse", "alpha_meta", "N", "eta_min", "eta_max"]
    env_df = aggregate_metric_df(raw_df, group_cols, metric_cols)
    env_df = env_df.sort_values(["env_name", "method"]).reset_index(drop=True)
    param_map = {
        env_name: {"ADAPTIVE_IPW": float(THEORY_C_ETA), "ADAPTIVE_DR": float(THEORY_C_ETA)}
        for env_name in env_df["env_name"].drop_duplicates().tolist()
    }
    return raw_df, env_df, param_map


def run_stage2_repeat(task: Tuple[Dict[str, object], str, str, int, int, int, bool]) -> Dict[str, object]:
    env_info, target_path, good_path, seed, hidden_dim, batch_size, force_linear = task
    env_name = str(env_info["env_name"])
    bundle = load_bundle(target_path, good_path, seed, hidden_dim, batch_size, force_linear)
    env_target = bundle["env_target"]
    env_good = bundle["env_good"]
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None

    source_models = {
        "ADAPTIVE_IPW": fit_source_old_classifier_adaptive(env_good, "ADAPTIVE_IPW"),
        "ADAPTIVE_DR": fit_source_old_classifier_adaptive(env_good, "ADAPTIVE_DR"),
    }

    rows: List[Dict[str, object]] = []
    for spec in COMPARISON_SPECS:
        out = run_exact_otl_transfer_adaptive_core(
            env_name=env_name,
            seed=seed,
            env_target=env_target,
            transfer_method=str(spec["transfer_method"]),
            target_method=str(spec["target_method"]),
            source_old_model=source_models[str(spec["source_method"])],
            record_trajectory=False,
            repeat=repeat,
        )

        rows.append(attach_env_meta({
            "method": str(spec["transfer_method"]),
            "comparison_label": str(spec["comparison_label"]),
            "target_method": str(spec["target_method"]),
            "source_method": str(spec["source_method"]),
            "baseline_method": str(spec["baseline_method"]),
            "core_method": str(spec["target_method"]),
            "core_c_eta": float(THEORY_C_ETA),
            "stage1_inherited_c_eta": float(THEORY_C_ETA),
            "otl_meta_eta": float(out["otl_meta_eta"]),
            "otl_loss_variant": str(out["otl_loss_variant"]),
            "cumhalfPEHE": float(out["cumhalfPEHE"]),
            "running_mean_mse": float(out["running_mean_mse"]),
            "running_rmse": float(out["running_rmse"]),
            "otl_weight_old_final": float(out["otl_weight_old_final"]),
            "otl_weight_new_final": float(out["otl_weight_new_final"]),
            "transfer_strategy": str(out["transfer_strategy"]),
        }, env_info, seed))

    return {"env_name": env_name, "seed": int(seed), "repeat": repeat, "rows": rows}


def build_stage2_outputs(stage2_results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([row for res in stage2_results for row in res["rows"]])
    group_cols = [
        "env_name", "dataset", "surface", "drift", "K", "method", "comparison_label", "target_method",
        "source_method", "baseline_method", "core_method", "core_c_eta", "stage1_inherited_c_eta",
        "family", "otl_meta_eta", "otl_loss_variant", "transfer_strategy"
    ]
    metric_cols = [
        "cumhalfPEHE", "running_mean_mse", "running_rmse", "otl_weight_old_final", "otl_weight_new_final"
    ]
    mean_df = aggregate_metric_df(raw_df, group_cols, metric_cols).sort_values(["env_name", "comparison_label"]).reset_index(drop=True)
    mean_df["selection_basis"] = "paper_exact_otl_no_search"
    return raw_df, mean_df


def run_final_repeat(task: Tuple[Dict[str, object], str, str, int, int, int, bool, bool]) -> Dict[str, object]:
    env_info, target_path, good_path, seed, hidden_dim, batch_size, force_linear, record_trajectory = task
    env_name = str(env_info["env_name"])
    bundle = load_bundle(target_path, good_path, seed, hidden_dim, batch_size, force_linear)
    env_target = bundle["env_target"]
    env_good = bundle["env_good"]
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None

    source_old_ipw = fit_source_old_classifier_adaptive(env_good, "ADAPTIVE_IPW")
    source_old_dr = fit_source_old_classifier_adaptive(env_good, "ADAPTIVE_DR")

    rows: List[Dict[str, object]] = []
    trajs: List[pd.DataFrame] = []

    # target-only methods
    for method in ["ADAPTIVE_IPW", "ADAPTIVE_DR", "ETHOS_IPW", "ETHOS_DR"]:
        if method.startswith("ADAPTIVE"):
            out = run_adaptive_core(env_name, seed, env_target, method, record_trajectory=record_trajectory, repeat=repeat)
            row = attach_env_meta({
                "method": method,
                "c_eta": float(THEORY_C_ETA),
                "cumhalfPEHE": float(out["cumhalfPEHE"]),
                "running_mean_mse": float(out["running_mean_mse"]),
                "running_rmse": float(out["running_rmse"]),
            }, env_info, seed)
        else:
            out = run_ethos_core(env_name, seed, env_target, method, record_trajectory=record_trajectory, repeat=repeat)
            row = attach_env_meta({
                "method": method,
                "c_eta": None,
                "cumhalfPEHE": float(out["cumhalfPEHE"]),
                "running_mean_mse": float(out["running_mean_mse"]),
                "running_rmse": float(out["running_rmse"]),
                "alpha_meta": float(out["alpha_meta"]),
                "N": int(out["N"]),
                "eta_min": float(out["eta_min"]),
                "eta_max": float(out["eta_max"]),
            }, env_info, seed)
        rows.append(row)
        if record_trajectory:
            trajs.append(out["trajectory_df"])

    # exact OTL adaptive transfer methods
    for spec in COMPARISON_SPECS:
        source_old_model = source_old_ipw if spec["source_method"] == "ADAPTIVE_IPW" else source_old_dr
        out = run_exact_otl_transfer_adaptive_core(
            env_name=env_name,
            seed=seed,
            env_target=env_target,
            transfer_method=str(spec["transfer_method"]),
            target_method=str(spec["target_method"]),
            source_old_model=source_old_model,
            record_trajectory=record_trajectory,
            repeat=repeat,
        )
        row = attach_env_meta({
            "method": str(spec["transfer_method"]),
            "comparison_label": str(spec["comparison_label"]),
            "target_method": str(spec["target_method"]),
            "source_method": str(spec["source_method"]),
            "baseline_method": str(spec["baseline_method"]),
            "core_method": str(spec["target_method"]),
            "core_c_eta": float(THEORY_C_ETA),
            "stage1_inherited_c_eta": float(THEORY_C_ETA),
            "otl_meta_eta": float(out["otl_meta_eta"]),
            "otl_loss_variant": str(out["otl_loss_variant"]),
            "cumhalfPEHE": float(out["cumhalfPEHE"]),
            "running_mean_mse": float(out["running_mean_mse"]),
            "running_rmse": float(out["running_rmse"]),
            "otl_weight_old_final": float(out["otl_weight_old_final"]),
            "otl_weight_new_final": float(out["otl_weight_new_final"]),
            "transfer_strategy": str(out["transfer_strategy"]),
        }, env_info, seed)
        rows.append(row)
        if record_trajectory:
            trajs.append(out["trajectory_df"])

    boundaries = build_segment_boundaries(env_target["df"], env_name)
    return {
        "env_name": env_name,
        "seed": int(seed),
        "repeat": repeat,
        "rows": rows,
        "trajectories": concat_nonempty(trajs),
        "segment_boundaries": boundaries,
    }


def run_tasks_with_pool(fn, tasks: List[Tuple], n_jobs: int) -> List[Dict[str, object]]:
    if not tasks:
        return []
    n_jobs = effective_n_jobs(n_jobs)
    if n_jobs <= 1 or len(tasks) == 1:
        return [fn(t) for t in tasks]
    results: List[Dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=n_jobs, mp_context=mp.get_context("spawn")) as ex:
        futures = {ex.submit(fn, t): t for t in tasks}
        for fut in as_completed(futures):
            results.append(fut.result())
    return results


# =========================================================
# Aggregation
# =========================================================


def aggregate_final_method_results(raw_method_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if raw_method_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [
        "env_name", "dataset", "surface", "drift", "K", "method", "family", "comparison_label",
        "target_method", "source_method", "baseline_method", "core_method", "core_c_eta",
        "stage1_inherited_c_eta", "otl_meta_eta", "otl_loss_variant", "transfer_strategy"
    ]
    metric_cols = [
        "cumhalfPEHE", "running_mean_mse", "running_rmse", "c_eta", "alpha_meta", "N", "eta_min", "eta_max",
        "otl_weight_old_final", "otl_weight_new_final"
    ]
    env_df = aggregate_metric_df(raw_method_df, group_cols, metric_cols).sort_values(["env_name", "method"]).reset_index(drop=True)
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    env_df["_rank"] = env_df["method"].map(lambda x: method_rank.get(str(x), 999))
    env_df = env_df.sort_values(["env_name", "cumhalfPEHE_mean", "_rank", "method"]).reset_index(drop=True)
    env_df["env_rank"] = env_df.groupby("env_name").cumcount() + 1
    best_df = env_df.groupby("env_name", as_index=False, group_keys=False).head(1).reset_index(drop=True)
    return env_df.drop(columns=["_rank"]), best_df.drop(columns=["_rank"])


def aggregate_trajectory_df(raw_traj_df: pd.DataFrame) -> pd.DataFrame:
    if raw_traj_df.empty:
        return pd.DataFrame()

    group_cols = ["env_name", "method", "family", "step", "segment", "segment_round", "t"]
    metric_cols = [
        "pred_tau", "tau_true", "instant_mse", "instant_pehe",
        "cumulative_half_mse", "cumulative_squared_error", "running_mean_mse", "running_rmse",
        "pred_old_component", "pred_new_component", "pred_mix_component",
        "otl_weight_old", "otl_weight_new", "otl_weight_old_next", "otl_weight_new_next",
        "otl_meta_eta", "pseudo_tau_raw", "otl_loss_old", "otl_loss_new",
        "otl_loss_gap", "otl_err_old_sq", "otl_err_new_sq"
    ]

    agg = aggregate_metric_df(raw_traj_df, group_cols, metric_cols)
    rename_map = {f"{c}_mean": c for c in metric_cols}
    out = agg.rename(columns=rename_map).sort_values(["env_name", "method", "step"]).reset_index(drop=True)

    if "otl_loss_variant" in raw_traj_df.columns:
        variant_df = (
            raw_traj_df[group_cols + ["otl_loss_variant"]]
            .drop_duplicates(subset=group_cols)
            .copy()
        )
        out = out.merge(variant_df, on=group_cols, how="left")

    return out.reset_index(drop=True)


def build_pairwise_best_df(env_method_mean_df: pd.DataFrame) -> pd.DataFrame:
    if env_method_mean_df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, object]] = []
    for env_name, env_df in env_method_mean_df.groupby("env_name"):
        row_map = {str(r["method"]): r for _, r in env_df.iterrows()}
        meta = env_df.iloc[0]
        for spec in COMPARISON_SPECS:
            baseline = str(spec["baseline_method"])
            transfer = str(spec["transfer_method"])
            if baseline not in row_map or transfer not in row_map:
                continue
            base_row = row_map[baseline]
            tr_row = row_map[transfer]
            baseline_cum = float(base_row["cumhalfPEHE_mean"])
            transfer_cum = float(tr_row["cumhalfPEHE_mean"])
            gain = baseline_cum - transfer_cum
            rows.append({
                "env_name": env_name,
                "dataset": meta["dataset"],
                "surface": meta["surface"],
                "drift": meta["drift"],
                "K": meta["K"],
                "comparison_label": spec["comparison_label"],
                "family": spec["family"],
                "target_method": spec["target_method"],
                "source_method": spec["source_method"],
                "baseline_method": baseline,
                "transfer_method": transfer,
                "baseline_cumhalfPEHE_mean": baseline_cum,
                "baseline_cumhalfPEHE_std": float(base_row.get("cumhalfPEHE_std", 0.0)),
                "transfer_cumhalfPEHE_mean": transfer_cum,
                "transfer_cumhalfPEHE_std": float(tr_row.get("cumhalfPEHE_std", 0.0)),
                "transfer_abs_gain": gain,
                "transfer_rel_gain_pct": 100.0 * gain / (abs(baseline_cum) + EPS_NUM),
                "transfer_better_than_baseline": int(gain > 1e-12),
                "tie": int(abs(gain) <= 1e-12),
                "baseline_better": int(gain < -1e-12),
                "otl_meta_eta": tr_row.get("otl_meta_eta_mean"),
                "n_repeats": int(tr_row["n_repeats"]),
            })
    return pd.DataFrame(rows).sort_values(["comparison_label", "env_name"]).reset_index(drop=True)


def build_pairwise_overall_summary(pairwise_best_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_best_df.empty:
        return pd.DataFrame()
    out = pairwise_best_df.groupby(
        ["comparison_label", "family", "target_method", "source_method", "baseline_method", "transfer_method"],
        as_index=False,
        dropna=False,
    ).agg(
        n_env=("env_name", "count"),
        transfer_win_count=("transfer_better_than_baseline", "sum"),
        tie_count=("tie", "sum"),
        loss_count=("baseline_better", "sum"),
        mean_baseline_cumhalfPEHE=("baseline_cumhalfPEHE_mean", "mean"),
        mean_transfer_cumhalfPEHE=("transfer_cumhalfPEHE_mean", "mean"),
        mean_transfer_abs_gain=("transfer_abs_gain", "mean"),
        median_transfer_abs_gain=("transfer_abs_gain", "median"),
        mean_transfer_rel_gain_pct=("transfer_rel_gain_pct", "mean"),
    )
    out["transfer_win_rate"] = out["transfer_win_count"] / out["n_env"].clip(lower=1)
    return out.sort_values(["family", "comparison_label"]).reset_index(drop=True)


def build_pairwise_running_gap(traj_mean_df: pd.DataFrame) -> pd.DataFrame:
    if traj_mean_df.empty:
        return pd.DataFrame()
    results: List[pd.DataFrame] = []
    keys = ["env_name", "step", "segment", "segment_round", "t"]
    for env_name, group in traj_mean_df.groupby("env_name"):
        for spec in COMPARISON_SPECS:
            transfer_method = str(spec["transfer_method"])
            baseline_method = str(spec["baseline_method"])
            tr_df = group[group["method"] == transfer_method].copy()
            bl_df = group[group["method"] == baseline_method].copy()
            if tr_df.empty or bl_df.empty:
                continue
            merged = pd.merge(tr_df, bl_df, on=keys, suffixes=("_transfer", "_baseline"), how="inner").sort_values("step")
            delta = merged["instant_mse_transfer"].to_numpy(dtype=float) - merged["instant_mse_baseline"].to_numpy(dtype=float)
            results.append(pd.DataFrame({
                "env_name": env_name,
                "comparison_label": spec["comparison_label"],
                "family": spec["family"],
                "target_method": spec["target_method"],
                "source_method": spec["source_method"],
                "baseline_method": baseline_method,
                "competitor_method": transfer_method,
                "step": merged["step"].to_numpy(dtype=int),
                "segment": merged["segment"].to_numpy(dtype=int),
                "segment_round": merged["segment_round"].to_numpy(dtype=int),
                "t": merged["t"].to_numpy(dtype=int),
                "instant_mse_gap": delta,
                "running_mean_mse_gap": np.cumsum(delta) / np.arange(1, len(delta) + 1),
            }))
    return concat_nonempty(results)


def build_env_rank_table(env_method_mean_df: pd.DataFrame) -> pd.DataFrame:
    if env_method_mean_df.empty:
        return pd.DataFrame()
    cols = [
        "env_name", "dataset", "surface", "drift", "K", "method", "env_rank", "cumhalfPEHE_mean", "cumhalfPEHE_std",
        "running_mean_mse_mean", "running_rmse_mean", "n_repeats", "c_eta_mean", "otl_meta_eta_mean"
    ]
    out = env_method_mean_df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    return out[cols].sort_values(["env_name", "env_rank", "method"]).reset_index(drop=True)


# =========================================================
# Plotting
# =========================================================


def _lazy_import_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def select_showcase_environment(env_infos: List[Dict[str, object]], requested_env: Optional[str] = None) -> str:
    env_names = [str(x["env_name"]) for x in env_infos]
    if requested_env is not None:
        if requested_env not in env_names:
            raise ValueError(f"showcase env not found: {requested_env}")
        return requested_env
    best = sorted(env_infos, key=lambda x: (x["drift"] == "switching", x["surface"] == "nonlinear", x["K"]), reverse=True)[0]
    return str(best["env_name"])


def add_segment_lines(ax, boundaries: pd.DataFrame) -> None:
    if boundaries.empty:
        return
    for step_end in boundaries["step_end"].tolist()[:-1]:
        ax.axvline(step_end, linestyle="--", linewidth=0.8, alpha=0.18)


def plot_ablation_regret_showcase(traj_df: pd.DataFrame, boundaries_df: pd.DataFrame, showcase_env: str, spec: Dict[str, str], output_path: str) -> None:
    if traj_df.empty:
        return
    plt = _lazy_import_matplotlib()
    methods = [str(spec["baseline_method"]), str(spec["transfer_method"])]
    plot_df = traj_df[(traj_df["env_name"] == showcase_env) & (traj_df["method"].isin(methods))].copy()
    if plot_df.empty:
        return
    boundaries = boundaries_df[boundaries_df["env_name"] == showcase_env].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for method in methods:
        method_df = plot_df[plot_df["method"] == method].sort_values("step")
        ax.plot(method_df["step"], method_df["cumulative_half_mse"], linewidth=2.2, label=method)
    add_segment_lines(ax, boundaries)
    ax.set_title(f"{spec['comparison_label']}\nRepeat-mean cumulative regret", fontsize=13)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean cumulative 0.5 x MSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_convergence_showcase(traj_df: pd.DataFrame, boundaries_df: pd.DataFrame, showcase_env: str, spec: Dict[str, str], output_path: str, window: int) -> None:
    if traj_df.empty:
        return
    plt = _lazy_import_matplotlib()
    methods = [str(spec["baseline_method"]), str(spec["transfer_method"])]
    plot_df = traj_df[(traj_df["env_name"] == showcase_env) & (traj_df["method"].isin(methods))].copy()
    if plot_df.empty:
        return
    boundaries = boundaries_df[boundaries_df["env_name"] == showcase_env].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    for method in methods:
        method_df = plot_df[plot_df["method"] == method].sort_values("step").copy()
        rolling_mse = method_df["instant_mse"].rolling(window=window, min_periods=max(5, min(window, 25))).mean()
        pehe_smooth = np.sqrt(np.maximum(rolling_mse, 0.0))
        ax.plot(method_df["step"], pehe_smooth, linewidth=2.2, label=method)
    add_segment_lines(ax, boundaries)
    ax.set_title(f"{spec['comparison_label']}\nRepeat-mean rolling convergence", fontsize=13)
    ax.set_xlabel("Training step")
    ax.set_ylabel(f"Rolling PEHE (window={window})")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_gap_showcase(gap_df: pd.DataFrame, boundaries_df: pd.DataFrame, showcase_env: str, spec: Dict[str, str], output_path: str) -> None:
    if gap_df.empty:
        return
    plt = _lazy_import_matplotlib()
    plot_df = gap_df[(gap_df["env_name"] == showcase_env) & (gap_df["comparison_label"] == spec["comparison_label"])].sort_values("step")
    if plot_df.empty:
        return
    boundaries = boundaries_df[boundaries_df["env_name"] == showcase_env].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax.plot(plot_df["step"], plot_df["running_mean_mse_gap"], linewidth=2.2)
    ax.axhline(0.0, linestyle="--", linewidth=1.0, alpha=0.55)
    add_segment_lines(ax, boundaries)
    ax.set_title(f"{spec['comparison_label']}\nRepeat-mean running risk gap", fontsize=13)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Running mean MSE gap")
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_environment_rank_heatmap(env_rank_df: pd.DataFrame, output_path: str) -> None:
    if env_rank_df.empty:
        return
    plt = _lazy_import_matplotlib()
    method_rank = {m: i for i, m in enumerate(METHOD_ORDER)}
    env_order = env_rank_df[["env_name", "surface", "drift", "K"]].drop_duplicates().sort_values(["surface", "drift", "K", "env_name"])["env_name"].tolist()
    method_order = sorted(env_rank_df["method"].dropna().astype(str).unique().tolist(), key=lambda m: method_rank.get(m, 999))
    pivot = env_rank_df.pivot(index="env_name", columns="method", values="env_rank").reindex(index=env_order, columns=method_order)
    values = pivot.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.85 * len(method_order)), max(4.8, 0.5 * len(env_order))), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", origin="upper", cmap="YlOrRd_r", vmin=1.0, vmax=max(float(np.nanmax(values)), 1.0))
    ax.set_title("Repeat-mean method rank by environment", fontsize=14)
    ax.set_xlabel("Method")
    ax.set_ylabel("Environment")
    ax.set_xticks(np.arange(len(method_order)))
    ax.set_xticklabels(method_order, rotation=35, ha="right")
    ax.set_yticks(np.arange(len(env_order)))
    ax.set_yticklabels(env_order)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if not np.isnan(values[i, j]):
                ax.text(j, i, str(int(values[i, j])), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_otl_weight_showcase(traj_df: pd.DataFrame, boundaries_df: pd.DataFrame, showcase_env: str, spec: Dict[str, str], output_path: str) -> None:
    if traj_df.empty:
        return
    plt = _lazy_import_matplotlib()
    method = str(spec["transfer_method"])
    plot_df = traj_df[(traj_df["env_name"] == showcase_env) & (traj_df["method"] == method)].copy().sort_values("step")
    if plot_df.empty or "otl_weight_old" not in plot_df.columns:
        return
    boundaries = boundaries_df[boundaries_df["env_name"] == showcase_env].copy()
    fig, ax = plt.subplots(figsize=(9.2, 5.8), constrained_layout=True)
    ax.plot(plot_df["step"], plot_df["otl_weight_old"], linewidth=2.2, label="old/source weight")
    ax.plot(plot_df["step"], plot_df["otl_weight_new"], linewidth=2.2, label="new/target weight")
    add_segment_lines(ax, boundaries)
    ax.set_title(f"{spec['comparison_label']}\nRoute B OTL expert weights", fontsize=13)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Weight")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_all_ablation_showcase_figures(traj_df: pd.DataFrame, boundaries_df: pd.DataFrame, gap_df: pd.DataFrame, showcase_env: str, output_dir: str, window: int) -> Dict[str, str]:
    ensure_dir(output_dir)
    regret_dir = os.path.join(output_dir, "regret")
    conv_dir = os.path.join(output_dir, "convergence")
    gap_dir = os.path.join(output_dir, "risk_gap")
    weight_dir = os.path.join(output_dir, "otl_weights")
    for d in [regret_dir, conv_dir, gap_dir, weight_dir]:
        ensure_dir(d)
    saved: Dict[str, str] = {}
    for spec in COMPARISON_SPECS:
        slug = safe_slug(spec["comparison_label"])
        regret_path = os.path.join(regret_dir, f"regret_{slug}.png")
        conv_path = os.path.join(conv_dir, f"convergence_{slug}.png")
        gap_path = os.path.join(gap_dir, f"risk_gap_{slug}.png")
        weight_path = os.path.join(weight_dir, f"otl_weights_{slug}.png")
        plot_ablation_regret_showcase(traj_df, boundaries_df, showcase_env, spec, regret_path)
        plot_ablation_convergence_showcase(traj_df, boundaries_df, showcase_env, spec, conv_path, window)
        plot_ablation_gap_showcase(gap_df, boundaries_df, showcase_env, spec, gap_path)
        plot_otl_weight_showcase(traj_df, boundaries_df, showcase_env, spec, weight_path)
        saved[f"regret_{slug}"] = regret_path
        saved[f"convergence_{slug}"] = conv_path
        saved[f"risk_gap_{slug}"] = gap_path
        saved[f"otl_weights_{slug}"] = weight_path
    return saved


# =========================================================
# Saving
# =========================================================


def write_summary_json(stage1_env_df: pd.DataFrame, stage2_mean_df: pd.DataFrame, env_method_mean_df: pd.DataFrame, env_best_df: pd.DataFrame, pairwise_best_df: pd.DataFrame, output_path: str) -> None:
    stage1_by_env = {env: g.to_dict(orient="records") for env, g in stage1_env_df.groupby("env_name")} if not stage1_env_df.empty else {}
    stage2_by_env = {env: g.to_dict(orient="records") for env, g in stage2_mean_df.groupby("env_name")} if not stage2_mean_df.empty else {}
    final_by_env = {env: g.to_dict(orient="records") for env, g in env_method_mean_df.groupby("env_name")} if not env_method_mean_df.empty else {}
    pairwise_by_env = {env: g.to_dict(orient="records") for env, g in pairwise_best_df.groupby("env_name")} if not pairwise_best_df.empty else {}
    env_best_map = {str(r["env_name"]): r.dropna().to_dict() for _, r in env_best_df.iterrows()} if not env_best_df.empty else {}
    all_envs = sorted(set(stage1_by_env) | set(stage2_by_env) | set(final_by_env) | set(pairwise_by_env))
    payload = []
    for env in all_envs:
        payload.append({
            "env_name": env,
            "stage1_target_only_repeat_mean": stage1_by_env.get(env, []),
            "stage2_exact_otl_repeat_mean": stage2_by_env.get(env, []),
            "final_method_repeat_mean": final_by_env.get(env, []),
            "pairwise_repeat_mean": pairwise_by_env.get(env, []),
            "env_best_repeat_mean": env_best_map.get(env, {}),
        })
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, default=json_default_safe)


def save_outputs(
    raw_stage1_df: pd.DataFrame,
    stage1_env_df: pd.DataFrame,
    raw_stage2_df: pd.DataFrame,
    stage2_mean_df: pd.DataFrame,
    raw_method_df: pd.DataFrame,
    env_method_mean_df: pd.DataFrame,
    env_best_df: pd.DataFrame,
    raw_traj_df: pd.DataFrame,
    traj_mean_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    pairwise_best_df: pd.DataFrame,
    pairwise_overall_df: pd.DataFrame,
    pairwise_gap_df: pd.DataFrame,
    env_rank_df: pd.DataFrame,
    output_dir: str,
    enable_plots: bool,
    showcase_env: str,
    smooth_window: int,
) -> Dict[str, str]:
    ensure_dir(output_dir)
    raw_dir = os.path.join(output_dir, "tables", "raw_repeat")
    sum_dir = os.path.join(output_dir, "tables", "repeat_summary")
    ensure_dir(raw_dir)
    ensure_dir(sum_dir)

    paths = {
        "raw_stage1_csv": os.path.join(raw_dir, "stage1_target_only_raw.csv"),
        "raw_stage2_csv": os.path.join(raw_dir, "stage2_exact_otl_raw.csv"),
        "raw_method_csv": os.path.join(raw_dir, "final_method_metrics_raw.csv"),
        "raw_traj_csv": os.path.join(raw_dir, "comparison_trajectories_raw.csv"),
        "stage1_env_csv": os.path.join(sum_dir, "stage1_target_only_repeat_mean.csv"),
        "stage2_env_csv": os.path.join(sum_dir, "stage2_exact_otl_repeat_mean.csv"),
        "env_method_mean_csv": os.path.join(sum_dir, "all_methods_repeat_mean.csv"),
        "env_best_csv": os.path.join(sum_dir, "env_best_repeat_mean.csv"),
        "traj_mean_csv": os.path.join(sum_dir, "showcase_trajectories_repeat_mean.csv"),
        "boundary_csv": os.path.join(sum_dir, "segment_boundaries.csv"),
        "pairwise_best_csv": os.path.join(sum_dir, "pairwise_best_repeat_mean.csv"),
        "pairwise_overall_csv": os.path.join(sum_dir, "pairwise_overall_repeat_mean.csv"),
        "pairwise_gap_csv": os.path.join(sum_dir, "showcase_pairwise_running_risk_gap_repeat_mean.csv"),
        "env_rank_csv": os.path.join(sum_dir, "environment_method_rank_repeat_mean.csv"),
        "summary_json": os.path.join(output_dir, "summary_repeat_mean.json"),
    }

    raw_stage1_df.to_csv(paths["raw_stage1_csv"], index=False)
    raw_stage2_df.to_csv(paths["raw_stage2_csv"], index=False)
    raw_method_df.to_csv(paths["raw_method_csv"], index=False)
    raw_traj_df.to_csv(paths["raw_traj_csv"], index=False)
    stage1_env_df.to_csv(paths["stage1_env_csv"], index=False)
    stage2_mean_df.to_csv(paths["stage2_env_csv"], index=False)
    env_method_mean_df.to_csv(paths["env_method_mean_csv"], index=False)
    env_best_df.to_csv(paths["env_best_csv"], index=False)
    traj_mean_df.to_csv(paths["traj_mean_csv"], index=False)
    boundaries_df.to_csv(paths["boundary_csv"], index=False)
    pairwise_best_df.to_csv(paths["pairwise_best_csv"], index=False)
    pairwise_overall_df.to_csv(paths["pairwise_overall_csv"], index=False)
    pairwise_gap_df.to_csv(paths["pairwise_gap_csv"], index=False)
    env_rank_df.to_csv(paths["env_rank_csv"], index=False)

    write_summary_json(stage1_env_df, stage2_mean_df, env_method_mean_df, env_best_df, pairwise_best_df, paths["summary_json"])

    if enable_plots:
        fig_root = os.path.join(output_dir, "figures")
        ensure_dir(fig_root)
        ensure_dir(os.path.join(fig_root, "ablation_showcase"))
        ensure_dir(os.path.join(fig_root, "env_rank"))
        rank_fig = os.path.join(fig_root, "env_rank", "environment_method_rank_heatmap.png")
        plot_environment_rank_heatmap(env_rank_df, rank_fig)
        paths["env_rank_fig"] = rank_fig
        paths.update(
            plot_all_ablation_showcase_figures(
                traj_mean_df,
                boundaries_df,
                pairwise_gap_df,
                showcase_env,
                os.path.join(fig_root, "ablation_showcase"),
                smooth_window,
            )
        )

    paths["showcase_env"] = showcase_env
    return paths


# =========================================================
# Main
# =========================================================


def main() -> None:
    args = parse_args()
    enable_plots = not args.disable_plots

    target_data_dir = args.target_data_dir or resolve_data_dir([
        "ihdp_target_streams_indep",
        "/mnt/data/ihdp_target_streams_indep",
        "ihdp_ethos_streams_indep",
        "/mnt/data/ihdp_ethos_streams_indep",
        "ihdp_ethos_streams",
        "/mnt/data/ihdp_ethos_streams",
    ])
    good_source_dir = args.good_source_dir or resolve_data_dir([
        "ihdp_ethos_streams_good_indep",
        "/mnt/data/ihdp_ethos_streams_good_indep",
        "ihdp_ethos_streams(good)",
        "/mnt/data/ihdp_ethos_streams(good)",
    ])

    env_repeat_files = discover_environment_files(target_data_dir)
    repeat_inventory = validate_repeat_inventory(env_repeat_files, args.n_repeats)
    env_files = unique_base_envs(env_repeat_files)
    showcase_env = select_showcase_environment(env_files, args.showcase_env)
    n_jobs = effective_n_jobs(args.n_jobs)

    base_tasks = []
    for env_info in env_repeat_files:
        repeat_id = int(env_info["repeat"])
        algo_seed = int(args.base_seed) + repeat_id - 1
        base_tasks.append(
            (
                env_info,
                str(env_info["full_path"]),
                resolve_source_path(good_source_dir, env_info, tag="good"),
                algo_seed,
            )
        )

    print(f"[INFO] TARGET_DATA_DIR      = {target_data_dir}")
    print(f"[INFO] GOOD_SOURCE_DIR     = {good_source_dir}")
    print(f"[INFO] ENV_COUNT           = {len(env_files)}")
    print(f"[INFO] N_REPEATS           = {args.n_repeats}")
    print(f"[INFO] TASK_COUNT          = {len(base_tasks)}")
    print(f"[INFO] THEORETICAL_C_ETA   = {THEORY_C_ETA:.12f}")
    print(f"[INFO] OTL_META_ETA        = {OTL_META_ETA:.6f}")
    print(f"[INFO] OTL_LOSS_VARIANT    = {OTL_LOSS_VARIANT}")
    print(f"[INFO] SHOWCASE_ENV        = {showcase_env}")
    print(f"[INFO] REPEAT_INVENTORY    = { {k: v for k, v in sorted(repeat_inventory.items())} }")

    stage1_tasks = [
        (env_info, target_path, good_path, seed, args.hidden_dim, args.feature_batch_size, args.force_linear)
        for env_info, target_path, good_path, seed in base_tasks
    ]
    stage1_results = sorted(
        run_tasks_with_pool(run_stage1_repeat, stage1_tasks, n_jobs),
        key=lambda x: (x["env_name"], x["repeat"], x["seed"])
    )
    raw_stage1_df, stage1_env_df, _ = build_stage1_outputs(stage1_results)

    stage2_tasks = [
        (env_info, target_path, good_path, seed, args.hidden_dim, args.feature_batch_size, args.force_linear)
        for env_info, target_path, good_path, seed in base_tasks
    ]
    stage2_results = sorted(
        run_tasks_with_pool(run_stage2_repeat, stage2_tasks, n_jobs),
        key=lambda x: (x["env_name"], x["repeat"], x["seed"])
    )
    raw_stage2_df, stage2_mean_df = build_stage2_outputs(stage2_results)

    final_tasks = []
    for env_info, target_path, good_path, seed in base_tasks:
        env_name = str(env_info["env_name"])
        record_trajectory = bool(args.save_all_trajectories or env_name == showcase_env)
        final_tasks.append((
            env_info,
            target_path,
            good_path,
            seed,
            args.hidden_dim,
            args.feature_batch_size,
            args.force_linear,
            record_trajectory,
        ))

    final_results = sorted(
        run_tasks_with_pool(run_final_repeat, final_tasks, n_jobs),
        key=lambda x: (x["env_name"], x["repeat"], x["seed"])
    )
    raw_method_df = pd.DataFrame([row for res in final_results for row in res["rows"]])
    raw_traj_df = concat_nonempty([res["trajectories"] for res in final_results])
    boundaries_df = (
        concat_nonempty([res["segment_boundaries"] for res in final_results])
        .drop_duplicates()
        .sort_values(["env_name", "segment"])
        .reset_index(drop=True)
    )

    env_method_mean_df, env_best_df = aggregate_final_method_results(raw_method_df)
    traj_mean_df = aggregate_trajectory_df(raw_traj_df)
    pairwise_best_df = build_pairwise_best_df(env_method_mean_df)
    pairwise_overall_df = build_pairwise_overall_summary(pairwise_best_df)
    pairwise_gap_df = build_pairwise_running_gap(traj_mean_df)
    env_rank_df = build_env_rank_table(env_method_mean_df)

    saved = save_outputs(
        raw_stage1_df,
        stage1_env_df,
        raw_stage2_df,
        stage2_mean_df,
        raw_method_df,
        env_method_mean_df,
        env_best_df,
        raw_traj_df,
        traj_mean_df,
        boundaries_df,
        pairwise_best_df,
        pairwise_overall_df,
        pairwise_gap_df,
        env_rank_df,
        args.output_dir,
        enable_plots,
        showcase_env,
        args.smooth_window,
    )

    print("\n[PAIRWISE OVERALL SUMMARY]")
    if pairwise_overall_df.empty:
        print("empty")
    else:
        for _, row in pairwise_overall_df.iterrows():
            print(
                f"{row['comparison_label']}: "
                f"transfer_win_rate={float(row['transfer_win_rate']):.3f}, "
                f"win/tie/loss={int(row['transfer_win_count'])}/{int(row['tie_count'])}/{int(row['loss_count'])}, "
                f"mean_transfer_gain={float(row['mean_transfer_abs_gain']):.6f}, "
                f"median_transfer_gain={float(row['median_transfer_abs_gain']):.6f}, "
                f"mean_transfer_rel_gain={float(row['mean_transfer_rel_gain_pct']):.2f}%"
            )

    print("\n[FILES SAVED]")
    for k, v in saved.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    mp.freeze_support()
    main()