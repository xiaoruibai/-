# -*- coding: utf-8 -*-
"""
style offline TL baseline with fairer prequential block evaluation.

What this script does
---------------------
1) Keeps PREQUENTIAL evaluation on the FULL target stream:
   - At each target step t, predict first.
   - Only after predicting does sample t become available for future fitting.
   - Metrics are emitted over the whole stream: cumulative 0.5 * squared error
     (cumhalfPEHE), running MSE / RMSE.

2) Matches the user's target-only offline baselines in evaluation/update protocol:
   - No expanding-history refits.
   - No sliding window over old history.
   - No hindsight hyperparameter search.
   - Fixed target update window = 100.
   - Refit ONLY after each block of 100 NEW target samples arrives.
   - Each refit uses exactly that just-arrived block of 100 samples, then moves on.

3) Keeps the requested source-side DR anchor change:
   - Source alpha0_s / alpha1_s / gamma_s are fitted in ONE PASS,
     aligned with the online ADAPTIVE_DR source-anchor order:
       current nuisance prediction -> current DR pseudo-outcome -> update gamma_s ->
       update observed-arm nuisance.

Interpretation
--------------
This is an offline TL method evaluated with the same block-prequential protocol as the
user's latest multi-method offline baselines:
- the prediction for each target point is made by the currently deployed TL model,
- every target point contributes exactly one prequential error,
- after 100 NEW target samples have arrived, we solve one anchored batch TL problem on
  that just-finished block and deploy the new model for the next block.
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
import os
import re
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
DEFAULT_OUTPUT_DIR = "ihdp_l1_tldr_fixed_window100_preq_outputs"

DEFAULT_HIDDEN_DIM = 128
DEFAULT_BATCH_SIZE = 128

EPS = 0.05
EPS_NUM = 1e-8

DEFAULT_WINDOW_SIZE = 50
DEFAULT_SOURCE_C_ETA = 1.0 / math.sqrt(2.0)
DEFAULT_TARGET_C_ETA = 1.0 / math.sqrt(2.0)  # CLI compatibility / documentation only

DEFAULT_LAMBDA_IPW = 0.02
DEFAULT_LAMBDA_OR = 0.02
DEFAULT_LAMBDA_DR = 0.02

DEFAULT_MAX_PROX_ITERS = 200
DEFAULT_PROX_TOL = 1e-6

META_COLS = {
    "env_name", "surface", "drift", "K",
    "repeat", "seed",
    "segment", "segment_round", "t",
    "alpha", "source_row", "q",
    "w", "p", "y", "mu_-1", "mu_1", "tau",
}

REPEAT_ENV_FILE_REGEX = re.compile(r"^(ihdp_(linear|nonlinear)_(switching|linear)_K(\d+))_repeat(\d+)\.csv$")
LEGACY_ENV_FILE_REGEX = re.compile(r"^(ihdp_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv)$")

METHOD_ORDER = [
    "L1_TLDR_PQ_FIXED100_PREQ",
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


def get_scale_free_eta(acc_sq: float, radius: float, c_eta: float) -> float:
    return float(c_eta * radius / math.sqrt(acc_sq + EPS_NUM))


def family_from_method(method: str) -> str:
    return "IPW" if "IPW" in method else "DR"


def soft_threshold_vec(x: np.ndarray, threshold: float) -> np.ndarray:
    return np.sign(x) * np.maximum(np.abs(x) - float(threshold), 0.0)


# =========================================================
# Data loading
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ihdp offline TL with fixed-window prequential evaluation using oracle p/q")
    parser.add_argument("--target-data-dir", type=str, default=None)
    parser.add_argument("--good-source-dir", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)

    parser.add_argument("--base-seed", type=int, default=BASE_SEED)
    parser.add_argument("--n-repeats", type=int, default=N_REPEATS)
    parser.add_argument("--n-jobs", type=int, default=DEFAULT_N_JOBS)

    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--feature-batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--force-linear", action="store_true")

    parser.add_argument("--window-size", type=int, default=DEFAULT_WINDOW_SIZE)

    parser.add_argument("--source-c-eta", type=float, default=DEFAULT_SOURCE_C_ETA)
    parser.add_argument("--target-c-eta", type=float, default=DEFAULT_TARGET_C_ETA)

    parser.add_argument("--lambda-ipw", type=float, default=DEFAULT_LAMBDA_IPW)
    parser.add_argument("--lambda-or", type=float, default=DEFAULT_LAMBDA_OR)
    parser.add_argument("--lambda-dr", type=float, default=DEFAULT_LAMBDA_DR)

    parser.add_argument("--max-prox-iters", type=int, default=DEFAULT_MAX_PROX_ITERS)
    parser.add_argument("--prox-tol", type=float, default=DEFAULT_PROX_TOL)

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
            "Found legacy ihdp environment files without repeat suffix, but no independent-repeat files. "
            "Expected files like ihdp_linear_switching_K5_repeat01.csv. "
            f"Examples found instead: {sorted(legacy_hits)[:5]}"
        )

    raise FileNotFoundError(f"No ihdp independent-repeat environment files found in {data_dir}")


def validate_repeat_inventory(env_files: List[Dict[str, object]], expected_n_repeats: int) -> Dict[str, List[int]]:
    repeat_map: Dict[str, List[int]] = defaultdict(list)
    for info in env_files:
        repeat_map[str(info["env_name"])].append(int(info["repeat"]))

    problems: List[str] = []
    for env_name, repeats in sorted(repeat_map.items()):
        repeats_sorted = sorted(repeats)
        if len(repeats_sorted) != expected_n_repeats:
            problems.append(f"{env_name}: found {len(repeats_sorted)} repeats {repeats_sorted}, expected {expected_n_repeats}")
        elif repeats_sorted != list(range(1, expected_n_repeats + 1)):
            problems.append(f"{env_name}: repeats are {repeats_sorted}, expected contiguous 1..{expected_n_repeats}")

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

    source_filename = target_filename.replace("ihdp_", "ihdp_good_") if tag == "good" else target_filename.replace("ihdp_", "ihdp_good_")
    exact_path = os.path.join(source_dir, source_filename)
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

    raise FileNotFoundError(f"Cannot find source file aligned with target repeat file {target_filename} under {source_dir}")


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
    return df.sort_values(["segment", "segment_round", "t"]).reset_index(drop=True)


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in META_COLS]


def init_random_mlp(input_dim: int, seed: int, hidden_dim: int) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    dims = [input_dim, 256, 256, 256, hidden_dim]
    weights: List[np.ndarray] = []
    biases: List[np.ndarray] = []
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
    outputs: List[np.ndarray] = []
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
    q = np.clip(df["q"].to_numpy(dtype=np.float32), EPS, 1.0 - EPS) if "q" in df.columns else np.where(w == 1.0, p, 1.0 - p).astype(np.float32)
    return {
        "df": df,
        "Phi": Phi,
        "w": w,
        "w_unique": np.unique(w).astype(np.float32),
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

    def add(self, pred_tau: float, tau_true: float, segment: int, segment_round: int, t: int, extra: Optional[Dict[str, object]] = None) -> None:
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


def unpack_stream(env: Dict[str, object]) -> Tuple[np.ndarray, ...]:
    return (
        env["Phi"], env["w"], env["p"], env["q"], env["y"], env["tau"],
        env["segment"], env["segment_round"], env["t"],
    )


def predict_tau(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(np.dot(theta, phi))


def predict_outcome(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(np.dot(theta, phi))


def compute_ipw_pseudo_oracle(w: np.ndarray, y: np.ndarray, p: np.ndarray) -> np.ndarray:
    return (w * y / p).astype(np.float32)


def compute_dr_pseudo_oracle(w: np.ndarray, y: np.ndarray, q: np.ndarray, m0_hat: np.ndarray, m1_hat: np.ndarray) -> np.ndarray:
    out = np.empty_like(y, dtype=np.float32)
    treat_mask = (w == 1.0)
    out[treat_mask] = ((m1_hat[treat_mask] - m0_hat[treat_mask]) + (y[treat_mask] - m1_hat[treat_mask]) / q[treat_mask]).astype(np.float32)
    out[~treat_mask] = ((m1_hat[~treat_mask] - m0_hat[~treat_mask]) - (y[~treat_mask] - m0_hat[~treat_mask]) / (1.0 - q[~treat_mask])).astype(np.float32)
    return out.astype(np.float32)


def update_ogd_squared(theta: np.ndarray, phi: np.ndarray, target: float, acc_sq: float, c_eta: float, radius: float = 1.0) -> Tuple[np.ndarray, float]:
    pred = float(np.dot(theta, phi))
    grad = ((pred - target) * phi).astype(np.float32)
    acc_sq += float(np.dot(grad, grad))
    eta = get_scale_free_eta(acc_sq, radius, c_eta)
    theta = (theta - eta * grad).astype(np.float32, copy=False)
    return theta, float(acc_sq)


def solve_anchor_l1_least_squares(
    Phi: np.ndarray,
    target: np.ndarray,
    anchor: np.ndarray,
    lam: float,
    init_theta: Optional[np.ndarray] = None,
    max_iter: int = DEFAULT_MAX_PROX_ITERS,
    tol: float = DEFAULT_PROX_TOL,
) -> np.ndarray:
    d = int(anchor.shape[0])
    if Phi.shape[0] == 0:
        return anchor.astype(np.float32, copy=True)

    rhs = target.astype(np.float32) - Phi @ anchor.astype(np.float32)

    if init_theta is None or init_theta.shape != anchor.shape:
        delta = np.zeros(d, dtype=np.float32)
    else:
        delta = (init_theta.astype(np.float32) - anchor.astype(np.float32)).copy()

    z = delta.copy()
    t_k = 1.0

    gram_scale = float(np.linalg.norm(Phi.astype(np.float64), ord=2) ** 2) if Phi.shape[0] > 0 else 1.0
    step = 1.0 / max(gram_scale / max(Phi.shape[0], 1), 1e-6)
    lam_step = step * float(max(lam, 0.0))

    for _ in range(max(int(max_iter), 1)):
        resid = Phi @ z - rhs
        grad = (Phi.T @ resid) / max(Phi.shape[0], 1)
        delta_next = soft_threshold_vec(z - step * grad.astype(np.float32), lam_step).astype(np.float32)

        if np.linalg.norm(delta_next - delta) <= tol * max(1.0, np.linalg.norm(delta)):
            delta = delta_next
            break

        t_next = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t_k * t_k))
        z = delta_next + ((t_k - 1.0) / max(t_next, EPS_NUM)) * (delta_next - delta)
        delta = delta_next
        t_k = t_next

    return (anchor.astype(np.float32) + delta.astype(np.float32)).astype(np.float32)


# =========================================================
# Source rough estimators (one-pass DR anchor)
# =========================================================


def fit_source_gamma_dr_ogd(env_source: Dict[str, object], c_eta: float) -> Dict[str, np.ndarray]:
    """One-pass source DR anchor fit aligned to the online ADAPTIVE_DR runner."""
    Phi, w, _p, q, y, _tau, *_ = unpack_stream(env_source)
    d = Phi.shape[1]

    alpha0_s = np.zeros(d, dtype=np.float32)
    alpha1_s = np.zeros(d, dtype=np.float32)
    gamma_s = np.zeros(d, dtype=np.float32)
    acc0_sq = 0.0
    acc1_sq = 0.0
    acc_gamma_sq = 0.0

    for idx, phi in enumerate(Phi):
        m0_hat = predict_outcome(alpha0_s, phi)
        m1_hat = predict_outcome(alpha1_s, phi)
        pseudo_i = compute_dr_pseudo_oracle(
            np.asarray([w[idx]], dtype=np.float32),
            np.asarray([y[idx]], dtype=np.float32),
            np.asarray([q[idx]], dtype=np.float32),
            np.asarray([m0_hat], dtype=np.float32),
            np.asarray([m1_hat], dtype=np.float32),
        )[0]

        gamma_s, acc_gamma_sq = update_ogd_squared(gamma_s, phi, float(pseudo_i), acc_gamma_sq, c_eta)

        if float(w[idx]) == 1.0:
            alpha1_s, acc1_sq = update_ogd_squared(alpha1_s, phi, float(y[idx]), acc1_sq, c_eta)
        else:
            alpha0_s, acc0_sq = update_ogd_squared(alpha0_s, phi, float(y[idx]), acc0_sq, c_eta)

    return {
        "alpha0_s": alpha0_s.astype(np.float32),
        "alpha1_s": alpha1_s.astype(np.float32),
        "gamma_s": gamma_s.astype(np.float32),
    }


# =========================================================
# Target batch TL fitting on one fresh block
# =========================================================


def fit_target_dr_batch(
    Phi_hist: np.ndarray,
    w_hist: np.ndarray,
    y_hist: np.ndarray,
    q_hist: np.ndarray,
    alpha0_s: np.ndarray,
    alpha1_s: np.ndarray,
    gamma_s: np.ndarray,
    lambda_or: float,
    lambda_dr: float,
    init_alpha0: Optional[np.ndarray],
    init_alpha1: Optional[np.ndarray],
    init_gamma: Optional[np.ndarray],
    max_prox_iters: int,
    prox_tol: float,
) -> Dict[str, np.ndarray]:
    mask1 = w_hist == 1.0
    mask0 = w_hist != 1.0

    alpha0_t = alpha0_s.astype(np.float32).copy()
    alpha1_t = alpha1_s.astype(np.float32).copy()

    if mask0.sum() > 0:
        alpha0_t = solve_anchor_l1_least_squares(
            Phi_hist[mask0].astype(np.float32),
            y_hist[mask0].astype(np.float32),
            alpha0_s.astype(np.float32),
            lam=float(lambda_or),
            init_theta=init_alpha0 if init_alpha0 is not None else alpha0_s,
            max_iter=max_prox_iters,
            tol=prox_tol,
        )
    if mask1.sum() > 0:
        alpha1_t = solve_anchor_l1_least_squares(
            Phi_hist[mask1].astype(np.float32),
            y_hist[mask1].astype(np.float32),
            alpha1_s.astype(np.float32),
            lam=float(lambda_or),
            init_theta=init_alpha1 if init_alpha1 is not None else alpha1_s,
            max_iter=max_prox_iters,
            tol=prox_tol,
        )

    m0_hist = Phi_hist @ alpha0_t
    m1_hist = Phi_hist @ alpha1_t
    pseudo_hist = compute_dr_pseudo_oracle(
        w_hist.astype(np.float32),
        y_hist.astype(np.float32),
        q_hist.astype(np.float32),
        m0_hist.astype(np.float32),
        m1_hist.astype(np.float32),
    )

    gamma_t = solve_anchor_l1_least_squares(
        Phi_hist.astype(np.float32),
        pseudo_hist.astype(np.float32),
        gamma_s.astype(np.float32),
        lam=float(lambda_dr),
        init_theta=init_gamma if init_gamma is not None else gamma_s,
        max_iter=max_prox_iters,
        tol=prox_tol,
    )

    return {
        "alpha0_t": alpha0_t.astype(np.float32),
        "alpha1_t": alpha1_t.astype(np.float32),
        "gamma_t": gamma_t.astype(np.float32),
    }


# =========================================================
# Fixed-window prequential runner
# =========================================================


def run_fixed_window_tl_preq_core(
    env_name: str,
    seed: int,
    env_target: Dict[str, object],
    env_good: Dict[str, object],
    method: str,
    window_size: int,
    source_c_eta: float,
    lambda_ipw: float,
    lambda_or: float,
    lambda_dr: float,
    max_prox_iters: int,
    prox_tol: float,
    record_trajectory: bool = False,
    repeat: Optional[int] = None,
) -> Dict[str, object]:
    del lambda_ipw  # DR-only runner in this script
    Phi_t, w_t, p_t, q_t, y_t, tau_t, segment, segment_round, original_t = unpack_stream(env_target)
    _ = p_t

    tracker = MetricTracker(env_name, method, seed, record_trajectory, repeat=repeat)

    if method != "L1_TLDR_PQ_FIXED100_PREQ":
        raise ValueError(f"Unknown method: {method}")

    if int(window_size) <= 0:
        raise ValueError("window_size must be positive")

    source_state = fit_source_gamma_dr_ogd(env_good, c_eta=source_c_eta)
    alpha0_s = source_state["alpha0_s"]
    alpha1_s = source_state["alpha1_s"]
    gamma_s = source_state["gamma_s"]

    current_alpha0 = alpha0_s.copy()
    current_alpha1 = alpha1_s.copy()
    current_gamma = gamma_s.copy()

    block_start = 0
    samples_in_current_block = 0

    for idx, phi in enumerate(Phi_t):
        pred = predict_tau(current_gamma, phi)
        extra = None
        if record_trajectory:
            extra = {
                "theta_norm": float(np.linalg.norm(current_gamma)),
                "alpha0_norm": float(np.linalg.norm(current_alpha0)),
                "alpha1_norm": float(np.linalg.norm(current_alpha1)),
                "history_size_before": int(samples_in_current_block),
                "source_theta_norm": float(np.linalg.norm(gamma_s)),
                "model_origin": "source_anchor" if block_start == 0 and samples_in_current_block < window_size else "fixed_window_block_fit",
                "current_block_size_before": int(samples_in_current_block),
                "current_block_start_index": int(block_start),
            }
        tracker.add(pred, float(tau_t[idx]), int(segment[idx]), int(segment_round[idx]), int(original_t[idx]), extra=extra)

        # only after predicting does sample idx become available for future fitting
        samples_in_current_block += 1

        # match the user's latest offline baselines: refit only after 100 NEW samples
        if samples_in_current_block < window_size:
            continue

        fit_idx = np.arange(block_start, idx + 1, dtype=np.int64)
        target_state = fit_target_dr_batch(
            Phi_hist=Phi_t[fit_idx],
            w_hist=w_t[fit_idx],
            y_hist=y_t[fit_idx],
            q_hist=q_t[fit_idx],
            alpha0_s=alpha0_s,
            alpha1_s=alpha1_s,
            gamma_s=gamma_s,
            lambda_or=lambda_or,
            lambda_dr=lambda_dr,
            init_alpha0=current_alpha0,
            init_alpha1=current_alpha1,
            init_gamma=current_gamma,
            max_prox_iters=max_prox_iters,
            prox_tol=prox_tol,
        )
        current_alpha0 = target_state["alpha0_t"]
        current_alpha1 = target_state["alpha1_t"]
        current_gamma = target_state["gamma_t"]

        block_start = idx + 1
        samples_in_current_block = 0

    out = tracker.summary()
    out.update({
        "alpha0_s": alpha0_s.astype(np.float32),
        "alpha1_s": alpha1_s.astype(np.float32),
        "gamma_s": gamma_s.astype(np.float32),
        "alpha0_t_final": current_alpha0.astype(np.float32),
        "alpha1_t_final": current_alpha1.astype(np.float32),
        "gamma_t_final": current_gamma.astype(np.float32),
        "window_size": int(window_size),
        "lambda_or": float(lambda_or),
        "lambda_dr": float(lambda_dr),
        "uses_oracle_pq": True,
        "treatment_coding": "raw_stream_coding_kept_no_conversion",
        "training_regime": "fixed_disjoint_block_prequential_evaluation_no_hindsight_search",
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

    df = df.copy()
    for c in group_cols:
        if c not in df.columns:
            df[c] = np.nan
    for c in metric_cols:
        if c not in df.columns:
            df[c] = np.nan

    agg_metric_cols = [c for c in metric_cols if c not in group_cols]
    grouped = df.groupby(group_cols, as_index=False, dropna=False).agg({c: ["mean", "std"] for c in agg_metric_cols})
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns]
    counts = df.groupby(group_cols, as_index=False, dropna=False).size().rename(columns={"size": repeat_col_name})
    grouped = pd.merge(grouped, counts, on=group_cols, how="left")
    for c in agg_metric_cols:
        std_col = f"{c}_std"
        if std_col in grouped.columns:
            grouped[std_col] = grouped[std_col].fillna(0.0)
    return grouped


def run_repeat(task: Tuple) -> Dict[str, object]:
    (
        env_info, target_path, good_path, seed, hidden_dim, batch_size, force_linear,
        window_size, source_c_eta, lambda_ipw, lambda_or, lambda_dr,
        max_prox_iters, prox_tol, record_trajectory,
    ) = task

    env_name = str(env_info["env_name"])
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None
    bundle = load_bundle(target_path, good_path, seed, hidden_dim, batch_size, force_linear)
    env_target = bundle["env_target"]
    env_good = bundle["env_good"]

    rows: List[Dict[str, object]] = []
    trajs: List[pd.DataFrame] = []

    out_dr = run_fixed_window_tl_preq_core(
        env_name=env_name,
        seed=seed,
        env_target=env_target,
        env_good=env_good,
        method="L1_TLDR_PQ_FIXED100_PREQ",
        window_size=window_size,
        source_c_eta=source_c_eta,
        lambda_ipw=lambda_ipw,
        lambda_or=lambda_or,
        lambda_dr=lambda_dr,
        max_prox_iters=max_prox_iters,
        prox_tol=prox_tol,
        record_trajectory=record_trajectory,
        repeat=repeat,
    )
    rows.append(attach_env_meta({
        "method": "L1_TLDR_PQ_FIXED100_PREQ",
        "cumhalfPEHE": float(out_dr["cumhalfPEHE"]),
        "running_mean_mse": float(out_dr["running_mean_mse"]),
        "running_rmse": float(out_dr["running_rmse"]),
        "window_size": int(out_dr["window_size"]),
        "source_c_eta": float(source_c_eta),
        "lambda_ipw": None,
        "lambda_or": float(out_dr["lambda_or"]),
        "lambda_dr": float(out_dr["lambda_dr"]),
        "uses_oracle_pq": int(bool(out_dr["uses_oracle_pq"])),
        "training_regime": str(out_dr["training_regime"]),
        "selection_basis": "fixed_hyperparameters_no_hindsight_tuning",
    }, env_info, seed))
    if record_trajectory:
        trajs.append(out_dr["trajectory_df"])

    return {
        "env_name": env_name,
        "seed": int(seed),
        "repeat": repeat,
        "rows": rows,
        "trajectories": concat_nonempty(trajs),
        "segment_boundaries": build_segment_boundaries(env_target["df"], env_name),
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
# Aggregation / saving
# =========================================================


def aggregate_method_results(raw_method_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if raw_method_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    group_cols = [
        "env_name", "dataset", "surface", "drift", "K", "method", "family",
        "selection_basis", "training_regime", "uses_oracle_pq",
        "window_size", "source_c_eta", "lambda_ipw", "lambda_or", "lambda_dr",
    ]
    metric_cols = [
        "cumhalfPEHE", "running_mean_mse", "running_rmse"
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
        "theta_norm", "alpha0_norm", "alpha1_norm", "history_size_before", "source_theta_norm",
        "current_block_size_before", "current_block_start_index",
    ]
    agg = aggregate_metric_df(raw_traj_df, group_cols, metric_cols)
    rename_map = {f"{c}_mean": c for c in metric_cols}
    out = agg.rename(columns=rename_map).sort_values(["env_name", "method", "step"]).reset_index(drop=True)
    if "model_origin" in raw_traj_df.columns:
        origin_df = raw_traj_df[group_cols + ["model_origin"]].drop_duplicates(subset=group_cols)
        out = out.merge(origin_df, on=group_cols, how="left")
    return out


def save_outputs(
    raw_method_df: pd.DataFrame,
    env_method_mean_df: pd.DataFrame,
    env_best_df: pd.DataFrame,
    raw_traj_df: pd.DataFrame,
    traj_mean_df: pd.DataFrame,
    boundaries_df: pd.DataFrame,
    output_dir: str,
    run_config: Dict[str, Any],
) -> Dict[str, str]:
    ensure_dir(output_dir)
    saved: Dict[str, str] = {}
    items = {
        "raw_method_results": raw_method_df,
        "env_method_mean": env_method_mean_df,
        "env_best_method": env_best_df,
        "raw_trajectories": raw_traj_df,
        "traj_mean": traj_mean_df,
        "segment_boundaries": boundaries_df,
    }
    for key, df in items.items():
        path = os.path.join(output_dir, f"{key}.csv")
        df.to_csv(path, index=False)
        saved[key] = path

    config_path = os.path.join(output_dir, "run_config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(run_config, f, ensure_ascii=False, indent=2, default=json_default_safe)
    saved["run_config"] = config_path
    return saved


# =========================================================
# Main
# =========================================================


def main() -> None:
    args = parse_args()

    target_data_dir = args.target_data_dir or resolve_data_dir(["ihdp_ethos_streams_repeats10_target"])
    good_source_dir = args.good_source_dir or resolve_data_dir(["ihdp_ethos_streams_repeats10_source"])

    env_repeat_files = discover_environment_files(target_data_dir)
    repeat_inventory = validate_repeat_inventory(env_repeat_files, args.n_repeats)
    env_files = unique_base_envs(env_repeat_files)
    n_jobs = effective_n_jobs(args.n_jobs)

    print(f"[INFO] TARGET_DATA_DIR    = {target_data_dir}")
    print(f"[INFO] GOOD_SOURCE_DIR   = {good_source_dir}")
    print(f"[INFO] BASE_ENV_COUNT    = {len(env_files)}")
    print(f"[INFO] TASK_COUNT        = {len(env_repeat_files)}")
    print(f"[INFO] N_REPEATS         = {args.n_repeats}")
    print(f"[INFO] WINDOW_SIZE       = {args.window_size}")
    print(f"[INFO] LAMBDA_OR         = {args.lambda_or}")
    print(f"[INFO] LAMBDA_DR         = {args.lambda_dr}")
    print(f"[INFO] USES_ORACLE_PQ    = True")
    print(f"[INFO] METHOD_SET        = DR only")
    print("[INFO] TREATMENT_CODING = raw stream coding kept as-is (expected w in {-1, 1})")
    print(f"[INFO] SELECTION_BASIS   = fixed_hyperparameters_no_hindsight_tuning")
    print(f"[INFO] REPEAT_INVENTORY  = {repeat_inventory}")

    tasks: List[Tuple] = []
    for env_info in env_repeat_files:
        repeat_id = int(env_info["repeat"])
        algo_seed = int(args.base_seed) + repeat_id - 1
        target_path = str(env_info["full_path"])
        good_path = resolve_source_path(good_source_dir, env_info)
        record_trajectory = bool(args.save_all_trajectories)
        tasks.append((
            env_info, target_path, good_path, algo_seed, args.hidden_dim, args.feature_batch_size, args.force_linear,
            args.window_size, args.source_c_eta, args.lambda_ipw, args.lambda_or, args.lambda_dr,
            args.max_prox_iters, args.prox_tol, record_trajectory
        ))

    results = sorted(run_tasks_with_pool(run_repeat, tasks, n_jobs), key=lambda x: (x["env_name"], x["repeat"], x["seed"]))

    raw_method_df = pd.DataFrame([row for res in results for row in res["rows"]])
    raw_traj_df = concat_nonempty([res["trajectories"] for res in results])
    boundaries_df = concat_nonempty([res["segment_boundaries"] for res in results]).drop_duplicates().sort_values(["env_name", "segment"]).reset_index(drop=True)
    env_method_mean_df, env_best_df = aggregate_method_results(raw_method_df)
    traj_mean_df = aggregate_trajectory_df(raw_traj_df)

    run_config = {
        "target_data_dir": target_data_dir,
        "good_source_dir": good_source_dir,
        "output_dir": args.output_dir,
        "base_seed": args.base_seed,
        "n_repeats": args.n_repeats,
        "n_jobs": n_jobs,
        "hidden_dim": args.hidden_dim,
        "feature_batch_size": args.feature_batch_size,
        "force_linear": bool(args.force_linear),
        "window_size": args.window_size,
        "source_c_eta": args.source_c_eta,
        "target_c_eta_note": "target fit is anchored batch TL on each disjoint fresh block; source rough uses one-pass OGD-family",
        "lambda_ipw": None,
        "lambda_or": args.lambda_or,
        "lambda_dr": args.lambda_dr,
        "max_prox_iters": args.max_prox_iters,
        "prox_tol": args.prox_tol,
        "methods": METHOD_ORDER,
        "uses_oracle_pq": True,
        "treatment_coding": "raw_stream_coding_kept_no_conversion",
        "selection_basis": "fixed_hyperparameters_no_hindsight_tuning",
        "training_regime": "fixed_disjoint_block_prequential_evaluation_no_hindsight_search",
        "repeat_inventory": repeat_inventory,
    }

    saved = save_outputs(
        raw_method_df=raw_method_df,
        env_method_mean_df=env_method_mean_df,
        env_best_df=env_best_df,
        raw_traj_df=raw_traj_df,
        traj_mean_df=traj_mean_df,
        boundaries_df=boundaries_df,
        output_dir=args.output_dir,
        run_config=run_config,
    )

    print("\n[FILES SAVED]")
    for k, v in saved.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    mp.freeze_support()
    main()


