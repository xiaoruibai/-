# -*- coding: utf-8 -*-
"""
 online OTL runner
(Two-expert exponential weighting with raw squared pseudo-loss, time-varying eta_meta, no shared-parameter/share layer, parameter ensemble, independent repeats)

核心原则
1) 保留目标端核心算法、数据读取、特征映射、评估与重复实验框架。
2) 迁移层保留 Zhao & Hoi (ICML 2010) homogeneous OTL 的两专家指数加权骨架，但最终输出改为参数集成：
   - 固定 old/source predictor h
   - 在线学习 new/target predictor f
   - 两专家初始权重 w_old = w_new = 1/2
   - 最终输出使用 theta_mix = w_old * theta_source + w_new * theta_target 的参数集成
   - 每步使用原始 0.5 * squared pseudo-loss 作为元损失（不做额外缩放）
   - eta_meta(t) = sqrt(8 ln 2 / t) / L_max(method)
   - 不使用 shared Hedge / Fixed-Share 的 share/lambda 机制
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
   - OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR

推荐：
python ihdp_otl_exact_adaptive_indep_runner_no_share.py --n-repeats 10 --n 1
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
N_REPEATS = 50
DEFAULT_N_JOBS = 1
DEFAULT_OUTPUT_DIR = "ihdp_otl_dr_repeats50_outputs"
DEFAULT_SMOOTH_WINDOW = 200
DEFAULT_HIDDEN_DIM = 128
DEFAULT_BATCH_SIZE = 128
DEFAULT_TARGET_DATA_DIR = "ihdp_ethos_streams_repeats50_target"
DEFAULT_GOOD_SOURCE_DIR = "ihdp_ethos_streams_repeats50_source"
DEFAULT_BAD_SOURCE_DIR = "ihdp_bad_ethos_streams_repeats50_source"

EPS = 0.05
EPS_NUM = 1e-8

# ETHOS-aligned theory constants.
D = 1.0
G = 1.0
H = 1.0

# Explicit names used by the implementation.
B = 1.0
OBSERVED_OUTCOME_ABS_BOUND = B
HTE_PRED_ABS_BOUND = H
OUTCOME_NUISANCE_PRED_ABS_BOUND = 1.0

# Keep norm bounds only inside adaptive step-size formulas.
# Do NOT use an explicit radius-D/2 projection or an explicit prediction clip.
HTE_PARAM_NORM_BOUND = 1.0
OUTCOME_NUISANCE_PARAM_NORM_BOUND = 1.0

# Keep non-target pieces unchanged; only the target-domain main learner uses sqrt(2).
THEORY_C_ETA = 1.0 / math.sqrt(2.0)

# Two-expert meta layer (no shared-share smoothing)
OTL_INITIAL_WEIGHT = 0.5
OTL_LOSS_VARIANT = "raw_half_squared_pseudo_loss_no_share"
OTL_LAMBDA_SCHEDULE = "none"
OTL_META_ETA_SCHEDULE = "sqrt_8ln2_over_t"
OTL_WEIGHT_UPDATE = "paper_exponential_weights_no_share"
OTL_WEIGHTING_INTERPRETATION = "engineering_hedge_style_adaptive_weighting_no_regret_claim"

# Robustness variants for the meta-loss. The raw variant is the main method.
# The default experiment keeps only the clipped DR robustness check to keep
# the paper story focused on DR-based double-line OTL.
OTL_MAIN_LOSS_MODE = "raw"
OTL_LOSS_MODES = ["raw", "clipped", "empirical_scaled"]
DEFAULT_ROBUST_LOSS_MODES = ["clipped"]
OTL_LOSS_CLIP = 5.0
OTL_EMPIRICAL_SCALE_BETA = 0.95
OTL_EMPIRICAL_SCALED_LOSS_CLIP = 5.0

SOURCE_PROFILES = ["good", "bad"]
MAIN_SOURCE_PROFILE = "good"

DR_TRANSFER_LINE_MODES = ["full", "theta_only", "nuisance_only"]
MAIN_TRANSFER_LINE_MODE = "full"

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
        "comparison_label": "ADAPTIVE_DR <- ADAPTIVE_DR (No Share)",
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
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR_CLIPPED",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR_THETA_ONLY",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR_NUISANCE_ONLY",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR_BAD_SOURCE",
    "OTL_ADAPTIVE_DR_FROM_ADAPTIVE_DR_BAD_SOURCE_CLIPPED",
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


def q25(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.25))


def q75(x: pd.Series) -> float:
    return float(pd.to_numeric(x, errors="coerce").quantile(0.75))


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
    m = str(method).upper()
    return "IPW" if "IPW" in m and "DR" not in m else "DR"


def theoretical_c_eta() -> float:
    return THEORY_C_ETA


def get_scale_free_eta(acc_sq: float, radius: float, c_eta: float) -> float:
    return float(c_eta * radius / math.sqrt(acc_sq + EPS_NUM))


def get_tau_abs_bound(method: str) -> float:
    if method.endswith("IPW"):
        return float(B / EPS)
    return float((B + OUTCOME_NUISANCE_PRED_ABS_BOUND) / EPS)


def get_method_g_tilde(method: str) -> float:
    return float((H + get_tau_abs_bound(method)) * G)


def get_shared_hedge_loss_upper_bound(method: str) -> float:
    tau_abs_bound = get_tau_abs_bound(method)
    return float(0.5 * (H + tau_abs_bound) ** 2)


def get_otl_meta_eta(step_1based: int, method: str) -> float:
    _ = method
    return float(math.sqrt(8.0 * math.log(2.0) / max(int(step_1based), 1)))


# =========================================================
# Data loading
# =========================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ihdp Route-B OTL runner aligned to independent-repeat data (no share layer)")
    parser.add_argument("--target-data-dir", type=str, default=None)
    parser.add_argument("--good-source-dir", type=str, default=None)
    parser.add_argument("--bad-source-dir", type=str, default=None)
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
    parser.add_argument("--disable-bad-source", action="store_true")
    parser.add_argument("--disable-robust-loss-variants", action="store_true")
    parser.add_argument("--disable-dr-line-ablations", action="store_true")
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

    # ====================== 修复在这里！！！ ======================

    # ==============================================================
    prefix_candidates = ["ihdp_good_", f"ihdp_{tag}_", "ihdp_source_"]
    for prefix in prefix_candidates:
        source_filename = target_filename.replace("ihdp_", prefix, 1)
        exact_path = os.path.join(source_dir, source_filename)
        if os.path.exists(exact_path):
            return exact_path

    # 下面这些保留不动
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


def safe_numeric_summary(values: pd.Series, prefix: str) -> Dict[str, float]:
    x = pd.to_numeric(values, errors="coerce").dropna()
    if x.empty:
        return {
            f"{prefix}_mean": np.nan,
            f"{prefix}_var": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_max": np.nan,
        }
    return {
        f"{prefix}_mean": float(x.mean()),
        f"{prefix}_var": float(x.var(ddof=1)) if len(x) > 1 else 0.0,
        f"{prefix}_min": float(x.min()),
        f"{prefix}_q25": float(x.quantile(0.25)),
        f"{prefix}_median": float(x.quantile(0.50)),
        f"{prefix}_q75": float(x.quantile(0.75)),
        f"{prefix}_max": float(x.max()),
    }


def build_data_diagnostic_row(env_info: Dict[str, object], target_path: str, source_path: str, source_profile: str) -> Dict[str, object]:
    target_df = load_stream_csv(target_path)
    source_df = load_stream_csv(source_path)
    n_aligned = min(len(target_df), len(source_df))

    same_position_count = np.nan
    same_position_rate = np.nan
    row_overlap_count = np.nan
    row_overlap_rate = np.nan
    if "source_row" in target_df.columns and "source_row" in source_df.columns:
        target_rows = target_df["source_row"].iloc[:n_aligned].to_numpy()
        source_rows = source_df["source_row"].iloc[:n_aligned].to_numpy()
        same_position_count = int(np.sum(target_rows == source_rows))
        same_position_rate = float(same_position_count / max(n_aligned, 1))
        target_set = set(pd.to_numeric(target_df["source_row"], errors="coerce").dropna().astype(int).tolist())
        source_set = set(pd.to_numeric(source_df["source_row"], errors="coerce").dropna().astype(int).tolist())
        row_overlap_count = int(len(target_set.intersection(source_set)))
        row_overlap_rate = float(row_overlap_count / max(len(target_set), 1))

    row: Dict[str, object] = {
        "env_name": env_info["env_name"],
        "dataset": env_info["dataset"],
        "surface": env_info["surface"],
        "drift": env_info["drift"],
        "K": env_info["K"],
        "repeat": env_info.get("repeat"),
        "source_profile": source_profile,
        "target_file": os.path.basename(target_path),
        "source_file": os.path.basename(source_path),
        "target_n": int(len(target_df)),
        "source_n": int(len(source_df)),
        "aligned_n": int(n_aligned),
        "same_position_source_row_count": same_position_count,
        "same_position_source_row_rate": same_position_rate,
        "source_row_overlap_count": row_overlap_count,
        "source_row_overlap_rate": row_overlap_rate,
    }

    for label, df in [("target", target_df), ("source", source_df)]:
        row.update(safe_numeric_summary(df["tau"], f"{label}_tau"))
        row.update(safe_numeric_summary(df["q"], f"{label}_q"))
        row.update(safe_numeric_summary(df["p"], f"{label}_observed_arm_prob"))
        inv_w = 1.0 / np.maximum(pd.to_numeric(df["p"], errors="coerce").to_numpy(dtype=float), EPS_NUM)
        row.update(safe_numeric_summary(pd.Series(inv_w), f"{label}_inverse_observed_weight"))
        w_num = pd.to_numeric(df["w"], errors="coerce")
        row[f"{label}_treatment_rate"] = float((w_num == 1.0).mean())

    row["same_covariate_support_note"] = (
        "IHDP semi-synthetic streams share the same covariate support; this diagnostic quantifies whether this repeat also uses the same source_row order."
    )
    return row


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
    return float(np.dot(theta, phi))


def predict_tau(theta: np.ndarray, phi: np.ndarray) -> float:
    return float(np.dot(theta, phi))


def get_otl_parameter_ensemble(theta_source: np.ndarray, theta_target: np.ndarray, weights: np.ndarray) -> np.ndarray:
    theta_mix = float(weights[0]) * theta_source + float(weights[1]) * theta_target
    return theta_mix.astype(np.float32, copy=False)


def get_ethos_hyperparams(T_run: int, method: str) -> Dict[str, object]:
    g_tilde = get_method_g_tilde(method)
    alpha_meta = math.sqrt(8.0 / (T_run * (g_tilde ** 2) * (D ** 2)))
    N = int(math.floor(0.5 * math.log2(1.0 + 4.0 * T_run / 7.0))) + 1
    etas = [(2.0 ** (i - 1)) * D / g_tilde * math.sqrt(7.0 / (2.0 * T_run)) for i in range(1, N + 1)]
    return {"alpha_meta": float(alpha_meta), "N": int(N), "etas": np.asarray(etas, dtype=np.float32)}


def update_outcome(
    theta: np.ndarray,
    phi: np.ndarray,
    y: float,
    acc_sq: float,
    c_eta: float = 1.0,
) -> Tuple[np.ndarray, float]:
    pred = predict_outcome(theta, phi)
    grad = ((pred - y) * phi).astype(np.float32)
    acc_sq = acc_sq + float(np.dot(grad, grad))
    eta = get_scale_free_eta(
        acc_sq=acc_sq,
        radius=OUTCOME_NUISANCE_PARAM_NORM_BOUND,
        c_eta=float(c_eta),
    )
    theta = (theta - eta * grad).astype(np.float32, copy=False)
    return theta.astype(np.float32), float(acc_sq)


def unpack_target_stream(env: Dict[str, object]) -> Tuple[np.ndarray, ...]:
    return (
        env["Phi"], env["w"], env["p"], env["q"], env["y"], env["tau"],
        env["segment"], env["segment_round"], env["t"],
    )


def get_otl_raw_losses(
    pred_old: float,
    pred_new: float,
    pseudo_tau: float,
) -> Tuple[float, float, float]:
    loss_old = float(0.5 * (pred_old - pseudo_tau) ** 2)
    loss_new = float(0.5 * (pred_new - pseudo_tau) ** 2)
    loss_gap = loss_old - loss_new
    return float(loss_old), float(loss_new), float(loss_gap)


def make_transfer_method_name(base_method: str, source_profile: str, loss_mode: str, transfer_line_mode: str) -> str:
    parts = [str(base_method)]
    if str(source_profile) != MAIN_SOURCE_PROFILE:
        parts.append(f"{source_profile}_source")
    if str(loss_mode) != OTL_MAIN_LOSS_MODE:
        parts.append(str(loss_mode))
    if str(transfer_line_mode) != MAIN_TRANSFER_LINE_MODE:
        parts.append(str(transfer_line_mode))
    if len(parts) == 1:
        return str(base_method)
    return "_".join(parts).upper()


def make_comparison_label(spec: Dict[str, object], source_profile: str, loss_mode: str, transfer_line_mode: str) -> str:
    label = str(spec["comparison_label"])
    extras: List[str] = []
    if source_profile != MAIN_SOURCE_PROFILE:
        extras.append(f"{source_profile} source")
    if loss_mode != OTL_MAIN_LOSS_MODE:
        extras.append(f"{loss_mode} loss")
    if transfer_line_mode != MAIN_TRANSFER_LINE_MODE:
        extras.append(f"{transfer_line_mode} DR transfer")
    if extras:
        return f"{label} [{', '.join(extras)}]"
    return label


def init_loss_scale_state() -> Dict[str, float]:
    return {"scale": 1.0, "initialized": 0.0}


def transform_otl_losses(
    loss_old: float,
    loss_new: float,
    loss_mode: str,
    loss_state: Dict[str, float],
) -> Tuple[float, float, float]:
    if loss_mode == "raw":
        effective_old = float(loss_old)
        effective_new = float(loss_new)
        loss_scale = 1.0
    elif loss_mode == "clipped":
        effective_old = float(min(float(loss_old), OTL_LOSS_CLIP))
        effective_new = float(min(float(loss_new), OTL_LOSS_CLIP))
        loss_scale = float(OTL_LOSS_CLIP)
    elif loss_mode == "empirical_scaled":
        loss_scale = float(max(loss_state.get("scale", 1.0), EPS_NUM))
        effective_old = float(min(float(loss_old) / loss_scale, OTL_EMPIRICAL_SCALED_LOSS_CLIP))
        effective_new = float(min(float(loss_new) / loss_scale, OTL_EMPIRICAL_SCALED_LOSS_CLIP))
    else:
        raise ValueError(f"Unknown OTL loss_mode: {loss_mode}")
    return effective_old, effective_new, float(loss_scale)


def update_loss_scale_state(loss_state: Dict[str, float], loss_old: float, loss_new: float) -> None:
    current = float(max(float(loss_old), float(loss_new), EPS_NUM))
    if loss_state.get("initialized", 0.0) <= 0.0:
        loss_state["scale"] = current
        loss_state["initialized"] = 1.0
    else:
        beta = float(OTL_EMPIRICAL_SCALE_BETA)
        loss_state["scale"] = beta * float(loss_state["scale"]) + (1.0 - beta) * current


def update_otl_weights_exponential_no_share(
    weights: np.ndarray,
    loss_old: float,
    loss_new: float,
    meta_eta: float,
) -> np.ndarray:
    s_old = math.exp(-meta_eta * float(loss_old))
    s_new = math.exp(-meta_eta * float(loss_new))
    num_old = float(weights[0]) * s_old
    num_new = float(weights[1]) * s_new
    denom = num_old + num_new + EPS_NUM
    normalized = np.asarray([num_old / denom, num_new / denom], dtype=np.float32)
    normalized = np.clip(normalized, EPS_NUM, 1.0)
    normalized = normalized / (np.sum(normalized) + EPS_NUM)
    return normalized.astype(np.float32)


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
    main_c_eta: float = THEORY_C_ETA,
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
        eta_t = get_scale_free_eta(acc_sq=acc_sq, radius=HTE_PARAM_NORM_BOUND, c_eta=main_c_eta)
        theta = (theta - eta_t * grad).astype(np.float32, copy=False)

        if method.endswith("DR"):
            if float(w[idx]) == 1.0:
                alpha1, acc1_sq = update_outcome(alpha1, phi, float(y[idx]), acc1_sq)
            else:
                alpha0, acc0_sq = update_outcome(alpha0, phi, float(y[idx]), acc0_sq)

    out = tracker.summary()
    out.update({"theta_final": theta, "c_eta": float(main_c_eta)})
    if method.endswith("DR"):
        out.update({
            "alpha0_final": alpha0.astype(np.float32),
            "alpha1_final": alpha1.astype(np.float32),
            "alpha0_c_eta": 1.0,
            "alpha1_c_eta": 1.0,
        })
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
        Theta = (Theta - etas[:, None] * ((preds - pseudo_tau)[:, None] * phi[None, :])).astype(np.float32, copy=False)

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


def fit_source_old_classifier_adaptive(env_source: Dict[str, object], method: str) -> Dict[str, np.ndarray]:
    source_env = {
        **env_source,
        "tau": np.zeros_like(env_source["y"]),
        "segment": np.zeros_like(env_source["y"], dtype=np.int64),
        "segment_round": np.arange(len(env_source["y"]), dtype=np.int64),
        "t": np.arange(len(env_source["y"]), dtype=np.int64),
    }
    out = run_adaptive_core(
        "__source__",
        0,
        source_env,
        method,
        record_trajectory=False,
        main_c_eta=1.0 / math.sqrt(2.0),
    )
    bundle: Dict[str, np.ndarray] = {
        "theta_final": out["theta_final"].astype(np.float32),
    }
    if method.endswith("DR"):
        bundle.update({
            "alpha0_final": out["alpha0_final"].astype(np.float32),
            "alpha1_final": out["alpha1_final"].astype(np.float32),
        })
    return bundle


# =========================================================
# Route B OTL transfer learners (adaptive source + adaptive target, no share layer)
# =========================================================



def run_exact_otl_transfer_adaptive_core(
    env_name: str,
    seed: int,
    env_target: Dict[str, object],
    transfer_method: str,
    target_method: str,
    source_old_bundle: Dict[str, np.ndarray],
    record_trajectory: bool = False,
    repeat: Optional[int] = None,
    main_c_eta: float = THEORY_C_ETA,
    source_profile: str = MAIN_SOURCE_PROFILE,
    loss_mode: str = OTL_MAIN_LOSS_MODE,
    transfer_line_mode: str = MAIN_TRANSFER_LINE_MODE,
) -> Dict[str, object]:
    Phi, w, p, q, y, tau, segment, segment_round, original_t = unpack_target_stream(env_target)
    d = Phi.shape[1]
    if loss_mode not in OTL_LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {OTL_LOSS_MODES}, got {loss_mode}")
    if transfer_line_mode not in DR_TRANSFER_LINE_MODES:
        raise ValueError(f"transfer_line_mode must be one of {DR_TRANSFER_LINE_MODES}, got {transfer_line_mode}")

    source_old_model = source_old_bundle["theta_final"].astype(np.float32)
    theta = np.zeros(d, dtype=np.float32)  # f_1 = 0
    acc_sq = 0.0

    # [old/source h, new/target f]
    expert_weights = np.asarray([OTL_INITIAL_WEIGHT, OTL_INITIAL_WEIGHT], dtype=np.float32)

    use_dr = target_method.endswith("DR")
    theta_transfer_enabled = transfer_line_mode in {"full", "theta_only"}
    nuisance_transfer_enabled = use_dr and transfer_line_mode in {"full", "nuisance_only"}
    if not use_dr and transfer_line_mode != MAIN_TRANSFER_LINE_MODE:
        raise ValueError("DR line ablations are only defined for DR transfer methods")
    if not theta_transfer_enabled:
        expert_weights = np.asarray([0.0, 1.0], dtype=np.float32)
    source_alpha0_old = source_old_bundle.get("alpha0_final", np.zeros(d, dtype=np.float32)).astype(np.float32)
    source_alpha1_old = source_old_bundle.get("alpha1_final", np.zeros(d, dtype=np.float32)).astype(np.float32)
    alpha0 = np.zeros(d, dtype=np.float32)
    alpha1 = np.zeros(d, dtype=np.float32)
    acc0_sq = 0.0
    acc1_sq = 0.0
    alpha0_expert_weights = np.asarray([OTL_INITIAL_WEIGHT, OTL_INITIAL_WEIGHT], dtype=np.float32)
    alpha1_expert_weights = np.asarray([OTL_INITIAL_WEIGHT, OTL_INITIAL_WEIGHT], dtype=np.float32)
    if use_dr and not nuisance_transfer_enabled:
        alpha0_expert_weights = np.asarray([0.0, 1.0], dtype=np.float32)
        alpha1_expert_weights = np.asarray([0.0, 1.0], dtype=np.float32)

    tracker = MetricTracker(env_name, transfer_method, seed, record_trajectory, repeat=repeat)
    otl_meta_eta = get_otl_meta_eta(1, target_method)
    theta_loss_state = init_loss_scale_state()
    alpha0_loss_state = init_loss_scale_state()
    alpha1_loss_state = init_loss_scale_state()

    for idx, phi in enumerate(Phi):
        pred_old = predict_tau(source_old_model, phi)
        pred_new = predict_tau(theta, phi)

        theta_mix = get_otl_parameter_ensemble(source_old_model, theta, expert_weights) if theta_transfer_enabled else theta
        pred_mix = predict_tau(theta_mix, phi)

        if use_dr:
            alpha0_mix = get_otl_parameter_ensemble(source_alpha0_old, alpha0, alpha0_expert_weights) if nuisance_transfer_enabled else alpha0
            alpha1_mix = get_otl_parameter_ensemble(source_alpha1_old, alpha1, alpha1_expert_weights) if nuisance_transfer_enabled else alpha1
            m0_hat = predict_outcome(alpha0_mix, phi)
            m1_hat = predict_outcome(alpha1_mix, phi)
        else:
            alpha0_mix = np.zeros(d, dtype=np.float32)
            alpha1_mix = np.zeros(d, dtype=np.float32)
            m0_hat = 0.0
            m1_hat = 0.0

        extra = None
        if record_trajectory:
            extra = {
                "pred_old_component": float(pred_old),
                "pred_new_component": float(pred_new),
                "pred_mix_component": float(pred_mix),
                "otl_weight_old": float(expert_weights[0]),
                "otl_weight_new": float(expert_weights[1]),
                "otl_meta_eta": float(get_otl_meta_eta(idx + 1, target_method)),
                "otl_loss_variant": OTL_LOSS_VARIANT,
                "otl_loss_mode": str(loss_mode),
                "source_profile": str(source_profile),
                "transfer_line_mode": str(transfer_line_mode),
                "otl_loss_scale": float(get_shared_hedge_loss_upper_bound(target_method)),
                "theta_source_norm": float(np.linalg.norm(source_old_model)),
                "theta_target_norm": float(np.linalg.norm(theta)),
                "theta_mix_norm": float(np.linalg.norm(theta_mix)),
            }
            if use_dr:
                extra.update({
                    "m0_hat_mix": float(m0_hat),
                    "m1_hat_mix": float(m1_hat),
                    "alpha0_source_norm": float(np.linalg.norm(source_alpha0_old)),
                    "alpha0_target_norm": float(np.linalg.norm(alpha0)),
                    "alpha0_mix_norm": float(np.linalg.norm(alpha0_mix)),
                    "alpha1_source_norm": float(np.linalg.norm(source_alpha1_old)),
                    "alpha1_target_norm": float(np.linalg.norm(alpha1)),
                    "alpha1_mix_norm": float(np.linalg.norm(alpha1_mix)),
                    "alpha0_weight_old": float(alpha0_expert_weights[0]),
                    "alpha0_weight_new": float(alpha0_expert_weights[1]),
                    "alpha1_weight_old": float(alpha1_expert_weights[0]),
                    "alpha1_weight_new": float(alpha1_expert_weights[1]),
                })

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
            pseudo_tau = get_dr_pseudo_outcome(float(w[idx]), float(y[idx]), float(q[idx]), float(m0_hat), float(m1_hat))

        loss_old, loss_new, loss_gap = get_otl_raw_losses(
            pred_old=float(pred_old),
            pred_new=float(pred_new),
            pseudo_tau=float(pseudo_tau),
        )
        loss_scale = float(get_shared_hedge_loss_upper_bound(target_method))
        raw_loss_old = float(loss_old)
        raw_loss_new = float(loss_new)
        otl_meta_eta = get_otl_meta_eta(idx + 1, target_method)
        meta_loss_old, meta_loss_new, theta_loss_scale = transform_otl_losses(
            float(loss_old),
            float(loss_new),
            str(loss_mode),
            theta_loss_state,
        )
        if theta_transfer_enabled:
            expert_weights = update_otl_weights_exponential_no_share(
                expert_weights,
                float(meta_loss_old),
                float(meta_loss_new),
                otl_meta_eta,
            )
        update_loss_scale_state(theta_loss_state, float(loss_old), float(loss_new))

        nuisance_loss_old = np.nan
        nuisance_loss_new = np.nan
        nuisance_meta_loss_old = np.nan
        nuisance_meta_loss_new = np.nan
        nuisance_loss_scale = np.nan
        nuisance_obs_arm = "none"

        if use_dr:
            if float(w[idx]) == 1.0:
                nuisance_obs_arm = "treated"
                pred_old_alpha1 = predict_outcome(source_alpha1_old, phi)
                pred_new_alpha1 = predict_outcome(alpha1, phi)
                nuisance_loss_old, nuisance_loss_new, _ = get_otl_raw_losses(
                    pred_old=float(pred_old_alpha1),
                    pred_new=float(pred_new_alpha1),
                    pseudo_tau=float(y[idx]),
                )
                nuisance_meta_loss_old, nuisance_meta_loss_new, nuisance_loss_scale = transform_otl_losses(
                    float(nuisance_loss_old),
                    float(nuisance_loss_new),
                    str(loss_mode),
                    alpha1_loss_state,
                )
                if nuisance_transfer_enabled:
                    alpha1_expert_weights = update_otl_weights_exponential_no_share(
                        alpha1_expert_weights,
                        float(nuisance_meta_loss_old),
                        float(nuisance_meta_loss_new),
                        otl_meta_eta,
                    )
                update_loss_scale_state(alpha1_loss_state, float(nuisance_loss_old), float(nuisance_loss_new))
                alpha1, acc1_sq = update_outcome(alpha1, phi, float(y[idx]), acc1_sq, c_eta=1.0)
            else:
                nuisance_obs_arm = "control"
                pred_old_alpha0 = predict_outcome(source_alpha0_old, phi)
                pred_new_alpha0 = predict_outcome(alpha0, phi)
                nuisance_loss_old, nuisance_loss_new, _ = get_otl_raw_losses(
                    pred_old=float(pred_old_alpha0),
                    pred_new=float(pred_new_alpha0),
                    pseudo_tau=float(y[idx]),
                )
                nuisance_meta_loss_old, nuisance_meta_loss_new, nuisance_loss_scale = transform_otl_losses(
                    float(nuisance_loss_old),
                    float(nuisance_loss_new),
                    str(loss_mode),
                    alpha0_loss_state,
                )
                if nuisance_transfer_enabled:
                    alpha0_expert_weights = update_otl_weights_exponential_no_share(
                        alpha0_expert_weights,
                        float(nuisance_meta_loss_old),
                        float(nuisance_meta_loss_new),
                        otl_meta_eta,
                    )
                update_loss_scale_state(alpha0_loss_state, float(nuisance_loss_old), float(nuisance_loss_new))
                alpha0, acc0_sq = update_outcome(alpha0, phi, float(y[idx]), acc0_sq, c_eta=1.0)

        if record_trajectory:
            tracker.rows[-1].update({
                "pseudo_tau_raw": float(pseudo_tau),
                "otl_loss_old": float(loss_old),
                "otl_loss_new": float(loss_new),
                "otl_loss_gap": float(loss_gap),
                "otl_meta_loss_old": float(meta_loss_old),
                "otl_meta_loss_new": float(meta_loss_new),
                "otl_meta_loss_gap": float(meta_loss_old - meta_loss_new),
                "otl_loss_old_raw": float(raw_loss_old),
                "otl_loss_new_raw": float(raw_loss_new),
                "otl_theory_loss_scale": float(loss_scale),
                "otl_loss_scale": float(theta_loss_scale),
                "otl_err_old_sq": float(2.0 * raw_loss_old),
                "otl_err_new_sq": float(2.0 * raw_loss_new),
                "otl_weight_old_next": float(expert_weights[0]),
                "otl_weight_new_next": float(expert_weights[1]),
            })
            if use_dr:
                tracker.rows[-1].update({
                    "nuisance_obs_arm": nuisance_obs_arm,
                    "nuisance_loss_old": float(nuisance_loss_old) if not np.isnan(nuisance_loss_old) else np.nan,
                    "nuisance_loss_new": float(nuisance_loss_new) if not np.isnan(nuisance_loss_new) else np.nan,
                    "nuisance_meta_loss_old": float(nuisance_meta_loss_old) if not np.isnan(nuisance_meta_loss_old) else np.nan,
                    "nuisance_meta_loss_new": float(nuisance_meta_loss_new) if not np.isnan(nuisance_meta_loss_new) else np.nan,
                    "nuisance_loss_scale": float(nuisance_loss_scale) if not np.isnan(nuisance_loss_scale) else np.nan,
                    "alpha0_weight_old_next": float(alpha0_expert_weights[0]),
                    "alpha0_weight_new_next": float(alpha0_expert_weights[1]),
                    "alpha1_weight_old_next": float(alpha1_expert_weights[0]),
                    "alpha1_weight_new_next": float(alpha1_expert_weights[1]),
                })

        # target/new predictor 仍按原 ADAPTIVE 核心更新；不使用 shared/share 层，仅保留两专家指数加权与参数集成
        grad = ((float(np.dot(theta, phi)) - pseudo_tau) * phi).astype(np.float32)
        acc_sq += float(np.dot(grad, grad))
        eta_t = get_scale_free_eta(acc_sq=acc_sq, radius=HTE_PARAM_NORM_BOUND, c_eta=main_c_eta)
        theta = (theta - eta_t * grad).astype(np.float32, copy=False)

    out = tracker.summary()
    theta_param_mix_final = get_otl_parameter_ensemble(source_old_model, theta, expert_weights)

    out.update({
        "theta_target_final": theta.astype(np.float32),
        "theta_source_fixed": source_old_model.astype(np.float32),
        "theta_param_mix_final": theta_param_mix_final.astype(np.float32),
        "c_eta": float(main_c_eta),
        "otl_meta_eta": float(otl_meta_eta),
        "otl_share_lambda": 0,
        "otl_share_lambda_schedule": OTL_LAMBDA_SCHEDULE,
        "otl_loss_variant": OTL_LOSS_VARIANT,
        "otl_loss_mode": str(loss_mode),
        "otl_weighting_interpretation": OTL_WEIGHTING_INTERPRETATION,
        "source_profile": str(source_profile),
        "transfer_line_mode": str(transfer_line_mode),
        "theta_transfer_enabled": int(theta_transfer_enabled),
        "nuisance_transfer_enabled": int(nuisance_transfer_enabled),
        "otl_weight_old_final": float(expert_weights[0]),
        "otl_weight_new_final": float(expert_weights[1]),
        "transfer_strategy": f"exponential_weights_parameter_ensemble_{loss_mode}_loss_no_share_{transfer_line_mode}",
    })
    if use_dr:
        alpha0_param_mix_final = get_otl_parameter_ensemble(source_alpha0_old, alpha0, alpha0_expert_weights)
        alpha1_param_mix_final = get_otl_parameter_ensemble(source_alpha1_old, alpha1, alpha1_expert_weights)
        out.update({
            "alpha0_target_final": alpha0.astype(np.float32),
            "alpha1_target_final": alpha1.astype(np.float32),
            "alpha0_source_fixed": source_alpha0_old.astype(np.float32),
            "alpha1_source_fixed": source_alpha1_old.astype(np.float32),
            "alpha0_param_mix_final": alpha0_param_mix_final.astype(np.float32),
            "alpha1_param_mix_final": alpha1_param_mix_final.astype(np.float32),
            "alpha0_weight_old_final": float(alpha0_expert_weights[0]),
            "alpha0_weight_new_final": float(alpha0_expert_weights[1]),
            "alpha1_weight_old_final": float(alpha1_expert_weights[0]),
            "alpha1_weight_new_final": float(alpha1_expert_weights[1]),
            "nuisance_transfer_strategy": "per_arm_exponential_weights_parameter_ensemble_raw_squared_loss_no_share",
            "nuisance_c_eta": 1.0,
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
    grouped = df.groupby(group_cols, as_index=False, dropna=False).agg({c: ["mean", "std", "median", q25, q75] for c in agg_metric_cols})
    grouped.columns = ["_".join(col).strip("_") if isinstance(col, tuple) else col for col in grouped.columns]
    counts = df.groupby(group_cols, as_index=False, dropna=False).size().rename(columns={"size": repeat_col_name})
    grouped = pd.merge(grouped, counts, on=group_cols, how="left")
    iqr_cols: Dict[str, pd.Series] = {}
    for c in agg_metric_cols:
        std_col = f"{c}_std"
        if std_col in grouped.columns:
            grouped[std_col] = grouped[std_col].fillna(0.0)
        q25_col = f"{c}_q25"
        q75_col = f"{c}_q75"
        if q25_col in grouped.columns and q75_col in grouped.columns:
            iqr_cols[f"{c}_iqr"] = grouped[q75_col] - grouped[q25_col]
    if iqr_cols:
        grouped = pd.concat([grouped, pd.DataFrame(iqr_cols)], axis=1)
    return grouped


def run_stage1_repeat(task: Tuple[Dict[str, object], str, str, int, int, int, bool]) -> Dict[str, object]:
    env_info, target_path, good_path, seed, hidden_dim, batch_size, force_linear = task
    env_name = str(env_info["env_name"])
    bundle = load_bundle(target_path, good_path, seed, hidden_dim, batch_size, force_linear)
    env_target = bundle["env_target"]
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None

    adaptive_ipw = run_adaptive_core(env_name, seed, env_target, "ADAPTIVE_IPW", repeat=repeat, main_c_eta=THEORY_C_ETA)
    adaptive_dr = run_adaptive_core(env_name, seed, env_target, "ADAPTIVE_DR", repeat=repeat, main_c_eta=THEORY_C_ETA)
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


def build_transfer_cases(enable_bad_source: bool, enable_robust_loss_variants: bool, enable_dr_line_ablations: bool) -> List[Dict[str, object]]:
    source_profiles = [MAIN_SOURCE_PROFILE] + (["bad"] if enable_bad_source else [])
    loss_modes = [OTL_MAIN_LOSS_MODE] + (DEFAULT_ROBUST_LOSS_MODES if enable_robust_loss_variants else [])
    cases: List[Dict[str, object]] = []
    for spec in COMPARISON_SPECS:
        if str(spec["family"]) != "DR":
            continue
        for source_profile in source_profiles:
            for loss_mode in loss_modes:
                cases.append({
                    "spec": spec,
                    "source_profile": source_profile,
                    "loss_mode": loss_mode,
                    "transfer_line_mode": MAIN_TRANSFER_LINE_MODE,
                    "method": make_transfer_method_name(
                        str(spec["transfer_method"]),
                        source_profile,
                        loss_mode,
                        MAIN_TRANSFER_LINE_MODE,
                    ),
                    "comparison_label": make_comparison_label(spec, source_profile, loss_mode, MAIN_TRANSFER_LINE_MODE),
                })
        if enable_dr_line_ablations and str(spec["family"]) == "DR":
            for transfer_line_mode in ["theta_only", "nuisance_only"]:
                cases.append({
                    "spec": spec,
                    "source_profile": MAIN_SOURCE_PROFILE,
                    "loss_mode": OTL_MAIN_LOSS_MODE,
                    "transfer_line_mode": transfer_line_mode,
                    "method": make_transfer_method_name(
                        str(spec["transfer_method"]),
                        MAIN_SOURCE_PROFILE,
                        OTL_MAIN_LOSS_MODE,
                        transfer_line_mode,
                    ),
                    "comparison_label": make_comparison_label(spec, MAIN_SOURCE_PROFILE, OTL_MAIN_LOSS_MODE, transfer_line_mode),
                })
    return cases


def make_transfer_metric_row(
    env_info: Dict[str, object],
    seed: int,
    case: Dict[str, object],
    out: Dict[str, object],
) -> Dict[str, object]:
    spec = case["spec"]
    return attach_env_meta({
        "method": str(case["method"]),
        "comparison_label": str(case["comparison_label"]),
        "target_method": str(spec["target_method"]),
        "source_method": str(spec["source_method"]),
        "baseline_method": str(spec["baseline_method"]),
        "core_method": str(spec["target_method"]),
        "core_c_eta": float(THEORY_C_ETA),
        "stage1_inherited_c_eta": float(THEORY_C_ETA),
        "source_profile": str(case["source_profile"]),
        "otl_meta_eta": float(out["otl_meta_eta"]),
        "otl_loss_variant": str(out["otl_loss_variant"]),
        "otl_loss_mode": str(out["otl_loss_mode"]),
        "otl_weighting_interpretation": str(out["otl_weighting_interpretation"]),
        "transfer_line_mode": str(out["transfer_line_mode"]),
        "theta_transfer_enabled": int(out["theta_transfer_enabled"]),
        "nuisance_transfer_enabled": int(out["nuisance_transfer_enabled"]),
        "cumhalfPEHE": float(out["cumhalfPEHE"]),
        "running_mean_mse": float(out["running_mean_mse"]),
        "running_rmse": float(out["running_rmse"]),
        "otl_weight_old_final": float(out["otl_weight_old_final"]),
        "otl_weight_new_final": float(out["otl_weight_new_final"]),
        "alpha0_weight_old_final": float(out.get("alpha0_weight_old_final", np.nan)),
        "alpha0_weight_new_final": float(out.get("alpha0_weight_new_final", np.nan)),
        "alpha1_weight_old_final": float(out.get("alpha1_weight_old_final", np.nan)),
        "alpha1_weight_new_final": float(out.get("alpha1_weight_new_final", np.nan)),
        "transfer_strategy": str(out["transfer_strategy"]),
        "otl_share_lambda": float(out["otl_share_lambda"]),
        "otl_share_lambda_schedule": str(out["otl_share_lambda_schedule"]),
    }, env_info, seed)


def run_stage2_repeat(task: Tuple[Dict[str, object], str, Dict[str, str], int, int, int, bool, List[Dict[str, object]]]) -> Dict[str, object]:
    env_info, target_path, source_paths, seed, hidden_dim, batch_size, force_linear, transfer_cases = task
    env_name = str(env_info["env_name"])
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None
    env_target = None
    source_models_by_profile: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for profile, source_path in source_paths.items():
        bundle = load_bundle(target_path, source_path, seed, hidden_dim, batch_size, force_linear)
        if env_target is None:
            env_target = bundle["env_target"]
        env_source = bundle["env_good"]
        source_models_by_profile[profile] = {
            "ADAPTIVE_IPW": fit_source_old_classifier_adaptive(env_source, "ADAPTIVE_IPW"),
            "ADAPTIVE_DR": fit_source_old_classifier_adaptive(env_source, "ADAPTIVE_DR"),
        }
    if env_target is None:
        raise RuntimeError("No source profiles available for stage2 transfer")

    rows: List[Dict[str, object]] = []
    for case in transfer_cases:
        spec = case["spec"]
        source_profile = str(case["source_profile"])
        if source_profile not in source_models_by_profile:
            continue
        out = run_exact_otl_transfer_adaptive_core(
            env_name=env_name,
            seed=seed,
            env_target=env_target,
            transfer_method=str(case["method"]),
            target_method=str(spec["target_method"]),
            source_old_bundle=source_models_by_profile[source_profile][str(spec["source_method"])],
            record_trajectory=False,
            repeat=repeat,
            main_c_eta=THEORY_C_ETA,
            source_profile=source_profile,
            loss_mode=str(case["loss_mode"]),
            transfer_line_mode=str(case["transfer_line_mode"]),
        )

        rows.append(make_transfer_metric_row(env_info, seed, case, out))

    return {"env_name": env_name, "seed": int(seed), "repeat": repeat, "rows": rows}


def build_stage2_outputs(stage2_results: List[Dict[str, object]]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    raw_df = pd.DataFrame([row for res in stage2_results for row in res["rows"]])
    group_cols = [
        "env_name", "dataset", "surface", "drift", "K", "method", "comparison_label", "target_method",
        "source_method", "baseline_method", "core_method", "core_c_eta", "stage1_inherited_c_eta",
        "family", "source_profile", "otl_meta_eta", "otl_share_lambda", "otl_share_lambda_schedule",
        "otl_loss_variant", "otl_loss_mode", "otl_weighting_interpretation",
        "transfer_line_mode", "theta_transfer_enabled", "nuisance_transfer_enabled", "transfer_strategy"
    ]
    metric_cols = [
        "cumhalfPEHE", "running_mean_mse", "running_rmse",
        "otl_weight_old_final", "otl_weight_new_final",
        "alpha0_weight_old_final", "alpha0_weight_new_final",
        "alpha1_weight_old_final", "alpha1_weight_new_final",
    ]
    mean_df = aggregate_metric_df(raw_df, group_cols, metric_cols).sort_values(["env_name", "comparison_label"]).reset_index(drop=True)
    mean_df["selection_basis"] = "paper_exact_otl_no_search"
    return raw_df, mean_df


def run_final_repeat(task: Tuple[Dict[str, object], str, Dict[str, str], int, int, int, bool, bool, List[Dict[str, object]]]) -> Dict[str, object]:
    env_info, target_path, source_paths, seed, hidden_dim, batch_size, force_linear, record_trajectory, transfer_cases = task
    env_name = str(env_info["env_name"])
    repeat = int(env_info.get("repeat")) if env_info.get("repeat") is not None else None

    env_target = None
    source_models_by_profile: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}
    for profile, source_path in source_paths.items():
        bundle = load_bundle(target_path, source_path, seed, hidden_dim, batch_size, force_linear)
        if env_target is None:
            env_target = bundle["env_target"]
        env_source = bundle["env_good"]
        source_models_by_profile[profile] = {
            "ADAPTIVE_IPW": fit_source_old_classifier_adaptive(env_source, "ADAPTIVE_IPW"),
            "ADAPTIVE_DR": fit_source_old_classifier_adaptive(env_source, "ADAPTIVE_DR"),
        }
    if env_target is None:
        raise RuntimeError("No source profiles available for final transfer")

    rows: List[Dict[str, object]] = []
    trajs: List[pd.DataFrame] = []

    # target-only methods
    for method in ["ADAPTIVE_IPW", "ADAPTIVE_DR", "ETHOS_IPW", "ETHOS_DR"]:
        if method.startswith("ADAPTIVE"):
            out = run_adaptive_core(env_name, seed, env_target, method, record_trajectory=record_trajectory, repeat=repeat, main_c_eta=THEORY_C_ETA)
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
    for case in transfer_cases:
        spec = case["spec"]
        source_profile = str(case["source_profile"])
        if source_profile not in source_models_by_profile:
            continue
        source_old_bundle = source_models_by_profile[source_profile][str(spec["source_method"])]
        out = run_exact_otl_transfer_adaptive_core(
            env_name=env_name,
            seed=seed,
            env_target=env_target,
            transfer_method=str(case["method"]),
            target_method=str(spec["target_method"]),
            source_old_bundle=source_old_bundle,
            record_trajectory=record_trajectory,
            repeat=repeat,
            main_c_eta=THEORY_C_ETA,
            source_profile=source_profile,
            loss_mode=str(case["loss_mode"]),
            transfer_line_mode=str(case["transfer_line_mode"]),
        )
        row = make_transfer_metric_row(env_info, seed, case, out)
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
        "stage1_inherited_c_eta", "source_profile", "otl_meta_eta", "otl_share_lambda",
        "otl_share_lambda_schedule", "otl_loss_variant", "otl_loss_mode", "otl_weighting_interpretation",
        "transfer_line_mode", "theta_transfer_enabled", "nuisance_transfer_enabled", "transfer_strategy"
    ]
    metric_cols = [
        "cumhalfPEHE", "running_mean_mse", "running_rmse", "c_eta", "alpha_meta", "N", "eta_min", "eta_max",
        "otl_weight_old_final", "otl_weight_new_final",
        "alpha0_weight_old_final", "alpha0_weight_new_final",
        "alpha1_weight_old_final", "alpha1_weight_new_final",
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
        "otl_meta_eta", "otl_share_lambda_t", "pseudo_tau_raw", "otl_loss_old", "otl_loss_new",
        "otl_loss_gap", "otl_meta_loss_old", "otl_meta_loss_new", "otl_meta_loss_gap",
        "otl_loss_old_raw", "otl_loss_new_raw", "otl_loss_scale", "otl_theory_loss_scale",
        "otl_err_old_sq", "otl_err_new_sq",
        "m0_hat_mix", "m1_hat_mix",
        "alpha0_weight_old", "alpha0_weight_new",
        "alpha0_weight_old_next", "alpha0_weight_new_next",
        "alpha1_weight_old", "alpha1_weight_new",
        "alpha1_weight_old_next", "alpha1_weight_new_next",
        "nuisance_loss_old", "nuisance_loss_new", "nuisance_meta_loss_old", "nuisance_meta_loss_new",
        "nuisance_loss_scale",
    ]

    agg = aggregate_metric_df(raw_traj_df, group_cols, metric_cols)
    rename_map = {f"{c}_mean": c for c in metric_cols}
    out = agg.rename(columns=rename_map).sort_values(["env_name", "method", "step"]).reset_index(drop=True)

    for meta_col in ["otl_loss_variant", "otl_loss_mode", "source_profile", "transfer_line_mode"]:
        if meta_col not in raw_traj_df.columns:
            continue
        variant_df = (
            raw_traj_df[group_cols + [meta_col]]
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
        transfer_df = env_df[env_df["baseline_method"].notna() & env_df["comparison_label"].notna()].copy()
        for _, tr_row in transfer_df.iterrows():
            baseline = str(tr_row["baseline_method"])
            transfer = str(tr_row["method"])
            if baseline not in row_map or transfer not in row_map:
                continue
            base_row = row_map[baseline]
            baseline_cum = float(base_row["cumhalfPEHE_mean"])
            transfer_cum = float(tr_row["cumhalfPEHE_mean"])
            gain = baseline_cum - transfer_cum
            rows.append({
                "env_name": env_name,
                "dataset": tr_row["dataset"],
                "surface": tr_row["surface"],
                "drift": tr_row["drift"],
                "K": tr_row["K"],
                "comparison_label": tr_row["comparison_label"],
                "family": tr_row["family"],
                "target_method": tr_row["target_method"],
                "source_method": tr_row["source_method"],
                "source_profile": tr_row.get("source_profile"),
                "otl_loss_mode": tr_row.get("otl_loss_mode"),
                "transfer_line_mode": tr_row.get("transfer_line_mode"),
                "baseline_method": baseline,
                "transfer_method": transfer,
                "baseline_cumhalfPEHE_mean": baseline_cum,
                "baseline_cumhalfPEHE_std": float(base_row.get("cumhalfPEHE_std", 0.0)),
                "baseline_cumhalfPEHE_median": float(base_row.get("cumhalfPEHE_median", np.nan)),
                "baseline_cumhalfPEHE_iqr": float(base_row.get("cumhalfPEHE_iqr", np.nan)),
                "transfer_cumhalfPEHE_mean": transfer_cum,
                "transfer_cumhalfPEHE_std": float(tr_row.get("cumhalfPEHE_std", 0.0)),
                "transfer_cumhalfPEHE_median": float(tr_row.get("cumhalfPEHE_median", np.nan)),
                "transfer_cumhalfPEHE_iqr": float(tr_row.get("cumhalfPEHE_iqr", np.nan)),
                "transfer_abs_gain": gain,
                "transfer_rel_gain_pct": 100.0 * gain / (abs(baseline_cum) + EPS_NUM),
                "transfer_better_than_baseline": int(gain > 1e-12),
                "tie": int(abs(gain) <= 1e-12),
                "baseline_better": int(gain < -1e-12),
                "negative_transfer_flag": int(gain < -1e-12),
                "otl_meta_eta": tr_row.get("otl_meta_eta_mean"),
                "n_repeats": int(tr_row["n_repeats"]),
            })
    return pd.DataFrame(rows).sort_values(["comparison_label", "env_name"]).reset_index(drop=True)


def build_pairwise_overall_summary(pairwise_best_df: pd.DataFrame) -> pd.DataFrame:
    if pairwise_best_df.empty:
        return pd.DataFrame()
    out = pairwise_best_df.groupby(
        [
            "comparison_label", "family", "target_method", "source_method", "source_profile",
            "otl_loss_mode", "transfer_line_mode", "baseline_method", "transfer_method",
        ],
        as_index=False,
        dropna=False,
    ).agg(
        n_env=("env_name", "count"),
        transfer_win_count=("transfer_better_than_baseline", "sum"),
        tie_count=("tie", "sum"),
        loss_count=("baseline_better", "sum"),
        negative_transfer_count=("negative_transfer_flag", "sum"),
        mean_baseline_cumhalfPEHE=("baseline_cumhalfPEHE_mean", "mean"),
        mean_transfer_cumhalfPEHE=("transfer_cumhalfPEHE_mean", "mean"),
        mean_transfer_abs_gain=("transfer_abs_gain", "mean"),
        median_transfer_abs_gain=("transfer_abs_gain", "median"),
        mean_transfer_rel_gain_pct=("transfer_rel_gain_pct", "mean"),
    )
    out["transfer_win_rate"] = out["transfer_win_count"] / out["n_env"].clip(lower=1)
    out["negative_transfer_rate"] = out["negative_transfer_count"] / out["n_env"].clip(lower=1)
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
        "cumhalfPEHE_median", "cumhalfPEHE_iqr", "running_mean_mse_mean", "running_rmse_mean",
        "n_repeats", "source_profile", "otl_loss_mode", "transfer_line_mode", "c_eta_mean", "otl_meta_eta_mean", "otl_share_lambda_mean",
        "otl_weight_old_final_mean", "otl_weight_new_final_mean",
        "alpha0_weight_old_final_mean", "alpha0_weight_new_final_mean",
        "alpha1_weight_old_final_mean", "alpha1_weight_new_final_mean"
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
    ax.set_title(f"{spec['comparison_label']}\nRoute B OTL expert weights (no share)", fontsize=13)
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
    data_diagnostics_df: pd.DataFrame,
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
        "data_diagnostics_csv": os.path.join(sum_dir, "data_diagnostics_repeat.csv"),
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

    data_diagnostics_df.to_csv(paths["data_diagnostics_csv"], index=False)
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
        DEFAULT_TARGET_DATA_DIR,

    ])
    good_source_dir = args.good_source_dir or resolve_data_dir([
        DEFAULT_GOOD_SOURCE_DIR,
 
    ])
    bad_source_dir = args.bad_source_dir
    if bad_source_dir is None and not args.disable_bad_source:
        try:
            bad_source_dir = resolve_data_dir([DEFAULT_BAD_SOURCE_DIR])
        except FileNotFoundError:
            bad_source_dir = None

    env_repeat_files = discover_environment_files(target_data_dir)
    repeat_inventory = validate_repeat_inventory(env_repeat_files, args.n_repeats)
    env_files = unique_base_envs(env_repeat_files)
    showcase_env = select_showcase_environment(env_files, args.showcase_env)
    n_jobs = effective_n_jobs(args.n_jobs)
    transfer_cases = build_transfer_cases(
        enable_bad_source=(bad_source_dir is not None and not args.disable_bad_source),
        enable_robust_loss_variants=not args.disable_robust_loss_variants,
        enable_dr_line_ablations=not args.disable_dr_line_ablations,
    )

    base_tasks = []
    for env_info in env_repeat_files:
        repeat_id = int(env_info["repeat"])
        algo_seed = int(args.base_seed) + repeat_id - 1
        source_paths = {
            MAIN_SOURCE_PROFILE: resolve_source_path(good_source_dir, env_info, tag="good"),
        }
        if bad_source_dir is not None and not args.disable_bad_source:
            source_paths["bad"] = resolve_source_path(str(bad_source_dir), env_info, tag="bad")
        base_tasks.append(
            (
                env_info,
                str(env_info["full_path"]),
                source_paths,
                algo_seed,
            )
        )

    print(f"[INFO] TARGET_DATA_DIR      = {target_data_dir}")
    print(f"[INFO] GOOD_SOURCE_DIR     = {good_source_dir}")
    print(f"[INFO] BAD_SOURCE_DIR      = {bad_source_dir if bad_source_dir is not None else 'disabled/not found'}")
    print(f"[INFO] ENV_COUNT           = {len(env_files)}")
    print(f"[INFO] N_REPEATS           = {args.n_repeats}")
    print(f"[INFO] TASK_COUNT          = {len(base_tasks)}")
    print(f"[INFO] TRANSFER_CASES      = {len(transfer_cases)}")
    print(f"[INFO] THEORY_C_ETA        = {THEORY_C_ETA:.12f}")
    print(f"[INFO] OTL_META_ETA(t)     = sqrt(8 ln 2 / t)")
    print(f"[INFO] OTL_WEIGHT_UPDATE   = {OTL_WEIGHT_UPDATE}")
    print(f"[INFO] OTL_LAMBDA_SCHEDULE = {OTL_LAMBDA_SCHEDULE}")
    print(f"[INFO] OTL_LOSS_VARIANT    = {OTL_LOSS_VARIANT}")
    print(f"[INFO] OTL_MAIN_META_LOSS  = raw 0.5*(pred - pseudo_tau)^2 ({OTL_WEIGHTING_INTERPRETATION})")
    print(f"[INFO] OTL_LOSS_MODES     = {[OTL_MAIN_LOSS_MODE] + (DEFAULT_ROBUST_LOSS_MODES if not args.disable_robust_loss_variants else [])}")
    print("[INFO] TRANSFER_FOCUS     = DR-only double-line OTL; IPW is retained as target-only/ETHOS baseline, not as OTL transfer")
    print(f"[INFO] SHOWCASE_ENV        = {showcase_env}")
    print(f"[INFO] REPEAT_INVENTORY    = { {k: v for k, v in sorted(repeat_inventory.items())} }")

    diagnostic_rows: List[Dict[str, object]] = []
    for env_info, target_path, source_paths, _seed in base_tasks:
        for source_profile, source_path in source_paths.items():
            diagnostic_rows.append(build_data_diagnostic_row(env_info, target_path, source_path, source_profile))
    data_diagnostics_df = pd.DataFrame(diagnostic_rows)

    stage1_tasks = [
        (env_info, target_path, source_paths[MAIN_SOURCE_PROFILE], seed, args.hidden_dim, args.feature_batch_size, args.force_linear)
        for env_info, target_path, source_paths, seed in base_tasks
    ]
    stage1_results = sorted(
        run_tasks_with_pool(run_stage1_repeat, stage1_tasks, n_jobs),
        key=lambda x: (x["env_name"], x["repeat"], x["seed"])
    )
    raw_stage1_df, stage1_env_df, _ = build_stage1_outputs(stage1_results)

    stage2_tasks = [
        (env_info, target_path, source_paths, seed, args.hidden_dim, args.feature_batch_size, args.force_linear, transfer_cases)
        for env_info, target_path, source_paths, seed in base_tasks
    ]
    stage2_results = sorted(
        run_tasks_with_pool(run_stage2_repeat, stage2_tasks, n_jobs),
        key=lambda x: (x["env_name"], x["repeat"], x["seed"])
    )
    raw_stage2_df, stage2_mean_df = build_stage2_outputs(stage2_results)

    final_tasks = []
    for env_info, target_path, source_paths, seed in base_tasks:
        env_name = str(env_info["env_name"])
        record_trajectory = bool(args.save_all_trajectories or env_name == showcase_env)
        final_tasks.append((
            env_info,
            target_path,
            source_paths,
            seed,
            args.hidden_dim,
            args.feature_batch_size,
            args.force_linear,
            record_trajectory,
            transfer_cases,
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
        data_diagnostics_df,
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
