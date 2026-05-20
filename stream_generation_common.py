#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


BetaSampler = Callable[[np.random.Generator, int], np.ndarray]
ResponseSurface = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class StreamGenerationConfig:
    dataset_prefix: str
    data_path: str
    outdir: str
    t: int
    repeats: int
    eps: float
    noise_var: float
    base_seed: int
    prop_subsample: int
    env_prefix: str
    beta_sampler: BetaSampler
    response_surface: ResponseSurface
    propensity_gamma_scale: float
    done_message: str

    @property
    def noise_std(self) -> float:
        return float(np.sqrt(self.noise_var))


KNOWN_MULTICLASS_CANONICAL: set[str] = set()
EXPECTED_CATEGORIES: dict[str, set[str]] = {}


def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def row_l2_normalize_leq1(X: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(X, axis=1)
    scale = np.maximum(1.0, norms)
    return X / (scale[:, None] + 1e-12)


def series_to_numeric_if_possible(s: pd.Series) -> pd.Series | None:
    s_num = pd.to_numeric(s, errors="coerce")
    if int(s_num.notna().sum()) == int(s.notna().sum()):
        return s_num
    return None


def is_binary_01_series(s: pd.Series) -> bool:
    s_num = series_to_numeric_if_possible(s)
    if s_num is None:
        return False
    non_na = s_num.dropna().astype(float)
    if len(non_na) == 0:
        return False
    uniq = set(np.unique(non_na).tolist())
    return uniq.issubset({0.0, 1.0})


def canonicalize_col_name(name: str) -> str:
    s = str(name).strip().lower()
    return s.replace("_", "").replace("-", "").replace(" ", "")


def pick_covariate_columns(df: pd.DataFrame) -> list[str]:
    exclude_substrings = [
        "outcome", "y0", "y1", "mu_", "tau", "cate", "ite",
        "treat", "treatment", "assigned", "propensity", "pscore", "score",
        "p_t", "segment", "time", "round",
    ]
    exclude_exact = {
        "w", "t", "p", "q", "y", "mu_-1", "mu_1", "tau",
        "id", "row_id", "sample_id", "repeat", "seed",
    }

    cov_cols: list[str] = []
    for c in df.columns:
        cl = str(c).strip().lower()
        if cl in exclude_exact:
            continue
        if any(key in cl for key in exclude_substrings):
            continue
        cov_cols.append(c)

    if not cov_cols:
        raise ValueError("No covariate columns found; please check the input CSV column names.")
    return cov_cols


def validate_known_multiclass_columns(Xraw: pd.DataFrame, categorical_cols: list[str]) -> None:
    if not EXPECTED_CATEGORIES:
        return
    cat_canon_to_raw = {canonicalize_col_name(c): c for c in categorical_cols}
    for canon_name, allowed_values in EXPECTED_CATEGORIES.items():
        if canon_name not in cat_canon_to_raw:
            continue
        raw_col = cat_canon_to_raw[canon_name]
        s = Xraw[raw_col].dropna().astype(str).str.strip()
        obs = set(s.unique().tolist())
        bad = obs.difference(allowed_values)
        if bad:
            raise ValueError(
                f"Column {raw_col} is expected to be a known categorical variable, "
                f"but unexpected values {sorted(bad)} were observed."
            )


def preprocess_covariates(df: pd.DataFrame, dataset_label: str) -> tuple[np.ndarray, list[str]]:
    drop_cols = ["Unnamed: 0.1", "Unnamed: 0", "bord_0", "bord_1"]
    keep = [c for c in df.columns if c not in drop_cols]
    df = df[keep].copy()

    cov_cols = pick_covariate_columns(df)
    Xraw = df[cov_cols].copy()

    binary_cols: list[str] = []
    categorical_cols: list[str] = []
    numeric_cols: list[str] = []

    for c in Xraw.columns:
        s = Xraw[c]
        canon = canonicalize_col_name(c)
        if canon in KNOWN_MULTICLASS_CANONICAL:
            categorical_cols.append(c)
        elif is_binary_01_series(s):
            binary_cols.append(c)
        elif pd.api.types.is_numeric_dtype(s) or series_to_numeric_if_possible(s) is not None:
            numeric_cols.append(c)
        else:
            categorical_cols.append(c)

    validate_known_multiclass_columns(Xraw, categorical_cols)

    parts: list[pd.DataFrame] = []
    if numeric_cols:
        X_num = Xraw[numeric_cols].copy()
        for c in numeric_cols:
            X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
            med = X_num[c].median()
            X_num[c] = X_num[c].fillna(0.0 if pd.isna(med) else med).astype(float)
        parts.append(X_num)

    if binary_cols:
        X_bin = Xraw[binary_cols].copy()
        for c in binary_cols:
            X_bin[c] = pd.to_numeric(X_bin[c], errors="coerce").fillna(0.0).astype(float)
        parts.append(X_bin)

    if categorical_cols:
        X_cat = Xraw[categorical_cols].copy()
        for c in categorical_cols:
            X_cat[c] = X_cat[c].astype("string").fillna("__MISSING__")
        parts.append(pd.get_dummies(X_cat, drop_first=False, dtype=float))

    Xdf = pd.concat(parts, axis=1)
    Xdf = Xdf.loc[:, ~Xdf.columns.duplicated()].copy()
    Xdf.columns = [str(c) for c in Xdf.columns]
    Xdf = Xdf.reindex(sorted(Xdf.columns), axis=1)

    X = row_l2_normalize_leq1(Xdf.to_numpy(dtype=float))

    print(f"[INFO] {dataset_label} preprocessing complete:")
    print(f" dropped columns = {drop_cols}")
    print(f" numeric_cols = {len(numeric_cols)}")
    print(f" binary_cols = {len(binary_cols)}")
    print(f" categorical_cols = {categorical_cols}")
    print(f" final feature count = {X.shape[1]}")
    return X, list(Xdf.columns)


def fit_propensity_lr(
    rng: np.random.Generator,
    X_all: np.ndarray,
    prop_subsample: int,
    gamma_scale: float,
) -> dict[str, np.ndarray | float]:
    n, d = X_all.shape
    m = min(prop_subsample, n)
    idx = rng.choice(n, size=m, replace=False)
    X_sub = X_all[idx]

    gamma = rng.normal(loc=0.0, scale=gamma_scale, size=d)
    prob = sigmoid_stable(X_sub @ gamma)
    y_sub = (rng.random(m) < prob).astype(int)

    if int(y_sub.min()) == int(y_sub.max()):
        y_sub[: m // 2] = 0
        y_sub[m // 2:] = 1
        rng.shuffle(y_sub)

    clf = LogisticRegression(solver="lbfgs", max_iter=2000)
    clf.fit(X_sub, y_sub)
    return {
        "coef": clf.coef_.reshape(-1).astype(float),
        "intercept": float(clf.intercept_.reshape(-1)[0]),
    }


def predict_propensity(model: dict[str, np.ndarray | float], X: np.ndarray) -> np.ndarray:
    return sigmoid_stable(X @ model["coef"] + model["intercept"])


def make_nonlinear_env(
    rng: np.random.Generator,
    X_all: np.ndarray,
    config: StreamGenerationConfig,
) -> dict[str, np.ndarray | float | dict]:
    beta3 = config.beta_sampler(rng, X_all.shape[1])
    rho = fit_propensity_lr(
        rng,
        X_all,
        prop_subsample=config.prop_subsample,
        gamma_scale=config.propensity_gamma_scale,
    )
    return {"beta3": beta3, "rho": rho}


def alpha_schedule(L: int) -> np.ndarray:
    pos = np.arange(L, dtype=float)
    alpha = (pos - 0.5 * L) / (0.5 * L)
    return np.minimum(1.0, np.maximum(0.0, alpha))


def sample_full_online_indices(rng: np.random.Generator, n: int, total_len: int) -> np.ndarray:
    return rng.choice(n, size=total_len, replace=total_len > n)


def build_segment_slices(total_len: int, K: int) -> list[tuple[int, int, int]]:
    if total_len % K != 0:
        raise ValueError(f"T={total_len} must be divisible by K={K}.")
    L = total_len // K
    return [(seg, seg * L, (seg + 1) * L) for seg in range(K)]


def generate_one_stream(
    rng: np.random.Generator,
    X_all: np.ndarray,
    feature_names: list[str],
    drift: str,
    K: int,
    repeat_id: int,
    seed_value: int,
    config: StreamGenerationConfig,
) -> pd.DataFrame:
    n = X_all.shape[0]
    segment_slices = build_segment_slices(config.t, K)
    online_idx_full = sample_full_online_indices(rng, n, config.t)
    X_stream = X_all[online_idx_full]
    rows: list[dict[str, float | int | str]] = []

    env_count = K if drift == "switching" else (K + 1)
    envs = [make_nonlinear_env(rng, X_all, config) for _ in range(env_count)]

    for seg, start, end in segment_slices:
        X_seg = X_stream[start:end]
        idx_seg = online_idx_full[start:end]
        L = end - start
        alpha = alpha_schedule(L)

        if drift == "switching":
            env = envs[seg]
            mu_m1, mu_p1 = config.response_surface(X_seg, env["beta3"])
            q = predict_propensity(env["rho"], X_seg)
        elif drift == "linear":
            env_a = envs[seg]
            env_b = envs[seg + 1]
            mu_m1_a, mu_p1_a = config.response_surface(X_seg, env_a["beta3"])
            mu_m1_b, mu_p1_b = config.response_surface(X_seg, env_b["beta3"])
            q_a = predict_propensity(env_a["rho"], X_seg)
            q_b = predict_propensity(env_b["rho"], X_seg)
            mu_m1 = (1.0 - alpha) * mu_m1_a + alpha * mu_m1_b
            mu_p1 = (1.0 - alpha) * mu_p1_a + alpha * mu_p1_b
            q = (1.0 - alpha) * q_a + alpha * q_b
        else:
            raise ValueError("drift must be 'switching' or 'linear'")

        q = np.clip(q, config.eps, 1.0 - config.eps)
        w = np.where(rng.random(L) < q, 1, -1)
        p = np.where(w == 1, q, 1.0 - q)
        y_m1 = mu_m1 + config.noise_std * rng.normal(size=L)
        y_p1 = mu_p1 + config.noise_std * rng.normal(size=L)
        y = np.where(w == 1, y_p1, y_m1)
        tau = mu_p1 - mu_m1

        for j in range(L):
            t_global = start + j + 1
            row: dict[str, float | int | str] = {
                "env_name": f"{config.env_prefix}_nonlinear_{drift}_K{K}_repeat{repeat_id:02d}",
                "surface": "nonlinear",
                "drift": drift,
                "K": int(K),
                "repeat": int(repeat_id),
                "seed": int(seed_value),
                "segment": int(seg),
                "segment_round": int(j),
                "t": int(t_global),
                "alpha": float(alpha[j] if drift == "linear" else 0.0),
                "source_row": int(idx_seg[j]),
                "q": float(q[j]),
                "w": int(w[j]),
                "p": float(p[j]),
                "y": float(y[j]),
                "mu_-1": float(mu_m1[j]),
                "mu_1": float(mu_p1[j]),
                "tau": float(tau[j]),
            }
            for k, fn in enumerate(feature_names):
                row[fn] = float(X_seg[j, k])
            rows.append(row)

    df_stream = pd.DataFrame(rows)
    if len(df_stream) != config.t:
        raise RuntimeError(f"Generated stream length mismatch: expected {config.t}, got {len(df_stream)}")
    return df_stream


def run_generation(config: StreamGenerationConfig) -> None:
    os.makedirs(config.outdir, exist_ok=True)
    df_raw = pd.read_csv(config.data_path)
    X_all, feature_names = preprocess_covariates(df_raw, config.dataset_prefix)

    configs: list[tuple[str, int]] = [
        ("switching", 5),
        ("switching", 20),
        ("linear", 5),
        ("linear", 20),
    ]
    total_jobs = len(configs) * config.repeats
    child_seqs = np.random.SeedSequence(config.base_seed).spawn(total_jobs)

    manifest_rows: list[dict[str, int | str]] = []
    job_id = 0
    for drift, K in configs:
        for repeat_id in range(1, config.repeats + 1):
            rng = np.random.default_rng(child_seqs[job_id])
            seed_value = int(rng.integers(1000, 10_000_000))
            df_stream = generate_one_stream(
                rng=rng,
                X_all=X_all,
                feature_names=feature_names,
                drift=drift,
                K=K,
                repeat_id=repeat_id,
                seed_value=seed_value,
                config=config,
            )

            fname = f"{config.env_prefix}_nonlinear_{drift}_K{K}_repeat{repeat_id:02d}.csv"
            df_stream.to_csv(os.path.join(config.outdir, fname), index=False)
            manifest_rows.append({
                "surface": "nonlinear",
                "drift": drift,
                "K": int(K),
                "repeat": int(repeat_id),
                "seed": int(seed_value),
                "rows": int(len(df_stream)),
                "file": fname,
            })
            print(f"[OK] {fname} | repeat={repeat_id:02d} | seed={seed_value} | rows={len(df_stream)}")
            job_id += 1

    manifest_path = os.path.join(config.outdir, "manifest.csv")
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"\n[ALL DONE] {config.done_message}")
    print(f"[ALL DONE] Output directory: {config.outdir}")
    print(f"[ALL DONE] Manifest: {manifest_path}")
