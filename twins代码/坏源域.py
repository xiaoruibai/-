#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


# ============================================================
# 0) Path and experiment settings
# ============================================================
DATA_PATH = "twin_pairs_X_3years_samesex.csv"
OUTDIR = "twins_bad_ethos_streams_repeats10_source"

T = 3000
REPEATS = 50
EPS = 0.05
NOISE_VAR = 0.1
NOISE_STD = float(np.sqrt(NOISE_VAR))
BASE_SEED = 1500
PROP_SUBSAMPLE = 3000

KNOWN_MULTICLASS_CANONICAL: set[str] = set()
EXPECTED_CATEGORIES: dict[str, set[str]] = {}


# ============================================================
# 1) Utility functions
# ============================================================
def sigmoid_stable(z: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid."""
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z, dtype=float)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def clip_prob_eps(q: np.ndarray, eps: float) -> np.ndarray:
    """Clip probabilities into [eps, 1-eps]."""
    return np.clip(q, eps, 1.0 - eps)



def row_l2_normalize_leq1(X: np.ndarray) -> np.ndarray:
    """Scale each row so its L2 norm is at most 1."""
    norms = np.linalg.norm(X, axis=1)
    scale = np.maximum(1.0, norms)
    return X / (scale[:, None] + 1e-12)


def series_to_numeric_if_possible(s: pd.Series) -> pd.Series | None:
    """Attempt to convert a series to numeric; return None if any parse fails."""
    s_num = pd.to_numeric(s, errors="coerce")
    if int(s_num.notna().sum()) == int(s.notna().sum()):
        return s_num
    return None


def is_binary_01_series(s: pd.Series) -> bool:
    """Check whether the series contains only 0 and 1 (ignoring NaNs)."""
    s_num = series_to_numeric_if_possible(s)
    if s_num is None:
        return False
    non_na = s_num.dropna().astype(float)
    if len(non_na) == 0:
        return False
    uniq = set(np.unique(non_na).tolist())
    return uniq.issubset({0.0, 1.0})


def canonicalize_col_name(name: str) -> str:
    """Lowercase and strip underscores/hyphens/spaces from a column name."""
    s = str(name).strip().lower()
    s = s.replace("_", "").replace("-", "").replace(" ", "")
    return s


def pick_covariate_columns(df: pd.DataFrame) -> list[str]:
    """Select twins covariate columns by exclusion rules, preserving twins structure."""
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

    if len(cov_cols) == 0:
        raise ValueError("No covariate columns found; please check the input CSV column names.")
    return cov_cols


def validate_known_multiclass_columns(Xraw: pd.DataFrame, categorical_cols: list[str]) -> None:
    """Validate known multiclass columns if EXPECTED_CATEGORIES is provided."""
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
        if len(bad) > 0:
            raise ValueError(
                f"Column {raw_col} is expected to be a known categorical variable, "
                f"but unexpected values {sorted(bad)} were observed."
            )


# ============================================================
# 2) Covariate preprocessing
# ============================================================
def preprocess_covariates(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Process raw twins covariates into a numeric matrix while preserving twins features."""
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
            continue
        if is_binary_01_series(s):
            binary_cols.append(c)
            continue
        if pd.api.types.is_numeric_dtype(s) or (series_to_numeric_if_possible(s) is not None):
            numeric_cols.append(c)
            continue
        categorical_cols.append(c)

    validate_known_multiclass_columns(Xraw, categorical_cols)

    parts: list[pd.DataFrame] = []

    if numeric_cols:
        X_num = Xraw[numeric_cols].copy()
        for c in numeric_cols:
            X_num[c] = pd.to_numeric(X_num[c], errors="coerce")
            med = X_num[c].median()
            if pd.isna(med):
                med = 0.0
            X_num[c] = X_num[c].fillna(med).astype(float)
        parts.append(X_num)

    if binary_cols:
        X_bin = Xraw[binary_cols].copy()
        for c in binary_cols:
            X_bin[c] = pd.to_numeric(X_bin[c], errors="coerce")
            X_bin[c] = X_bin[c].fillna(0.0).astype(float)
        parts.append(X_bin)

    if categorical_cols:
        X_cat = Xraw[categorical_cols].copy()
        for c in categorical_cols:
            X_cat[c] = X_cat[c].astype("string")
            X_cat[c] = X_cat[c].fillna("__MISSING__")
        X_cat = pd.get_dummies(X_cat, drop_first=False, dtype=float)
        parts.append(X_cat)

    Xdf = pd.concat(parts, axis=1)
    Xdf = Xdf.loc[:, ~Xdf.columns.duplicated()].copy()
    Xdf.columns = [str(c) for c in Xdf.columns]
    Xdf = Xdf.reindex(sorted(Xdf.columns), axis=1)

    X = Xdf.to_numpy(dtype=float)
    X = row_l2_normalize_leq1(X)

    print("[INFO] Preprocessing complete:")
    print(f" dropped columns = {drop_cols}")
    print(f" numeric_cols = {len(numeric_cols)}")
    print(f" binary_cols = {len(binary_cols)}")
    print(f" categorical_cols = {categorical_cols}")
    print(f" final feature count = {X.shape[1]}")

    return X, list(Xdf.columns)


# ============================================================
# 3) Semi-synthetic parameter generation (nonlinear only)
# ============================================================
def sample_beta_nonlinear(rng: np.random.Generator, d: int) -> np.ndarray:
    """
    Keep the current uploaded nonlinear beta distribution unchanged.
    """
    vals = np.array([0.0, 0.1, -0.2, -0.3, -0.4], dtype=float)
    probs = np.array([0.2, 0.2, 0.2, 0.2, 0.2], dtype=float)
    beta = rng.choice(vals, size=d, p=probs).astype(float)
    if np.linalg.norm(beta) < 1e-12:
        beta[rng.integers(0, d)] = 0.1
    return beta


def fit_propensity_lr(
    rng: np.random.Generator,
    X_all: np.ndarray
) -> dict[str, np.ndarray | float]:
    """Fit a logistic regression model used to generate treatment propensities."""
    n, d = X_all.shape
    m = min(PROP_SUBSAMPLE, n)
    idx = rng.choice(n, size=m, replace=False)
    X_sub = X_all[idx]

    gamma = rng.normal(loc=0.0, scale=1.5, size=d)
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


def predict_propensity(
    model: dict[str, np.ndarray | float],
    X: np.ndarray
) -> np.ndarray:
    """Compute predicted propensities given logistic regression parameters."""
    return sigmoid_stable(X @ model["coef"] + model["intercept"])


def make_nonlinear_env(
    rng: np.random.Generator,
    X_all: np.ndarray
) -> dict[str, np.ndarray | float | dict]:
    """Create a nonlinear environment."""
    d = X_all.shape[1]
    beta3 = sample_beta_nonlinear(rng, d)
    rho = fit_propensity_lr(rng, X_all)
    return {"beta3": beta3, "rho": rho}


# ============================================================
# 4) Nonlinear response surface
# ============================================================
def mu_nonlinear(X: np.ndarray, beta3: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute potential outcomes for the nonlinear surface."""
    nb = np.linalg.norm(beta3) + 1e-12
    mu_m1 = np.exp((X + 0.25) @ beta3) / np.exp(nb)
    mu_p1 = ( X**2  @ beta3 + 0.25 ) / nb
    return mu_m1, mu_p1


# ============================================================
# 5) Online environment scheduling
# ============================================================
def alpha_schedule(L: int) -> np.ndarray:
    """Schedule alpha from 0 to 1 across L points for linear drift."""
    pos = np.arange(L, dtype=float)
    alpha = (pos - 0.5 * L) / (0.5 * L)
    alpha = np.maximum(0.0, alpha)
    alpha = np.minimum(1.0, alpha)
    return alpha


def sample_full_online_indices(
    rng: np.random.Generator,
    n: int,
    total_len: int
) -> np.ndarray:
    """Sample an entire online stream directly from the full covariate pool, no train/test split."""
    replace = total_len > n
    return rng.choice(n, size=total_len, replace=replace)


def build_segment_slices(T_total: int, K: int) -> list[tuple[int, int, int]]:
    """Build (segment, start, end) slices for the full stream."""
    if T_total % K != 0:
        raise ValueError(f"T={T_total} must be divisible by K={K}.")
    L = T_total // K
    return [(seg, seg * L, (seg + 1) * L) for seg in range(K)]


# ============================================================
# 6) Generate a single stream (one repeat, nonlinear only)
# ============================================================
def generate_one_stream(
    rng: np.random.Generator,
    X_all: np.ndarray,
    feature_names: list[str],
    drift: str,
    K: int,
    repeat_id: int,
    seed_value: int,
) -> pd.DataFrame:
    """Generate one full online environment stream for a single repeat."""
    n = X_all.shape[0]
    segment_slices = build_segment_slices(T, K)
    online_idx_full = sample_full_online_indices(rng, n, T)
    X_stream = X_all[online_idx_full]

    rows: list[dict[str, float | int | str]] = []

    env_count = K if drift == "switching" else (K + 1)
    envs = [make_nonlinear_env(rng, X_all) for _ in range(env_count)]

    for seg, start, end in segment_slices:
        X_seg = X_stream[start:end]
        idx_seg = online_idx_full[start:end]
        L = end - start
        alpha = alpha_schedule(L)

        if drift == "switching":
            env = envs[seg]
            mu_m1, mu_p1 = mu_nonlinear(X_seg, env["beta3"])
            q = predict_propensity(env["rho"], X_seg)

        elif drift == "linear":
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
            raise ValueError("drift must be 'switching' or 'linear'")

        q = clip_prob_eps(q, EPS)
        w = np.where(rng.random(L) < q, 1, -1)
        p = np.where(w == 1, q, 1.0 - q)

        y_m1 = mu_m1 + NOISE_STD * rng.normal(size=L)
        y_p1 = mu_p1 + NOISE_STD * rng.normal(size=L)
        y = np.where(w == 1, y_p1, y_m1)
        tau = mu_p1 - mu_m1

        for j in range(L):
            t_global = start + j + 1
            row: dict[str, float | int | str] = {
                "env_name": f"twins_good_nonlinear_{drift}_K{K}_repeat{repeat_id:02d}",
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
            xj = X_seg[j]
            for k, fn in enumerate(feature_names):
                row[fn] = float(xj[k])
            rows.append(row)

    df_stream = pd.DataFrame(rows)
    if len(df_stream) != T:
        raise RuntimeError(f"Generated stream length mismatch: expected {T}, got {len(df_stream)}")
    return df_stream


# ============================================================
# 7) Main function
# ============================================================
def main() -> None:
    os.makedirs(OUTDIR, exist_ok=True)

    df_raw = pd.read_csv(DATA_PATH)
    X_all, feature_names = preprocess_covariates(df_raw)

    configs: list[tuple[str, int]] = [
        ("switching", 5),
        ("switching", 20),
        ("linear", 5),
        ("linear", 20),
    ]

    total_jobs = len(configs) * REPEATS
    child_seqs = np.random.SeedSequence(BASE_SEED).spawn(total_jobs)

    manifest_rows: list[dict[str, int | str]] = []
    job_id = 0

    for drift, K in configs:
        for repeat_id in range(1, REPEATS + 1):
            child_ss = child_seqs[job_id]
            rng = np.random.default_rng(child_ss)
            seed_value = int(rng.integers(1000, 10_000_000))

            df_stream = generate_one_stream(
                rng=rng,
                X_all=X_all,
                feature_names=feature_names,
                drift=drift,
                K=K,
                repeat_id=repeat_id,
                seed_value=seed_value,
            )

            fname = f"twins_good_nonlinear_{drift}_K{K}_repeat{repeat_id:02d}.csv"
            fpath = os.path.join(OUTDIR, fname)
            df_stream.to_csv(fpath, index=False)

            manifest_rows.append(
                {
                    "surface": "nonlinear",
                    "drift": drift,
                    "K": int(K),
                    "repeat": int(repeat_id),
                    "seed": int(seed_value),
                    "rows": int(len(df_stream)),
                    "file": fname,
                }
            )

            print(
                f"[OK] {fname} | repeat={repeat_id:02d} | "
                f"seed={seed_value} | rows={len(df_stream)}"
            )
            job_id += 1

    manifest = pd.DataFrame(manifest_rows)
    manifest_path = os.path.join(OUTDIR, "manifest.csv")
    manifest.to_csv(manifest_path, index=False)

    print("\n[ALL DONE] All twins good/source nonlinear-only repeat streams generated successfully.")
    print(f"[ALL DONE] Output directory: {OUTDIR}")
    print(f"[ALL DONE] Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
