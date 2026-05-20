# -*- coding: utf-8 -*-

import os

# ============================================================
# 0) 非常关键：必须在 numpy / sklearn / torch 之前限制每个子进程内部线程数
#    否则多进程 * 每进程多线程，会导致 Windows 上极慢甚至卡死
# ============================================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import re
import json
import copy
import random
import warnings
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from scipy.linalg import LinAlgWarning
except Exception:
    LinAlgWarning = Warning

warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=LinAlgWarning)
warnings.filterwarnings(
    "ignore",
    message="Singular matrix in solving dual problem. Using least-squares solution instead."
)

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "未检测到 PyTorch。\n"
        "如果你现在是 Windows + Python 3.9，建议先运行：\n"
        "python -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 "
        "--index-url https://download.pytorch.org/whl/cpu"
    ) from e


# ============================================================
# 1) 全局配置
#    现在改成与 acic16 独立 repeat 数据生成脚本对齐：
#      - 读取 twins_{surface}_{drift}_K{K}_repeat01.csv ... repeat10.csv
#      - 对每个 repeat 文件单独做一次完整离线滑窗实验
#      - 再按 base environment 聚合 repeat 均值/方差
#
#    评估口径保持不变：
#      cumhalfPEHE = Σ_t 0.5 * (tau_hat_t - tau_t)^2
# ============================================================

def resolve_data_dir(candidates):
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    expanded = []
    for cand in candidates:
        expanded.append(cand)
        expanded.append(os.path.join(script_dir, cand))
        expanded.append(os.path.join(os.getcwd(), cand))

    seen = []
    for p in expanded:
        if p not in seen:
            seen.append(p)

    for p in seen:
        if os.path.isdir(p):
            return p

    raise FileNotFoundError(f"找不到数据目录，候选路径: {seen}")


TARGET_DATA_DIR = resolve_data_dir([
    "acic16_ethos_streams_repeats10_target",
 
])

OUTPUT_DIR = "acic16_offline_baseline_prequential_outputs"
BASE_SEED = 2026
N_REPEATS = 10

REQUESTED_N_JOBS = min(24, os.cpu_count() or 24)

if os.name == "nt":
    N_JOBS_CLASSICAL = min(8, REQUESTED_N_JOBS)
    N_JOBS_NEURAL = min(2, REQUESTED_N_JOBS)
else:
    N_JOBS_CLASSICAL = min(12, REQUESTED_N_JOBS)
    N_JOBS_NEURAL = min(3, REQUESTED_N_JOBS)

METHOD_WINDOW_GRID = {
    "OLS": [50, 100],
    "KNN": [50, 100],
    "PSM": [50, 100],
    "RF": [50, 100],
    "CF": [50, 100],
    "BNN": [100],
}
WINDOW_GRID = sorted({w for ws in METHOD_WINDOW_GRID.values() for w in ws})

INPUT_REPR = "raw"
NEURAL_INPUT_REPR = "raw"

EPS = 1e-6
DEVICE = torch.device("cpu")

METHODS_TO_RUN = ["OLS", "KNN", "PSM", "RF", "CF", "BNN"]
NEURAL_METHODS = {"BNN"}
CLASSICAL_METHODS = [m for m in METHODS_TO_RUN if m not in NEURAL_METHODS]

META_COLS = {
    "env_name", "surface", "drift", "K",
    "repeat", "seed",
    "segment", "segment_round", "t",
    "alpha", "source_row", "q",
    "w", "p", "y", "mu_-1", "mu_1", "tau"
}

FORBIDDEN_RAW_COLS = {
    "unnamed: 0", "treatment", "y_factual", "y_cfactual", "yf", "ycf",
    "mu0", "mu1", "y0", "y1", "potential_outcome_0", "potential_outcome_1",
    "outcome_0", "outcome_1", "cate_true", "ite_true",
}

REPEAT_ENV_FILE_REGEX = re.compile(r"^(acic16_(linear|nonlinear)_(switching|linear)_K(\d+))_repeat(\d+)\.csv$")
LEGACY_ENV_FILE_REGEX = re.compile(r"^(acic16_(linear|nonlinear)_(switching|linear)_K(\d+)\.csv)$")

_TORCH_RUNTIME_CONFIGURED = False
_BUNDLE_CACHE = {}


# ============================================================
# 2) 基础工具
# ============================================================

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def configure_torch_runtime():
    global _TORCH_RUNTIME_CONFIGURED
    if _TORCH_RUNTIME_CONFIGURED:
        return

    try:
        torch.set_num_threads(1)
    except Exception:
        pass

    try:
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    _TORCH_RUNTIME_CONFIGURED = True


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    try:
        torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def row_l2_normalize_leq1(X):
    norms = np.linalg.norm(X, axis=1)
    scale = np.maximum(1.0, norms)
    return X / (scale[:, None] + 1e-12)


def canonical_col(c):
    return str(c).strip().lower()


def sort_stream_df(df):
    sort_cols = [c for c in ["segment", "segment_round", "t"] if c in df.columns]
    if sort_cols:
        return df.sort_values(sort_cols).reset_index(drop=True)
    return df.reset_index(drop=True)


def parse_env_file_info(path_or_name):
    name = os.path.basename(path_or_name)
    m = REPEAT_ENV_FILE_REGEX.match(name)
    if m is not None:
        return {
            "dataset": m.group(1),
            "surface": m.group(2),
            "drift": m.group(3),
            "K": int(m.group(4)),
            "repeat": int(m.group(5)),
            "filename": name,
            "env_name": m.group(1),
            "env_repeat_name": name[:-4],
        }

    m_legacy = LEGACY_ENV_FILE_REGEX.match(name)
    if m_legacy is not None:
        return {
            "dataset": m_legacy.group(1)[:-4],
            "surface": m_legacy.group(2),
            "drift": m_legacy.group(3),
            "K": int(m_legacy.group(4)),
            "repeat": None,
            "filename": name,
            "env_name": m_legacy.group(1)[:-4],
            "env_repeat_name": name[:-4],
        }

    raise ValueError(f"环境文件名不符合约定格式: {name}")


def discover_environment_files(data_dir):
    matched = []
    legacy_hits = []

    for fname in os.listdir(data_dir):
        fpath = os.path.join(data_dir, fname)
        if not os.path.isfile(fpath):
            continue

        m = REPEAT_ENV_FILE_REGEX.match(fname)
        if m is not None:
            info = parse_env_file_info(fname)
            info["full_path"] = fpath
            matched.append(info)
            continue

        if LEGACY_ENV_FILE_REGEX.match(fname):
            legacy_hits.append(fname)

    if matched:
        surface_order = {"linear": 0, "nonlinear": 1}
        drift_order = {"switching": 0, "linear": 1}
        matched.sort(
            key=lambda x: (
                surface_order.get(x["surface"], 999),
                drift_order.get(x["drift"], 999),
                x["K"],
                x["repeat"],
                x["filename"],
            )
        )
        return matched

    if legacy_hits:
        raise FileNotFoundError(
            "发现旧格式 acic16 环境文件，但没有发现独立 repeat 文件。"
            "当前脚本期望文件名形如 acic16_linear_switching_K5_repeat01.csv。"
            f"目前找到的旧文件示例: {sorted(legacy_hits)[:5]}"
        )

    raise FileNotFoundError(f"在 {data_dir} 中未找到 acic16 独立 repeat 环境文件。")


def validate_repeat_inventory(env_files, expected_n_repeats):
    repeat_map = defaultdict(list)
    for info in env_files:
        repeat_map[str(info["env_name"])].append(int(info["repeat"]))

    problems = []
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


def unique_base_envs(env_files):
    seen = set()
    out = []
    for info in env_files:
        key = str(info["env_name"])
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "env_name": info["env_name"],
            "surface": info["surface"],
            "drift": info["drift"],
            "K": info["K"],
        })
    out.sort(key=lambda x: (x["surface"], x["drift"], x["K"], x["env_name"]))
    return out


def get_feature_columns(df):
    """
    训练时只返回真正可用的协变量列。

    关键修复：
    - 数据文件里允许存在 mu0/mu1、y_factual/y_cfactual、ite_true 等真值或潜在结果列；
    - 这些列不能进入训练特征，但也不应该触发报错；
    - 它们会被自动忽略，仅保留在原始 dataframe 中，供评估或调试使用。
    """
    feat_cols = []
    ignored_leakage_cols = []

    for c in df.columns:
        c_can = canonical_col(c)

        if c in META_COLS:
            continue

        if c_can in FORBIDDEN_RAW_COLS:
            ignored_leakage_cols.append(c)
            continue

        feat_cols.append(c)

    if len(feat_cols) == 0:
        raise ValueError(
            "未识别到协变量列。请检查是否把所有协变量都写进了 META_COLS，"
            "或者数据文件里只剩下评估/真值列。"
        )

    non_numeric = [c for c in feat_cols if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"检测到非数值型协变量列: {non_numeric}")

    return feat_cols


def infer_surface(df, fallback_name=""):
    if "surface" in df.columns:
        vals = df["surface"].dropna().astype(str).unique().tolist()
        if len(vals) == 1:
            s = vals[0].strip().lower()
            if s in {"linear", "nonlinear"}:
                return s

    name = fallback_name.lower()
    if "nonlinear" in name:
        return "nonlinear"
    if "linear" in name:
        return "linear"

    raise ValueError("无法识别 surface。")


def encode_treatment_to_binary(w_raw):
    """
    统一把处理标签转成 0/1。
    兼容 {0,1}、{-1,1}，以及任意两个不同取值。
    """
    w_raw = np.asarray(w_raw)
    uniq = np.unique(w_raw)

    if uniq.size != 2:
        raise ValueError(f"w 列必须恰好只有两个不同取值，当前为: {uniq.tolist()}")

    uniq_set = set(uniq.tolist())

    if uniq_set == {0, 1}:
        return w_raw.astype(np.int64)

    if uniq_set == {-1, 1}:
        return (w_raw == 1).astype(np.int64)

    lo = np.min(uniq)
    hi = np.max(uniq)
    return (w_raw == hi).astype(np.int64)


# ============================================================
# 3) 特征映射
#    linear: [x, 1]
#    nonlinear: 固定随机 MLP(256,256,256,2048)
# ============================================================

def init_random_mlp(input_dim, seed):
    rng = np.random.default_rng(seed)
    dims = [input_dim, 256, 256, 256, 2048]
    weights, biases = [], []
    for din, dout in zip(dims[:-1], dims[1:]):
        W = rng.normal(0.0, 1.0 / np.sqrt(din), size=(din, dout)).astype(np.float32)
        b = rng.normal(0.0, 0.01, size=(dout,)).astype(np.float32)
        weights.append(W)
        biases.append(b)
    return {"surface": "nonlinear", "weights": weights, "biases": biases}


def make_feature_map(surface, input_dim, seed):
    if surface == "linear":
        return {"surface": "linear"}
    if surface == "nonlinear":
        return init_random_mlp(input_dim, seed)
    raise ValueError("surface 必须为 linear 或 nonlinear。")


def apply_feature_map_batch(feature_map, X):
    if feature_map["surface"] == "linear":
        Phi = np.concatenate(
            [X.astype(np.float64), np.ones((X.shape[0], 1), dtype=np.float64)],
            axis=1
        )
        return row_l2_normalize_leq1(Phi)

    Hx = np.asarray(X, dtype=np.float32)
    for idx, (W, b) in enumerate(zip(feature_map["weights"], feature_map["biases"])):
        Hx = Hx @ W + b
        if idx < len(feature_map["weights"]) - 1:
            Hx = np.maximum(Hx, 0.0)

    return row_l2_normalize_leq1(Hx.astype(np.float64))


# ============================================================
# 4) 数据读取
#    直接按完整排序后的 target stream 进行 prequential 评估
# ============================================================

def prepare_env(df, feature_map):
    df = sort_stream_df(df)
    feature_cols = get_feature_columns(df)

    X_raw = df[feature_cols].to_numpy(dtype=np.float64)
    Phi = apply_feature_map_batch(feature_map, X_raw)

    w = encode_treatment_to_binary(df["w"].to_numpy())
    p = np.clip(df["p"].to_numpy(dtype=np.float64), EPS, 1.0 - EPS)
    y = df["y"].to_numpy(dtype=np.float64)
    tau = df["tau"].to_numpy(dtype=np.float64)

    segment = df["segment"].to_numpy(dtype=np.int64) if "segment" in df.columns else np.zeros(len(df), dtype=np.int64)
    segment_round = (
        df["segment_round"].to_numpy(dtype=np.int64)
        if "segment_round" in df.columns
        else np.arange(len(df), dtype=np.int64)
    )
    t = df["t"].to_numpy(dtype=np.int64) if "t" in df.columns else np.arange(len(df), dtype=np.int64)

    return {
        "df": df,
        "X_raw": X_raw,
        "Phi": Phi,
        "w": w,
        "p": p,
        "y": y,
        "tau": tau,
        "segment": segment,
        "segment_round": segment_round,
        "t": t,
    }


def load_bundle(target_path, seed):
    configure_torch_runtime()
    set_global_seed(seed)

    df_target = pd.read_csv(target_path)
    feature_cols = get_feature_columns(df_target)
    surface_t = infer_surface(df_target, target_path)
    feature_map = make_feature_map(surface_t, len(feature_cols), seed)

    env_target = prepare_env(df_target, feature_map)

    return {
        "surface": surface_t,
        "env_target": env_target,
    }


def get_bundle_cached(target_path, seed):
    key = (target_path, seed)
    if key not in _BUNDLE_CACHE:
        _BUNDLE_CACHE[key] = load_bundle(target_path, seed)
    return _BUNDLE_CACHE[key]


# ============================================================
# 5) 评估口径
#    cumhalfPEHE = Σ_t 0.5 * (pred_tau_t - tau_t)^2
# ============================================================

class MetricTracker:
    def __init__(self):
        self.cumulative_loss_half = 0.0

    def add_scalar(self, pred_tau, tau_true):
        sqerr = float((pred_tau - tau_true) ** 2)
        self.cumulative_loss_half += 0.5 * sqerr

    def to_dict(self):
        return {
            "cumhalfPEHE": float(self.cumulative_loss_half),
        }


def choose_input(env_target, repr_name):
    if repr_name == "raw":
        return env_target["X_raw"]
    if repr_name == "mapped":
        return env_target["Phi"]
    raise ValueError(f"未知输入表示: {repr_name}")


def safe_clip_propensity(p):
    return np.clip(np.asarray(p, dtype=np.float64), 0.025, 0.975)


# ============================================================
# 6) 基础模型
# ============================================================

class ZeroCateModel:
    def predict_tau(self, X):
        return np.zeros(X.shape[0], dtype=np.float64)


class ConstantRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, value=0.0):
        self.value = float(value)

    def fit(self, X, y):
        y = np.asarray(y, dtype=np.float64)
        self.value_ = float(np.mean(y)) if y.size > 0 else self.value
        return self

    def predict(self, X):
        return np.full(X.shape[0], getattr(self, "value_", self.value), dtype=np.float64)


def fit_propensity_model(X, w, C=1.0):
    w = np.asarray(w, dtype=np.int64)
    if len(np.unique(w)) < 2:
        return {"type": "constant", "p": float(np.mean(w))}

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=float(C),
            solver="liblinear",
            max_iter=2000,
            random_state=0
        )),
    ])

    try:
        model.fit(X, w)
        return {"type": "model", "model": model}
    except Exception:
        return {"type": "constant", "p": float(np.mean(w))}


def predict_propensity(prop_model, X):
    if prop_model["type"] == "constant":
        return np.full(X.shape[0], prop_model["p"], dtype=np.float64)
    return safe_clip_propensity(prop_model["model"].predict_proba(X)[:, 1])


class OLSCateModel:
    def __init__(self, model):
        self.model = model

    @staticmethod
    def _design(X, t):
        t_arr = np.asarray(t, dtype=np.float64).reshape(-1, 1)
        if t_arr.shape[0] == 1 and X.shape[0] > 1:
            t_arr = np.full((X.shape[0], 1), float(t_arr[0, 0]), dtype=np.float64)
        inter = X * t_arr
        return np.concatenate([X, t_arr, inter], axis=1)

    def predict_tau(self, X):
        X1 = self._design(X, 1.0)
        X0 = self._design(X, 0.0)
        return np.asarray(self.model.predict(X1) - self.model.predict(X0), dtype=np.float64)


def fit_ols_cate(X, w, y, config, seed):
    design = OLSCateModel._design(X, w)
    alpha = max(float(config["alpha"]), 1e-3)
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", Ridge(
            alpha=alpha,
            solver="lsqr",
            fit_intercept=True,
            random_state=seed
        ))
    ])
    model.fit(design, y)
    return OLSCateModel(model)


class TLearnerCateModel:
    def __init__(self, model0, model1):
        self.model0 = model0
        self.model1 = model1

    def predict_tau(self, X):
        return np.asarray(self.model1.predict(X) - self.model0.predict(X), dtype=np.float64)


def fit_knn_cate(X, w, y, config, seed):
    mask1 = (w == 1)
    mask0 = (w == 0)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return ZeroCateModel()

    k = int(config["n_neighbors"])
    weights = str(config.get("weights", "distance"))

    model0 = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=max(1, min(k, int(mask0.sum()))),
            weights=weights
        ))
    ])
    model1 = Pipeline([
        ("scaler", StandardScaler()),
        ("knn", KNeighborsRegressor(
            n_neighbors=max(1, min(k, int(mask1.sum()))),
            weights=weights
        ))
    ])

    model0.fit(X[mask0], y[mask0])
    model1.fit(X[mask1], y[mask1])
    return TLearnerCateModel(model0, model1)


class MatchingCateModel:
    def __init__(self, prop_model, ps_treat, y_treat, ps_ctrl, y_ctrl, k):
        self.prop_model = prop_model
        self.ps_treat = np.asarray(ps_treat, dtype=np.float64).reshape(-1, 1)
        self.y_treat = np.asarray(y_treat, dtype=np.float64)
        self.ps_ctrl = np.asarray(ps_ctrl, dtype=np.float64).reshape(-1, 1)
        self.y_ctrl = np.asarray(y_ctrl, dtype=np.float64)
        self.k = int(k)

        self.nn_treat = None
        self.nn_ctrl = None

        if self.ps_treat.shape[0] > 0:
            self.nn_treat = NearestNeighbors(n_neighbors=min(self.k, self.ps_treat.shape[0]))
            self.nn_treat.fit(self.ps_treat)

        if self.ps_ctrl.shape[0] > 0:
            self.nn_ctrl = NearestNeighbors(n_neighbors=min(self.k, self.ps_ctrl.shape[0]))
            self.nn_ctrl.fit(self.ps_ctrl)

    def predict_tau(self, X):
        q = predict_propensity(self.prop_model, X).reshape(-1, 1)
        if self.nn_treat is None or self.nn_ctrl is None:
            return np.zeros(X.shape[0], dtype=np.float64)

        idx_t = self.nn_treat.kneighbors(q, return_distance=False)
        idx_c = self.nn_ctrl.kneighbors(q, return_distance=False)
        mu1 = self.y_treat[idx_t].mean(axis=1)
        mu0 = self.y_ctrl[idx_c].mean(axis=1)
        return np.asarray(mu1 - mu0, dtype=np.float64)


def fit_psm_cate(X, w, y, config, seed):
    prop_model = fit_propensity_model(X, w, C=float(config["ps_C"]))
    ps = predict_propensity(prop_model, X)

    mask1 = (w == 1)
    mask0 = (w == 0)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return ZeroCateModel()

    return MatchingCateModel(
        prop_model,
        ps[mask1], y[mask1],
        ps[mask0], y[mask0],
        int(config["match_k"])
    )


def make_rf_regressor(seed, n_estimators=300, max_depth=None, min_samples_leaf=1, max_features="sqrt"):
    return RandomForestRegressor(
        n_estimators=int(n_estimators),
        max_depth=max_depth,
        min_samples_leaf=int(min_samples_leaf),
        max_features=max_features,
        bootstrap=True,
        n_jobs=1,
        random_state=seed
    )


def fit_rf_cate(X, w, y, config, seed):
    mask1 = (w == 1)
    mask0 = (w == 0)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return ZeroCateModel()

    n_estimators = int(config.get("n_estimators", 300))

    model0 = make_rf_regressor(
        seed=seed,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"])
    )
    model1 = make_rf_regressor(
        seed=seed + 1,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"])
    )

    model0.fit(X[mask0], y[mask0])
    model1.fit(X[mask1], y[mask1])
    return TLearnerCateModel(model0, model1)


class DirectCateModel:
    def __init__(self, model):
        self.model = model

    def predict_tau(self, X):
        return np.asarray(self.model.predict(X), dtype=np.float64)


def fit_cf_cate(X, w, y, config, seed):
    # sklearn 无原生 causal forest；这里用 forest-based DR learner 近似 CF 基线
    mask1 = (w == 1)
    mask0 = (w == 0)
    if mask0.sum() == 0 or mask1.sum() == 0:
        return ZeroCateModel()

    prop_model = fit_propensity_model(X, w, C=float(config.get("ps_C", 1.0)))
    ehat = predict_propensity(prop_model, X)
    n_estimators = int(config.get("n_estimators", 300))

    m0 = make_rf_regressor(
        seed=seed,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"])
    )
    m1 = make_rf_regressor(
        seed=seed + 1,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"])
    )

    m0.fit(X[mask0], y[mask0])
    m1.fit(X[mask1], y[mask1])

    m0_hat = m0.predict(X)
    m1_hat = m1.predict(X)

    pseudo_tau = (
        (m1_hat - m0_hat)
        + w * (y - m1_hat) / ehat
        - (1 - w) * (y - m0_hat) / (1.0 - ehat)
    )

    tau_model = make_rf_regressor(
        seed=seed + 2,
        n_estimators=n_estimators,
        max_depth=config["max_depth"],
        min_samples_leaf=int(config["min_samples_leaf"])
    )
    tau_model.fit(X, pseudo_tau)
    return DirectCateModel(tau_model)


# ============================================================
# 7) 神经网络模型：TARNet / BNN / CFR
# ============================================================

class MLPBlock(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, depth=2, dropout=0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, depth - 1)):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class TARNetModule(nn.Module):
    def __init__(self, input_dim, rep_dim=32, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.rep = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, rep_dim),
            nn.ReLU()
        )
        self.head0 = MLPBlock(rep_dim, hidden_dim, 1, depth=2, dropout=dropout)
        self.head1 = MLPBlock(rep_dim, hidden_dim, 1, depth=2, dropout=dropout)

    def forward(self, x):
        z = self.rep(x)
        y0 = self.head0(z).squeeze(-1)
        y1 = self.head1(z).squeeze(-1)
        return z, y0, y1


class NeuralCateModel:
    def __init__(self, net):
        self.net = net

    def predict_tau(self, X):
        self.net.eval()
        with torch.no_grad():
            xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
            _, y0, y1 = self.net(xt)
            return (y1 - y0).cpu().numpy().astype(np.float64)


def split_indices_for_early_stop(n, seed):
    if n < 8:
        idx = np.arange(n)
        return idx, idx

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    n_val = max(2, int(round(0.2 * n)))
    n_val = min(n_val, n - 2)
    val_idx = perm[:n_val]
    tr_idx = perm[n_val:]
    return np.sort(tr_idx), np.sort(val_idx)


def factual_loss(y0, y1, w, y):
    y_hat = torch.where(w > 0.5, y1, y0)
    return torch.mean((y_hat - y) ** 2)


def subsample_tensor_rows(x, max_points):
    if x.shape[0] <= max_points:
        return x
    idx = torch.randperm(x.shape[0], device=x.device)[:max_points]
    return x[idx]


def mmd_rbf(x_t, x_c, max_points=128):
    if x_t.shape[0] == 0 or x_c.shape[0] == 0:
        return torch.tensor(0.0, device=x_t.device if x_t.numel() else x_c.device)

    x_t = subsample_tensor_rows(x_t, max_points)
    x_c = subsample_tensor_rows(x_c, max_points)

    xx = torch.cdist(x_t, x_t, p=2) ** 2
    cc = torch.cdist(x_c, x_c, p=2) ** 2
    xc = torch.cdist(x_t, x_c, p=2) ** 2

    with torch.no_grad():
        bw = torch.median(torch.cat([xx.flatten(), cc.flatten(), xc.flatten()], dim=0))
        bw = torch.clamp(bw, min=1e-3)

    return torch.exp(-xx / bw).mean() + torch.exp(-cc / bw).mean() - 2.0 * torch.exp(-xc / bw).mean()


def sinkhorn_distance(x_t, x_c, reg=0.1, iters=8, max_points=64):
    if x_t.shape[0] == 0 or x_c.shape[0] == 0:
        return torch.tensor(0.0, device=x_t.device if x_t.numel() else x_c.device)

    x_t = subsample_tensor_rows(x_t, max_points)
    x_c = subsample_tensor_rows(x_c, max_points)

    C = torch.cdist(x_t, x_c, p=2) ** 2
    C = C / (torch.median(C.detach()) + 1e-6)

    a = torch.full((x_t.shape[0],), 1.0 / x_t.shape[0], device=x_t.device)
    b = torch.full((x_c.shape[0],), 1.0 / x_c.shape[0], device=x_c.device)

    logK = -C / max(float(reg), 1e-3)
    K = torch.exp(torch.clamp(logK, min=-30.0, max=30.0))

    u = torch.ones_like(a)
    v = torch.ones_like(b)

    for _ in range(int(iters)):
        Kv = torch.clamp(K @ v, min=1e-8)
        KTu = torch.clamp(K.t() @ u, min=1e-8)
        u = a / Kv
        v = b / KTu

    Tm = u[:, None] * K * v[None, :]
    return torch.sum(Tm * C)


def fit_tarnet_family(X, w, y, config, seed, method_name):
    configure_torch_runtime()
    set_global_seed(seed)

    X = np.asarray(X, dtype=np.float32)
    w = np.asarray(w, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)

    if X.shape[0] < 4 or len(np.unique(w.astype(int))) < 2:
        return ZeroCateModel()

    tr_idx, val_idx = split_indices_for_early_stop(X.shape[0], seed)

    x_tr = torch.as_tensor(X[tr_idx], dtype=torch.float32, device=DEVICE)
    w_tr = torch.as_tensor(w[tr_idx], dtype=torch.float32, device=DEVICE)
    y_tr = torch.as_tensor(y[tr_idx], dtype=torch.float32, device=DEVICE)

    x_val = torch.as_tensor(X[val_idx], dtype=torch.float32, device=DEVICE)
    w_val = torch.as_tensor(w[val_idx], dtype=torch.float32, device=DEVICE)
    y_val = torch.as_tensor(y[val_idx], dtype=torch.float32, device=DEVICE)

    net = TARNetModule(
        input_dim=X.shape[1],
        rep_dim=int(config.get("rep_dim", 32)),
        hidden_dim=int(config.get("hidden_dim", 64)),
        dropout=float(config.get("dropout", 0.0))
    ).to(DEVICE)

    optimizer = optim.Adam(
        net.parameters(),
        lr=float(config.get("lr", 1e-3)),
        weight_decay=float(config.get("weight_decay", 1e-4))
    )

    best_state = copy.deepcopy(net.state_dict())
    best_val = float("inf")
    bad_rounds = 0

    for _ in range(int(config.get("epochs", 80))):
        net.train()
        optimizer.zero_grad()

        z_tr, y0_tr, y1_tr = net(x_tr)
        loss = factual_loss(y0_tr, y1_tr, w_tr, y_tr)

        z_t = z_tr[w_tr > 0.5]
        z_c = z_tr[w_tr <= 0.5]

        if method_name == "BNN":
            loss = loss + float(config.get("balance_alpha", 0.0)) * mmd_rbf(
                z_t, z_c,
                max_points=int(config.get("mmd_max_points", 128))
            )
        elif method_name == "CFR":
            loss = loss + float(config.get("balance_alpha", 0.0)) * sinkhorn_distance(
                z_t, z_c,
                reg=float(config.get("sinkhorn_reg", 0.2)),
                iters=int(config.get("sinkhorn_iters", 8)),
                max_points=int(config.get("sinkhorn_max_points", 64))
            )

        loss.backward()
        optimizer.step()

        net.eval()
        with torch.no_grad():
            _, y0_val, y1_val = net(x_val)
            val_loss = factual_loss(y0_val, y1_val, w_val, y_val).item()

        if val_loss < best_val - 1e-8:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= int(config.get("patience", 10)):
                break

    net.load_state_dict(best_state)
    net.eval()
    return NeuralCateModel(net)


# ============================================================
# 8) 统一拟合接口
# ============================================================

def fit_cate_model(method, X, w, y, config, seed):
    if method == "OLS":
        return fit_ols_cate(X, w, y, config, seed)
    if method == "KNN":
        return fit_knn_cate(X, w, y, config, seed)
    if method == "PSM":
        return fit_psm_cate(X, w, y, config, seed)
    if method == "RF":
        return fit_rf_cate(X, w, y, config, seed)
    if method == "CF":
        return fit_cf_cate(X, w, y, config, seed)
    if method in {"TARNet", "BNN", "CFR"}:
        return fit_tarnet_family(X, w, y, config, seed, method)
    raise ValueError(f"未知方法: {method}")


def build_method_grid(method):
    configs = []
    method_windows = METHOD_WINDOW_GRID[method]

    if method == "OLS":
        for window in method_windows:
            for alpha in [1e-2, 1e-1, 1.0]:
                configs.append({
                    "window": int(window),
                    "alpha": float(alpha),
                    "repr_name": INPUT_REPR
                })
        return configs

    if method == "KNN":
        for window in method_windows:
            for k in [5, 10, 20]:
                configs.append({
                    "window": int(window),
                    "n_neighbors": int(k),
                    "weights": "distance",
                    "repr_name": INPUT_REPR
                })
        return configs

    if method == "PSM":
        for window in method_windows:
            for match_k in [1, 3, 5]:
                configs.append({
                    "window": int(window),
                    "ps_C": 1.0,
                    "match_k": int(match_k),
                    "repr_name": INPUT_REPR
                })
        return configs

    if method == "RF":
        for window in method_windows:
            for max_depth in [10, None]:
                configs.append({
                    "window": int(window),
                    "max_depth": max_depth,
                    "min_samples_leaf": 5,
                    "n_estimators": 200,
                    "repr_name": INPUT_REPR
                })
        return configs

    if method == "CF":
        for window in method_windows:
            for max_depth in [10, None]:
                configs.append({
                    "window": int(window),
                    "max_depth": max_depth,
                    "min_samples_leaf": 5,
                    "n_estimators": 200,
                    "ps_C": 1.0,
                    "repr_name": INPUT_REPR
                })
        return configs

    if method == "BNN":
        for window in method_windows:
            for alpha in [0.1, 0.3]:
                configs.append({
                    "window": int(window),
                    "lr": 1e-3,
                    "weight_decay": 1e-4,
                    "balance_alpha": float(alpha),
                    "epochs": 40,
                    "patience": 5,
                    "hidden_dim": 32,
                    "rep_dim": 16,
                    "dropout": 0.0,
                    "mmd_max_points": 64,
                    "repr_name": NEURAL_INPUT_REPR
                })
        return configs

    raise ValueError(f"未知方法: {method}")


# ============================================================
# 9) 离线滑窗运行（online-comparable prequential protocol）
#    每一步只对当前到达样本记一次误差，不再对固定 test set 反复记 MSE
# ============================================================

def run_offline_window_core(env_target, method, config, seed):
    repr_name = config["repr_name"]
    X_all = choose_input(env_target, repr_name)
    w_all = env_target["w"]
    y_all = env_target["y"]
    tau_all = env_target["tau"]

    tracker = MetricTracker()
    current_model = ZeroCateModel()
    block_buffer = []
    window = int(config["window"])

    for idx in range(len(tau_all)):
        x_now = X_all[idx:idx + 1]
        pred_tau = float(current_model.predict_tau(x_now)[0])
        tracker.add_scalar(pred_tau, float(tau_all[idx]))

        # 当前样本在预测完成后才进入历史缓冲区
        block_buffer.append(int(idx))

        # 保持原脚本的“每累计 window 个新样本后重训一次”节奏
        if len(block_buffer) >= window:
            fit_idx = np.asarray(block_buffer[-window:], dtype=int)
            current_model = fit_cate_model(
                method,
                X_all[fit_idx],
                w_all[fit_idx],
                y_all[fit_idx],
                config,
                seed
            )
            block_buffer = []

    return tracker.to_dict()


def offline_worker(task):
    target_path, env_name, env_repeat_name, surface, drift, K, repeat, seed, method, config = task

    if method in NEURAL_METHODS:
        configure_torch_runtime()
        set_global_seed(seed)

    bundle = get_bundle_cached(target_path, seed)
    out = run_offline_window_core(
        bundle["env_target"],
        method,
        config,
        seed
    )

    row = {
        "env_name": env_name,
        "env_repeat_name": env_repeat_name,
        "surface": surface,
        "drift": drift,
        "K": int(K),
        "repeat": int(repeat),
        "seed": int(seed),
        "method": method,
        "cumhalfPEHE": float(out["cumhalfPEHE"]),
        "config_json": json.dumps(config, ensure_ascii=False, sort_keys=True),
        "window": int(config["window"]),
    }
    for k, v in config.items():
        if k not in row:
            row[k] = v
    return row


def choose_best(rows):
    best = None
    for r in rows:
        if best is None or r["cumhalfPEHE"] < best["cumhalfPEHE"] - 1e-12:
            best = r
    return best


# ============================================================
# 10) 并行执行
# ============================================================

def _worker_init():
    configure_torch_runtime()


def _progress_step(total):
    if total <= 20:
        return 1
    return max(1, total // 10)


def run_tasks_in_thread_pool(tasks, worker, max_workers, label):
    """
    Windows 下 classical 任务用线程池，避免大量 spawn 进程反复 import torch。
    """
    if len(tasks) == 0:
        return []

    results = []
    total = len(tasks)
    done = 0
    step = _progress_step(total)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(worker, t) for t in tasks]
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % step == 0 or done == total:
                print(f"[PROGRESS] {label}: {done}/{total} finished", flush=True)

    return results


def run_tasks_in_process_pool(tasks, worker, max_workers, label):
    if len(tasks) == 0:
        return []

    if max_workers <= 1:
        results = []
        total = len(tasks)
        step = _progress_step(total)
        for i, t in enumerate(tasks, start=1):
            results.append(worker(t))
            if i % step == 0 or i == total:
                print(f"[PROGRESS] {label}: {i}/{total} finished", flush=True)
        return results

    ctx = mp.get_context("spawn")
    results = []
    total = len(tasks)
    done = 0
    step = _progress_step(total)

    with ctx.Pool(
        processes=max_workers,
        initializer=_worker_init,
        maxtasksperchild=32
    ) as pool:
        for out in pool.imap_unordered(worker, tasks, chunksize=1):
            results.append(out)
            done += 1
            if done % step == 0 or done == total:
                print(f"[PROGRESS] {label}: {done}/{total} finished", flush=True)

    return results


def run_tasks_safely(tasks, worker):
    classical_tasks = [t for t in tasks if t[8] in CLASSICAL_METHODS]
    neural_tasks = [t for t in tasks if t[8] in NEURAL_METHODS]

    results = []

    if classical_tasks:
        print(
            f"[INFO] classical tasks = {len(classical_tasks)}, "
            f"workers = {min(N_JOBS_CLASSICAL, len(classical_tasks))}",
            flush=True
        )

        if os.name == "nt":
            results.extend(
                run_tasks_in_thread_pool(
                    classical_tasks,
                    worker,
                    min(N_JOBS_CLASSICAL, len(classical_tasks)),
                    label="classical"
                )
            )
        else:
            results.extend(
                run_tasks_in_process_pool(
                    classical_tasks,
                    worker,
                    min(N_JOBS_CLASSICAL, len(classical_tasks)),
                    label="classical"
                )
            )

    if neural_tasks:
        print(
            f"[INFO] neural tasks = {len(neural_tasks)}, "
            f"workers = {min(N_JOBS_NEURAL, len(neural_tasks))}",
            flush=True
        )
        results.extend(
            run_tasks_in_process_pool(
                neural_tasks,
                worker,
                min(N_JOBS_NEURAL, len(neural_tasks)),
                label="neural"
            )
        )

    return results


# ============================================================
# 11) 环境级运行（每个 repeat 文件单独跑）
# ============================================================

def run_one_environment_repeat(env_info, seed):
    target_filename = env_info["filename"]
    env_name = env_info["env_name"]
    env_repeat_name = env_info["env_repeat_name"]
    surface = env_info["surface"]
    drift = env_info["drift"]
    K = int(env_info["K"])
    repeat = int(env_info["repeat"])

    get_bundle_cached(env_info["full_path"], seed)

    tasks = []
    for method in METHODS_TO_RUN:
        for config in build_method_grid(method):
            tasks.append((
                env_info["full_path"],
                env_name,
                env_repeat_name,
                surface,
                drift,
                K,
                repeat,
                seed,
                method,
                config,
            ))

    rows = run_tasks_safely(tasks, offline_worker)

    best_by_method = {}
    for method in METHODS_TO_RUN:
        method_rows = [r for r in rows if r["method"] == method]
        best_by_method[method] = choose_best(method_rows)

    return {
        "env_name": env_name,
        "env_repeat_name": env_repeat_name,
        "surface": surface,
        "drift": drift,
        "K": K,
        "repeat": repeat,
        "seed": int(seed),
        "best_by_method": best_by_method,
    }


# ============================================================
# 12) 聚合与输出
# ============================================================

def print_environment_report(res):
    print("\n" + "=" * 150, flush=True)
    print(
        f"[ENV-REPEAT] {res['env_repeat_name']} | base_env={res['env_name']} | "
        f"repeat={res['repeat']:02d} | seed={res['seed']}",
        flush=True
    )
    print("[BEST-BY-METHOD]", flush=True)

    for method in METHODS_TO_RUN:
        row = res["best_by_method"][method]
        print(
            f"{method:<8s} | "
            f"cumhalfPEHE={row['cumhalfPEHE']:.6f} | "
            f"window={row['window']} | "
            f"config={row['config_json']}",
            flush=True
        )


def collect_best_rows_raw(all_env_reports):
    rows = []
    for res in all_env_reports:
        for method in METHODS_TO_RUN:
            rows.append(copy.deepcopy(res["best_by_method"][method]))

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    front_cols = [
        "env_name", "env_repeat_name", "surface", "drift", "K", "repeat", "seed",
        "method", "cumhalfPEHE", "window", "config_json"
    ]
    other_cols = [c for c in df.columns if c not in front_cols]
    df = df[front_cols + sorted(other_cols)]
    df = df.sort_values(["env_name", "repeat", "method"]).reset_index(drop=True)
    return df


def aggregate_metric_df(df, group_cols, metric_cols, repeat_col_name="n_repeats"):
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


def build_repeat_summary_tables(raw_best_df):
    if raw_best_df.empty:
        empty = pd.DataFrame()
        return empty, empty, empty

    group_cols = ["env_name", "surface", "drift", "K", "method"]
    summary_df = aggregate_metric_df(
        raw_best_df,
        group_cols=group_cols,
        metric_cols=["cumhalfPEHE"],
        repeat_col_name="n_repeats",
    ).sort_values(["env_name", "method"]).reset_index(drop=True)

    method_rank = {m: i for i, m in enumerate(METHODS_TO_RUN)}
    summary_df["_rank"] = summary_df["method"].map(lambda x: method_rank.get(str(x), 999))
    summary_df = summary_df.sort_values(["env_name", "cumhalfPEHE_mean", "_rank", "method"]).reset_index(drop=True)
    summary_df["env_rank"] = summary_df.groupby("env_name").cumcount() + 1
    best_env_df = summary_df.groupby("env_name", as_index=False, group_keys=False).head(1).reset_index(drop=True)

    raw_with_stats_df = pd.merge(
        raw_best_df,
        summary_df[group_cols + ["n_repeats", "cumhalfPEHE_mean", "cumhalfPEHE_std", "env_rank"]],
        on=group_cols,
        how="left",
    )
    raw_with_stats_df = raw_with_stats_df.sort_values(["env_name", "repeat", "method"]).reset_index(drop=True)

    front_cols = [
        "env_name", "env_repeat_name", "surface", "drift", "K", "repeat", "seed",
        "method", "cumhalfPEHE", "cumhalfPEHE_mean", "cumhalfPEHE_std", "n_repeats",
        "env_rank", "window", "config_json"
    ]
    other_cols = [c for c in raw_with_stats_df.columns if c not in front_cols]
    raw_with_stats_df = raw_with_stats_df[front_cols + sorted(other_cols)]

    summary_df = summary_df.drop(columns=["_rank"])
    best_env_df = best_env_df.drop(columns=["_rank"])
    return raw_with_stats_df, summary_df, best_env_df


def main():
    ensure_dir(OUTPUT_DIR)

    print(f"[INFO] TARGET_DATA_DIR   = {TARGET_DATA_DIR}", flush=True)
    print(f"[INFO] OUTPUT_DIR        = {OUTPUT_DIR}", flush=True)
    print(f"[INFO] REQUESTED_N_JOBS = {REQUESTED_N_JOBS}", flush=True)
    print(f"[INFO] N_JOBS_CLASSICAL = {N_JOBS_CLASSICAL}", flush=True)
    print(f"[INFO] N_JOBS_NEURAL    = {N_JOBS_NEURAL}", flush=True)
    print(f"[INFO] WINDOW_GRID      = {WINDOW_GRID}", flush=True)
    print(f"[INFO] METHODS_TO_RUN   = {METHODS_TO_RUN}", flush=True)
    print(f"[INFO] INPUT_REPR       = {INPUT_REPR}", flush=True)
    print(f"[INFO] NEURAL_REPR      = {NEURAL_INPUT_REPR}", flush=True)
    print(f"[INFO] EXPECT_REPEATS   = {N_REPEATS}", flush=True)
    print("[INFO] EVAL_PROTOCOL    = prequential scalar cumhalfPEHE (online-comparable)", flush=True)

    if os.name == "nt":
        print("[INFO] Windows detected.", flush=True)
        print("[INFO] Classical tasks use thread pool; neural tasks use process pool.", flush=True)

    env_repeat_files = discover_environment_files(TARGET_DATA_DIR)
    repeat_inventory = validate_repeat_inventory(env_repeat_files, N_REPEATS)
    env_files = unique_base_envs(env_repeat_files)

    print(f"[INFO] BASE_ENV_COUNT    = {len(env_files)}", flush=True)
    print(f"[INFO] TASK_ENV_REPEATS  = {len(env_repeat_files)}", flush=True)
    print(f"[INFO] REPEAT_INVENTORY  = { {k: v for k, v in sorted(repeat_inventory.items())} }", flush=True)

    all_env_reports = []
    for env_info in env_repeat_files:
        seed = BASE_SEED + int(env_info["repeat"]) - 1
        res = run_one_environment_repeat(env_info, seed)
        print_environment_report(res)
        all_env_reports.append(res)

    raw_best_df = collect_best_rows_raw(all_env_reports)
    raw_with_stats_df, repeat_mean_df, env_best_df = build_repeat_summary_tables(raw_best_df)

    raw_csv = os.path.join(OUTPUT_DIR, "acic16_offline_baseline_best_by_method_raw_repeat.csv")
    raw_with_stats_csv = os.path.join(OUTPUT_DIR, "acic16_offline_baseline_best_by_method_raw_repeat_with_std.csv")
    mean_csv = os.path.join(OUTPUT_DIR, "acic16_offline_baseline_best_by_method_repeat_mean.csv")
    best_csv = os.path.join(OUTPUT_DIR, "acic16_offline_baseline_env_best_repeat_mean.csv")

    raw_best_df.to_csv(raw_csv, index=False, encoding="utf-8-sig")
    raw_with_stats_df.to_csv(raw_with_stats_csv, index=False, encoding="utf-8-sig")
    repeat_mean_df.to_csv(mean_csv, index=False, encoding="utf-8-sig")
    env_best_df.to_csv(best_csv, index=False, encoding="utf-8-sig")

    print("\n" + "#" * 150, flush=True)
    print(f"[DONE] 已输出原始 repeat 结果: {raw_csv}", flush=True)
    print(f"[DONE] 已输出 raw + mean/std 汇总: {raw_with_stats_csv}", flush=True)
    print(f"[DONE] 已输出 repeat 均值/标准差汇总: {mean_csv}", flush=True)
    print(f"[DONE] 已输出环境最优方法表: {best_csv}", flush=True)
    print(f"[DONE] raw 行数 = {len(raw_best_df)}", flush=True)
    print(f"[DONE] raw_with_stats 行数 = {len(raw_with_stats_df)}", flush=True)
    print(f"[DONE] repeat_mean 行数 = {len(repeat_mean_df)}", flush=True)
    print(f"[DONE] env_best 行数 = {len(env_best_df)}", flush=True)
    print("#" * 150, flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()