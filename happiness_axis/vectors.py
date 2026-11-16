"""Denoised difference-in-means, projection AUCs and z-scores (the paper's recipe, section 3.2)."""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold

DENOISE_VARIANCE = 0.5   # project out the PCs that explain 50% of the control variance
N_FOLDS = 5
SEED = 42


def denoise(vec: np.ndarray, control_acts: np.ndarray, variance: float = DENOISE_VARIANCE) -> np.ndarray:
    """Remove the top principal components of the (centred) control cloud from `vec`."""
    if len(control_acts) < 2:
        return vec
    pca = PCA()
    pca.fit(control_acts - control_acts.mean(0))
    cum = np.cumsum(pca.explained_variance_ratio_)
    k = min(int(np.searchsorted(cum, variance)) + 1, len(pca.components_))
    out = vec.copy()
    for d in pca.components_[:k]:
        out = out - np.dot(out, d) * d
    return out


def difference_in_means(acts: np.ndarray, cats: np.ndarray, pos: list[str], ctrl: list[str],
                        do_denoise: bool = True) -> np.ndarray:
    """mean(positive categories) - mean(control categories), then denoised against the controls."""
    p = acts[np.isin(cats, pos)]
    c = acts[np.isin(cats, ctrl)]
    v = p.mean(0) - c.mean(0)
    return denoise(v, c) if do_denoise else v


def projection_auc(acts: np.ndarray, cats: np.ndarray, vec: np.ndarray, pos: list[str], ctrl: list[str]) -> float:
    u = vec / (np.linalg.norm(vec) + 1e-8)
    proj = acts @ u
    m_p, m_c = np.isin(cats, pos), np.isin(cats, ctrl)
    y = np.r_[np.ones(m_p.sum()), np.zeros(m_c.sum())]
    return float(roc_auc_score(y, np.r_[proj[m_p], proj[m_c]]))


def zscore(acts: np.ndarray, vec: np.ndarray, ref_acts: np.ndarray) -> np.ndarray:
    u = vec / (np.linalg.norm(vec) + 1e-8)
    ref = ref_acts @ u
    return (acts @ u - ref.mean()) / (ref.std() + 1e-8)


def cv_layer_auc(layer_acts: np.ndarray, cats: np.ndarray, sets: np.ndarray, pos, ctrl,
                 neutral: str) -> dict[str, float]:
    """5-fold CV *split by sentence set*: fit on 4/5 of the sets, score AUC on the held-out 1/5."""
    uniq = np.array(sorted(set(sets.tolist())))
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    a_all, a_neu = [], []
    for tr, te in kf.split(uniq):
        m_tr, m_te = np.isin(sets, uniq[tr]), np.isin(sets, uniq[te])
        v_all = difference_in_means(layer_acts[m_tr], cats[m_tr], pos, ctrl)
        v_neu = difference_in_means(layer_acts[m_tr], cats[m_tr], pos, [neutral])
        a_all.append(projection_auc(layer_acts[m_te], cats[m_te], v_all, pos, ctrl))
        a_neu.append(projection_auc(layer_acts[m_te], cats[m_te], v_neu, pos, [neutral]))
    return {"auc_vs_all_controls": float(np.mean(a_all)), "auc_vs_neutral": float(np.mean(a_neu)),
            "auc_std": float(np.std(a_all))}


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))
