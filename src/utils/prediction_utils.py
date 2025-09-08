import numpy as np
import pandas as pd
from mapie.metrics import (
    classification_coverage_score,
    classification_mean_width_score,
)

# --- helpers identiques à l'entraînement ---
def pset_from_spci(intervals, threshold: float) -> np.ndarray:
    """
    intervals: array-like de shape (n_samples, 2) ou liste de (L, U)
    Retourne un p-set bool (n_samples, 2).
    """
    iv = np.asarray(intervals)
    if iv.ndim != 2 or iv.shape[1] != 2:
        raise ValueError(f"intervals doit être (n,2), reçu {iv.shape}")
    L, U = iv[:, 0], iv[:, 1]
    n = len(L)
    pset = np.zeros((n, 2), dtype=bool)
    mask1 = threshold > U
    mask0 = threshold < L
    amb = ~(mask1 | mask0)
    pset[mask1, 1] = True
    pset[mask0, 0] = True
    pset[amb, :] = True
    return pset

def pset_from_mcp(model_mcp, X_CP: np.ndarray, alpha: float, partition) -> np.ndarray:
    """
    Appelle le MCP et récupère un p-set bool (n_samples, 2).
    Hyp: model_mcp.predict(...) -> (y_pred, yps) avec yps booléisable.
    """
    _, yps = model_mcp.predict(X_CP, alpha=alpha, partition=partition)
    yps = yps[:, :, 0] if yps.ndim == 3 else yps
    return yps.astype(bool)

def _meta_features(X_CP: np.ndarray, pset_mcp: np.ndarray, pset_spci: np.ndarray) -> np.ndarray:
    """
    Construit les meta-features attendues par le gate: [X_CP | w_cls | w_spc | diff]
    """
    w_cls = pset_mcp.sum(axis=1).astype(int).reshape(-1, 1)
    w_spc = pset_spci.sum(axis=1).astype(int).reshape(-1, 1)
    diff = (w_cls - w_spc)
    return np.hstack([X_CP, w_cls, w_spc, diff])

# --- prédicteur utilisant des modèles déjà entraînés ---
def predict_with_gate(
    model_mcp,          # ex. models_c_ng[('RF', n, 'vanilla')]
    model_spci,         # ex. models_lg[n]
    gate_clf,           # ex. models_gate[('RF', n)] ou similaire
    *,
    X_CP: np.ndarray,   # features côté MCP pour le même n (ceux utilisés au train du gate)
    X_SPCI: np.ndarray, # features côté SPCI pour le même n
    partition,          # partition/cluster utilisée par MCP
    threshold: float,
    alpha: float = 0.05,
    return_details: bool = False,
):
    """
    Retourne final_pset (n,2) bool. Si return_details, renvoie aussi un dict.
    Décodage gate: 0=MCP, 1=SPCI, 2=UNION
    """
    # 1) p-sets des deux branches
    pset_mcp = pset_from_mcp(model_mcp, X_CP, alpha, partition)
    intervals = model_spci.predict_interval(X_SPCI)
    pset_spc = pset_from_spci(intervals, threshold)

    # 2) features meta pour le gate
    X_gate = _meta_features(X_CP, pset_mcp, pset_spc)

    # 3) décision du gate
    decisions = gate_clf.predict(X_gate).astype(int)  # 0/1/2

    # 4) composition finale
    final = np.zeros_like(pset_mcp, dtype=bool)
    mcp_mask   = decisions == 0
    spci_mask  = decisions == 1
    union_mask = decisions == 2

    final[mcp_mask]   = pset_mcp[mcp_mask]
    final[spci_mask]  = pset_spc[spci_mask]
    final[union_mask] = (pset_mcp[union_mask] | pset_spc[union_mask])

    if return_details:
        return final, {
            "pset_mcp": pset_mcp,
            "pset_spci": pset_spc,
            "intervals": np.asarray(intervals),
            "decisions": decisions,  # 0/1/2
        }
    return final

def metrics_for_pset_mapie(pset: np.ndarray, y_true: np.ndarray) -> dict:
    """
    Utilise MAPIE pour la couverture & largeur moyenne.
    Ajoute des stats utiles : taux d’ambiguïté, taux singleton, précision conditionnelle si singleton.
    """
    y_true = np.asarray(y_true).astype(int).reshape(-1)
    if pset.ndim != 2 or pset.shape[1] != 2:
        raise ValueError("pset doit être (n, 2) bool")
    if len(y_true) != pset.shape[0]:
        raise ValueError("y_true doit avoir la même longueur que pset.")

    coverage = classification_coverage_score(y_true, pset)
    mean_width = classification_mean_width_score(pset)

    width = pset.sum(axis=1)  # 1 ou 2
    amb_rate = (width == 2).mean()
    single_rate = (width == 1).mean()

    cond_acc = np.nan
    mask_single = (width == 1)
    if mask_single.any():
        point_pred = pset[mask_single].argmax(axis=1)
        cond_acc = (point_pred == y_true[mask_single]).mean()

    return {
        "n": int(pset.shape[0]),
        "coverage": float(coverage),
        "mean_width": float(mean_width),
        "ambiguous_rate": float(amb_rate),
        "singleton_rate": float(single_rate),
        "cond_acc_if_singleton": float(cond_acc) if not np.isnan(cond_acc) else np.nan,
    }

def summarize_three_mapie(pset_mcp: np.ndarray, pset_spci: np.ndarray, pset_gate: np.ndarray, y_true: np.ndarray) -> pd.DataFrame:
    """
    Compare MCP / SPCI / GATE avec métriques MAPIE (+ extras).
    """
    rows = []
    rows.append(("MCP",  metrics_for_pset_mapie(pset_mcp,  y_true)))
    rows.append(("SPCI", metrics_for_pset_mapie(pset_spci, y_true)))
    rows.append(("GATE", metrics_for_pset_mapie(pset_gate, y_true)))
    df = pd.DataFrame({name: m for name, m in rows}).T
    return df[["n", "coverage", "mean_width", "ambiguous_rate", "singleton_rate", "cond_acc_if_singleton"]]

def print_sample_predictions(y_true, pset_mcp, pset_spci, pset_gate, decisions=None, k=10):
    """
    Affiche un petit échantillon lisible (k lignes).
    """
    n = len(y_true)
    idx = np.arange(min(k, n))
    def fmt_set(ps):
        if ps[0] and ps[1]: return "{0,1}"
        if ps[0]:           return "{0}"
        if ps[1]:           return "{1}"
        return "∅"
    print("\n--- Échantillon de prédictions ---")
    header = ["i", "y", "MCP", "SPCI", "GATE"]
    if decisions is not None:
        header += ["gate_dec"]
    print("\t".join(header))
    for i in idx:
        row = [str(i), str(int(y_true[i])), fmt_set(pset_mcp[i]), fmt_set(pset_spci[i]), fmt_set(pset_gate[i])]
        if decisions is not None:
            # 0=MCP, 1=SPCI, 2=UNION
            dec = int(decisions[i])
            row.append({0:"MCP", 1:"SPCI", 2:"UNION"}.get(dec, str(dec)))
        print("\t".join(row))