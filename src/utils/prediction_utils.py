import numpy as np

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
