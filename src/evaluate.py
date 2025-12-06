from __future__ import annotations
import os, json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (roc_curve, auc, precision_recall_curve)
from sklearn.inspection import permutation_importance

from .data_utils import load_uci_heart, make_targets
from .model_zoo import build_model_zoo, get_feature_schema
from .metrics_utils import compute_metrics, mean_std_ci, paired_t_test

def evaluate_models(X, y, task: str = "binary", cv_folds:int = 10, seed:int = 42,
                    models_to_run: List[str] = None,
                    out_dir:str = "outputs") -> Dict[str, Any]:

    rng = np.random.RandomState(seed)
    models = build_model_zoo(task=task, random_state=seed)
    if models_to_run:
        models = {k:v for k,v in models.items() if k in models_to_run}

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_rows = []
    oof_store = {name: {"proba": [], "y": []} for name in models.keys()}

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y[tr], y[te]
        classes = np.unique(y_tr)

        for name, model in models.items():
            # Fit
            model.fit(X_tr, y_tr)
            # Predict
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba(X_te)
            else:
                # For pipelines returning final estimators w/o proba (shouldn't happen due to calibration)
                # use decision_function if available + sigmoid mapping
                if hasattr(model, "decision_function"):
                    dec = model.decision_function(X_te)
                    # map to probabilities via softmax/sigmoid
                    if dec.ndim == 1:
                        p1 = 1.0 / (1.0 + np.exp(-dec))
                        proba = np.column_stack([1-p1, p1])
                    else:
                        # softmax
                        e = np.exp(dec - dec.max(axis=1, keepdims=True))
                        proba = e / e.sum(axis=1, keepdims=True)
                else:
                    raise ValueError(f"Model {name} does not support predict_proba.")

            pred = np.argmax(proba, axis=1)
            # Align classes for logloss if needed
            metrics = compute_metrics(y_te, proba, pred, task, classes=np.unique(y))

            row = dict(fold=fold, model=name, **metrics)
            fold_rows.append(row)

            # store for oof curves
            oof_store[name]["proba"].append(proba)
            oof_store[name]["y"].append(y_te)

            # Optionally: export TAN structure for one fold
            if name == "tan_bayes":
                # try to access structure and save
                try:
                    est = model
                    st = getattr(est, "structure_", None)
                    if st is not None:
                        feat = getattr(est, "feature_names_", [])
                        parent = st.parent
                        root = st.root
                        order = st.order
                        d = {"root": int(root),
                             "parent": [None if p is None else int(p) for p in parent],
                             "order": list(map(int, order)),
                             "feature_names": feat}
                        os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
                        with open(os.path.join(out_dir, "reports", f"tan_structure_fold{fold}.json"), "w") as f:
                            json.dump(d, f, indent=2)
                except Exception:
                    pass

    fold_df = pd.DataFrame(fold_rows)
    os.makedirs(os.path.join(out_dir, "reports"), exist_ok=True)
    fold_df.to_csv(os.path.join(out_dir, "reports", "fold_metrics.csv"), index=False)

    # Summary table
    summary_rows = []
    all_models = sorted(models.keys())
    primary_metric = "roc_auc" if task=="binary" else "roc_auc_macro"
    for name in all_models:
        sub = fold_df[fold_df.model==name]
        row = {"model": name}
        for col in sub.columns:
            if col in ["fold","model"]:
                continue
            m, s, (lo, hi) = mean_std_ci(sub[col].tolist())
            row[f"{col}_mean"] = m
            row[f"{col}_std"] = s
            row[f"{col}_ci95_lo"] = lo
            row[f"{col}_ci95_hi"] = hi
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.sort_values(by=f"{primary_metric}_mean", ascending=False, inplace=True)
    summary_df.to_csv(os.path.join(out_dir, "reports", "summary_metrics.csv"), index=False)

    # Leaderboard (top by AUROC)
    leader = summary_df[["model", f"{primary_metric}_mean", f"{primary_metric}_ci95_lo", f"{primary_metric}_ci95_hi"]].copy()
    leader.rename(columns={f"{primary_metric}_mean":"auroc_mean",
                           f"{primary_metric}_ci95_lo":"auroc_ci95_lo",
                           f"{primary_metric}_ci95_hi":"auroc_ci95_hi"}, inplace=True)
    leader.to_csv(os.path.join(out_dir, "reports", "leaderboard.csv"), index=False)

    # Paired t-tests vs logistic baseline (binary or macro AUC)
    if "logreg_L2" in all_models:
        base_scores = fold_df[fold_df.model=="logreg_L2"][primary_metric].tolist()
        tests = []
        for name in all_models:
            if name == "logreg_L2": continue
            s2 = fold_df[fold_df.model==name][primary_metric].tolist()
            if len(s2) == len(base_scores):
                t, p = paired_t_test(base_scores, s2)
                tests.append({"model": name, "t_stat": t, "p_value": p})
        pd.DataFrame(tests).to_csv(os.path.join(out_dir, "reports", "paired_t_tests_vs_logreg.csv"), index=False)

    # Build OOF curves for top 5 (always include tan_bayes if it exists)
    top = leader.head(5)["model"].tolist() if len(leader) >= 5 else leader["model"].tolist()
    # Always include tan_bayes (Bayesian network) in graphs if it exists
    if "tan_bayes" in all_models and "tan_bayes" not in top:
        top.append("tan_bayes")
    for name in top:
        y_all = np.concatenate(oof_store[name]["y"], axis=0)
        p_all = np.concatenate(oof_store[name]["proba"], axis=0)

        if task == "binary":
            from sklearn.metrics import roc_curve, auc, precision_recall_curve
            fpr, tpr, _ = roc_curve(y_all, p_all[:,1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
            plt.plot([0,1],[0,1], ls='--')
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"ROC (OOF): {name}")
            plt.legend(loc="lower right")
            os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
            plt.savefig(os.path.join(out_dir, "figures", f"roc_oof_{name}.png"), dpi=160, bbox_inches='tight')
            plt.close()

            prec, rec, _ = precision_recall_curve(y_all, p_all[:,1])
            ap = np.trapz(prec[::-1], rec[::-1])
            plt.figure()
            plt.plot(rec, prec, lw=2, label=f"{name} (AP~{ap:.3f})")
            plt.xlabel("Recall")
            plt.ylabel("Precision")
            plt.title(f"PR (OOF): {name}")
            plt.legend(loc="lower left")
            plt.savefig(os.path.join(out_dir, "figures", f"pr_oof_{name}.png"), dpi=160, bbox_inches='tight')
            plt.close()
        else:
            # For multiclass: plot ROC per-class for top
            unique = np.unique(y_all)
            for i, c in enumerate(unique):
                y_bin = (y_all == c).astype(int)
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_bin, p_all[:, i])
                roc_auc = auc(fpr, tpr)
                plt.plot(fpr, tpr, lw=1.2, label=f"class {c} (AUC={roc_auc:.3f})")
            plt.plot([0,1],[0,1], ls='--', lw=1)
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title(f"Multiclass ROC (OOF): {name}")
            plt.legend(fontsize=8, loc="lower right")
            os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
            plt.savefig(os.path.join(out_dir, "figures", f"roc_oof_multiclass_{name}.png"), dpi=160, bbox_inches='tight')
            plt.close()

    return {
        "fold_metrics_path": os.path.join(out_dir, "reports", "fold_metrics.csv"),
        "summary_path": os.path.join(out_dir, "reports", "summary_metrics.csv"),
        "leaderboard_path": os.path.join(out_dir, "reports", "leaderboard.csv")
    }
