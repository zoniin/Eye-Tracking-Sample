"""
Train and validate the learner-state classifier (poster section 5: "The model
will be trained and validated against student task performance and
self-reports").

1. Build a dataset from recorded sessions (probe answers -> labelled windows)::

       python training/train_behavior.py build --sessions output/* --window 30 --out data/behavior.csv

2. Train with **leave-one-participant-out** cross-validation and save the model::

       python training/train_behavior.py train --dataset data/behavior.csv --out models/behavior.joblib

   Reported metrics: balanced accuracy and macro-F1 per held-out participant,
   compared against (a) always predicting the majority class and (b) the
   rule-based classifier (for labels it knows).  Only beating these baselines
   on *unseen participants* shows the model generalises.

3. Use it live::

       python -m eyetrack live --behavior-model models/behavior.joblib

``--participant-normalize`` z-scores each feature against the session's own
baseline (all sliding windows, labelled or not), which removes stable
individual differences (e.g. someone who always blinks a lot) without leaking
label information.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from eyetrack.behavior import RuleBasedClassifier  # noqa: E402
from eyetrack.dataset import build_examples, write_dataset  # noqa: E402
from eyetrack.features import BASE_FEATURES  # noqa: E402

META_COLS = {"participant", "session", "t", "label", "window_s"}


def cmd_build(args) -> None:
    rows: List[Dict] = []
    stats: Dict = {}
    dirs = [d for pat in args.sessions for d in sorted(glob.glob(pat)) if Path(d).is_dir()]
    event_cfg = None
    if args.config:
        from eyetrack.app import load_config
        event_cfg = load_config(args.config)[0].events
    for d in dirs:
        ex = build_examples(d, window_s=args.window, label_source=args.labels,
                            min_presence=args.min_presence, event_cfg=event_cfg,
                            stats_out=stats)
        print(f"{d}: {len(ex)} labelled windows")
        rows.extend(ex)
    write_dataset(rows, args.out)
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats))
    labels = {}
    for r in rows:
        labels[r["label"]] = labels.get(r["label"], 0) + 1
    print(f"Wrote {len(rows)} rows to {args.out}; labels: {labels}")


def make_pipeline(kind: str, seed: int):
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline as mk
    from sklearn.preprocessing import StandardScaler
    if kind == "logreg":
        return mk(SimpleImputer(strategy="median"), StandardScaler(),
                  LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    if kind == "rf":
        return mk(SimpleImputer(strategy="median"),
                  RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                         class_weight="balanced", random_state=seed, n_jobs=-1))
    return mk(SimpleImputer(strategy="median"),
              HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=300,
                                             min_samples_leaf=5, class_weight="balanced",
                                             random_state=seed))


def session_zscore(df, cols, stats: Dict):
    """Normalise each row by its session's baseline statistics.

    The statistics come from *all* sliding windows of the session (written by
    ``build``), not from the labelled rows, so they do not depend on labels -
    the same information is available live (running statistics)."""
    out = df.copy()
    for sess, idx in df.groupby("session").groups.items():
        st = stats.get(str(sess), {})
        for c in cols:
            mu, sd = st.get(c, [np.nan, 1.0])
            sd = sd if sd and sd > 1e-6 else 1.0
            out.loc[idx, c] = (df.loc[idx, c].astype(float) - mu) / sd
    return out


def cmd_train(args) -> None:
    import joblib
    import pandas as pd
    from sklearn.metrics import (balanced_accuracy_score, classification_report,
                                 confusion_matrix, f1_score)
    from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold

    df = pd.read_csv(args.dataset)
    for spec in args.merge or []:
        src, dst = spec.split(":")
        df.loc[df["label"] == src, "label"] = dst
    if args.drop_labels:
        df = df[~df["label"].isin(args.drop_labels)]
    counts = df["label"].value_counts()
    keep = counts[counts >= args.min_count].index
    if len(keep) < len(counts):
        print(f"Dropping rare labels (<{args.min_count}): {sorted(set(counts.index) - set(keep))}")
    df = df[df["label"].isin(keep)].reset_index(drop=True)
    feat_cols = [c for c in df.columns if c not in META_COLS]
    if args.features == "base":
        feat_cols = [c for c in feat_cols if c in BASE_FEATURES]
    window_s = float(df["window_s"].iloc[0]) if "window_s" in df.columns else args.window
    if args.participant_normalize:
        stats_path = Path(args.dataset).with_suffix(".stats.json")
        if not stats_path.is_file():
            raise SystemExit(f"{stats_path} not found - rebuild the dataset with `build`.")
        X_df = session_zscore(df, feat_cols, json.loads(stats_path.read_text()))
    else:
        X_df = df
    X = X_df[feat_cols].astype(float).to_numpy()
    y = df["label"].astype(str).to_numpy()
    groups = df["participant"].astype(str).to_numpy()
    labels = sorted(set(y))
    n_part = len(set(groups))
    print(f"{len(y)} windows, {n_part} participants, labels: {counts[keep].to_dict()}")
    if args.model == "auto":
        args.model = "logreg" if len(y) < 300 else "hgb"
    if len(labels) < 2:
        raise SystemExit("Need at least two labels to train a classifier.")

    if n_part >= 3:
        cv, cv_name = LeaveOneGroupOut(), "leave-one-participant-out"
        splits = list(cv.split(X, y, groups))
    else:
        k = int(min(5, counts[keep].min()))
        cv, cv_name = StratifiedKFold(k, shuffle=True, random_state=args.seed), f"stratified {k}-fold"
        splits = list(cv.split(X, y))
        print("WARNING: fewer than 3 participants - results are NOT person-independent.")

    pred = np.empty(len(y), dtype=object)
    per_fold = []
    for tr, te in splits:
        model = make_pipeline(args.model, args.seed).fit(X[tr], y[tr])
        pred[te] = model.predict(X[te])
        per_fold.append({"held_out": str(groups[te][0]) if n_part >= 3 else len(per_fold),
                         "n": int(len(te)),
                         "balanced_acc": float(balanced_accuracy_score(y[te], pred[te]))})

    majority = counts[keep].idxmax()
    rules = RuleBasedClassifier()
    rule_pred = np.array([rules.predict({c: (v if np.isfinite(v) else float("nan"))
                                         for c, v in zip(feat_cols, row)}).state
                          for row in df[feat_cols].astype(float).to_numpy()])
    known = np.isin(y, rules.labels)
    report = {
        "cv": cv_name,
        "model": args.model,
        "n_windows": int(len(y)),
        "n_participants": n_part,
        "labels": labels,
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro")),
        "chance_balanced_accuracy": 1.0 / len(labels),
        "majority_baseline_macro_f1": float(f1_score(y, np.full(len(y), majority), average="macro")),
        "rule_baseline_balanced_accuracy_on_shared_labels":
            float(balanced_accuracy_score(y[known], rule_pred[known])) if known.any() else None,
        "per_fold": per_fold,
        "confusion_matrix": {"labels": labels,
                             "matrix": confusion_matrix(y, pred, labels=labels).tolist()},
    }
    print(f"\n=== {cv_name} ({args.model}) ===")
    print(classification_report(y, pred, labels=labels, zero_division=0))
    print(f"balanced accuracy {report['balanced_accuracy']:.3f} "
          f"(chance {report['chance_balanced_accuracy']:.3f}); macro-F1 {report['macro_f1']:.3f} "
          f"(majority baseline {report['majority_baseline_macro_f1']:.3f})")
    if report["rule_baseline_balanced_accuracy_on_shared_labels"] is not None:
        print(f"rule-based baseline balanced accuracy (shared labels): "
              f"{report['rule_baseline_balanced_accuracy_on_shared_labels']:.3f}")

    final = make_pipeline(args.model, args.seed).fit(X, y)
    importances = {}
    try:
        from sklearn.inspection import permutation_importance
        pi = permutation_importance(final, X, y, n_repeats=5, random_state=args.seed,
                                    scoring="balanced_accuracy")
        order = np.argsort(pi.importances_mean)[::-1][:15]
        importances = {feat_cols[i]: float(pi.importances_mean[i]) for i in order}
        print("Top features (permutation importance, training data):")
        for k, v in importances.items():
            print(f"  {k:28s} {v:.3f}")
    except Exception as exc:  # pragma: no cover
        print(f"(skipped feature importance: {exc})")
    report["feature_importance"] = importances

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": final, "feature_names": feat_cols, "labels": labels,
                 "window_s": window_s, "participant_normalized": bool(args.participant_normalize),
                 "meta": report}, out)
    out.with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    print(f"\nSaved {out} and {out.with_suffix('.report.json')}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="sessions -> labelled feature table")
    b.add_argument("--sessions", nargs="+", required=True, help="session directories (globs ok)")
    b.add_argument("--window", type=float, default=30.0)
    b.add_argument("--labels", default="probes", choices=["probes", "labels", "both"])
    b.add_argument("--min-presence", type=float, default=0.0)
    b.add_argument("--config", help="YAML config with the event thresholds used live")
    b.add_argument("--out", default="data/behavior.csv")
    t = sub.add_parser("train", help="cross-validate and fit the classifier")
    t.add_argument("--dataset", required=True)
    t.add_argument("--model", default="auto", choices=["auto", "hgb", "rf", "logreg"],
                   help="auto = logistic regression below 300 windows, gradient boosting above")
    t.add_argument("--features", default="all", choices=["all", "base"],
                   help="'base' drops layout-specific dwell_<aoi> columns")
    t.add_argument("--participant-normalize", action="store_true",
                   help="z-score features against each session's own baseline")
    t.add_argument("--merge", nargs="*", help="relabel, e.g. frustrated:confused")
    t.add_argument("--drop-labels", nargs="*")
    t.add_argument("--min-count", type=int, default=5)
    t.add_argument("--window", type=float, default=30.0,
                   help="window length if the dataset has no window_s column")
    t.add_argument("--out", default="models/behavior.joblib")
    t.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    {"build": cmd_build, "train": cmd_train}[args.cmd](args)


if __name__ == "__main__":
    main()
