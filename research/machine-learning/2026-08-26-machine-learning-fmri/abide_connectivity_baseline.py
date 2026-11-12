"""Educational ABIDE functional-connectivity baseline.

This script downloads preprocessed regional time series to NILEARN_DATA outside
the repository, constructs one correlation feature vector per participant, and
evaluates logistic regression while leaving one acquisition site out at a time.
It is intentionally a baseline, not a clinical diagnostic system.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from nilearn.connectome import ConnectivityMeasure
from nilearn.datasets import fetch_abide_pcp
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


os.environ.setdefault("NILEARN_DATA", str(Path.home() / ".cache" / "nilearn"))


def main() -> None:
    # rois_cc200 is a compact PCP derivative: one time series per CC200 parcel.
    abide = fetch_abide_pcp(
        n_subjects=200,
        pipeline="cpac",
        derivatives=["rois_cc200"],
        quality_checked=True,
    )
    phenotype = pd.DataFrame(abide.phenotypic)
    timeseries = [np.asarray(values, dtype=float) for values in abide.rois_cc200]
    usable = np.array([values.ndim == 2 and values.shape[0] > 20 for values in timeseries])
    phenotype = phenotype.loc[usable].reset_index(drop=True)
    timeseries = [values for values, keep in zip(timeseries, usable) if keep]

    connectivity = ConnectivityMeasure(
        kind="correlation", vectorize=True, discard_diagonal=True
    )
    features = connectivity.fit_transform(timeseries)
    labels = (phenotype["DX_GROUP"].astype(int).to_numpy() == 1).astype(int)
    sites = phenotype["SITE_ID"].astype(str).to_numpy()

    predictions = np.full(labels.size, np.nan)
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(C=0.1, max_iter=2000, class_weight="balanced"),
    )

    for site in np.unique(sites):
        test = sites == site
        train = ~test
        # A fold with one class cannot support this binary training/evaluation.
        if np.unique(labels[train]).size < 2 or np.unique(labels[test]).size < 2:
            continue
        model.fit(features[train], labels[train])
        predictions[test] = model.predict_proba(features[test])[:, 1]

    evaluated = np.isfinite(predictions)
    hard_labels = (predictions[evaluated] >= 0.5).astype(int)
    print(f"Participants evaluated: {evaluated.sum()} / {labels.size}")
    print(f"Sites represented in evaluated folds: {np.unique(sites[evaluated]).size}")
    print(f"Leave-one-site-out balanced accuracy: {balanced_accuracy_score(labels[evaluated], hard_labels):.3f}")
    print(f"Leave-one-site-out ROC AUC: {roc_auc_score(labels[evaluated], predictions[evaluated]):.3f}")
    print("Treat these as exploratory outputs; inspect QC, confounding, and uncertainty before interpretation.")


if __name__ == "__main__":
    main()
