import argparse
import time

import pandas as pd
from scipy.stats import randint, loguniform
from sklearn.base import clone
from sklearn.model_selection import RandomizedSearchCV, cross_val_score

import config
import pipeline as pl

PARAM_DISTS = {
    "Ridge": {"regressor__alpha": loguniform(1e-3, 1e3)},
    "KNN": {"regressor__n_neighbors": randint(2, 30), "regressor__weights": ["uniform", "distance"]},
    "RandomForest": {
        "regressor__n_estimators": randint(100, 300),
        "regressor__max_depth": randint(5, 25),
        "regressor__min_samples_leaf": randint(1, 10),
        "regressor__max_features": [0.5, 0.8, 1.0],
    },
    "HistGradientBoosting": {
        "regressor__max_iter": randint(100, 400),
        "regressor__learning_rate": loguniform(0.02, 0.3),
        "regressor__max_depth": [None, 3, 5, 8],
        "regressor__min_samples_leaf": randint(5, 50),
        "regressor__l2_regularization": loguniform(1e-3, 10),
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=config.EXPERIMENT)
    ap.add_argument("--model", default=None, help="podrazumevano: model iz best_model.json")
    ap.add_argument("--n-iter", type=int, default=20)
    args = ap.parse_args()
    exp, t0 = args.experiment, time.time()

    model_name = args.model or pl.load_best_info(exp)["model"]
    if model_name not in PARAM_DISTS:
        print(f"Model '{model_name}' nema hiperparametara za podešavanje - ništa da se radi.")
        return

    _, _, metrics_dir = pl.setup_dirs(exp)
    df = pl.load_data(verbose=False)
    X, y, groups = pl.get_xyg(df, exp)
    train_idx, test_idx = pl.make_split(X, y, groups)   
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]

    pl.header(f"RandomizedSearchCV: {model_name} (eksperiment: {exp}, n_iter={args.n_iter})")
    default_pipe = pl.make_pipe(pl.get_models()[model_name])
    cv = pl.make_cv(g_tr)

    default_cv = cross_val_score(default_pipe, X_tr, y_tr, cv=cv, groups=g_tr, scoring="r2", n_jobs=config.N_JOBS).mean()

    search = RandomizedSearchCV(default_pipe, PARAM_DISTS[model_name], n_iter=args.n_iter, cv=cv,
                                scoring="r2", random_state=config.RANDOM_STATE, n_jobs=config.N_JOBS,
                                refit=True, verbose=1)
    search.fit(X_tr, y_tr, groups=g_tr)
    print(f"\nCV R2: podrazumevani={default_cv:.4f} | tjunirani={search.best_score_:.4f}")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    # Poređenje na test skupu 
    m_def, _, _ = pl.evaluate_on_test(default_pipe, X_tr, y_tr, X_te, y_te)
    m_tun, preds_tun, _ = pl.evaluate_on_test(search.best_estimator_, X_tr, y_tr, X_te, y_te)
    cmp_df = pd.DataFrame([
        {"model": "Podrazumevani", "cv_r2": default_cv, "test_r2": m_def["r2"], "rmse": m_def["rmse"], "mae": m_def["mae"]},
        {"model": "Tjunirani", "cv_r2": search.best_score_, "test_r2": m_tun["r2"], "rmse": m_tun["rmse"], "mae": m_tun["mae"]},
    ])
    pl.header("POREĐENJE: PODRAZUMEVANI vs TJUNIRANI")
    print(cmp_df.round(4).to_string(index=False))
    cmp_df.to_csv(metrics_dir / "tuning_before_after.csv", index=False)
    (pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
       .to_csv(metrics_dir / "hyperparameter_search_results.csv", index=False))

    use_tuned = search.best_score_ > default_cv + 0.002
    print("\nZAKLJUČAK (po CV):")
    if use_tuned:
        print(f"  Tuning poboljšava CV R2 za {search.best_score_ - default_cv:.4f} -> koristi se tjunirani model.")
    else:
        print("  Tuning ne daje značajno poboljšanje po CV ")
    print(f"  Test R2: {m_def['r2']:.4f} -> {m_tun['r2']:.4f}")

    final = clone(search.best_estimator_ if use_tuned else default_pipe).fit(X, y)
    assert final.named_steps["scaler"].n_features_in_ == X.shape[1]
    path = pl.save_bundle(final, X, exp, model_name, tuned=use_tuned,
                          params=search.best_params_ if use_tuned else None)
    print(f"\nFinalni model sačuvan : {path}")
    print(f"Ukupno vreme: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()