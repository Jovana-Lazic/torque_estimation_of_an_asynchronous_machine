import json
import time

import matplotlib
matplotlib.use("Agg")  
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib
import sklearn

from sklearn.base import clone
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import (GroupKFold, GroupShuffleSplit, cross_val_predict,
                                     cross_val_score, cross_validate)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config
from features import add_engineered_features

LINE_SEP = "=" * 60
COLOR_LIGHT, COLOR_MID, COLOR_DARK, COLOR_ACCENT = "#D2B48C", "#8C6D58", "#5C4033", "#A89F91"

BASELINE_NAME = "Baseline (Mean)"

COMPLEXITY_ORDER = ["LinearRegression", "Ridge", "KNN", "RandomForest", "HistGradientBoosting"]

DROPPED_COLUMNS = {
    "series_id": "ključ za grupisanje (GroupKFold)",
    "window / t": "redni broj / vreme prozora ",
    "p_mech": "curenje cilja: P_mech ≈ moment · ω",
    "n_samples_hf": "tehnička veličina",
    "torque_std": "koristi se samo za filtriranje tranzijenata ",
    "faze (i_a/b/c, u_1/2/3, d_a/b/c)": "u 'merged' setovima spojene u prosek (korelacija ~1)",
}


def header(text):
    print("\n" + LINE_SEP)
    print(text)
    print(LINE_SEP)


# DIREKTORIJUMI
def setup_dirs(experiment):
    model_dir = config.model_dir(experiment)
    results = config.results_dir(experiment)
    fig_dir, metrics_dir = results / "figures", results / "metrics"
    for d in (model_dir, fig_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)
    return model_dir, fig_dir, metrics_dir

# UČITAVANJE + ČIŠĆENJE
def load_data(verbose=True):
    df = pd.read_csv(config.DATA_PATH, encoding="utf-8")
    n0 = len(df)
    if verbose:
        header("UČITAVANJE I ČIŠĆENJE PODATAKA")
        print(f"Učitano redova: {n0:,}")

    na = df.isna().sum()
    na = na[na > 0]
    if verbose:
        print("Nedostajuće vrednosti po kolonama:", "nema" if na.empty else "")
        if not na.empty:
            print(na.to_string())
    df = df.dropna()
    n_dup = int(df.duplicated().sum())
    df = df.drop_duplicates()
    if verbose:
        print(f"Uklonjeno zbog NaN: {n0 - len(df) - n_dup:,} | duplikata: {n_dup:,}")

    num_cols = [c for c in df.select_dtypes("number").columns if c not in ("series_id", "window")]
    df[num_cols] = df[num_cols].astype("float64")

    df = _filter_incomplete_windows(df, verbose)
    df = _filter_transients(df, verbose)
    df = _remove_impossible(df, verbose)
    if verbose:
        _report_outliers(df)
    df = add_engineered_features(df).reset_index(drop=True)

    if verbose:
        print(f"\nFinalno: {len(df):,} prozora, {df['series_id'].nunique()} mernih serija")
    return df


def _filter_incomplete_windows(df, verbose):
    if "n_samples_hf" not in df.columns:
        return df
    thr = config.MIN_SAMPLES_FRAC * df["n_samples_hf"].max()
    keep = df["n_samples_hf"] >= thr
    if verbose:
        print(f"Nepotpuni prozori (< {config.MIN_SAMPLES_FRAC:.0%} uzoraka) uklonjeni: {int((~keep).sum())}")
    return df[keep]


def _filter_transients(df, verbose):
    if not config.TRANSIENT_QUANTILE or "torque_std" not in df.columns:
        return df
    thr = df["torque_std"].quantile(config.TRANSIENT_QUANTILE)
    keep = df["torque_std"] <= thr
    if verbose:
        print(f"Tranzijenti (torque_std > {thr:.4g}, gornjih {1 - config.TRANSIENT_QUANTILE:.0%}) uklonjeni: {int((~keep).sum())}")
    return df[keep]


def _remove_impossible(df, verbose):
    bad = (df["u_dc"] <= 0)
    for c in ("i_a_rms", "i_b_rms", "i_c_rms", "u_1_rms", "u_2_rms", "u_3_rms"):
        bad |= df[c] < 0
    for c in ("theta_s_mean", "theta_r_mean"):
        bad |= (df[c] < -50) | (df[c] > 250)
    if verbose:
        print(f"Fizički nemoguće vrednosti uklonjene: {int(bad.sum())}")
    return df[~bad]


def _report_outliers(df):
    for col in ("torque", "i_a_rms", "i_b_rms", "i_c_rms"):
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        n_out = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        print(f"  IQR outlieri '{col}': {n_out} ({100 * n_out / len(df):.2f}%) - zadržani (validne radne tačke)")


def print_dropped_columns():
    header("ATRIBUTI KOJI NISU ULAZ MODELA")
    for k, v in DROPPED_COLUMNS.items():
        print(f"  {k:35s} {v}")


# SETOVI ATRIBUTA
def get_feature_sets():
    base = ["speed_rpm", "u_dc", "d_a_std", "d_b_std", "d_c_std",
            "theta_s_mean", "theta_s_max", "theta_r_mean", "theta_r_max",
            "i_a_rms", "i_b_rms", "i_c_rms", "u_1_rms", "u_2_rms", "u_3_rms"]
    power = ["p_in_mot", "p_in_inv"]
    merged = ["speed_rpm", "u_dc", "d_std", "theta_s_mean", "theta_r_mean", "i_rms", "u_rms"] + power
    return {
        "no_power": base,
        "with_power": base + power,
        "merged": merged,
        "merged_power_over_omega": merged + ["p_in_mot_over_omega"],
    }


def get_xyg(df, experiment):
    sets = get_feature_sets()
    if experiment not in sets:
        raise ValueError(f"Nepoznat eksperiment '{experiment}'. Dozvoljeno: {list(sets)}")
    cols = sets[experiment]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Nedostaju kolone u datasetu: {missing}")
    return df[cols], df["torque"], df["series_id"]


# MODELI + CV
def get_models():
    rs = config.RANDOM_STATE
    return {
        BASELINE_NAME: DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "RandomForest": RandomForestRegressor(n_estimators=100, max_depth=15, random_state=rs, n_jobs=1),
        "HistGradientBoosting": HistGradientBoostingRegressor(max_iter=200, learning_rate=0.1, random_state=rs),
    }


def make_pipe(model):
    return Pipeline([("scaler", StandardScaler()), ("regressor", model)])


def make_cv(groups):
    return GroupKFold(n_splits=min(config.N_SPLITS_CV, groups.nunique()))


def make_split(X, y, groups):
    splitter = GroupShuffleSplit(n_splits=1, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    return next(splitter.split(X, y, groups))


def metrics_dict(y_true, preds):
    mse = mean_squared_error(y_true, preds)
    return {"r2": r2_score(y_true, preds), "rmse": float(np.sqrt(mse)), "mae": mean_absolute_error(y_true, preds)}


def compare_models(X, y, groups, verbose=True):
    cv = make_cv(groups)
    rows = []
    for name, model in get_models().items():
        t0 = time.time()
        res = cross_validate(make_pipe(model), X, y, cv=cv, groups=groups,
                             scoring={"r2": "r2", "rmse": "neg_root_mean_squared_error"},
                             n_jobs=config.N_JOBS)
        rows.append({
            "model": name,
            "cv_r2_mean": res["test_r2"].mean(),
            "cv_r2_std": res["test_r2"].std(),
            "cv_rmse_mean": -res["test_rmse"].mean(),
            "cv_time_s": time.time() - t0,
        })
        if verbose:
            r = rows[-1]
            print(f"  {name:22s} CV R2={r['cv_r2_mean']:.4f} (+/-{r['cv_r2_std']:.4f})  "
                  f"RMSE={r['cv_rmse_mean']:.4f}  [{r['cv_time_s']:.1f}s]")
    return pd.DataFrame(rows)


def select_model(df_results, verbose=True):

    cand = df_results[df_results["model"] != BASELINE_NAME]
    best = cand.loc[cand["cv_r2_mean"].idxmax()]
    thr = best["cv_r2_mean"] - best["cv_r2_std"]
    ok = cand[cand["cv_r2_mean"] >= thr]["model"].tolist()
    order = {m: i for i, m in enumerate(COMPLEXITY_ORDER)}
    chosen = min(ok, key=lambda m: order.get(m, 99))
    if verbose:
        print(f"\n[IZBOR MODELA] Najbolji CV: {best['model']} ({best['cv_r2_mean']:.4f} +/- {best['cv_r2_std']:.4f})")
        print(f"[IZBOR MODELA] Ravnopravni po CV: {ok} -> najjednostavniji: {chosen}")
    return chosen


def physics_baselines(df, train_idx, test_idx, groups_train):

    if "p_in_mot_over_omega" not in df.columns:
        return []
    y, p = df["torque"], df["p_in_mot_over_omega"]
    ytr, yte = y.iloc[train_idx], y.iloc[test_idx]

    raw = {"model": "Physics: p_in_mot/omega (bez učenja)",
           "cv_r2_mean": r2_score(ytr, p.iloc[train_idx]), "cv_r2_std": np.nan,
           "cv_rmse_mean": metrics_dict(ytr, p.iloc[train_idx])["rmse"]}
    m = metrics_dict(yte, p.iloc[test_idx])
    raw.update(test_r2=m["r2"], test_rmse=m["rmse"], test_mae=m["mae"])

    X1 = df[["p_in_mot_over_omega"]]
    pipe = make_pipe(LinearRegression())
    scores = cross_val_score(pipe, X1.iloc[train_idx], ytr, cv=make_cv(groups_train),
                             groups=groups_train, scoring="r2", n_jobs=config.N_JOBS)
    pipe.fit(X1.iloc[train_idx], ytr)
    m = metrics_dict(yte, pipe.predict(X1.iloc[test_idx]))
    cal = {"model": "Physics: kalibrisan (LR na p/omega)",
           "cv_r2_mean": scores.mean(), "cv_r2_std": scores.std(), "cv_rmse_mean": np.nan,
           "test_r2": m["r2"], "test_rmse": m["rmse"], "test_mae": m["mae"]}
    return [raw, cal]


def evaluate_on_test(pipe, X_train, y_train, X_test, y_test):
    fitted = clone(pipe).fit(X_train, y_train)
    preds = fitted.predict(X_test)
    return metrics_dict(y_test, preds), preds, fitted


def run_experiment(name, df, verbose=True):
    X, y, groups = get_xyg(df, name)
    train_idx, test_idx = make_split(X, y, groups)
    X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
    y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
    g_tr = groups.iloc[train_idx]

    if verbose:
        header(f"EKSPERIMENT: {name} ({X.shape[1]} atributa) | serija train={g_tr.nunique()} "
               f"test={groups.iloc[test_idx].nunique()}")
    df_models = compare_models(X_tr, y_tr, g_tr, verbose)
    chosen = select_model(df_models, verbose)

    pipe = make_pipe(get_models()[chosen])
    test_m, preds, fitted = evaluate_on_test(pipe, X_tr, y_tr, X_te, y_te)
    row = df_models[df_models["model"] == chosen].iloc[0]
    if verbose:
        print(f"[TEST, samo izabrani model] {chosen}: R2={test_m['r2']:.4f} RMSE={test_m['rmse']:.4f} MAE={test_m['mae']:.4f}")

    summary = {
        "experiment": name, "n_features": X.shape[1], "best_model": chosen,
        "cv_r2_mean": row["cv_r2_mean"], "cv_r2_std": row["cv_r2_std"], "cv_rmse": row["cv_rmse_mean"],
        "test_r2": test_m["r2"], "test_rmse": test_m["rmse"], "test_mae": test_m["mae"],
    }
    ctx = dict(X=X, y=y, groups=groups, train_idx=train_idx, test_idx=test_idx,
               X_train=X_tr, X_test=X_te, y_train=y_tr, y_test=y_te, groups_train=g_tr,
               chosen=chosen, pipe=pipe, fitted=fitted, preds=preds, test_metrics=test_m)
    return summary, df_models, ctx


# ODABIR NAJZNAČAJNIJIH ATRIBUTA 
def rank_features(pipe, X, y, groups):
    imps = []
    for tr, va in make_cv(groups).split(X, y, groups):
        p = clone(pipe).fit(X.iloc[tr], y.iloc[tr])
        perm = permutation_importance(p, X.iloc[va], y.iloc[va], n_repeats=5,
                                      random_state=config.RANDOM_STATE, scoring="r2", n_jobs=config.N_JOBS)
        imps.append(perm.importances_mean)
    return (pd.DataFrame({"feature": X.columns, "importance": np.mean(imps, axis=0)})
              .sort_values("importance", ascending=False).reset_index(drop=True))


def top_k_table(pipe, X_train, y_train, X_test, y_test, groups_train, ranking, ks=(3, 5, 8)):
    n = X_train.shape[1]
    k_list = [k for k in ks if k < n] + [n]
    cv = make_cv(groups_train)
    rows = []
    for k in k_list:
        feats = ranking["feature"].head(k).tolist()
        scores = cross_val_score(pipe, X_train[feats], y_train, cv=cv, groups=groups_train,
                                 scoring="r2", n_jobs=config.N_JOBS)
        fitted = clone(pipe).fit(X_train[feats], y_train)
        rows.append({"k": k if k < n else f"svi ({n})", "n": k, "cv_r2_mean": scores.mean(), "cv_r2_std": scores.std(),
                     "test_r2": r2_score(y_test, fitted.predict(X_test[feats])), "features": ", ".join(feats)})
    tab = pd.DataFrame(rows)
    full = tab.iloc[-1]
    ok = tab[tab["cv_r2_mean"] >= full["cv_r2_mean"] - full["cv_r2_std"]]
    rec = int(ok["n"].min())
    return tab, rec



# EDA
def compute_vif(X):
    Xs = StandardScaler().fit_transform(X)
    out = []
    for i, c in enumerate(X.columns):
        others = np.delete(Xs, i, axis=1)
        if others.shape[1] == 0:
            out.append(1.0)
            continue
        r2 = LinearRegression().fit(others, Xs[:, i]).score(others, Xs[:, i])
        out.append(np.inf if r2 > 1 - 1e-12 else 1 / (1 - r2))
    return pd.DataFrame({"feature": X.columns, "VIF": out}).sort_values("VIF", ascending=False)


def eda(df, X, y, fig_dir, metrics_dir, plots=True):
    header("EKSPLORATIVNA ANALIZA")
    corr = X.corr()
    pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
             if abs(corr.loc[a, b]) > 0.95]
    pairs_df = pd.DataFrame(pairs, columns=["feature_a", "feature_b", "corr"]).sort_values("corr", key=abs, ascending=False)
    pairs_df.to_csv(metrics_dir / "high_corr_pairs.csv", index=False)
    print(f"Parova atributa sa |r| > 0.95: {len(pairs_df)}")
    if len(pairs_df):
        print(pairs_df.head(10).to_string(index=False))

    vif = compute_vif(X)
    vif.to_csv(metrics_dir / "vif.csv", index=False)
    print("\nVIF (>10 = jaka multikolinearnost):")
    print(vif.head(8).to_string(index=False))
    print(f"\nKorelacija sa momentom (top 5 po |r|):")
    print(X.corrwith(y).sort_values(key=abs, ascending=False).head(5).to_string())

    if not plots:
        return

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.boxplot(y, vert=False, patch_artist=True, boxprops=dict(facecolor=COLOR_LIGHT, color=COLOR_DARK),
               medianprops=dict(color=COLOR_DARK, linewidth=2))
    ax.set_title("Raspodela momenta (ekstremne vrednosti)", color=COLOR_DARK, fontweight="bold")
    ax.set_xlabel("Moment [Nm]")
    fig.tight_layout(); fig.savefig(fig_dir / "torque_boxplot.png", dpi=150); plt.close(fig)

    ncol = 4
    nrow = int(np.ceil(X.shape[1] / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.2 * ncol, 2.6 * nrow))
    for ax, c in zip(np.atleast_1d(axes).ravel(), X.columns):
        ax.hist(X[c], bins=40, color=COLOR_MID)
        ax.set_title(c, fontsize=9)
    for ax in np.atleast_1d(axes).ravel()[X.shape[1]:]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(fig_dir / "feature_histograms.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.scatter(df["speed_rpm"], y, s=6, alpha=0.4, color=COLOR_MID)
    ax.set_xlabel("Brzina [o/min]"); ax.set_ylabel("Moment [Nm]")
    ax.set_title("Moment vs brzina (radne tačke)", color=COLOR_DARK, fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "torque_vs_speed.png", dpi=150); plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 9))
    cmap = sns.blend_palette([COLOR_LIGHT, COLOR_MID, COLOR_DARK], as_cmap=True)
    sns.heatmap(X.assign(torque=y).corr(), annot=X.shape[1] <= 18, fmt=".2f", cmap=cmap, ax=ax,
                annot_kws={"color": "white", "size": 7})
    ax.set_title("Matrica korelacije", color=COLOR_DARK, fontweight="bold")
    fig.tight_layout(); fig.savefig(fig_dir / "correlation_matrix.png", dpi=150); plt.close(fig)



# GRAFICI EVALUACIJE
def plot_model_comparison(df_results, path, title):
    d = df_results.sort_values("cv_r2_mean")
    fig, ax = plt.subplots(figsize=(9, 0.6 * len(d) + 1.5))
    ax.barh(d["model"], d["cv_r2_mean"], xerr=d["cv_r2_std"].fillna(0), color=COLOR_MID, ecolor=COLOR_DARK)
    ax.set_xlabel("CV R2 (GroupKFold, train skup)")
    ax.set_title(title, color=COLOR_DARK, fontweight="bold")
    lo = min(0.0, d["cv_r2_mean"].min())
    ax.set_xlim(lo - 0.05, 1.0)
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def plot_test_diagnostics(y_test, preds, speed_test, path, name):
    res = y_test.to_numpy() - preds
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    axes[0].scatter(y_test, preds, s=8, alpha=0.4, color=COLOR_MID)
    lim = [y_test.min(), y_test.max()]
    axes[0].plot(lim, lim, "--", color=COLOR_DARK)
    axes[0].set_xlabel("Stvarni moment [Nm]"); axes[0].set_ylabel("Predviđeni moment [Nm]")
    axes[0].set_title(f"Stvarno vs predviđeno: {name}")
    axes[1].scatter(speed_test, res, s=8, alpha=0.4, color=COLOR_MID)
    axes[1].axhline(0, color=COLOR_DARK, ls="--")
    axes[1].set_xlabel("Brzina [o/min]"); axes[1].set_ylabel("Rezidual [Nm]")
    axes[1].set_title("Rezidual po brzini")
    axes[2].hist(res, bins=40, color=COLOR_MID)
    axes[2].set_xlabel("Rezidual [Nm]"); axes[2].set_title("Raspodela reziduala")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)


def grouped_cv_report(pipe, X, y, groups, fig_dir, metrics_dir):
    preds = cross_val_predict(pipe, X, y, cv=make_cv(groups), groups=groups, n_jobs=config.N_JOBS)
    overall = metrics_dict(y, preds)
    tab = (pd.DataFrame({"series_id": groups.to_numpy(), "err": preds - y.to_numpy()})
             .groupby("series_id")["err"]
             .agg(n="size", bias="mean", rmse=lambda e: float(np.sqrt(np.mean(e ** 2))),
                  mae=lambda e: float(np.mean(np.abs(e)))).reset_index())
    tab.to_csv(metrics_dir / "error_per_series.csv", index=False)

    fig, ax = plt.subplots(figsize=(max(6, 0.3 * len(tab)), 4))
    ax.bar(tab["series_id"].astype(str), tab["rmse"], color=COLOR_MID)
    ax.set_xlabel("Merna serija"); ax.set_ylabel("RMSE [Nm]")
    ax.set_title("Greška po mernoj seriji (leave-series-out)", color=COLOR_DARK, fontweight="bold")
    ax.tick_params(axis="x", rotation=90, labelsize=7)
    fig.tight_layout(); fig.savefig(fig_dir / "error_per_series.png", dpi=150); plt.close(fig)
    return overall, tab



# ČUVANJE MODELA + META PODACI
def save_bundle(pipe, X, experiment, model_name, tuned=False, params=None):
    """Snima finalni model + model_meta.json (atributi, opsezi ulaza, verzija scikit-learna)."""
    model_dir, _, metrics_dir = setup_dirs(experiment)
    path = model_dir / "final_torque_model.joblib"
    joblib.dump(pipe, path)
    meta = {
        "experiment": experiment, "model": model_name, "tuned": tuned,
        "params": {k: (v.item() if hasattr(v, "item") else v) for k, v in (params or {}).items()},
        "sklearn_version": sklearn.__version__,
        "features": list(X.columns),
        "feature_ranges": {c: [float(X[c].min()), float(X[c].max())] for c in X.columns},
        "target": "torque [Nm]",
    }
    (model_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    (metrics_dir / "best_model.json").write_text(
        json.dumps({"experiment": experiment, "model": model_name, "tuned": tuned}, indent=2), encoding="utf-8")
    return path


def load_best_info(experiment):
    p = config.results_dir(experiment) / "metrics" / "best_model.json"
    if not p.exists():
        raise FileNotFoundError(f"Nema {p}. Pokreni prvo main.py za eksperiment '{experiment}'.")
    return json.loads(p.read_text(encoding="utf-8"))