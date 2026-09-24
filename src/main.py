import argparse
import time

import pandas as pd

import config
import pipeline as pl


def analyze_prediction(ctx, df_models, extra_rows):
    m, y_test = ctx["test_metrics"], ctx["y_test"]
    rng = y_test.max() - y_test.min()
    pl.header(f"ANALIZA REZULTATA PREDIKCIJE ({ctx['chosen']}, test skup)")
    print(f"R2   = {m['r2']:.4f}")
    print(f"RMSE = {m['rmse']:.4f} Nm ({100 * m['rmse'] / rng:.2f}% opsega momenta u test skupu: {rng:.2f} Nm)")
    print(f"MAE  = {m['mae']:.4f} Nm ")

    print("\nZAKLJUČAK:")
    if m["r2"] >= 0.9:
        print("  - Vrlo dobro slaganje predikcije sa stvarnim momentom.")
    elif m["r2"] >= 0.7:
        print("  - Solidno slaganjeb predikcije sa stvarnim momentom")
    else:
        print("  - Nizak R2")
    phys = [r for r in extra_rows if r["model"].startswith("Physics: kalibrisan")]
    if phys:
        gap = m["r2"] - phys[0]["test_r2"]
        if gap > 0.005:
            print(f"  - ML model je bolji od fizičkog baseline-a{gap:.4f}.")
        else:
            print(f"  - ML NE pobeđuje jednostavan fizički baseline (ΔR2={gap:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=config.EXPERIMENT)
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args()
    exp, t0 = args.experiment, time.time()

    model_dir, fig_dir, metrics_dir = pl.setup_dirs(exp)
    df = pl.load_data()
    pl.print_dropped_columns()
    X_all, y_all, _ = pl.get_xyg(df, exp)
    pl.eda(df, X_all, y_all, fig_dir, metrics_dir, plots=not args.no_plots)

    summary, df_models, ctx = pl.run_experiment(exp, df)

    extra = pl.physics_baselines(df, ctx["train_idx"], ctx["test_idx"], ctx["groups_train"])
    table = pd.concat([df_models, pd.DataFrame(extra)], ignore_index=True)
    table.to_csv(metrics_dir / "model_comparison.csv", index=False)
    pl.header("TABELA POREĐENJA (CV na train skupu; test_* samo za fizičke baseline-e i izabrani model)")
    print(table.round(4).to_string(index=False))
    pl.plot_model_comparison(table, fig_dir / "model_comparison.png", f"Poređenje modela ({exp})")

    analyze_prediction(ctx, df_models, extra)
    pl.plot_test_diagnostics(ctx["y_test"], ctx["preds"], df["speed_rpm"].iloc[ctx["test_idx"]],
                             fig_dir / "test_diagnostics.png", ctx["chosen"])

    # Odabir najznačajnijih atributa
    pl.header("ODABIR NAJZNAČAJNIJIH ATRIBUTA (permutation importance na CV validacionim foldovima)")
    ranking = pl.rank_features(ctx["pipe"], ctx["X_train"], ctx["y_train"], ctx["groups_train"])
    ranking.to_csv(metrics_dir / "feature_importance.csv", index=False)
    print(ranking.round(4).to_string(index=False))
    topk, rec = pl.top_k_table(ctx["pipe"], ctx["X_train"], ctx["y_train"], ctx["X_test"], ctx["y_test"],
                               ctx["groups_train"], ranking)
    topk.to_csv(metrics_dir / "top_k_comparison.csv", index=False)
    print("\nSvi vs najbitniji atributi:")
    print(topk.drop(columns=["n"]).round(4).to_string(index=False))
    print(f"-> Najmanji k čiji je CV R2 unutar 1 std od punog skupa: k = {rec}")

    overall, per_series = pl.grouped_cv_report(ctx["pipe"], ctx["X"], ctx["y"], ctx["groups"], fig_dir, metrics_dir)
    pl.header("LEAVE-SERIES-OUT PROCENA (ceo skup)")
    print(f"R2={overall['r2']:.4f} RMSE={overall['rmse']:.4f} MAE={overall['mae']:.4f}")
    print(f"RMSE po seriji: min={per_series['rmse'].min():.4f} medijana={per_series['rmse'].median():.4f} "
          f"max={per_series['rmse'].max():.4f}")

    # Finalni model na celom skupu
    final = pl.make_pipe(pl.get_models()[ctx["chosen"]])
    final.fit(ctx["X"], ctx["y"])
    assert final.named_steps["scaler"].n_features_in_ == ctx["X"].shape[1]
    path = pl.save_bundle(final, ctx["X"], exp, ctx["chosen"])
    print(f"\nFinalni model ({ctx['chosen']}) sačuvan: {path}")
    print(f"Ukupno vreme: {time.time() - t0:.0f}s. Sledeće: python hyperparameter_tuning.py --experiment {exp}")


if __name__ == "__main__":
    main()