import time

import matplotlib.pyplot as plt
import pandas as pd

import config
import pipeline as pl


def main():
    t0 = time.time()
    out_dir = config.BASE_DIR / "results" / "all_experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pl.load_data()
    rows = []
    for name in pl.get_feature_sets():
        summary, _, _ = pl.run_experiment(name, df)
        rows.append(summary)

    summ = pd.DataFrame(rows).sort_values("cv_r2_mean", ascending=False).reset_index(drop=True)
    summ.to_csv(out_dir / "experiments_summary.csv", index=False)

    pl.header("ZBIRNA TABELA - POREĐENJE SETOVA ATRIBUTA (sortirano po CV R2)")
    print(summ.round(4).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4))
    d = summ.sort_values("cv_r2_mean")
    ax.barh(d["experiment"], d["cv_r2_mean"], xerr=d["cv_r2_std"], color=pl.COLOR_MID, ecolor=pl.COLOR_DARK)
    ax.set_xlabel("CV R2 najboljeg modela po setu")
    ax.set_title("Poređenje setova atributa", color=pl.COLOR_DARK, fontweight="bold")
    fig.tight_layout(); fig.savefig(out_dir / "experiments_comparison.png", dpi=150); plt.close(fig)

    w = summ.iloc[0]
    print(f"\nPOBEDNIČKI SET (po CV): '{w['experiment']}' sa modelom {w['best_model']} "
          f"(CV R2 = {w['cv_r2_mean']:.4f})")
    print(f"Sledeći koraci:\n  python main.py --experiment {w['experiment']}\n"
          f"  python hyperparameter_tuning.py --experiment {w['experiment']}")
    print(f"Ukupno vreme: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()