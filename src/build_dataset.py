
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

import config
from features import OMEGA_MIN

WINDOW_S = config.WINDOW_S
CHUNK_ROWS = 2_000_000  
KEYS = ["series_id", "window"]

FIRST_RENAME = {
    "l": "series_id", "t": "t", "n_{m}": "speed_rpm", "T_{TS,m}": "torque",
    "u_{dc,m}": "u_dc", "d_{a,-2}": "d_a", "d_{b,-2}": "d_b", "d_{c,-2}": "d_c",
}

THIRD_RENAME = {
    "l": "series_id",
    "P_{in,mot,m}": "p_in_mot", "P_{in,minv,m}": "p_in_inv", "P_{mech,m}": "p_mech",
    "i_{a,r,m}": "i_a_rms", "i_{b,r,m}": "i_b_rms", "i_{c,r,m}": "i_c_rms",
    "u_{1,r,m}": "u_1_rms", "u_{2,r,m}": "u_2_rms", "u_{3,r,m}": "u_3_rms",
}


def add_window(df):
    df["window"] = (df["t"] // WINDOW_S).astype("int64")
    return df


def aggregate_first_part(path):
    mean_cols = ["speed_rpm", "torque", "u_dc"]
    ac_cols = ["d_a", "d_b", "d_c", "torque"]  # std se racuna za njih
    partials = []

    reader = pd.read_csv(path, usecols=list(FIRST_RENAME.keys()),
                         dtype={"l": "int32"}, chunksize=CHUNK_ROWS)
    for i, chunk in enumerate(reader, start=1):
        chunk = add_window(chunk.rename(columns=FIRST_RENAME))
        for c in ac_cols:
            chunk[c + "_sq"] = chunk[c] ** 2

        sum_cols = mean_cols + ["d_a", "d_b", "d_c"] + [c + "_sq" for c in ac_cols]
        grouped = chunk.groupby(KEYS)
        part = grouped[sum_cols].sum()
        part["n"] = grouped.size()
        partials.append(part)
        print(f"   First_Part: obrađen deo {i} ({len(chunk):,} redova)")

    total = pd.concat(partials).groupby(level=KEYS).sum()
    n = total["n"]

    out = pd.DataFrame(index=total.index)
    for c in mean_cols:
        out[c] = total[c] / n
    for c in ac_cols:
        var = total[c + "_sq"] / n - (total[c] / n) ** 2
        out[c + "_std"] = np.sqrt(var.clip(lower=0))
    out["n_samples_hf"] = n
    return out.reset_index()


def _row_mean(df, cols):
    total = df[cols[0]].to_numpy(dtype="float64").copy()
    for c in cols[1:]:
        total += df[c].to_numpy(dtype="float64")
    return total / len(cols)


def _row_max(df, cols):
    arrays = [df[c].to_numpy(dtype="float64") for c in cols]
    return np.maximum.reduce(arrays)


def aggregate_second_part(path):
    df = pd.read_csv(path, dtype={"l": "int32"}).rename(columns={"l": "series_id"})
    df = add_window(df)

    s_cols = [c for c in df.columns if c.startswith("theta_{S,")]
    r_cols = [c for c in df.columns if c.startswith("theta_{R,")]

    per_row = pd.DataFrame({
        "series_id": df["series_id"],
        "window": df["window"],
        "theta_s_mean": _row_mean(df, s_cols),
        "theta_s_max": _row_max(df, s_cols),
        "theta_r_mean": _row_mean(df, r_cols),
        "theta_r_max": _row_max(df, r_cols),
    })
    agg_map = {"theta_s_mean": "mean", "theta_s_max": "max",
               "theta_r_mean": "mean", "theta_r_max": "max"}
    return per_row.groupby(KEYS, as_index=False).agg(agg_map)


def aggregate_third_part(path):
    df = pd.read_csv(path, dtype={"l": "int32"})
    df = add_window(df.rename(columns=THIRD_RENAME))
    cols = [c for c in THIRD_RENAME.values() if c != "series_id"]
    return df.groupby(KEYS, as_index=False)[cols].mean()


def check_window_shift(first, third):
    print("\nProvera vremenskog pomaka Third_Part :")
    best = None
    for k in (-1, 0, 1):
        t = third[KEYS + ["p_in_mot"]].copy()
        t["window"] += k
        m = first[KEYS + ["speed_rpm", "torque"]].merge(t, on=KEYS)
        omega = m["speed_rpm"] * 2 * np.pi / 60
        m, omega = m[omega.abs() >= OMEGA_MIN], omega[omega.abs() >= OMEGA_MIN]
        if len(m) < 3:
            continue
        corr = np.corrcoef(m["p_in_mot"] / omega, m["torque"])[0, 1]
        print(f"   pomak {k:+d}: corr = {corr:.4f}  ({len(m):,} prozora)")
        if best is None or corr > best[1]:
            best = (k, corr)
    if best is not None:
        print(f"   -> najbolji pomak: {best[0]:+d} "
              f"(trenutno u config.THIRD_WINDOW_SHIFT = {config.THIRD_WINDOW_SHIFT})")


def build_dataset(raw_dir=config.RAW_DIR, out_path=config.DATA_PATH, force=False):
    raw_dir, out_path = Path(raw_dir), Path(out_path)
    if out_path.exists() and not force:
        print(f"Izlaz već postoji ({out_path}) - preskačem. Koristi --force za ponovnu izgradnju.")
        return pd.read_csv(out_path)

    print("Obrada First_Part.csv...")
    first = aggregate_first_part(raw_dir / "First_Part.csv")
    print("Obrada Second_Part.csv...")
    second = aggregate_second_part(raw_dir / "Second_Part.csv")
    print("Obrada Third_Part.csv...")
    third = aggregate_third_part(raw_dir / "Third_Part.csv")

    print(f"\nProzora po delu: first={len(first):,} | second={len(second):,} | third={len(third):,}")
    print(f"Mernih serija (l): first={first['series_id'].nunique()} | "
          f"second={second['series_id'].nunique()} | third={third['series_id'].nunique()}")

    check_window_shift(first, third)
    if config.THIRD_WINDOW_SHIFT:
        third = third.copy()
        third["window"] += config.THIRD_WINDOW_SHIFT
        print(f"Primenjen pomak Third_Part prozora: {config.THIRD_WINDOW_SHIFT:+d}")

    df = (first.merge(second, on=KEYS, how="inner")
               .merge(third, on=KEYS, how="inner")
               .dropna()
               .sort_values(KEYS)
               .reset_index(drop=True))
    print(f"Nakon spajanja (inner join): {len(df):,} redova, {df['series_id'].nunique()} serija")

    # P_mech ~ moment * omega => curenje cilja ako uđe u ulazne atribute
    omega = df["speed_rpm"] * 2 * np.pi / 60
    corr = np.corrcoef(df["p_mech"], df["torque"] * omega)[0, 1]
    print(f"corr(P_mech, moment*omega) = {corr:.4f}  (blizu 1 => P_mech NE sme u ulazne atribute)")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"\nSačuvano: {out_path}")
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="ponovo izgradi i ako izlaz postoji")
    build_dataset(force=ap.parse_args().force)