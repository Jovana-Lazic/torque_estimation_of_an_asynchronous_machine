
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn

import config
from features import OMEGA_MIN, add_engineered_features, required_raw_columns


class ModelLoadError(RuntimeError):
    pass


class InputValidationError(ValueError):
    pass


def model_path(experiment):
    return config.model_dir(experiment) / "final_torque_model.joblib"


def load_model(experiment=config.EXPERIMENT):
    path = model_path(experiment)
    if not path.exists():
        raise ModelLoadError(f"Model nije pronađen: {path}\nPokreni prvo: python main.py --experiment {experiment}")
    try:
        model = joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(f"Neuspešno učitavanje modela '{path}': {exc}") from exc
    if not hasattr(model, "predict"):
        raise ModelLoadError("Učitani objekat nema metodu 'predict'.")
    return model


def load_meta(experiment=config.EXPERIMENT):
    p = config.model_dir(experiment) / "model_meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def raw_columns_for(model):
    return required_raw_columns(list(model.feature_names_in_))


def validate_and_align_input(model, input_df):
    if not hasattr(model, "feature_names_in_"):
        raise InputValidationError("Model ne izlaže feature_names_in_.")
    expected = list(model.feature_names_in_)
    missing = set(expected) - set(input_df.columns)
    if missing:
        raise InputValidationError(f"Nedostaju kolone: {sorted(missing)}. Potrebne sirove kolone: {required_raw_columns(expected)}")
    aligned = input_df.reindex(columns=expected).apply(pd.to_numeric, errors="coerce")
    if aligned.isnull().any().any():
        raise InputValidationError(f"Kolone sadrže NaN / nenumeričke vrednosti: {aligned.columns[aligned.isnull().any()].tolist()}")
    return aligned


def collect_warnings(input_df, model, meta=None):
    warns = []
    if meta and meta.get("sklearn_version") != sklearn.__version__:
        warns.append(f"Model treniran u scikit-learn {meta['sklearn_version']}, sada je {sklearn.__version__}.")
    if "speed_rpm" in input_df.columns:
        omega = input_df["speed_rpm"].astype(float) * 2 * np.pi / 60
        for i in np.where(omega.abs() < OMEGA_MIN)[0]:
            warns.append(f"Uzorak {i}: |ω| < {OMEGA_MIN} rad/s (~{OMEGA_MIN * 60 / (2 * np.pi):.0f} o/min) - "
                         f"atribut p_in_mot/ω je ograničen, predikcija je manje pouzdana.")
    if meta:
        eng = add_engineered_features(input_df)
        for f, (lo, hi) in meta["feature_ranges"].items():
            if f not in eng.columns:
                continue
            margin = 0.1 * (hi - lo)
            out = np.where((eng[f] < lo - margin) | (eng[f] > hi + margin))[0]
            for i in out:
                warns.append(f"Uzorak {i}: '{f}'={eng[f].iloc[i]:.4g} je van opsega treninga [{lo:.4g}, {hi:.4g}] - ekstrapolacija.")
    return warns


def predict_torque(model, input_df):
    engineered = add_engineered_features(input_df)
    aligned = validate_and_align_input(model, engineered)
    return model.predict(aligned).tolist()


def default_sample():
    return pd.DataFrame([{
        "speed_rpm": 460.0, "u_dc": 574.0,
        "d_a_std": 0.02, "d_b_std": 0.02, "d_c_std": 0.02,
        "theta_s_mean": 45.0, "theta_s_max": 47.0, "theta_r_mean": 58.0, "theta_r_max": 59.5,
        "i_a_rms": 1.50, "i_b_rms": 1.50, "i_c_rms": 1.50,
        "u_1_rms": 29.8, "u_2_rms": 29.9, "u_3_rms": 29.6,
        "p_in_mot": 14.8, "p_in_inv": 56.3,
    }])


def print_report(input_df, predictions, warns):
    sep = "=" * 40
    print(f"{sep}\nULAZNI PODACI\n{sep}")
    print(input_df.to_string(index=False))
    print(f"\n{sep}\nPREDIKCIJA MODELA\n{sep}")
    for i, p in enumerate(predictions):
        print(f"Uzorak {i}: procenjeni moment motora = {p:.4f} Nm")
    for w in warns:
        print(f"[UPOZORENJE] {w}")
    print(sep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default=config.EXPERIMENT)
    ap.add_argument("--csv", default=None, help="CSV sa sirovim ulaznim kolonama")
    ap.add_argument("--out", default=None, help="gde snimiti CSV sa predikcijama")
    args = ap.parse_args()
    try:
        model = load_model(args.experiment)
        meta = load_meta(args.experiment)
        input_df = pd.read_csv(args.csv) if args.csv else default_sample()
        preds = predict_torque(model, input_df)
        warns = collect_warnings(input_df, model, meta)
        print_report(input_df, preds, warns)
        if args.out:
            out = input_df.assign(torque_pred_Nm=preds)
            out.to_csv(args.out, index=False)
            print(f"Predikcije snimljene: {args.out}")
    except (ModelLoadError, InputValidationError, FileNotFoundError) as exc:
        print(f"[GREŠKA] {exc}", file=sys.stderr)
        sys.exit(1)
    except Exception as exc:
        print(f"[NEOČEKIVANA GREŠKA] {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()