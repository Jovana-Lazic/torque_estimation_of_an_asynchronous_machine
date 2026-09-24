import numpy as np

OMEGA_MIN = 5.0

# Izvedeni atribut -> sirove kolone od kojih nastaje
DERIVED_SOURCES = {
    "u_rms": ["u_1_rms", "u_2_rms", "u_3_rms"],
    "i_rms": ["i_a_rms", "i_b_rms", "i_c_rms"],
    "d_std": ["d_a_std", "d_b_std", "d_c_std"],
    "p_in_mot_over_omega": ["speed_rpm", "p_in_mot"],
}


def required_raw_columns(feature_names):
    raw = []
    for f in feature_names:
        for c in DERIVED_SOURCES.get(f, [f]):
            if c not in raw:
                raw.append(c)
    return raw


def add_engineered_features(df):
    df = df.copy()


    for name in ("u_rms", "i_rms", "d_std"):
        cols = DERIVED_SOURCES[name]
        if all(c in df.columns for c in cols):
            df[name] = df[cols].mean(axis=1)

    # Moment ~ snaga / ugaona brzina 
    if all(c in df.columns for c in DERIVED_SOURCES["p_in_mot_over_omega"]):
        omega = df["speed_rpm"] * 2 * np.pi / 60
        sign = np.where(omega >= 0, 1.0, -1.0)
        omega_safe = np.where(np.abs(omega) >= OMEGA_MIN, omega, sign * OMEGA_MIN)
        df["p_in_mot_over_omega"] = df["p_in_mot"] / omega_safe

    return df