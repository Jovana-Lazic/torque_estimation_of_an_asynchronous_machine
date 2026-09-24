

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_PATH = BASE_DIR / "data" / "processed" / "induction_motor_windows.csv"

RANDOM_STATE = 42
N_JOBS = 1          
TEST_SIZE = 0.2        
N_SPLITS_CV = 5        

# --- Obrada podataka---------------------------------------
WINDOW_S = 10.0        # širina vremenskog prozora [s]
THIRD_WINDOW_SHIFT = 0

# --- Čišćenje--------------------------------------------
MIN_SAMPLES_FRAC = 0.9    
TRANSIENT_QUANTILE = 0.99   

# --- Eksperiment (skup ulaznih atributa) --------------------------------------
#   "no_power"                = brzina, naponi, struje, temperature (bez snaga)
#   "with_power"              = no_power + snage (p_in_mot, p_in_inv)
#   "merged"                  = spojene faze + snage
#   "merged_power_over_omega" = merged + p_in_mot / ugaona brzina
EXPERIMENT = "merged_power_over_omega"


def model_dir(experiment: str) -> Path:
    return BASE_DIR / "models" / experiment


def results_dir(experiment: str) -> Path:
    return BASE_DIR / "results" / experiment