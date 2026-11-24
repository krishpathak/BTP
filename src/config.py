from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
TABLES = RESULTS / "tables"
LOGS = ROOT / "logs"

for _p in (DATA_RAW, DATA_PROC, RESULTS, FIGURES, TABLES, LOGS):
    _p.mkdir(parents=True, exist_ok=True)

INCART_DB = "incartdb"
INCART_FS = 257

# INCART records whose header lists coronary artery disease
CAD_RECORDS = {
    "patient 1":  ["I01", "I02"],
    "patient 10": ["I20", "I21", "I22"],
    "patient 16": ["I35", "I36", "I37"],
    "patient 17": ["I38", "I39"],
    "patient 20": ["I44", "I45", "I46"],
    "patient 25": ["I57", "I58"],
    "patient 31": ["I72", "I73"],
}

FANTASIA_DB = "fantasia"
FANTASIA_FS = 250
NORMAL_RECORDS = (
    [f"f1o{i:02d}" for i in range(1, 11)] + [f"f1y{i:02d}" for i in range(1, 11)] +
    [f"f2o{i:02d}" for i in range(1, 11)] + [f"f2y{i:02d}" for i in range(1, 11)]
)

TARGET_FS = 257

WAVELET = "db6"
WAVELET_LEVEL = 9
DENOISE_DETAIL_KILL = 1

PRE_R = 64
POST_R = 104
BEAT_LEN = PRE_R + 1 + POST_R

SEG_BOUNDS = [(0, 57), (57, 113), (113, 169)]
SEG_NAMES = ["Seg1 (P-wave)", "Seg2 (QRS-complex)", "Seg3 (T-wave)"]

SS_NUM_SCALES = 60
SS_SCALE_STEP = 0.5
NVM_MIN, NVM_MAX = 2, 12

N_MODES = 4

VMD_ALPHA = 500.0
VMD_TAU = 0.0
VMD_DC = False
VMD_INIT = 1
VMD_TOL = 1e-6
VMD_MAX_ITER = 500

ENERGY_OPERATORS = ["TKEO", "AGO", "EDO"]
STATS = ["mean", "variance", "skewness", "kurtosis"]
N_FEATURES = len(SEG_BOUNDS) * len(STATS) * (N_MODES + 1)

CV_FOLDS = 10
BO_ITERATIONS = 20
BO_SUBSAMPLE = 30000
BO_INNER_FOLDS = 3
RANDOM_STATE = 42
