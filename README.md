# CAD identification from ECG using improved energy estimation

B.Tech project based on the paper "Coronary Artery Disease Identification Using ECG:
An Improved Energy Estimation-Based Approach" (IEEE Access, vol. 12, 2024).

The idea: energy operators are nonlinear, so applying them straight to a signal that has
several components produces cross-terms. Splitting the beat into modes with VMD first and
adding up the per-mode energies avoids this. VMD, the scale-space boundary detector, Otsu
thresholding, Pan-Tompkins, the three energy operators and the Bayesian optimiser are all
written from scratch.

## Pipeline

```
lead-II ECG
  -> db6 wavelet denoising (remove A9 baseline wander and D1 noise)
  -> resample normal records 250 Hz -> 257 Hz
  -> Pan-Tompkins R-peak detection
  -> beats of 169 samples (64 before R, 104 after)
  -> VMD with 4 modes
  -> TKEO / AGO / EDO energy per mode, summed
  -> 3 segments x 4 moments x 5 signals = 60 features
  -> Bayesian-optimised tree ensemble, 10-fold CV
```

## Data

Both databases are open on PhysioNet.

| | CAD | Normal |
|---|---|---|
| Database | INCART 12-lead arrhythmia | Fantasia |
| Subjects | 7 | 40 |
| Records | 17 | 40 |

## Files

```
src/
  config.py             constants and paths
  data_download.py      downloads the records
  preprocessing.py      denoising, resampling, Pan-Tompkins, beat cutting
  scale_space.py        scale-space minima, Otsu, number of modes
  vmd.py                VMD (single signal and batch versions)
  energy_operators.py   TKEO, AGO, EDO
  features.py           segment statistics
  bayesopt.py           Gaussian-process Bayesian optimiser
  validate_detector.py  R-peak detector vs. annotations
  classify.py           training and cross-validation
  synthetic_demo.py     synthetic four-component test
  figures.py            figures
  build_dataset.py      feature extraction per record
run_all.py              runs everything in order
```

## Running

```
pip install numpy scipy scikit-learn matplotlib pandas wfdb PyWavelets joblib requests
python run_all.py
```

or one step at a time, e.g. `python src/build_dataset.py --n-jobs -1 --nvm-stride 5`.
`run_all.py` also takes `--from`, `--to` and `--only` with a stage name.
Feature extraction saves one file per record, so it can be stopped and restarted.
Random seeds are set in `src/config.py`.

## Limitations

1. The two classes come from different databases and different recording equipment, so
   some of the separation may come from the acquisition and not from the disease.
2. The 10-fold CV splits beats, not patients, so beats of one patient end up in both
   train and test. `classify.py` also runs a subject-wise evaluation, which is the
   stricter number.
