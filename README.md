# Data-Science-Coding-Challange

Churn-prediction exercise: one row per video-streaming subscription, and the task is to predict
whether that subscription is continued into the next month (binary `Churn`, scored as a
probability between 0 and 1).

---

## What is actually in this repository

Verified on this workstation **2026-09-27**. The repository tracks **4 files**
(`git ls-files | wc -l` → `4`), and these are the only files ever added to it
(`git log --all --diff-filter=A --name-only` shows exactly the same four, in the single commit
`33e9831`).

| File | What it really does (read from the source) |
|---|---|
| `Data Science Coding Challange.py` | Starter / exploration script. Imports `pandas`, `matplotlib.pyplot`, `seaborn`. Loads `train.csv`, prints `train_df.shape` and `train_df.head()`, then plots a 30-bin KDE histogram of `MonthlyCharges`. The required submission format is documented in the trailing comment block. |
| `updated_model.py` | The working model. Imports `pandas` and `scikit-learn` (`RandomForestClassifier`, `roc_auc_score`, `train_test_split`, `StandardScaler`). Median-fills the numeric columns, one-hot encodes the categorical ones, re-aligns the test columns to the train columns, scales train/validation/test with one `StandardScaler`, trains a 100-tree Random Forest (`random_state=42`) on an 80/20 split, prints `Validation ROC AUC Score: <value>`, and writes `predictions.csv` with columns `CustomerID,predicted_probability`. |
| `.gitignore` | Ignores `__pycache__/`, `*.pyc`, `.ipynb_checkpoints/`, `predictions.csv`. |
| `README.md` | This file. |

There is no package, no test suite and no build step — just the two scripts above.

---

## The datasets are NOT included

**`train.csv` and `test.csv` are not in this repository, and neither script can run as checked
out.** Verify it directly:

```console
$ git ls-files | wc -l
4

$ find . -not -path './.git/*' \( -iname '*.csv' -o -iname '*.json' \) | wc -l
0

$ git log --all --diff-filter=A --name-only --pretty=format: | sort -u | grep -v '^$'
Data Science Coding Challange.py
.gitignore
README.md
updated_model.py

$ python3 updated_model.py
Missing data file(s): train.csv, test.csv.
The challenge datasets are not part of this repository - obtain train.csv (with the Churn label)
and test.csv and place them in the working directory.
$ echo $?
1
```

Before the missing-file guard was added, the same run ended in the raw pandas error
`FileNotFoundError: [Errno 2] No such file or directory: 'train.csv'` with exit code `1`.

`data_descriptions.csv` is also absent. The original README carried a two-line snippet
(`data_descriptions = pd.read_csv('data_descriptions.csv')`, line 34 of `33e9831:README.md`) to print
the feature descriptions; the file itself has never been committed either
(`git log --all --diff-filter=A --name-only` lists only the four tracked files above).

To run either script you must obtain the challenge's `train.csv` and `test.csv` yourself and put
them **next to the scripts**. Columns that the code actually requires (traced from the lines that
use them):

- `train.csv` — `CustomerID`, `Churn` (the binary target), and the feature columns. The starter
  script also reads `MonthlyCharges`.
- `test.csv` — `CustomerID` plus the feature columns, **including `MonthlyCharges`**. The challenge
  says `test.csv` withholds the `Churn` ground truth; that cannot be checked here because the file
  is absent. Either way, `updated_model.py` discards any extra column, because it re-aligns the
  encoded test frame onto the exact train column set.

Each script now checks for its input files up front and exits with a clear message instead of a
raw traceback:

```console
$ python3 updated_model.py
Missing data file(s): train.csv, test.csv.
The challenge datasets are not part of this repository - obtain train.csv (with the Churn label)
and test.csv and place them in the working directory.
```

### Numbers from the original challenge text — quoted, not verified

The challenge description that shipped with this repository stated:

> train.csv contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly,
> will reveal whether or not the subscription was continued into the next month (the “ground truth”).
>
> The test.csv dataset contains the exact same information about the remaining segment of the overall
> sample (104,480 subscriptions to be exact), but does not disclose the “ground truth” for each
> subscription.

The quote itself *is* verifiable — it is recoverable from this repository's own git history, where it
was the initial `README.md` (blob `ea0f249`, first commit `33e9831`):

```console
$ git show 33e9831:README.md | grep -n 'subscriptions to be exact'
18:train.csv contains 70% of the overall sample (243,787 subscriptions to be exact) and importantly, will reveal whether or not the subscription was continued into the next month (the “ground truth”).
20:The test.csv dataset contains the exact same information about the remaining segment of the overall sample (104,480 subscriptions to be exact), but does not disclose the “ground truth” for each subscription. It’s your job to predict this outcome!
```

**What cannot be checked is the counts in those sentences**, because the CSVs are not here and no data
file has ever been committed to this repository (see the history check above). They are recorded above
as *quoted from the original challenge text* — not as verified facts about this repo.

Once you have the real CSVs, read the true counts instead of trusting any inherited number:

```bash
python3 -c "import pandas as pd; print('train rows:', len(pd.read_csv('train.csv'))); print('test rows:', len(pd.read_csv('test.csv')))"
```

Your submission must have exactly as many rows as `test.csv` has — read that number from the file,
never hard-code it.

---

## Requirements

Both scripts use only the standard library plus these four third-party packages:

| Package (PyPI name) | Imported as | Used by |
|---|---|---|
| `pandas` | `pandas` | both scripts |
| `scikit-learn` | `sklearn.ensemble`, `sklearn.metrics`, `sklearn.model_selection`, `sklearn.preprocessing` | `updated_model.py` |
| `matplotlib` | `matplotlib.pyplot` | `Data Science Coding Challange.py` |
| `seaborn` | `seaborn` | `Data Science Coding Challange.py` |

Installing those four also pulls in their own dependencies — `numpy` (required by both `pandas`
and `scikit-learn`), `scipy`, `joblib` and `threadpoolctl` (required by `scikit-learn`). Install
everything in one line:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

If the system Python is externally managed (common on recent Ubuntu), use one of:

```bash
pip install --user pandas scikit-learn matplotlib seaborn
# or a virtual environment
python3 -m venv .venv && . .venv/bin/activate && pip install pandas scikit-learn matplotlib seaborn
```

### Versions this README was verified against

```console
$ python3 --version
Python 3.10.12

$ python3 -c "from importlib.metadata import version; [print(f'{d}=={version(d)}') for d in ['pandas','numpy','scikit-learn','matplotlib','seaborn','scipy']]"
pandas==2.3.3
numpy==2.2.6
scikit-learn==1.7.2
matplotlib==3.10.8
seaborn==0.13.2
scipy==1.15.3
```

---

## Usage

```bash
# 1. put train.csv and test.csv next to the scripts
# 2. explore (draws a matplotlib window with the MonthlyCharges histogram)
python3 "Data Science Coding Challange.py"
# 3. train, validate, and write the submission
python3 updated_model.py
```

`updated_model.py` writes `predictions.csv` in the working directory in the format the challenge
expects: a header row plus one row per `test.csv` row, with columns
`CustomerID,predicted_probability`.

---

## Feature notes (from the code, not from a description file)

`updated_model.py` drops `CustomerID` and `Churn` from the feature matrix, one-hot encodes every
remaining column with `pandas.get_dummies`, and aligns the encoded test frame back onto the train
columns with `reindex(columns=X.columns, fill_value=0)`. The starter script reads `MonthlyCharges`
for its histogram. Anything beyond that — the full feature list and their meanings — lives in the
challenge's own `data_descriptions.csv`, which is not part of this repository.

---

## Verification log — 2026-09-27, this workstation

| Check | Command | Real result |
|---|---|---|
| Tracked file count | `git ls-files \| wc -l` | `4` |
| CSV/JSON files in tree | `find . -not -path './.git/*' \( -iname '*.csv' -o -iname '*.json' \) \| wc -l` | `0` |
| Starter script syntax | `python3 -m py_compile "Data Science Coding Challange.py"` | exit `0` |
| Model script syntax | `python3 -m py_compile updated_model.py` | exit `0` |
| Starter script run | `python3 "Data Science Coding Challange.py"` | `train.csv was not found in the working directory.` guard message, exit `1` |
| Model script run | `python3 updated_model.py` | `Missing data file(s): train.csv, test.csv.`, exit `1` |
| Import smoke test | import the four declared packages (`pandas`, `matplotlib.pyplot`, `seaborn`, and the sklearn submodules) | all 7 submodules import OK on Python 3.10.12 (`pandas 2.3.3`, `scikit-learn 1.7.2`) |
| End-to-end code path | generated a **seeded synthetic fixture** (see below) and ran `updated_model.py` on it | exit `0`; printed `Validation ROC AUC Score: 0.47529706066291433` and `Wrote predictions.csv with 120 rows`; `predictions.csv` = 121 lines (header + 120) |
| End-to-end code path | same fixture, ran the starter script with `MPLBACKEND=Agg` | exit `0`; printed `train_df Shape: (400, 5)` plus a 5-row head; `plt.show()` emitted `UserWarning: FigureCanvasAgg is non-interactive` |

The two end-to-end rows are reproducible: the fixture is seeded, and re-generating it produces the
identical `0.47529706066291433` on a second run (verified twice). Regenerate it with:

```bash
python3 - <<'PY'
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
def frame(n, with_label, start):
    d = {"CustomerID": [f"S{start+i:05d}" for i in range(n)],
         "MonthlyCharges": np.round(rng.uniform(20, 120, n), 2),
         "Tenure": rng.integers(0, 72, n),
         "Contract": rng.choice(["monthly", "1yr", "2yr"], n)}
    if with_label:
        d["Churn"] = rng.integers(0, 2, n)
    return pd.DataFrame(d)
frame(400, True, 100000).to_csv("train.csv", index=False)
frame(120, False, 200000).to_csv("test.csv", index=False)
PY
```

That fixture is **synthetic smoke-test input, not the challenge data**. Its `Churn` labels are
random, so the `0.475…` ROC AUC is meaningless as a model result. The run is included only to prove
the scripts execute start-to-finish — and, for `updated_model.py`, that its submission shape is one
row per `test.csv` row. No number in this README is a model result, because the model has never been
run on the real data: the real data is not here.

The pre-guard failure recorded above (`FileNotFoundError: [Errno 2] No such file or directory:
'train.csv'`, exit `1`) is the raw pandas error that `pd.read_csv` raises on a missing file, checked
directly with `python3 -c "import pandas as pd; pd.read_csv('train.csv')"`.
