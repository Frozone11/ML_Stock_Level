# Magazine Cover Part Classifier (Random Forest)

This script trains a **RandomForestClassifier** to predict how many **top covers** or **bottom covers** (labeled as `Part` = `0–20`) are present in a magazine, based on a single numeric sensor value called `Reading` (exported from an Arduino workflow into Excel).

It loads an Excel sheet, trains a model, prints metrics, and shows a confusion matrix heatmap.

---

## Repository Layout

├── StockClassifier.py\
├── requirements.txt\
└── data/\
├── ultrasound_top.xlsx\
├── ultrasound_bottom.xlsx\
├── toppart_data.xlsx\
└── bottompart_data.xlsx


---

## Data Format (Excel)

The Excel file **must** contain these columns:

- `Reading` (numeric): sensor reading (e.g., ultrasound distance)
- `Part` (integer): class label from `0` to `20`

Example:

| Reading | Part |
|--------:|-----:|
| 123.4   | 0    |
| 118.9   | 1    |
| ...     | ...  |

The script expects a sheet named:

- `Data` by default (changeable)

---

## Setup

### Create and activate a virtual environment (recommended)

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```
**Windows (PowerShell)**
```bash
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```
Run the script
```bash
python3 StockClassifier.py
```
### Choose Top vs Bottom Covers

At the top of StockClassifier.py, change:

EXCEL_PATH = "data/ultrasound_top.xlsx"
# or
EXCEL_PATH = "data/ultrasound_bottom.xlsx"

SHEET_NAME = "Data"

---
## What the Script Does

1) Loads an Excel dataset from **EXCEL_PATH**

2) Validates that **Reading** and **Part** exist

3) Cleans data (coerces numeric, drops missing values)

4) Splits data into train/test (80/20) with stratification

5) Trains a **RandomForestClassifier** (**n_estimators=200**)

6) Prints:
   * Accuracy
   * Full classification report (precision/recall/F1)

7) Plots:
   * Confusion matrix heatmap

8) Prints example predictions across the reading range

---

### Output

In the terminal you’ll see:
* A quick preview of the dataframe
* Unique part labels found (should be 0–20)
* Test accuracy
* Classification report
* Example predictions

A confusion matrix window will pop up (matplotlib).





