# 🔬 Semiconductor Yield & WAT Analysis Dashboard
### ADI Gandhinagar — Yield/Manufacturing Engineering Group

A production-grade Streamlit dashboard that demonstrates:
- **SPC Charts** (X-bar, R-chart, CUSUM with excursion detection)
- **Lot-by-lot Yield Trend** using Murphy's Yield Model
- **Root Cause Analysis** via PCA + Isolation Forest (ML)
- **Cpk / Process Capability** for all WAT parameters
- **WAT Parameter Correlation Heatmap**

---

## 🚀 How to Run (Step-by-Step)

### Step 1 — Prerequisites
Make sure Python 3.9 or newer is installed.
```
python --version
```

### Step 2 — Create a Virtual Environment (Recommended)
```
python -m venv venv
```

Activate it:
- **Windows:**  `venv\Scripts\activate`
- **Mac/Linux:** `source venv/bin/activate`

### Step 3 — Install Dependencies
```
pip install -r requirements.txt
```

### Step 4 — Run the Dashboard
```
streamlit run app.py
```

Your browser will automatically open at:
```
http://localhost:8501
```

---

## 📁 Project Structure
```
adi_yield_dashboard/
├── app.py              ← Main Streamlit application (all code here)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🧪 What the Dashboard Shows

| Tab | Content |
|-----|---------|
| 📈 SPC Charts | X-bar chart, R-chart, CUSUM with real excursion events highlighted |
| 📉 Yield Analysis | Lot yield trend, yield distribution, D0 vs Yield scatter (Murphy's model), yield by process step |
| 🔍 Root Cause & PCA | PCA scatter, explained variance, Isolation Forest anomaly flags, parameter correlation heatmap |
| 📊 Cpk & Parameter Health | Cpk table + bar chart, histograms vs spec limits, process health radar |

---

## 🎯 Key Features That Align with ADI JD

- **SPC with CUSUM** → "Generate and maintain SPC charts for excursion detection"
- **Defect density / Murphy's model** → "Real-time analysis of inline fab data on defects"
- **Cpk / D0 / yield opportunity** → "Identify systematic issues and D0/Cpk/yield opportunity"
- **PCA + Isolation Forest** → "Statistical and root cause analysis using advanced data analysis tools"
- **WAT parameter tracking** → "ETest/WAT related parameters and wafer sort parameters"
- **Process step breakdown** → "Root cause analysis for low yields"

---

## 📊 Simulated Data Parameters

| Parameter | Nominal | LSL | USL |
|-----------|---------|-----|-----|
| Vt NMOS (V) | 0.45 | 0.40 | 0.50 |
| Idsat NMOS (µA/µm) | 482 | 430 | 530 |
| Sheet Resistance (Ω/□) | 98 | 85 | 115 |
| Contact Resistance (Ω) | 48 | 42 | 58 |
| Defect Density (cm⁻²) | ~0.1 | 0.0 | 0.50 |

Excursion events are embedded at lots 28–30, 57–59, 73–75, 100–101.

---

## 🛠 Tech Stack
- **Streamlit** — Dashboard framework
- **Plotly** — Interactive charts
- **scikit-learn** — PCA, Isolation Forest, StandardScaler
- **NumPy / Pandas** — Data manipulation
- **statsmodels** — OLS trendline (via Plotly express)

---

*Built for ADI Campus Drive 2026 — Yield/Manufacturing Engineering Role*
