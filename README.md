# 🔬 Semiconductor Yield & WAT Analysis Dashboard

A **production-grade Streamlit dashboard** for analyzing semiconductor manufacturing data using **Statistical Process Control (SPC), yield modeling, process capability (Cpk), and machine learning-based root cause analysis**.

---

## 🚀 Overview

This project simulates real-world workflows in semiconductor manufacturing, focusing on:

* Monitoring process stability using SPC charts
* Tracking lot-wise yield trends
* Identifying process excursions and anomalies
* Evaluating process capability (Cpk)
* Performing data-driven root cause analysis

---

## 🧠 Key Features

### 📈 Statistical Process Control (SPC)

* X-bar chart (subgroup mean monitoring)
* R chart (process variation tracking)
* CUSUM chart (early drift detection)
* Automatic out-of-control detection

---

### 📉 Yield Analysis

* Lot-by-lot yield trend visualization
* Yield distribution across lots
* Defect density (D₀) vs yield relationship (Murphy’s model)
* Yield breakdown by process step

---

### 🔍 Root Cause Analysis (Machine Learning)

* PCA (Principal Component Analysis) for dimensionality reduction
* Isolation Forest for anomaly detection
* Correlation heatmap of WAT parameters
* Feature contribution insights

---

### 📊 Process Capability (Cpk)

* Cpk calculation for key WAT parameters
* Visualization against LSL/USL limits
* Capability classification:

  * ✅ Capable (Cpk ≥ 1.33)
  * ⚠ Marginal (1.0–1.33)
  * ❌ Not Capable (Cpk < 1.0)

---

## ⚙️ How to Run

### 1️⃣ Prerequisites

Ensure Python 3.9+ is installed:

```
python --version
```

---

### 2️⃣ Create Virtual Environment

```
python -m venv venv
```

Activate:

* **Windows:** `venv\Scripts\activate`
* **Mac/Linux:** `source venv/bin/activate`

---

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

---

### 4️⃣ Run the Dashboard

```
streamlit run app.py
```

Access locally at:

```
http://localhost:8501
```

---

## 📁 Project Structure

```
semiconductor-yield-spc-dashboard/
├── app.py
├── requirements.txt
└── README.md
```

---

## 🧪 Dashboard Modules

| Module              | Description                                                |
| ------------------- | ---------------------------------------------------------- |
| 📈 SPC Charts       | X-bar, R-chart, and CUSUM with excursion detection         |
| 📉 Yield Analysis   | Yield trends, distributions, and defect-based modeling     |
| 🔍 Root Cause & PCA | PCA visualization, anomaly detection, correlation analysis |
| 📊 Cpk & Health     | Process capability metrics and parameter health monitoring |

---

## 📊 Simulated Dataset

The dataset is synthetically generated to mimic semiconductor fab conditions, including:

* WAT parameters (Vt, Idsat, resistances)
* Defect density (D₀)
* Lot-level yield
* Embedded process excursions across selected lots

---

## 🛠 Tech Stack

* **Streamlit** — Interactive dashboard
* **Plotly** — Data visualization
* **scikit-learn** — PCA, Isolation Forest
* **NumPy & Pandas** — Data processing
* **SciPy / Statsmodels** — Statistical analysis

---

## 🎯 Applications

This project demonstrates concepts relevant to:

* Semiconductor Yield Engineering
* Process Control & SPC Monitoring
* Manufacturing Data Analytics
* Root Cause & Failure Analysis

---

## 🚀 Future Improvements

* Integration with real fab datasets
* Real-time data streaming
* Advanced anomaly detection (deep learning)
* Cloud deployment (Streamlit Cloud / AWS / GCP)

---

## 👩‍💻 Author

**Jiya Parmar**
Robotics & AI Engineer | Interested in Semiconductor Manufacturing & Intelligent Systems

