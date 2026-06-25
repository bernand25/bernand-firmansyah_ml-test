# AI Engineer Technical Test – Travel Time Prediction

## Project Overview

Proyek ini bertujuan membangun pipeline awal untuk prediksi travel time antar segmen halte menggunakan data historis GPS bus TransJakarta.

Dataset berisi informasi perjalanan bus pada level segmen, termasuk waktu tempuh aktual (`traveling_time_sec`), rata-rata historis (`average_time_sec`), metadata rute, urutan halte, dan waktu kedatangan.

---

## Dataset Issues Identified

Berdasarkan hasil Exploratory Data Analysis (EDA), ditemukan tiga isu utama:

1. Trip incompleteness (hilangnya segmen dalam suatu trip)
2. Distribusi travel time yang sangat skewed
3. Sparsity pada kombinasi route-segment-hour tertentu

---

## Feature Engineering

Fitur tambahan yang dibuat:

| Feature           | Deskripsi                     |
| ----------------- | ----------------------------- |
| hour              | Jam kedatangan                |
| day_of_week       | Hari dalam minggu             |
| is_weekend        | Indikator akhir pekan         |
| is_peak_hour      | Indikator jam sibuk           |
| prev_travel_time  | Travel time segmen sebelumnya |
| rolling_mean_3    | Rata-rata 3 segmen sebelumnya |
| segment_frequency | Frekuensi kemunculan segmen   |
| is_gap_suspected  | Indikator trip incompleteness |

---

## Assumptions and Thresholds

### 1. Trip Incompleteness

Gap sequence dianggap terjadi jika:

```python
seq_diff > 1
```

dimana:

```python
seq_diff =
current_stop_sequence - previous_stop_sequence
```

Jika gap terdeteksi maka:

```python
is_gap_suspected = 1
```

---

### 2. Peak Hour Definition

Jam sibuk didefinisikan sebagai:

* Pagi : 06.00 – 09.00
* Sore : 16.00 – 19.00

Implementasi:

```python
hour in [6,7,8,9,16,17,18,19]
```

---

### 3. Skewness Handling

Karena distribusi target sangat right-skewed, dilakukan transformasi:

```python
y_log = np.log1p(traveling_time_sec)
```

Prediksi kemudian dikembalikan ke skala asli menggunakan:

```python
np.expm1()
```

---

### 4. Missing Sequential Features

Nilai kosong pada:

* prev_travel_time
* rolling_mean_3

diisi menggunakan:

```python
average_time_sec
```

untuk mempertahankan informasi historis.

---

## Model

Model utama yang digunakan:

* XGBoost Regressor

Parameter:

```python
XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)
```

---

## Evaluation Metrics

Evaluasi dilakukan menggunakan:

* MAE (Mean Absolute Error)
* RMSE (Root Mean Squared Error)
* MAPE (Mean Absolute Percentage Error)
* Trip-Level MAE

---

## Result

### Segment-Level Performance

| Metric | Value       |
| ------ | ----------- |
| MAE    | 221.93 sec  |
| RMSE   | 6033.10 sec |
| MAPE   | 28.04%      |

### Trip-Level Performance

| Metric   | Value          |
| -------- | -------------- |
| Trip MAE | 786,249.27 sec |

---

## Feature Importance

Fitur dengan kontribusi terbesar:

1. average_time_sec
2. rolling_mean_3
3. prev_travel_time

Hal ini menunjukkan bahwa informasi historis dan pola sekuensial merupakan faktor dominan dalam prediksi travel time.

---

## Recommended Production Architecture

Direkomendasikan pendekatan ensemble:

* XGBoost untuk feature tabular
* LSTM untuk pola sekuensial antar segmen

Strategi ensemble:

```text
Final Prediction =
0.7 × XGBoost +
0.3 × LSTM
```

---

## How to Run

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost pyarrow joblib
```

### Run Notebook

Buka:

```text
AI_Engineer_Travel_Time_Prediction.ipynb
```

Upload file:

```text
AI_Engineer_dataset.parquet
```

Kemudian jalankan seluruh cell secara berurutan.

---

## Output Files

Notebook menghasilkan:

```text
processed_dataset.csv
xgboost_travel_time_model.pkl
```
