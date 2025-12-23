# Workflow-CI

Repositori ini menggabungkan pipeline pelatihan model Water Potability berbasis MLflow dengan monitoring metrik menggunakan Prometheus + Grafana. Eksperimen dilacak ke DagsHub sehingga setiap training menghasilkan artefak, metrik, serta model yang siap dipromosikan ke tahap berikutnya.

## Ikhtisar
- Dataset: water potability (fitur kualitas air) yang sudah dibersihkan dan dipisah train/test.
- Model: `RandomForestClassifier` dengan `RandomizedSearchCV` untuk tuning hyperparameter.
- Tracking: MLflow (remote tracking ke DagsHub, menyimpan metrik, artefak, dan model).
- Observability: exporter custom Python yang memancarkan 10 metrik ke Prometheus dan divisualisasikan di Grafana.

## Struktur Repositori
```
Workflow-CI/
|-- MLProject/
|   |-- MLProject            # Definisi MLflow Project (entry point: modelling.py)
|   |-- conda.yaml           # Lingkungan Conda untuk menjalankan eksperimen
|   |-- modelling.py         # Script utama training + logging MLflow
|   `-- water_potability_preprocessing/
|       |-- train_clean_automated.csv
|       `-- test_clean_automated.csv
|-- Monitoring/
|   |-- prometheus.yml       # Target scrape Prometheus
|   `-- prometheus_exporter.py
|-- requirements.txt         # Alternatif instalasi via pip
`-- README.md
```

## Dataset dan Preprocessing
Dataset telah dinormalisasi dan dibersihkan terlebih dahulu. File train/test berada di `MLProject/water_potability_preprocessing/`. Bila ingin mengganti dataset, pastikan kolom target bernama `Potability` dan struktur fitur konsisten antara train/test.

## Lingkungan & Dependensi
### Opsi 1 - Conda
```bash
conda env create -f MLProject/conda.yaml
conda activate mlflow-env
```

### Opsi 2 - Pip/venv
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Menjalankan Eksperimen MLflow
1. Masuk ke folder proyek MLflow:
	```bash
	cd MLProject
	```
2. Siapkan kredensial DagsHub agar MLflow bisa menulis ke remote registry. Contoh (PowerShell):
	```bash
	setx MLFLOW_TRACKING_USERNAME your_dagshub_username
	setx MLFLOW_TRACKING_PASSWORD your_dagshub_token
	```
3. Jalankan training secara langsung:
	```bash
	python modelling.py
	```
	atau manfaatkan definisi MLflow Project:
	```bash
	mlflow run .
	```

Script akan:
- Membaca dataset train/test lalu memisahkan fitur/label.
- Melakukan `RandomizedSearchCV` (5 iterasi, 3-fold CV) untuk mencari kombinasi `n_estimators`, `max_depth`, `min_samples_split`, dan `min_samples_leaf` terbaik.
- Menghitung Accuracy, Precision, Recall, dan F1 di set test.
- Mengunggah metrik, confusion matrix, feature importance, dan model terseri ke DagsHub melalui MLflow.
- Menyimpan `run_id` terakhir di `MLProject/run_id.txt` untuk memudahkan referensi di pipeline selanjutnya.

## Monitoring dan Observabilitas
### 1. Menjalankan Prometheus Exporter
```bash
cd Monitoring
python prometheus_exporter.py
```
Exporter membuka port 8000 dan memancarkan 10 metrik dummy yang berguna untuk menggambarkan health pipeline:
- `system_cpu_usage_percent`, `system_memory_usage_bytes`, `system_disk_usage_percent`
- `model_accuracy_score`, `model_prediction_total`, `model_prediction_errors_total`
- `request_latency_seconds`, `process_time_seconds`
- `active_user_sessions`, `api_calls_rate`

### 2. Prometheus
`Monitoring/prometheus.yml` sudah dikonfigurasi untuk men-scrape `host.docker.internal:8000` (memudahkan akses dari container ke host Windows). Jalankan Prometheus via Docker:
```powershell
docker run -d --name prometheus -p 9090:9090 `
  -v ${PWD}/Monitoring/prometheus.yml:/etc/prometheus/prometheus.yml `
  prom/prometheus
```

### 3. Grafana
```powershell
docker run -d -p 3000:3000 --name grafana grafana/grafana
```
Tambahkan data source Prometheus (URL: `http://host.docker.internal:9090`) lalu buat dashboard untuk 10 metrik di atas. Anda dapat menambahkan alert berbasis `Prediction Error` atau `Latency` untuk mengawasi regresi model.

