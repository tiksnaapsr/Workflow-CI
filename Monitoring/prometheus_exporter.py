import time
import random
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Summary

# --- DEFINISI 10 METRIKS ---

# System Metrics
CPU_USAGE = Gauge('system_cpu_usage_percent', 'Persentase penggunaan CPU')
MEMORY_USAGE = Gauge('system_memory_usage_bytes', 'Penggunaan memori dalam bytes')
DISK_USAGE = Gauge('system_disk_usage_percent', 'Persentase penggunaan Disk')

# Model Performance Metrics
MODEL_ACCURACY = Gauge('model_accuracy_score', 'Akurasi model pada batch terakhir')
PREDICTION_COUNT = Counter('model_prediction_total', 'Total prediksi yang dilakukan')
PREDICTION_ERROR = Counter('model_prediction_errors_total', 'Total error saat prediksi')

# Latency Metrics
LATENCY = Histogram('request_latency_seconds', 'Distribusi latency request')
PROCESS_TIME = Summary('process_time_seconds', 'Waktu pemrosesan data')

# Business/User Metrics
ACTIVE_USERS = Gauge('active_user_sessions', 'Jumlah user aktif saat ini')
API_CALLS_PER_MINUTE = Gauge('api_calls_rate', 'Rate panggilan API per menit')

def simulate_metrics():
    """Fungsi untuk mengubah angka secara acak agar grafik bergerak"""
    while True:
        # Update System Metrics
        CPU_USAGE.set(random.uniform(10, 90))  # CPU antara 10% - 90%
        MEMORY_USAGE.set(random.uniform(1000000, 5000000))
        DISK_USAGE.set(random.uniform(30, 40))

        # Update Model Metrics
        MODEL_ACCURACY.set(random.uniform(0.80, 0.95))
        PREDICTION_COUNT.inc(random.randint(1, 5))
        
        # Simulasi Error (sesekali error naik)
        if random.random() > 0.9:
            PREDICTION_ERROR.inc()

        # Update Latency
        LATENCY.observe(random.uniform(0.1, 0.5))
        PROCESS_TIME.observe(random.uniform(0.05, 0.2))

        # Update User Metrics
        ACTIVE_USERS.set(random.randint(50, 200))
        API_CALLS_PER_MINUTE.set(random.randint(10, 60))

        time.sleep(2) # Update setiap 2 detik

if __name__ == '__main__':
    # Jalankan server metrics di port 8000
    start_http_server(8000)
    print("Prometheus Exporter berjalan di port 8000...")
    print("Menghasilkan 10 Metriks Dummy untuk Grafana...")
    simulate_metrics()