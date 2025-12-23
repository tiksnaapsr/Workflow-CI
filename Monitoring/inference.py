import requests
import json

# URL Endpoint V2 
url = "http://localhost:5000/v2/models/water-potability-model/infer"

# Data Asli (9 Fitur per baris)
# ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity
input_data = [
    [7.0, 200.0, 20000.0, 7.0, 300.0, 400.0, 15.0, 80.0, 4.0],  # Baris 1
    [5.0, 150.0, 15000.0, 5.0, 200.0, 300.0, 10.0, 60.0, 3.0]   # Baris 2
]

# FORMAT V2 
# Flatten list 2D menjadi 1D list panjang
# Lalu memberi tahu server bentuk aslinya lewat "shape"
flat_data = [x for row in input_data for x in row] 

payload = {
    "inputs": [
        {
            "name": "input-0",       # Nama input default MLflow
            "shape": [2, 9],         # [Jumlah Baris, Jumlah Kolom] -> [2 baris, 9 fitur]
            "datatype": "FP64",      # Tipe data (Float 64-bit)
            "data": flat_data        # Data yang sudah didatarkan
        }
    ]
}

headers = {"Content-Type": "application/json"}

print(f"Mengirim data V2 ke {url}...")

try:
    response = requests.post(url, json=payload, headers=headers)
    
    if response.status_code == 200:
        print("\nSUKSES! Model Merespon.")
        print("Hasil Prediksi:", response.json())
        print("\nSelamat! Submission Anda Selesai.")
    else:
        print(f"\nGAGAL. Status Code: {response.status_code}")
        print("Pesan Error:", response.text)

except Exception as e:
    print(f"\nError Koneksi: {e}")