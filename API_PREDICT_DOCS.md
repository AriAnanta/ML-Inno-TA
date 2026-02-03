# Dokumentasi API Prediksi Status Gizi Balita


## Instalasi

1. Masuk ke folder backend `ml`.
2. Install dependencies dari `requirements.txt`.

## Jalankan Server

Jalankan aplikasi FastAPI dengan perintah:

```bash
uvicorn app:app --reload
```

atau

```bash
python -m uvicorn app:app --reload
```

## Endpoint

### 1) POST `/predict`
Model lomba (input sederhana dari kader).

### 2) POST `/predict-ta`
Model TA (input gabungan kader + orang tua).

## Request Headers
- `Content-Type: application/json`

## Payload (Request Body)

### A) Payload `/predict`
Field wajib:
- `Tgl_Lahir` (string, format `YYYY-MM-DD`)
- `Tanggal_Pengukuran` (string, format `YYYY-MM-DD`)
- `Jenis_Kelamin_Balita` (string, `Laki-Laki` atau `Perempuan`)
- `Berat` (number, kg)
- `Tinggi` (number, cm)

Field opsional:
- `LiLA` (number, cm)

### Contoh Payload
```json
{
  "Tgl_Lahir": "2022-05-10",
  "Tanggal_Pengukuran": "2024-12-01",
  "Jenis_Kelamin_Balita": "Laki-Laki",
  "Berat": 11.2,
  "Tinggi": 83.5,
  "LiLA": 14.0
}
```

### B) Payload `/predict-ta`
Payload dibungkus dalam object `data`. Contoh ringkas:

```json
{
  "data": {
    "umur_balita_bulan": 24,
    "jenis_kelamin": "Laki-Laki",
    "berat_badan_kg": 11.2,
    "tinggi_badan_cm": 83.5,
    "lila_cm": 14.0,
    "Tanggal_Lahir_Balita_kader": "2022-05-10",
    "is_bblr": "Tidak",
    "is_prematur": "Tidak",
    "jenis_suplemen_ibu": "Tablet tambah darah (zat besi)",
    "riwayat_vaksinasi": "BCG, Polio Tetes 1"
  }
}
```

Catatan:
- Field lainnya mengikuti struktur `example_request.json` di folder `ml_ta`.
- Field yang tidak dikirim akan diperlakukan sebagai default (0/kosong).



## Response
### Sukses (200 OK)
Response berisi hasil prediksi untuk 3 target:
- `BB/TB`
- `BB/U`
- `TB/U`

### Contoh Response
```json
{
  "BB/TB": "Gizi Baik",
  "BB/U": "Berat Badan Normal",
  "TB/U": "Normal"
}
```

### Sukses (200 OK) untuk `/predict-ta`
Response berisi `predictions`:

```json
{
  "predictions": {
    "status_gizi_bbtb": {"class_id": 2, "label": "Gizi Baik", "proba": [0.1, 0.2, 0.6, 0.1]},
    "status_gizi_bbu": {"class_id": 2, "label": "Berat Badan Normal", "proba": [0.1, 0.2, 0.6, 0.1]},
    "status_gizi_tbu": {"class_id": 2, "label": "Normal", "proba": [0.1, 0.2, 0.6, 0.1]}
  }
}
```

## Error Response
### 500 Internal Server Error
```json
{
  "detail": "<pesan error>"
}
```

## Catatan Penting
- Pastikan format tanggal valid.
- Satuan: `Berat` dalam kg, `Tinggi` dan `LiLA` dalam cm.
- Untuk `/predict-ta`, lihat `example_request.json` sebagai referensi field lengkap.
