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
    "berat_badan_kg": 12.5,
    "tinggi_badan_cm": 85.0,
    "lingkar_kepala_cm": 48.0,
    "lila_cm": 14.5,
    "tren_bb_bulan_lalu": "naik",
    "usia_kehamilan_lahir": 38,
    "berat_lahir_kg": 3.2,
    "panjang_lahir_cm": 49.0,
    "is_bblr": "tidak",
    "is_prematur": "tidak",
    "is_imd": "ya",
    "is_komplikasi_lahir": "tidak",
    "jenis_komplikasi_lahir": "Tidak Ada Komplikasi",
    "tinggi_ibu_cm": 155.0,
    "berat_ibu_kg": 55.0,
    "tinggi_ayah_cm": 165.0,
    "berat_ayah_kg": 65.0,
    "status_gizi_ibu_hamil": "Normal",
    "is_anemia_ibu": "tidak",
    "is_hamil_muda_u20": "tidak",
    "jarak_kelahiran": "lebih dari 2 tahun",
    "is_hipertensi_gestasional": "tidak",
    "is_diabetes_gestasional": "tidak",
    "is_infeksi_kehamilan": "tidak",
    "is_suplemen_kehamilan": "ya",
    "is_hamil_lagi": "tidak",
    "frekuensi_suplemen_minggu": "setiap hari",
    "jenis_suplemen_ibu": "TTD, Kalsium",
    "is_ttd_90_tablet": "ya",
    "is_asi_eksklusif": "ya",
    "usia_mulai_mpasi": 6,
    "is_mpasi_hewani": "ya",
    "frekuensi_makan_utama": 3,
    "is_susu_non_asi": "tidak",
    "frekuensi_susu_non_asi": 0,
    "terakhir_vitamin_a": "Februari",
    "is_tablet_besi_anak": "ya",
    "is_obat_cacing_anak": "tidak",
    "is_intervensi_gizi": "ya",
    "jenis_intervensi_gizi": "PMT",
    "riwayat_vaksinasi": "Lengkap",
    "is_sakit_2_minggu": "tidak",
    "jenis_penyakit_balita": "Tidak Sakit",
    "konsumsi_asi_h_1": "ya",
    "konsumsi_karbohidrat_h_1": "ya",
    "konsumsi_kacangan_h_1": "ya",
    "konsumsi_susu_hewani_h_1": "tidak",
    "is_susu_murni_100": "tidak",
    "konsumsi_daging_ikan_h_1": "ya",
    "konsumsi_telur_h_1": "ya",
    "konsumsi_vit_a_h_1": "ya",
    "konsumsi_buah_sayur_lain_h_1": "ya",
    "is_konsumsi_manis_berlebih": "tidak",
    "is_pernah_pmt": "ya",
    "is_pernah_rawat_inap": "tidak",
    "jam_tidur_harian": 12,
    "durasi_aktivitas_luar": 2,
    "tingkat_aktivitas_anak": "Aktif",
    "is_ibu_bekerja": "tidak",
    "skor_pengetahuan_ibu": 80,
    "skor_pola_asuh_makan": 90,
    "is_bpjs": "ya",
    "is_perokok_di_rumah": "tidak",
    "sumber_air_minum": "PDAM",
    "kualitas_air_minum": "Terlindungi",
    "jenis_sanitasi": "Jamban Sendiri",
    "kebersihan_lingkungan": "Bersih",
    "kebiasaan_cuci_tangan": "Rutin",
    "akses_faskes": "Mudah",
    "frekuensi_posyandu_bulan": 1,
    "is_penyakit_bawaan": "tidak",
    "is_baby_blues": "tidak",
    "is_gejala_depresi": "tidak",
    "pendidikan_ibu": "SMA",
    "pendidikan_ayah": "SMA",
    "is_pernah_penyuluhan_gizi": "ya",
    "frekuensi_ikut_kelas_ibu": 4,
    "is_paham_makanan_sehat": "ya",
    "pekerjaan_kepala_keluarga": "Buruh",
    "jumlah_art": 4,
    "pendapatan_bulanan": 3000000,
    "jarak_akses_pangan": "Dekat",
    "is_pantangan_makan": "tidak",
    "Siapa_yang_biasanya_menentukan_makanan_apa_yang_dimakan_oleh_anak_di_rumah_": "Ibu",
    "Tanggal_Lahir_Balita_kader": "2023-01-01"
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
