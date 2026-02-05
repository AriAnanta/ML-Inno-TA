import json
import re
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd
import numpy as np
import joblib
from datetime import date, datetime
from fastapi import FastAPI, HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator, create_model
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Security Configuration ---
API_KEY_NAME = "X-API-KEY"
API_KEY = os.getenv("API_KEY", "growell123")  # Gunakan default hanya jika .env tidak ada
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=True)

async def get_api_key(header_key: str = Security(api_key_header)):
    if header_key == API_KEY:
        return header_key
    raise HTTPException(
        status_code=403, detail="Engga ada API KEY nya"
    )

# Try importing pygrowup2 first, fallback to pygrowup
try:
    from pygrowup2 import Calculator
    print("Using pygrowup2 for Z-score calculation.")
except ImportError:
    try:
        from pygrowup import Calculator
        print("Using pygrowup for Z-score calculation.")
    except ImportError:
        print("Error: Neither pygrowup2 nor pygrowup found. Z-score calculation will not be available.")
        Calculator = None # Set to None if neither is found

app = FastAPI()

# Allow frontend origin (Vite dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# --- 1. Load Models and Preprocessing Components ---

# Load best models for each target
try:
    model_bbtb = joblib.load('best_model_deployment_BB_TB.joblib')
    model_bbu = joblib.load('best_model_deployment_BB_U.joblib')
    model_tbu = joblib.load('best_model_deployment_TB_U.joblib')
    print("Successfully loaded all best models.")
except Exception as e:
    raise RuntimeError(f"Error loading models: {e}. Ensure model files are present.")

# Load scaler and mapping dictionaries
try:
    scaler = joblib.load('scaler.joblib')
    map_gizi_bbtb = joblib.load('map_gizi_bbtb.joblib')
    map_gizi_bbu = joblib.load('map_gizi_bbu.joblib')
    map_gizi_tbu = joblib.load('map_gizi_tbu.joblib')
    map_jk = joblib.load('map_jk.joblib')
    print("Successfully loaded scaler and mapping dictionaries.")
except Exception as e:
    raise RuntimeError(f"Error loading preprocessing components: {e}. Ensure joblib files are present.")

# Create reverse mapping for human-readable output
reverse_map_bbtb = {v: k for k, v in map_gizi_bbtb.items()}
reverse_map_bbu = {v: k for k, v in map_gizi_bbu.items()}
reverse_map_tbu = {v: k for k, v in map_gizi_tbu.items()}

# --- 1b. Load TA Models and Preprocessing Components ---
TA_BASE_DIR = Path(__file__).resolve().parent
TA_MODELS_DIR = TA_BASE_DIR / "models_enhanced_final"
TA_SHARED_DIR = TA_BASE_DIR / "models"

TA_MODEL_PATHS = {
    "status_gizi_bbtb": TA_MODELS_DIR / "best_model_status_gizi_bbtb.joblib",
    "status_gizi_bbu": TA_MODELS_DIR / "best_model_status_gizi_bbu.joblib",
    "status_gizi_tbu": TA_MODELS_DIR / "best_model_status_gizi_tbu.joblib",
}

TA_MODEL_FEATURES_PATH = TA_SHARED_DIR / "model_features.json"
TA_SCALER_FEATURES_PATH = TA_SHARED_DIR / "scaler_features.json"
TA_SCALER_PATH = TA_SHARED_DIR / "scaler.joblib"
TA_LABEL_ENCODERS_PATH = TA_SHARED_DIR / "label_encoders.joblib"
TA_ENCODER_CLASSES_PATH = TA_SHARED_DIR / "encoder_classes.json"

TA_COLUMN_ALIASES = {
    "Siapa yang biasanya menentukan makanan apa yang dimakan oleh anak di rumah? ":
        "Siapa_yang_biasanya_menentukan_makanan_apa_yang_dimakan_oleh_anak_di_rumah_",
}

TA_RAW_ALIASES = {
    "umur_balita_bulan": "umur_balita",
    "umur_balita": "umur_balita",
    "jenis_komplikasi_lahir": "kategori_komplikasi_lahir",
    "jenis_penyakit_balita": "kategori_penyakit_balita",
    "jenis_intervensi_gizi": "kategori_intervensi_utama",
    "riwayat_vaksinasi": "kategori_vaksinasi",
    "pekerjaan_kepala_keluarga": "kategori_pekerjaan_kk",
    "terakhir_vitamin_a": "kategori_vitamin_a",
    "usia_mulai_mpasi": "usia_mulai_mpasi",
    "Siapa_yang_biasanya_menentukan_makanan_apa_yang_dimakan_oleh_anak_di_rumah_":
        "Siapa_yang_biasanya_menentukan_makanan_apa_yang_dimakan_oleh_anak_di_rumah_",
}

TA_LABEL_MAPS = {
    "status_gizi_tbu": {0: "Sangat Pendek", 1: "Pendek", 2: "Normal", 3: "Tinggi"},
    "status_gizi_bbtb": {0: "Gizi Buruk", 1: "Gizi Kurang", 2: "Gizi Baik", 3: "Gizi Lebih"},
    "status_gizi_bbu": {
        0: "Berat Badan Sangat Kurang",
        1: "Berat Badan Kurang",
        2: "Berat Badan Normal",
        3: "Berat Badan Lebih",
    },
}


def _ta_load_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


ta_model_features = _ta_load_json(TA_MODEL_FEATURES_PATH)
ta_scaler_features = _ta_load_json(TA_SCALER_FEATURES_PATH)
ta_encoder_classes = _ta_load_json(TA_ENCODER_CLASSES_PATH)

ta_label_encoders: Dict[str, Any] = joblib.load(TA_LABEL_ENCODERS_PATH)

# Ensure label encoder keys exist for both alias and canonical forms
for alias, canonical in TA_COLUMN_ALIASES.items():
    if alias in ta_label_encoders and canonical not in ta_label_encoders:
        ta_label_encoders[canonical] = ta_label_encoders[alias]
    if canonical in ta_label_encoders and alias not in ta_label_encoders:
        ta_label_encoders[alias] = ta_label_encoders[canonical]

ta_scaler = joblib.load(TA_SCALER_PATH)
ta_models: Dict[str, Any] = {target: joblib.load(path) for target, path in TA_MODEL_PATHS.items()}

ta_feature_order = None
for _model in ta_models.values():
    if hasattr(_model, "feature_names_in_"):
        ta_feature_order = list(_model.feature_names_in_)
        break
if ta_feature_order is None:
    ta_feature_order = ta_model_features

# --- 2. Pydantic Input Model ---
class BalitaInput(BaseModel):
    Tgl_Lahir: date = Field(..., description="Tanggal lahir balita (YYYY-MM-DD)")
    Tanggal_Pengukuran: date = Field(..., description="Tanggal pengukuran (YYYY-MM-DD)")
    Jenis_Kelamin_Balita: str = Field(..., description="Jenis kelamin balita (Laki-Laki/Perempuan)")
    Berat: float = Field(..., description="Berat badan saat ini (kg)")
    Tinggi: float = Field(..., description="Tinggi badan saat ini (cm)")
    LiLA: Optional[float] = Field(None, description="Lingkar Lengan Atas (cm), opsional")

    @validator('Jenis_Kelamin_Balita', pre=True)
    def validate_jenis_kelamin_balita(cls, v):
        if v is None or str(v).strip() == '':
            raise ValueError('Jenis_Kelamin_Balita wajib diisi (Laki-Laki/Perempuan).')
        return v


# --- 2b. Pydantic Input Model for TA ---
TA_RAW_INPUT_FIELDS = [
    "umur_balita_bulan",
    "jenis_kelamin",
    "berat_badan_kg",
    "tinggi_badan_cm",
    "lingkar_kepala_cm",
    "lila_cm",
    "tren_bb_bulan_lalu",
    "usia_kehamilan_lahir",
    "berat_lahir_kg",
    "panjang_lahir_cm",
    "is_bblr",
    "is_prematur",
    "is_imd",
    "is_komplikasi_lahir",
    "jenis_komplikasi_lahir",
    "tinggi_ibu_cm",
    "berat_ibu_kg",
    "tinggi_ayah_cm",
    "berat_ayah_kg",
    "status_gizi_ibu_hamil",
    "is_anemia_ibu",
    "is_hamil_muda_u20",
    "jarak_kelahiran",
    "is_hipertensi_gestasional",
    "is_diabetes_gestasional",
    "is_infeksi_kehamilan",
    "is_suplemen_kehamilan",
    "is_hamil_lagi",
    "frekuensi_suplemen_minggu",
    "jenis_suplemen_ibu",
    "is_ttd_90_tablet",
    "is_asi_eksklusif",
    "usia_mulai_mpasi",
    "is_mpasi_hewani",
    "frekuensi_makan_utama",
    "is_susu_non_asi",
    "frekuensi_susu_non_asi",
    "terakhir_vitamin_a",
    "is_tablet_besi_anak",
    "is_obat_cacing_anak",
    "is_intervensi_gizi",
    "jenis_intervensi_gizi",
    "riwayat_vaksinasi",
    "is_sakit_2_minggu",
    "jenis_penyakit_balita",
    "konsumsi_asi_h_1",
    "konsumsi_karbohidrat_h_1",
    "konsumsi_kacangan_h_1",
    "konsumsi_susu_hewani_h_1",
    "is_susu_murni_100",
    "konsumsi_daging_ikan_h_1",
    "konsumsi_telur_h_1",
    "konsumsi_vit_a_h_1",
    "konsumsi_buah_sayur_lain_h_1",
    "is_konsumsi_manis_berlebih",
    "is_pernah_pmt",
    "is_pernah_rawat_inap",
    "jam_tidur_harian",
    "durasi_aktivitas_luar",
    "tingkat_aktivitas_anak",
    "is_ibu_bekerja",
    "skor_pengetahuan_ibu",
    "skor_pola_asuh_makan",
    "is_bpjs",
    "is_perokok_di_rumah",
    "sumber_air_minum",
    "kualitas_air_minum",
    "jenis_sanitasi",
    "kebersihan_lingkungan",
    "kebiasaan_cuci_tangan",
    "akses_faskes",
    "frekuensi_posyandu_bulan",
    "is_penyakit_bawaan",
    "is_baby_blues",
    "is_gejala_depresi",
    "pendidikan_ibu",
    "pendidikan_ayah",
    "is_pernah_penyuluhan_gizi",
    "frekuensi_ikut_kelas_ibu",
    "is_paham_makanan_sehat",
    "pekerjaan_kepala_keluarga",
    "jumlah_art",
    "pendapatan_bulanan",
    "jarak_akses_pangan",
    "is_pantangan_makan",
    "Siapa_yang_biasanya_menentukan_makanan_apa_yang_dimakan_oleh_anak_di_rumah_",
    "Tanggal_Lahir_Balita_kader",
]

TA_RawInput = create_model(
    "TA_RawInput",
    **{
        name: (Optional[Union[float, int, str]], Field(default=None))
        for name in TA_RAW_INPUT_FIELDS
    },
)


class TAPredictRequest(BaseModel):
    data: TA_RawInput

    class Config:
        json_schema_extra = {
            "example": {
                "data": {key: 0 for key in TA_RAW_INPUT_FIELDS}
            }
        }


MEDIAN_BB_TB_RATIO = 0.13575

SCALE_COLUMNS = [
    'Berat', 'Tinggi', 'LiLA',
    'Age_Days', 'Age_Months',
    'BB_TB_Ratio', 'Berat_x_Tinggi', 'LiLA_x_Berat',
    'ZS BB/U', 'ZS TB/U', 'ZS BB/TB'
]

# Nama kolom yang diharapkan oleh model (XGBoost)
MODEL_FEATURES = [
    'JK', 'Berat', 'Tinggi', 'LiLA', 'Age_Days', 'Age_Months', 
    'BB_TB_Ratio', 'Berat_x_Tinggi', 'LiLA_x_Berat', 
    'ZS_BBU', 'ZS_TBU', 'ZS_BBTB'
]

def preprocess_input(data: BalitaInput) -> pd.DataFrame:
    # Convert Pydantic model to DataFrame
    input_df = pd.DataFrame([data.model_dump()])

    # Type conversion for dates
    input_df['Tgl_Lahir'] = pd.to_datetime(input_df['Tgl_Lahir'])
    input_df['Tanggal_Pengukuran'] = pd.to_datetime(input_df['Tanggal_Pengukuran'])

    # --- Feature Engineering (mirroring training notebook) ---

    # Age (days & months)
    input_df['Age_Days'] = (input_df['Tanggal_Pengukuran'] - input_df['Tgl_Lahir']).dt.days
    input_df['Age_Months'] = (input_df['Age_Days'] / 30.4375).round(2)
    input_df.loc[input_df['Age_Days'] < 0, ['Age_Days', 'Age_Months']] = np.nan

    # Map Jenis Kelamin (JK) and create 'sex' for pygrowup
    def _normalize_gender(val):
        if pd.isna(val):
            return None
        s = str(val).strip().lower()
        if s in ['l', 'laki-laki', 'laki laki', 'laki', 'Laki-Laki']:
            return 'L'
        if s in ['p', 'perempuan', 'wanita', 'Perempuan']:
            return 'P'
        return str(val).strip()

    def _map_gender_to_jk(val):
        if val is None:
            return np.nan
        # Direct match
        if val in map_jk:
            return map_jk[val]
        # Try normalized L/P
        norm = _normalize_gender(val)
        if norm in map_jk:
            return map_jk[norm]
        # Try common Indonesian labels
        label_map = {'Laki-Laki': 'L', 'Laki laki': 'L', 'Perempuan': 'P'}
        if val in label_map and label_map[val] in map_jk:
            return map_jk[label_map[val]]
        # Try case-insensitive matching against map_jk keys
        lower_keys = {str(k).strip().lower(): k for k in map_jk.keys()}
        val_lower = str(val).strip().lower()
        if val_lower in lower_keys:
            return map_jk[lower_keys[val_lower]]
        return np.nan

    input_df['JK'] = input_df['Jenis_Kelamin_Balita'].apply(_map_gender_to_jk)
    # Fallback jika map_jk tidak cocok dengan nilai yang masuk
    if input_df['JK'].isna().any():
        normalized = input_df['Jenis_Kelamin_Balita'].apply(_normalize_gender)
        input_df['JK'] = normalized.map({'L': 1, 'P': 0})
    input_df['sex'] = input_df['JK'].map({1: 'M', 0: 'F'})

    if input_df['JK'].isna().any():
        bad_value = input_df['Jenis_Kelamin_Balita'].iloc[0]
        raise ValueError(f'Jenis_Kelamin_Balita tidak valid: "{bad_value}". Gunakan Laki-Laki atau Perempuan.')

    # Ratios & interactions
    input_df['BB_TB_Ratio'] = input_df['Berat'] / input_df['Tinggi'].replace(0, np.nan)
    input_df['Berat_x_Tinggi'] = input_df['Berat'] * input_df['Tinggi']
    input_df['LiLA_x_Berat'] = input_df['LiLA'] * input_df['Berat'] if 'LiLA' in input_df.columns else np.nan

    # WHO Z-Score with pygrowup
    if Calculator:
        calc = Calculator()
        def _safe_zscores_api(row):
            if pd.isna(row.get('Age_Months')) or pd.isna(row.get('Berat')) or pd.isna(row.get('Tinggi')) or pd.isna(row.get('sex')):
                return pd.Series([np.nan, np.nan, np.nan])
            try:
                # pygrowup expects sex as 'M' or 'F', weight in kg, height in cm, age in months
                z_wfa = calc.wfa(row['Berat'], row['Age_Months'], row['sex'])
                z_lhfa = calc.lhfa(row['Tinggi'], row['Age_Months'], row['sex'])
                z_wfl = calc.wfl(row['Berat'], row['Age_Months'], row['sex'], row['Tinggi'])
                return pd.Series([z_wfa, z_lhfa, z_wfl])
            except Exception:
                return pd.Series([np.nan, np.nan, np.nan])
        
        # Apply _safe_zscores_api row-wise if all necessary columns exist
        if all(col in input_df.columns for col in ['Age_Months', 'Berat', 'Tinggi', 'sex']):
            input_df[['ZS BB/U', 'ZS TB/U', 'ZS BB/TB']] = input_df.apply(_safe_zscores_api, axis=1)
        else:
            input_df[['ZS BB/U', 'ZS TB/U', 'ZS BB/TB']] = np.nan # Default to nan if essential cols missing
    else:
        input_df[['ZS BB/U', 'ZS TB/U', 'ZS BB/TB']] = np.nan # Default to nan if pygrowup is not available

    # Handle inf/NaN from division and LiLA (LiLA itself can be NaN from input)
    input_df = input_df.replace([np.inf, -np.inf], np.nan).infer_objects(copy=False)
    
    # Impute missing values for 'LiLA_x_Berat' if LiLA was None
    input_df['LiLA_x_Berat'] = input_df['LiLA_x_Berat'].fillna(0) # or a more appropriate median if available

    # Impute BB_TB_Ratio using pre-calculated median
    input_df['BB_TB_Ratio'] = input_df['BB_TB_Ratio'].fillna(MEDIAN_BB_TB_RATIO)

    # Pastikan semua kolom fitur tersedia dan imputasi NaN dengan 0 (fallback)
    for col in SCALE_COLUMNS:
        if col not in input_df.columns:
            input_df[col] = np.nan
    input_df[SCALE_COLUMNS] = input_df[SCALE_COLUMNS].fillna(0.0)

    # Scale numerical features
    scaled_features = scaler.transform(input_df[SCALE_COLUMNS])
    scaled_df = pd.DataFrame(scaled_features, columns=SCALE_COLUMNS)

    final_df = scaled_df.copy()
    
    # Rename ZS columns from spaces/slashes to underscores for the model
    final_df = final_df.rename(columns={
        'ZS BB/U': 'ZS_BBU',
        'ZS TB/U': 'ZS_TBU',
        'ZS BB/TB': 'ZS_BBTB'
    })
    
    final_df['JK'] = input_df['JK'].values[0] # Assuming single row input

    final_df = final_df[MODEL_FEATURES]
    return final_df


# --- 3b. TA Preprocessing Helpers ---
def _ta_normalize_input(data: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in data.items():
        canonical = TA_RAW_ALIASES.get(key, key)
        canonical = TA_COLUMN_ALIASES.get(canonical, canonical)
        normalized[canonical] = value
    return normalized


def _ta_to_number(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace(",", "."))
    except Exception:
        return None


def _ta_map_binary(val: Any) -> int:
    if val is None:
        return 0
    if isinstance(val, (int, float)):
        return 1 if val > 0 else 0
    text = str(val).strip().lower()
    if text in {"ya", "y", "yes", "true", "1", "aktif", "iya"}:
        return 1
    if "ya" in text:
        return 1
    return 0


def _ta_age_months_from_date(value: Any) -> Optional[float]:
    if not value:
        return None
    dob = pd.to_datetime(value, errors="coerce")
    if pd.isna(dob):
        return None
    cutoff_date = pd.Timestamp("2025-12-31")
    months = (cutoff_date - dob).days / 30.44
    months = float(np.floor(months))
    return months if months >= 0 else None


def _ta_get_raw(raw: Dict[str, Any], *keys: str) -> Optional[Any]:
    for key in keys:
        if key in raw:
            val = raw.get(key)
            if val is not None and str(val).strip() != "":
                return val
    return None


def _ta_derive_features(raw: Dict[str, Any]) -> Dict[str, Any]:
    derived: Dict[str, Any] = {}

    def map_binary(val: Any) -> int:
        return 1 if "ya" in str(val).lower() else 0

    def kategorisasi_komplikasi(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Ada Komplikasi"
        text = str(text).lower()
        if any(keyword in text for keyword in ["jantung", "detak"]):
            return "Kelainan Jantung"
        if any(keyword in text for keyword in ["paru", "napas", "thorax", "atsma", "sesak"]):
            return "Gangguan Pernapasan"
        if any(keyword in text for keyword in ["pencernaan", "usus", "lambung"]):
            return "Gangguan Pencernaan"
        return "Lainnya"

    def kategorisasi_penyakit(text: Any) -> str:
        if pd.isna(text):
            return "Tidak Tahu"
        text = str(text).lower()
        if any(x in text for x in ["tidak", "tdk", "sehat", "no"]):
            return "Tidak Sakit"
        if any(x in text for x in ["ispa", "batuk", "pilek", "flu", "bapil", "radang"]):
            return "ISPA/Respirasi"
        if any(x in text for x in ["diare", "mencret", "sembelit", "usus"]):
            return "Pencernaan"
        if any(x in text for x in ["demam", "panas"]):
            return "Demam"
        if any(x in text for x in ["tb", "tbc", "paru", "jantung", "step", "malaria"]):
            return "Penyakit Serius/Kronis"
        return "Lainnya"

    def kategorisasi_frekuensi_suplemen(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Tahu/Kosong"
        text = str(text).lower()
        negative_keywords = ["tidak", "tdk", "belum", "ngga", "stop", "jarang", "lupa"]
        if any(keyword in text for keyword in negative_keywords):
            if "lupa" in text:
                return "Tidak Tahu/Lupa"
            return "Tidak Pernah"
        daily_keywords = ["setiap hari", "tiap hari", "rutin", "sehari", "daily", "1x1", "2x1", "3x1", "full"]
        if any(keyword in text for keyword in daily_keywords):
            return "Rutin (Setiap Hari)"
        numbers = re.findall(r"\d+", text)
        if numbers:
            try:
                num = int(numbers[0])
                if num >= 7:
                    return "Rutin (Setiap Hari)"
                if 3 <= num <= 6:
                    return "Sering (3-6 kali/minggu)"
                if 1 <= num <= 2:
                    return "Jarang (1-2 kali/minggu)"
                if num == 0:
                    return "Tidak Pernah"
            except Exception:
                pass
        if len(text) > 2:
            return "Konsumsi (Frekuensi Tidak Jelas)"
        return "Tidak Tahu/Kosong"

    def kategorisasi_jenis_suplemen(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Tahu/Kosong"
        text = str(text).lower()
        if any(x in text for x in ["tidak", "tdk", "stop"]):
            if "tidak ada" in text and len(text) < 15:
                return "Tidak Mengonsumsi"
            if "tidak mengonsumsi" in text:
                return "Tidak Mengonsumsi"
        has_iron = any(x in text for x in ["tambah darah", "zat besi", "ttd", "fe", "folamil", "gestiamin", "vitamom", "mms"])
        has_folat = any(x in text for x in ["asam folat", "folamil", "gestiamin", "mms"])
        has_calcium = any(x in text for x in ["kalsium", "calcifar", "mms", "cdr"])
        has_milk = "susu" in text
        is_package = any(x in text for x in ["semuanya", "dari bidan", "paket"])
        if (has_iron and has_folat and (has_calcium or has_milk)) or is_package:
            return "Lengkap (TTD + Folat + Kalsium/Susu)"
        if has_iron and has_folat:
            return "Baik (TTD + Asam Folat)"
        if has_iron:
            return "Cukup (Ada Zat Besi/TTD)"
        if has_milk or has_calcium or "vitamin" in text:
            return "Kurang (Hanya Susu/Vit Tanpa TTD)"
        if "buah" in text or "madu" in text:
            return "Alternatif (Non-Medis)"
        return "Tidak Mengonsumsi"

    def kategorisasi_mpasi(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Tahu/Kosong"
        text_lower = str(text).lower()
        if "lahir" in text_lower:
            return "Terlalu Dini (< 6 Bulan)"
        if any(x in text_lower for x in ["belum", "tidak", "blm", "nanti"]):
            return "Belum MPASI"
        if any(x in text_lower for x in ["januari", "februari", "agustus", "2025", "2026"]):
            return "Data Tidak Valid (Tanggal)"
        text_clean = text_lower.replace(",", ".").replace("½", ".5").replace("setengah", ".5")
        if "kurang" in text_clean:
            if "6" in text_clean:
                return "Terlalu Dini (< 6 Bulan)"
        numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text_clean)
        if numbers:
            try:
                val = float(numbers[0])
                if val < 6:
                    return "Terlalu Dini (< 6 Bulan)"
                if val == 6 or (6 < val < 7):
                    return "Tepat Waktu (6 Bulan)"
                if val >= 7:
                    return "Terlalu Lambat (> 6 Bulan)"
            except ValueError:
                pass
        if "enam" in text_lower:
            return "Tepat Waktu (6 Bulan)"
        return "Tidak Tahu/Kosong"

    def kategorisasi_vitamin_a(text: Any) -> str:
        if pd.isna(text) or str(text).strip() in {"", "_"}:
            return "Tidak Tahu/Lupa"
        text = str(text).lower()
        if any(x in text for x in ["belum", "tidak", "blm", "tdk", "no"]):
            if "6 bulan" in text and "masih" in text:
                return "Belum Cukup Umur (<6 Bulan)"
            return "Tidak/Belum Pernah"
        recent_months = ["agustus", "september", "oktober", "november", "desember", "aug"]
        if any(x in text for x in recent_months):
            return "Terkini (Agustus/Semester Ini)"
        if "minggu" in text:
            return "Terkini (Agustus/Semester Ini)"
        if "bulan" in text or "bln" in text:
            numbers = re.findall(r"\d+", text)
            if numbers:
                val = int(numbers[0])
                if val <= 5:
                    return "Terkini (Agustus/Semester Ini)"
                if val >= 6:
                    return "Sudah Lama (> 6 Bulan/Terlewat)"
        old_months = ["februari", "januari", "maret", "april", "mei", "juni", "juli", "pebruari"]
        if any(x in text for x in old_months):
            return "Sudah Lama (> 6 Bulan/Terlewat)"
        if "tahun" in text:
            return "Sudah Lama (> 6 Bulan/Terlewat)"
        if any(x in text for x in ["posyandu", "rutin", "jadwal", "imunisasi", "ya", "pernah", "susu"]):
            return "Pernah (Waktu Tidak Jelas)"
        return "Tidak Tahu/Lupa"

    def kategorisasi_intervensi(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Ada Intervensi"
        text = str(text).lower()
        if any(x in text for x in ["rutf", "rujukan", "rumah sakit", "gizi buruk"]):
            return "Intensif (RUTF/Rujukan)"
        if any(x in text for x in ["makanan tambahan", "pmt", "puding", "telur", "mbg"]):
            return "Suplementasi (PMT)"
        if "edukasi" in text or "penyuluhan" in text:
            return "Edukasi Gizi Saja"
        return "Lainnya/Tidak Jelas"

    def kategorisasi_penyakit_utama(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Tahu/Kosong"
        text = str(text).lower()
        if any(x in text for x in ["tb", "tbc", "paru", "step", "kejang"]):
            return "Penyakit Serius (TBC/Paru/Neurologis)"
        if any(x in text for x in ["diare", "mencret", "sembelit", "buang air besar", "pencernaan"]):
            return "Gangguan Pencernaan (Diare)"
        if any(x in text for x in ["batuk", "pilek", "flu", "bapil", "ispa", "sesak", "radang", "ingusan"]):
            return "ISPA (Batuk/Pilek)"
        if any(x in text for x in ["demam", "panas", "malaria"]):
            return "Demam/Malaria"
        if any(x in text for x in ["gigi", "sariawan", "kencing", "kecil"]):
            return "Penyakit Lainnya"
        if any(x in text for x in ["tidak", "tdk", "sehat", "aman", "no"]):
            return "Sehat/Tidak Sakit"
        return "Lainnya/Tidak Jelas"

    def kategorisasi_pekerjaan(text: Any) -> str:
        if pd.isna(text) or str(text).strip() in ["", "-", "dfhd", "pe", "supriyadi 3", "5;"]:
            return "Tidak Diketahui/Lainnya"
        text = str(text).lower()
        if any(x in text for x in ["tidak bekerja", "belum bekerja", "menganggur", "mencari nafkah", "mencari napkah", "ibu rumah tangga", "irt"]):
            return "Tidak Bekerja"
        if any(x in text for x in ["pns", "bumn", "pemda", "pppk", "p3k", "dosen", "guru"]):
            return "PNS/BUMN/Pemerintah"
        if any(x in text for x in ["ojek", "ojol", "gojek", "grab", "driver", "kurir", "sopir", "supir"]):
            return "Ojek Online/Driver"
        keywords_unstable = ["lepas", "harian", "serabutan", "bhl", "kuli", "pemulung", "parkir", "bangunan", "butuh"]
        if any(x in text for x in keywords_unstable):
            return "Buruh Harian/Serabutan"
        keywords_wira = ["wira", "dagang", "jual", "usaha", "bengkel", "toko", "warung", "home", "cetak", "bordir", "alumunium", "service", "sepatu", "montir", "teknisi"]
        if any(x in text for x in keywords_wira):
            return "Wiraswasta/Pedagang"
        keywords_karyawan = ["karyawan", "pegawai", "swasta", "pabrik", "security", "satpam", "scurity", "admin", "spv", "manager", "staff", "buruh", "jurnalis", "cleaning"]
        if any(x in text for x in keywords_karyawan):
            return "Karyawan Swasta/Buruh Tetap"
        return "Lainnya"

    def kategorisasi_vaksin(text: Any) -> str:
        if pd.isna(text) or str(text).strip() == "":
            return "Tidak Ada Data"
        text = str(text).lower()
        if "lanjutan" in text:
            return "Lengkap (Dasar + Lanjutan)"
        has_mr = any(x in text for x in ["campak", "mr", "rubella"])
        has_dpt3 = "dpt-hb-hib 3" in text
        if has_mr and has_dpt3:
            return "Lengkap (Dasar)"
        return "Belum Lengkap / Parsial"

    def hitung_jumlah_vaksin(text: Any) -> int:
        if pd.isna(text) or str(text).strip() == "":
            return 0
        items = [x.strip() for x in str(text).split(",")]
        return len(items)

    umur = _ta_to_number(_ta_get_raw(raw, "umur_balita"))
    if umur is None:
        umur = _ta_age_months_from_date(_ta_get_raw(raw, "Tanggal_Lahir_Balita_kader"))
    if umur is not None:
        derived["umur_balita"] = float(umur)

    bb = _ta_to_number(_ta_get_raw(raw, "berat_badan_kg"))
    tb = _ta_to_number(_ta_get_raw(raw, "tinggi_badan_cm"))

    if bb is not None and tb is not None:
        rasio_bb_tb = bb / (tb + 0.01)
        bmi_anak = bb / (((tb / 100) ** 2) + 0.01)
        derived["rasio_bb_tb_log"] = float(np.log1p(max(rasio_bb_tb, 0.01)))
        derived["bmi_log"] = float(np.log1p(max(bmi_anak, 0.01)))
        derived["bb_tb_interaction"] = float(bb * tb)

    if bb is not None and umur is not None:
        derived["laju_bb_umur"] = float(bb / (umur + 1))

    if umur is not None:
        bins = [0, 6, 12, 24, 36, 48, 60, 120]
        labels = ["0-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60+"]
        umur_bin = pd.cut([umur], bins=bins, labels=labels, include_lowest=True)[0]
        if pd.notna(umur_bin):
            derived["umur_bin"] = str(umur_bin)

    diet_cols = [
        "konsumsi_karbohidrat_h_1",
        "konsumsi_kacangan_h_1",
        "konsumsi_susu_hewani_h_1",
        "konsumsi_daging_ikan_h_1",
        "konsumsi_telur_h_1",
        "konsumsi_vit_a_h_1",
        "konsumsi_buah_sayur_lain_h_1",
    ]
    derived["score_dietary_diversity"] = int(sum(map_binary(raw.get(c)) for c in diet_cols))

    sumber_air_val = _ta_get_raw(raw, "sumber_air_minum")
    if sumber_air_val is not None:
        map_typo_air = {
            "sibel": "Sumur Bor/Sibel",
            "Sibel": "Sumur Bor/Sibel",
            "AQua": "Air Kemasan",
            "Aqua": "Air Kemasan",
            "Galon aqua": "Air Kemasan",
            "ron 88 ": "Air Kemasan",
            "Ron": "Air Kemasan",
            "Lemineral galon": "Air Kemasan",
            "Air Isi Ulang/Galon/Kemasan": "Air Kemasan",
            "Jetpam": "Sumur Bor/Sibel",
            "Sumur": "Sumur Gali/Timba",
        }
        derived["sumber_air_minum"] = map_typo_air.get(str(sumber_air_val), sumber_air_val)

    suplemen_text = str(_ta_get_raw(raw, "jenis_suplemen_ibu") or "")
    suplemen_lower = suplemen_text.lower()
    derived["kategori_nutrisi_ibu"] = kategorisasi_jenis_suplemen(suplemen_text)
    derived["kategori_kepatuhan_suplemen"] = kategorisasi_frekuensi_suplemen(
        _ta_get_raw(raw, "frekuensi_suplemen_minggu")
    )
    derived["flag_minum_ttd"] = 1 if suplemen_lower and any(k in suplemen_lower for k in ["tambah darah", "zat besi", "folamil"]) else 0
    derived["flag_minum_susu"] = 1 if suplemen_lower and "susu" in suplemen_lower else 0

    derived["kategori_mpasi"] = kategorisasi_mpasi(_ta_get_raw(raw, "usia_mulai_mpasi"))
    derived["kategori_vitamin_a"] = kategorisasi_vitamin_a(
        _ta_get_raw(raw, "terakhir_vitamin_a", "kategori_vitamin_a")
    )

    intervensi_text = str(_ta_get_raw(raw, "jenis_intervensi_gizi", "kategori_intervensi_utama") or "")
    intervensi_lower = intervensi_text.lower()
    derived["kategori_intervensi_utama"] = kategorisasi_intervensi(intervensi_text)
    derived["flag_dapat_pmt"] = 1 if any(k in intervensi_lower for k in ["makanan tambahan", "pmt", "puding", "telur"]) else 0
    derived["flag_dapat_rutf"] = 1 if "rutf" in intervensi_lower else 0
    derived["flag_dapat_edukasi"] = 1 if "edukasi" in intervensi_lower else 0
    derived["flag_dirujuk"] = 1 if "rujukan" in intervensi_lower or "rumah sakit" in intervensi_lower else 0

    penyakit_text = str(_ta_get_raw(raw, "jenis_penyakit_balita", "kategori_penyakit_balita") or "")
    penyakit_lower = penyakit_text.lower()
    derived["kategori_penyakit_balita"] = kategorisasi_penyakit(penyakit_text)
    derived["kategori_penyakit_dominan"] = kategorisasi_penyakit_utama(penyakit_text)
    derived["flag_sakit_diare"] = 1 if any(k in penyakit_lower for k in ["diare", "mencret", "buang air besar"]) else 0
    derived["flag_sakit_ispa"] = 1 if any(k in penyakit_lower for k in ["batuk", "pilek", "flu", "bapil", "ispa", "sesak"]) else 0
    derived["flag_sakit_tb_serius"] = 1 if any(k in penyakit_lower for k in ["tb", "tbc", "paru", "step"]) else 0

    derived["kategori_komplikasi_lahir"] = kategorisasi_komplikasi(
        _ta_get_raw(raw, "jenis_komplikasi_lahir", "kategori_komplikasi_lahir")
    )

    derived["kategori_pekerjaan_kk"] = kategorisasi_pekerjaan(
        _ta_get_raw(raw, "pekerjaan_kepala_keluarga", "kategori_pekerjaan_kk")
    )

    vaksin_text = _ta_get_raw(raw, "riwayat_vaksinasi", "kategori_vaksinasi")
    derived["kategori_vaksinasi"] = kategorisasi_vaksin(vaksin_text)
    derived["jumlah_vaksin_diterima"] = hitung_jumlah_vaksin(vaksin_text)

    env_score = 0
    if "sungai" in str(_ta_get_raw(raw, "sumber_air_minum") or "").lower():
        env_score += 2
    if "sungai" in str(_ta_get_raw(raw, "jenis_sanitasi") or "").lower():
        env_score += 2
    if "ya" in str(_ta_get_raw(raw, "is_perokok_di_rumah") or "").lower():
        env_score += 1
    derived["score_environmental_risk"] = env_score

    maternal_score = 0
    if str(_ta_get_raw(raw, "is_bblr") or "") == "Ya":
        maternal_score += 2
    if str(_ta_get_raw(raw, "is_prematur") or "") == "Ya":
        maternal_score += 2
    if str(_ta_get_raw(raw, "is_anemia_ibu") or "") == "Ya":
        maternal_score += 1
    derived["score_maternal_risk"] = maternal_score

    def map_inc(val: Any) -> int:
        s = str(val).lower()
        if "kurang" in s or "<" in s:
            return 1
        if "1.000" in s:
            return 2
        if "3.000" in s:
            return 3
        if "5.000" in s:
            return 4
        if "lebih" in s or ">" in s:
            return 5
        return 1

    inc_score = map_inc(_ta_get_raw(raw, "pendapatan_bulanan"))
    derived["score_income_level"] = inc_score
    jumlah_art = _ta_to_number(_ta_get_raw(raw, "jumlah_art"))
    if jumlah_art is not None:
        derived["ratio_economic_burden"] = float(jumlah_art / (inc_score + 0.1))

    edu_map = {
        "Tidak sekolah": 0,
        "SD/sederajat": 1,
        "SMP/sederajat": 2,
        "SMA/sederajat": 3,
        "Diploma (D1–D3)": 4,
        "S1": 5,
        "S2": 6,
        "S3": 6,
    }
    s_ibu = edu_map.get(str(_ta_get_raw(raw, "pendidikan_ibu") or ""), 1)
    s_ayah = edu_map.get(str(_ta_get_raw(raw, "pendidikan_ayah") or ""), 1)
    derived["score_edu_parents"] = s_ibu + s_ayah

    return derived


def _ta_apply_label_encoding(df: pd.DataFrame) -> pd.DataFrame:
    if "umur_bin" in df.columns:
        umur_labels = ["0-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60+"]
        umur_map = {label: idx for idx, label in enumerate(umur_labels)}
        df["umur_bin"] = df["umur_bin"].map(umur_map).fillna(0)

    for col, encoder in ta_label_encoders.items():
        target_col = col if col in df.columns else None
        if target_col is None:
            alias = TA_COLUMN_ALIASES.get(col)
            if alias and alias in df.columns:
                target_col = alias
            else:
                alias_inverse = {v: k for k, v in TA_COLUMN_ALIASES.items()}
                alias_back = alias_inverse.get(col)
                if alias_back and alias_back in df.columns:
                    target_col = alias_back
        if target_col is None:
            continue

        if isinstance(encoder, dict):
            df[target_col] = df[target_col].map(encoder).fillna(0)
        else:
            if hasattr(encoder, "classes_"):
                mapping = {str(cls): idx for idx, cls in enumerate(encoder.classes_)}
                df[target_col] = df[target_col].astype(str).map(mapping).fillna(0).astype(int)
            else:
                df[target_col] = df[target_col]

    return df


def _ta_apply_scaling(df: pd.DataFrame) -> pd.DataFrame:
    alias_inverse = {v: k for k, v in TA_COLUMN_ALIASES.items()}
    scale_cols = []
    source_cols = []
    for col in ta_scaler_features:
        if col in df.columns:
            scale_cols.append(col)
            source_cols.append(col)
        else:
            alias = TA_COLUMN_ALIASES.get(col)
            if alias and alias in df.columns:
                scale_cols.append(alias)
                source_cols.append(col)
            else:
                alias_back = alias_inverse.get(col)
                if alias_back and alias_back in df.columns:
                    scale_cols.append(alias_back)
                    source_cols.append(col)

    if scale_cols:
        scaled_values = ta_scaler.transform(df[scale_cols])
        for idx, target_col in enumerate(source_cols):
            if target_col in df.columns:
                df[target_col] = scaled_values[:, idx]
            else:
                alias = TA_COLUMN_ALIASES.get(target_col)
                if alias and alias in df.columns:
                    df[alias] = scaled_values[:, idx]
    return df


def preprocess_input_ta(raw_data: Dict[str, Any]) -> pd.DataFrame:
    normalized = _ta_normalize_input(raw_data)
    derived = _ta_derive_features(normalized)
    combined = {**normalized, **derived}
    alias_inverse = {v: k for k, v in TA_COLUMN_ALIASES.items()}
    for canonical, alias in TA_COLUMN_ALIASES.items():
        if alias in combined and canonical not in combined:
            combined[canonical] = combined.get(alias)
        if canonical in combined and alias not in combined:
            combined[alias] = combined.get(canonical)
    df = pd.DataFrame([combined])

    numeric_cols = [
        "umur_balita",
        "berat_badan_kg",
        "tinggi_badan_cm",
        "lingkar_kepala_cm",
        "lila_cm",
        "usia_kehamilan_lahir",
        "berat_lahir_kg",
        "panjang_lahir_cm",
        "tinggi_ibu_cm",
        "berat_ibu_kg",
        "tinggi_ayah_cm",
        "berat_ayah_kg",
        "frekuensi_susu_non_asi",
        "usia_mulai_mpasi",
        "jam_tidur_harian",
        "jumlah_art",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    for col in ta_feature_order:
        if col not in df.columns:
            df[col] = 0

    df = _ta_apply_label_encoding(df)
    df = _ta_apply_scaling(df)
    df = df.reindex(columns=ta_feature_order, fill_value=0)

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = pd.to_numeric(df[col], errors="ignore")

    return df


def _ta_predict_single(model: Any, features: pd.DataFrame, target: str) -> Dict[str, Any]:
    pred_class = int(model.predict(features)[0])

    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(features)[0].tolist()

    return {
        "class_id": pred_class,
        "label": TA_LABEL_MAPS.get(target, {}).get(pred_class, str(pred_class)),
        "proba": proba,
    }

# --- 4. Prediction Endpoint ---

@app.post("/predict", dependencies=[Depends(get_api_key)])
async def predict(data: BalitaInput):
    try:
        processed_input = preprocess_input(data)
        
        # Prediksi untuk BB/TB
        prediction_bbtb = int(model_bbtb.predict(processed_input)[0])
        status_bbtb = reverse_map_bbtb.get(prediction_bbtb, "Unknown")
        
        # Prediksi untuk BB/U
        prediction_bbu = int(model_bbu.predict(processed_input)[0])
        status_bbu = reverse_map_bbu.get(prediction_bbu, "Unknown")
        
        # Prediksi untuk TB/U
        prediction_tbu = int(model_tbu.predict(processed_input)[0])
        status_tbu = reverse_map_tbu.get(prediction_tbu, "Unknown")
        
        return {
            "BB/TB": status_bbtb,
            "BB/U": status_bbu,
            "TB/U": status_tbu
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/predict-ta", dependencies=[Depends(get_api_key)])
def predict_status_gizi_ta(payload: TAPredictRequest) -> Dict[str, Any]:
    try:
        if hasattr(payload.data, "model_dump"):
            data = payload.data.model_dump()
        else:
            data = payload.data.dict()

        features = preprocess_input_ta(data)

        results = {
            target: _ta_predict_single(model, features, target)
            for target, model in ta_models.items()
        }

        return {"predictions": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc))
