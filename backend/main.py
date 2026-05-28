from __future__ import annotations

import json
import os
import re
import uuid
from pathlib import Path
from typing import Any, Literal

import httpx
import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from lightgbm import LGBMRegressor
from pydantic import BaseModel, Field
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ROOT_DIR = Path(__file__).resolve().parents[1]
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT_DIR / ".env.local")
except ImportError:
    pass


def _path_from_env(name: str, default: Path) -> Path:
    return Path(os.getenv(name, str(default))).expanduser().resolve()


ARTIFACTS_DIR = _path_from_env("TIRELIFE_ARTIFACTS_DIR", ROOT_DIR / "artifacts")
DATA_PATH = _path_from_env("TIRELIFE_DATA_PATH", ARTIFACTS_DIR / "tyre_rul_cleaned.csv")
MODEL_DIR = _path_from_env("TIRELIFE_MODEL_DIR", ARTIFACTS_DIR / "trained_models")
FRONTEND_DIST_DIR = _path_from_env("FRONTEND_DIST_DIR", ROOT_DIR / "dist")

TARGET_COL = "remaining_useful_life(km)"
DEFAULT_MODEL = "lightgbm"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"

# Keep "normal" model free from obvious target leakage.
LEAKAGE_EXCLUDED_COLS = {
    "expected_tyre_life(km)",
    "km_used_ratio_vs_expected",
}

# Main user fields we ask for in chat before predicting.
REQUIRED_CHAT_FIELDS = [
    "current_tread_depth(mm)",
    "kilometers_driven(km)",
    "average_inflation_pressure(psi)",
    "tyre_age(years)",
]

# If a user previously got a prediction, require at least this many core fields
# in a new message before silently reusing old session values.
MIN_REQUIRED_FIELDS_UPDATED_PER_TURN = 2
AUTOFILL_OFFER_AFTER_FOLLOWUPS = 3
AUTO_PREDICT_AFTER_NOINFO_TURNS = 2

# Minimum info gates for a first usable estimate.
WEAR_SIGNAL_FIELDS = ["current_tread_depth(mm)", "kilometers_driven(km)"]
CONTEXT_SIGNAL_FIELDS = ["average_inflation_pressure(psi)", "tyre_age(years)"]

REQUIRED_CHAT_FIELD_HINTS = {
    "current_tread_depth(mm)": "Current tread depth in mm (example: 4.5 mm)",
    "kilometers_driven(km)": "Total kilometers already driven on this tyre (example: 28000 km)",
    "average_inflation_pressure(psi)": "Average inflation pressure in psi (example: 32 psi)",
    "tyre_age(years)": "Tyre age in years (example: 3 years)",
}

DEFAULT_INTENT_TOKENS = {
    "use defaults",
    "use default",
    "default values",
    "auto fill",
    "autofill",
    "go with defaults",
    "estimate anyway",
    "predict anyway",
}

DONE_SHARING_TOKENS = {
    "thats all i know",
    "that's all i know",
    "nothing else",
    "what if i dont have those",
    "what if i don't have those",
    "dont know anything else",
    "don't know anything else",
    "no more details",
    "no more info",
    "thats all i have",
    "that's all i have",
    "thats all",
    "that's all",
}

NO_VALID_RESPONSE_TOKENS = {
    "i dont know",
    "i don't know",
    "dont know",
    "don't know",
    "not sure",
    "no idea",
    "idk",
    "cant remember",
    "can't remember",
    "unknown",
}

SMALL_TALK_TOKENS = {
    "hi",
    "hello",
    "hey",
    "yo",
    "thanks",
    "thank you",
    "thx",
    "ok",
    "okay",
    "cool",
    "great",
    "awesome",
    "got it",
    "sounds good",
}

# Additional high-impact fields that improve prediction quality beyond the core set.
ACCURACY_BOOST_CHAT_FIELDS = [
    "vehicle_model",
    "number_of_punctures",
    "road_condition",
    "weather_condition",
    "recommended_inflation_pressure(psi)",
    "Standard_tread_depth(mm)",
]

ACCURACY_BOOST_FIELD_HINTS = {
    "vehicle_model": "Vehicle model (example: BMW X5)",
    "number_of_punctures": "Number of punctures so far (example: 0 or 1)",
    "road_condition": "Typical road condition (smooth / mixed / rough)",
    "weather_condition": "Typical weather (dry / rainy / snowy / mixed conditions)",
    "recommended_inflation_pressure(psi)": "Recommended pressure from sticker/manual in psi",
    "Standard_tread_depth(mm)": "New tire tread depth in mm (if known)",
}

MODEL_PROFILE_FIELDS = [
    "tyre_brand",
    "tyre_size",
    "tread_material",
    "recommended_inflation_pressure(psi)",
    "Standard_tread_depth(mm)",
]

MODEL_PROFILE_FALLBACK_FIELDS = [
    "tread_material",
    "recommended_inflation_pressure(psi)",
    "Standard_tread_depth(mm)",
]

UNKNOWN_TEXT_TOKENS = {
    "unknown",
    "unknown brand",
    "generic",
    "generic brand",
    "generic / unknown brand",
    "not sure",
    "dont know",
    "don't know",
    "na",
    "n/a",
    "none",
}

# Prompt for more details when many of these impact fields are still defaulted.
ACCURACY_FOLLOW_UP_TRIGGER_COUNT = 3

FRIENDLY_FIELD_NAMES = {
    "current_tread_depth(mm)": "tread depth",
    "kilometers_driven(km)": "distance driven",
    "average_inflation_pressure(psi)": "tire pressure",
    "tyre_age(years)": "tire age",
    "vehicle_model": "vehicle model",
    "tyre_brand": "tire brand",
    "number_of_punctures": "puncture count",
    "road_condition": "road condition",
    "weather_condition": "weather condition",
    "recommended_inflation_pressure(psi)": "recommended pressure",
    "Standard_tread_depth(mm)": "new tire tread depth",
}


class PredictRequest(BaseModel):
    features: dict[str, Any] = Field(default_factory=dict)
    model: Literal["lightgbm", "deeplearning_test"] = DEFAULT_MODEL


class PredictResponse(BaseModel):
    model: str
    predicted_rul_km: float
    predicted_rul_miles: float
    defaults_used: list[str]
    defaults_used_count: int
    normalized_features: dict[str, Any]


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    session_id: str | None = None
    model: Literal["lightgbm", "deeplearning_test"] = DEFAULT_MODEL
    force_predict: bool = False


class ChatResponse(BaseModel):
    session_id: str
    needs_follow_up: bool
    assistant_message: str
    missing_fields: list[str]
    suggest_autofill: bool = False
    extractor: str
    parsed_features: dict[str, Any]
    prediction: PredictResponse | None = None


class HealthResponse(BaseModel):
    status: str
    dataset_path: str
    artifacts_dir: str
    model_dir: str
    frontend_dist_dir: str
    frontend_static_enabled: bool
    models_available: list[str]
    required_chat_fields: list[str]
    gemini_extraction_enabled: bool
    gemini_chat_enabled: bool
    gemini_model: str | None = None
    gemini_key_source: str = "none"


class GeminiConfigRequest(BaseModel):
    api_key: str | None = None
    model: str | None = None
    persist_to_env: bool = False


class GeminiConfigResponse(BaseModel):
    gemini_enabled: bool
    gemini_model: str | None = None
    key_source: str


class TireRULService:
    def __init__(self, data_path: Path, model_dir: Path):
        self.data_path = data_path
        self.model_dir = model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.profile_path = self.model_dir / "feature_profile.joblib"

        self.runtime_gemini_api_key: str | None = None
        self.runtime_gemini_model: str | None = None
        self.runtime_gemini_disabled: bool = False
        self.sessions: dict[str, dict[str, Any]] = {}

        self.defaults: dict[str, Any] = {}
        self.numeric_cols: set[str] = set()
        self.categorical_cols: set[str] = set()
        self.feature_cols_all: list[str] = []
        self.feature_cols_lightgbm: list[str] = []
        self.feature_cols_deep: list[str] = []
        self.models: dict[str, Any] = {}
        self.vehicle_profiles_by_model: dict[str, dict[str, Any]] = {}
        self.vehicle_profiles_by_make: dict[str, dict[str, Any]] = {}

        self.feature_aliases = {
            "current_tread_depth_mm": "current_tread_depth(mm)",
            "current_tread_depth": "current_tread_depth(mm)",
            "standard_tread_depth_mm": "Standard_tread_depth(mm)",
            "standard_tread_depth": "Standard_tread_depth(mm)",
            "kilometers_driven_km": "kilometers_driven(km)",
            "kilometers_driven": "kilometers_driven(km)",
            "km_driven": "kilometers_driven(km)",
            "tread_wear_rating_utqg": "tread_wear_rating (UTQG)",
            "average_inflation_pressure_psi": "average_inflation_pressure(psi)",
            "recommended_inflation_pressure_psi": "recommended_inflation_pressure(psi)",
            "vehicle_sprung_mass_kg": "vehicle_sprung_mass(kg)",
            "vehicle_acceleration_0_100_km_h_in_seconds": "vehicle_acceleration(0-100 km/h in seconds)",
            "maximum_power_hp": "maximum_power(hp)",
            "maximum_torque_n_m": "maximum_torque(N/m)",
            "maximum_speed_km_h": "maximum_speed (km/h)",
            "vehicle_mileage_mpg": "vehicle_mileage(mpg)",
            "average_tread_temperature_celsius": "average_tread_temperature(celsius)",
            "tyre_age_years": "tyre_age(years)",
            "number_of_punctures": "number_of_punctures",
            "expected_tyre_life_km": "expected_tyre_life(km)",
            "axle_type": "axle_type(driven/dead)",
        }

    def startup(self) -> None:
        if self.data_path.exists():
            self._load_profile()
            self._save_profile()
        else:
            self._load_saved_profile()
        self._load_or_train_models()

    def _gemini_key_source(self) -> str:
        if self.runtime_gemini_disabled:
            return "runtime:disabled"
        if self.runtime_gemini_api_key:
            return "runtime"
        if os.getenv("GEMINI_API_KEY", "").strip():
            return "env:GEMINI_API_KEY"
        if os.getenv("GOOGLE_API_KEY", "").strip():
            return "env:GOOGLE_API_KEY"
        return "none"

    def _gemini_api_key(self) -> str:
        if self.runtime_gemini_disabled:
            return ""
        if self.runtime_gemini_api_key:
            return self.runtime_gemini_api_key
        env_primary = os.getenv("GEMINI_API_KEY", "").strip()
        if env_primary:
            return env_primary
        return os.getenv("GOOGLE_API_KEY", "").strip()

    def _gemini_model_name(self) -> str:
        return (
            (self.runtime_gemini_model or "").strip()
            or os.getenv("GEMINI_CHAT_MODEL", "").strip()
            or os.getenv("GEMINI_MODEL", "").strip()
            or DEFAULT_GEMINI_MODEL
        )

    def _persist_gemini_env(self, *, api_key: str | None, model: str | None) -> None:
        env_path = ROOT_DIR / ".env.local"
        existing: dict[str, str] = {}
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                raw = line.strip()
                if not raw or raw.startswith("#") or "=" not in raw:
                    continue
                k, v = raw.split("=", 1)
                existing[k.strip()] = v.strip()

        if api_key is None:
            existing.pop("GEMINI_API_KEY", None)
        else:
            api_key_clean = api_key.strip()
            if api_key_clean:
                existing["GEMINI_API_KEY"] = api_key_clean
            else:
                existing.pop("GEMINI_API_KEY", None)

        if model is not None:
            model_clean = model.strip()
            if model_clean:
                existing["GEMINI_CHAT_MODEL"] = model_clean
                existing["GEMINI_MODEL"] = model_clean
            else:
                existing.pop("GEMINI_CHAT_MODEL", None)
                existing.pop("GEMINI_MODEL", None)

        lines = [f"{k}={v}" for k, v in sorted(existing.items())]
        env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def _gemini_headers(api_key: str) -> dict[str, str]:
        return {
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        }

    def _validate_gemini_setup(self, *, api_key: str, model: str) -> None:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": "Reply with OK"}]}],
            "generationConfig": {"temperature": 0.0},
        }
        with httpx.Client(timeout=12.0) as client:
            resp = client.post(url, headers=self._gemini_headers(api_key), json=body)
        if resp.status_code >= 400:
            detail = ""
            try:
                payload = resp.json()
                detail = payload.get("error", {}).get("message", "")
            except Exception:
                detail = resp.text
            raise ValueError(
                f"Gemini configuration failed ({resp.status_code}). "
                f"{detail.strip() or 'Please verify API key and model name.'}"
            )

    def configure_gemini(self, request: GeminiConfigRequest) -> GeminiConfigResponse:
        incoming_key = request.api_key.strip() if request.api_key is not None else None
        incoming_model = request.model.strip() if request.model is not None else None

        prev_key = self.runtime_gemini_api_key
        prev_model = self.runtime_gemini_model
        prev_disabled = self.runtime_gemini_disabled

        if incoming_key is not None:
            if incoming_key:
                self.runtime_gemini_api_key = incoming_key
                self.runtime_gemini_disabled = False
            else:
                self.runtime_gemini_api_key = None
                self.runtime_gemini_disabled = True
        if incoming_model is not None:
            self.runtime_gemini_model = incoming_model or None

        key = self._gemini_api_key()
        model = self._gemini_model_name() if key else None
        if key and model:
            try:
                self._validate_gemini_setup(api_key=key, model=model)
            except Exception:
                self.runtime_gemini_api_key = prev_key
                self.runtime_gemini_model = prev_model
                self.runtime_gemini_disabled = prev_disabled
                raise

        if request.persist_to_env:
            persisted_key = None if self.runtime_gemini_disabled else self.runtime_gemini_api_key
            self._persist_gemini_env(api_key=persisted_key, model=self.runtime_gemini_model)

        key = self._gemini_api_key()
        model = self._gemini_model_name() if key else None
        return GeminiConfigResponse(
            gemini_enabled=bool(key),
            gemini_model=model,
            key_source=self._gemini_key_source(),
        )

    def _chat_model_name(self) -> str:
        return self._gemini_model_name()

    def _profile_nrows(self) -> int | None:
        raw = os.getenv("PROFILE_SAMPLE_ROWS", "220000").strip()
        if raw == "0":
            return None
        try:
            value = int(raw)
            return value if value > 0 else None
        except ValueError:
            return 220000

    def _train_nrows(self) -> int | None:
        raw = os.getenv("TRAIN_SAMPLE_ROWS", "220000").strip()
        if raw == "0":
            return None
        try:
            value = int(raw)
            return value if value > 0 else None
        except ValueError:
            return 220000

    def _load_profile(self) -> None:
        df = pd.read_csv(self.data_path, nrows=self._profile_nrows(), low_memory=False)
        if TARGET_COL not in df.columns:
            raise ValueError(f"Missing target column '{TARGET_COL}' in {self.data_path}")

        self.feature_cols_all = [c for c in df.columns if c != TARGET_COL]

        numeric_cols: list[str] = []
        categorical_cols: list[str] = []
        defaults: dict[str, Any] = {}

        for col in self.feature_cols_all:
            numeric_candidate = pd.to_numeric(df[col], errors="coerce")
            numeric_ratio = float(numeric_candidate.notna().mean())
            if numeric_ratio >= 0.95:
                numeric_cols.append(col)
                defaults[col] = float(numeric_candidate.median(skipna=True))
            else:
                categorical_cols.append(col)
                mode = df[col].dropna().astype(str).str.strip().str.lower().mode()
                defaults[col] = mode.iloc[0] if not mode.empty else "unknown"

        self.numeric_cols = set(numeric_cols)
        self.categorical_cols = set(categorical_cols)
        self.defaults = defaults

        self.feature_cols_lightgbm = [
            col for col in self.feature_cols_all if col not in LEAKAGE_EXCLUDED_COLS
        ]
        self.feature_cols_deep = list(self.feature_cols_lightgbm)
        self._build_vehicle_profiles(df)

    def _save_profile(self) -> None:
        profile = {
            "defaults": self.defaults,
            "numeric_cols": sorted(self.numeric_cols),
            "categorical_cols": sorted(self.categorical_cols),
            "feature_cols_all": self.feature_cols_all,
            "feature_cols_lightgbm": self.feature_cols_lightgbm,
            "feature_cols_deep": self.feature_cols_deep,
            "vehicle_profiles_by_model": self.vehicle_profiles_by_model,
            "vehicle_profiles_by_make": self.vehicle_profiles_by_make,
        }
        joblib.dump(profile, self.profile_path)

    def _load_saved_profile(self) -> None:
        if not self.profile_path.exists():
            raise FileNotFoundError(
                "Dataset is missing and no saved feature profile was found. "
                f"Expected dataset at {self.data_path} or profile at {self.profile_path}."
            )

        profile = joblib.load(self.profile_path)
        self.defaults = dict(profile["defaults"])
        self.numeric_cols = set(profile["numeric_cols"])
        self.categorical_cols = set(profile["categorical_cols"])
        self.feature_cols_all = list(profile["feature_cols_all"])
        self.feature_cols_lightgbm = list(profile["feature_cols_lightgbm"])
        self.feature_cols_deep = list(profile["feature_cols_deep"])
        self.vehicle_profiles_by_model = dict(profile.get("vehicle_profiles_by_model", {}))
        self.vehicle_profiles_by_make = dict(profile.get("vehicle_profiles_by_make", {}))

    @staticmethod
    def _normalize_vehicle_model(raw: str) -> str:
        return re.sub(r"\s+", " ", str(raw).strip().lower())

    @staticmethod
    def _extract_vehicle_make(normalized_model: str) -> str:
        token = normalized_model.split(" ", 1)[0].strip()
        return token

    def _representative_profile_value(self, series: pd.Series, field: str) -> Any | None:
        if field in self.numeric_cols:
            values = pd.to_numeric(series, errors="coerce").dropna()
            if values.empty:
                return None
            return float(values.median())

        values = (
            series.dropna()
            .astype(str)
            .str.strip()
            .str.lower()
        )
        values = values[values != ""]
        if values.empty:
            return None
        mode = values.mode()
        if mode.empty:
            return None
        return str(mode.iloc[0]).strip().lower()

    def _build_vehicle_profiles(self, df: pd.DataFrame) -> None:
        self.vehicle_profiles_by_model = {}
        self.vehicle_profiles_by_make = {}
        if "vehicle_model" not in df.columns:
            return

        candidate_fields = [field for field in MODEL_PROFILE_FIELDS if field in df.columns]
        if not candidate_fields:
            return

        profile_df = df[["vehicle_model", *candidate_fields]].copy()
        profile_df["vehicle_model_norm"] = (
            profile_df["vehicle_model"]
            .astype(str)
            .map(self._normalize_vehicle_model)
        )
        profile_df = profile_df[profile_df["vehicle_model_norm"] != ""]
        if profile_df.empty:
            return

        profile_df["vehicle_make"] = profile_df["vehicle_model_norm"].map(self._extract_vehicle_make)
        profile_df = profile_df[profile_df["vehicle_make"] != ""]
        if profile_df.empty:
            return

        for model_key, group in profile_df.groupby("vehicle_model_norm", dropna=False):
            profile: dict[str, Any] = {}
            for field in candidate_fields:
                value = self._representative_profile_value(group[field], field)
                if value is not None:
                    profile[field] = value
            if profile:
                self.vehicle_profiles_by_model[str(model_key)] = profile

        for make_key, group in profile_df.groupby("vehicle_make", dropna=False):
            profile = {}
            for field in candidate_fields:
                value = self._representative_profile_value(group[field], field)
                if value is not None:
                    profile[field] = value
            if profile:
                self.vehicle_profiles_by_make[str(make_key)] = profile

    def _resolve_vehicle_profile(self, vehicle_model: Any) -> tuple[dict[str, Any], str | None]:
        if vehicle_model in (None, ""):
            return {}, None

        normalized = self._normalize_vehicle_model(str(vehicle_model))
        if not normalized:
            return {}, None

        exact = self.vehicle_profiles_by_model.get(normalized)
        if exact:
            return exact, "model_exact"

        for known_model, profile in self.vehicle_profiles_by_model.items():
            if normalized.startswith(known_model) or known_model in normalized:
                return profile, "model_partial"

        make = self._extract_vehicle_make(normalized)
        if make:
            make_profile = self.vehicle_profiles_by_make.get(make)
            if make_profile:
                return make_profile, "make_default"

        return {}, None

    @staticmethod
    def _is_unknown_text(value: Any) -> bool:
        if value in (None, ""):
            return True
        text = str(value).strip().lower()
        if text in UNKNOWN_TEXT_TOKENS:
            return True
        if "dont know" in text or "don't know" in text or "unknown" in text:
            return True
        return False

    def _apply_vehicle_profile_inference(
        self,
        *,
        session_features: dict[str, Any],
        parsed_features: dict[str, Any],
    ) -> dict[str, Any]:
        profile, source = self._resolve_vehicle_profile(session_features.get("vehicle_model"))
        if not profile:
            return {}

        allowed_fields = (
            MODEL_PROFILE_FALLBACK_FIELDS if source == "make_default" else MODEL_PROFILE_FIELDS
        )

        inferred: dict[str, Any] = {}
        for field in allowed_fields:
            if field not in profile or field not in self.defaults:
                continue
            if field in parsed_features:
                continue

            existing = session_features.get(field)
            if field in self.numeric_cols:
                if existing not in (None, ""):
                    continue
            elif not self._is_unknown_text(existing):
                continue

            inferred[field] = profile[field]

        return inferred

    def _make_training_frame(self, df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, pd.Series]:
        y = pd.to_numeric(df[TARGET_COL], errors="coerce")
        mask = y.notna()
        y = y.loc[mask].astype(float)

        X = df.loc[mask, feature_cols].copy()
        for col in feature_cols:
            if col in self.numeric_cols:
                X[col] = pd.to_numeric(X[col], errors="coerce").fillna(float(self.defaults[col]))
            else:
                X[col] = (
                    X[col]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .fillna(str(self.defaults[col]))
                    .astype("category")
                )

        return X, y

    def _load_or_train_models(self) -> None:
        lgbm_path = self.model_dir / "lightgbm_normal_non_leakage.joblib"
        deep_path = self.model_dir / "deeplearning_test_mlp.joblib"

        if lgbm_path.exists():
            try:
                self.models["lightgbm"] = joblib.load(lgbm_path)
            except Exception:
                if not self.data_path.exists():
                    raise
                self.models["lightgbm"] = self._train_lightgbm()
                joblib.dump(self.models["lightgbm"], lgbm_path)
        else:
            if not self.data_path.exists():
                raise FileNotFoundError(
                    f"LightGBM model not found at {lgbm_path}. "
                    "Provide the prebuilt model artifact or the training dataset."
                )
            self.models["lightgbm"] = self._train_lightgbm()
            joblib.dump(self.models["lightgbm"], lgbm_path)

        if deep_path.exists():
            try:
                self.models["deeplearning_test"] = joblib.load(deep_path)
            except Exception:
                if not self.data_path.exists():
                    return
                self.models["deeplearning_test"] = self._train_deep_test_model()
                joblib.dump(self.models["deeplearning_test"], deep_path)
        elif self.data_path.exists():
            self.models["deeplearning_test"] = self._train_deep_test_model()
            joblib.dump(self.models["deeplearning_test"], deep_path)

    def _load_training_df(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Cannot train models because dataset was not found at {self.data_path}. "
                "Provide the dataset or prebuilt model joblib files in the model directory."
            )
        return pd.read_csv(self.data_path, nrows=self._train_nrows(), low_memory=False)

    def _train_lightgbm(self) -> dict[str, Any]:
        df = self._load_training_df()
        X, y = self._make_training_frame(df, self.feature_cols_lightgbm)
        cat_cols = [c for c in self.feature_cols_lightgbm if c in self.categorical_cols]

        model = LGBMRegressor(
            objective="regression",
            random_state=42,
            n_estimators=500,
            learning_rate=0.05,
            num_leaves=63,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_alpha=0.05,
            reg_lambda=0.05,
        )
        model.fit(X, y, categorical_feature=cat_cols)

        return {
            "model": model,
            "feature_cols": list(self.feature_cols_lightgbm),
            "cat_cols": cat_cols,
            "kind": "lightgbm",
        }

    def _train_deep_test_model(self) -> dict[str, Any]:
        df = self._load_training_df()
        if len(df) > 120000:
            df = df.sample(n=120000, random_state=42)

        X, y = self._make_training_frame(df, self.feature_cols_deep)
        cat_cols = [c for c in self.feature_cols_deep if c in self.categorical_cols]
        num_cols = [c for c in self.feature_cols_deep if c in self.numeric_cols]

        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "num",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="median")),
                            ("scaler", StandardScaler()),
                        ]
                    ),
                    num_cols,
                ),
                (
                    "cat",
                    Pipeline(
                        steps=[
                            ("imputer", SimpleImputer(strategy="most_frequent")),
                            (
                                "encoder",
                                OrdinalEncoder(
                                    handle_unknown="use_encoded_value",
                                    unknown_value=-1,
                                ),
                            ),
                        ]
                    ),
                    cat_cols,
                ),
            ],
            remainder="drop",
        )

        model = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "mlp",
                    MLPRegressor(
                        hidden_layer_sizes=(128, 64),
                        activation="relu",
                        learning_rate_init=0.001,
                        max_iter=90,
                        early_stopping=True,
                        random_state=42,
                    ),
                ),
            ]
        )
        model.fit(X, y)

        return {
            "model": model,
            "feature_cols": list(self.feature_cols_deep),
            "cat_cols": cat_cols,
            "kind": "deeplearning_test",
        }

    def _normalize_feature_key(self, raw_key: str) -> str | None:
        key = raw_key.strip()
        if key in self.defaults:
            return key
        compact = (
            key.lower()
            .replace("-", "_")
            .replace(" ", "_")
            .replace("(", "_")
            .replace(")", "")
            .replace("/", "_")
            .replace("%", "pct")
            .replace("__", "_")
        )
        if compact in self.feature_aliases:
            return self.feature_aliases[compact]
        return None

    def _coerce_value(self, feature: str, value: Any) -> Any:
        if value is None:
            return None

        if feature == "retreaded":
            text = str(value).strip().lower()
            if text in {"yes", "y", "true", "1"}:
                return 1.0
            if text in {"no", "n", "false", "0"}:
                return 0.0

        if feature in self.numeric_cols:
            if isinstance(value, (int, float, np.integer, np.floating)):
                return float(value)
            cleaned = str(value).strip().lower().replace(",", "")
            match = re.search(r"-?\d+(\.\d+)?", cleaned)
            if match:
                return float(match.group())
            return None

        return str(value).strip().lower()

    def _derive_engineered_features(self, features: dict[str, Any]) -> None:
        def _safe_float(name: str) -> float:
            value = self._coerce_value(name, features.get(name, self.defaults.get(name)))
            if value is None:
                return float(self.defaults.get(name, 0.0))
            return float(value)

        std_depth = _safe_float("Standard_tread_depth(mm)")
        current_depth = _safe_float("current_tread_depth(mm)")
        avg_pressure = _safe_float("average_inflation_pressure(psi)")
        rec_pressure = _safe_float("recommended_inflation_pressure(psi)")
        km_driven = _safe_float("kilometers_driven(km)")
        expected_life = _safe_float("expected_tyre_life(km)")

        tread_depth_used_mm = max(0.0, std_depth - current_depth)
        features["tread_depth_used_mm"] = tread_depth_used_mm
        features["tread_depth_used_pct"] = tread_depth_used_mm / std_depth if std_depth > 0 else 0.0
        features["pressure_gap_psi"] = avg_pressure - rec_pressure
        features["pressure_gap_pct"] = (avg_pressure - rec_pressure) / rec_pressure if rec_pressure > 0 else 0.0
        features["km_used_ratio_vs_expected"] = km_driven / expected_life if expected_life > 0 else 0.0

    def normalize_features(self, incoming: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
        normalized: dict[str, Any] = {}
        provided_keys: set[str] = set()

        for raw_key, raw_value in incoming.items():
            canonical = self._normalize_feature_key(raw_key)
            if not canonical:
                continue
            coerced = self._coerce_value(canonical, raw_value)
            if coerced is None:
                continue
            normalized[canonical] = coerced
            provided_keys.add(canonical)

        merged = dict(self.defaults)
        merged.update(normalized)
        self._derive_engineered_features(merged)

        defaulted = sorted(
            [
                col
                for col in self.feature_cols_all
                if col not in provided_keys and col in merged
            ]
        )
        return merged, defaulted

    def _predict_from_record(self, model_name: str, record: dict[str, Any]) -> float:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not available")
        payload = self.models[model_name]
        model = payload["model"]
        feature_cols: list[str] = payload["feature_cols"]

        row = pd.DataFrame([{col: record.get(col, self.defaults[col]) for col in feature_cols}])
        for col in feature_cols:
            if col in self.numeric_cols:
                row[col] = pd.to_numeric(row[col], errors="coerce").fillna(float(self.defaults[col]))
            else:
                row[col] = (
                    row[col]
                    .astype("string")
                    .str.strip()
                    .str.lower()
                    .fillna(str(self.defaults[col]))
                    .astype("category")
                )

        pred = float(model.predict(row)[0])
        return max(0.0, pred)

    def predict(self, features: dict[str, Any], model_name: str) -> PredictResponse:
        normalized, defaulted = self.normalize_features(features)
        pred_km = self._predict_from_record(model_name=model_name, record=normalized)
        pred_miles = pred_km * 0.621371
        model_feature_cols = set(self.models[model_name]["feature_cols"])
        model_defaulted = [col for col in defaulted if col in model_feature_cols]

        return PredictResponse(
            model=model_name,
            predicted_rul_km=round(pred_km, 2),
            predicted_rul_miles=round(pred_miles, 2),
            defaults_used=model_defaulted,
            defaults_used_count=len(model_defaulted),
            normalized_features=normalized,
        )

    def _extract_with_gemini(self, user_text: str) -> dict[str, Any]:
        api_key = self._gemini_api_key()
        if not api_key:
            return {}

        model = self._gemini_model_name()
        prompt = f"""
You extract tyre prediction features from user text.
Return only valid JSON object, no markdown, no extra keys.
If value not present, use null.

Allowed keys:
{json.dumps(self.feature_cols_all, ensure_ascii=True)}

Rules:
- Numeric fields must be numbers.
- Keep categorical fields lower-case strings.
- Convert miles to kilometers for any distance field.
- Convert booleans like retreaded yes/no into 1 or 0.

User message:
{user_text}
""".strip()

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
            },
        }

        with httpx.Client(timeout=20.0) as client:
            resp = client.post(url, headers=self._gemini_headers(api_key), json=body)
            resp.raise_for_status()
            payload = resp.json()

        candidates = payload.get("candidates", [])
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        if not parts:
            return {}

        raw_text = parts[0].get("text", "").strip()
        if not raw_text:
            return {}

        if raw_text.startswith("```"):
            raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        extracted = json.loads(raw_text)
        if isinstance(extracted, dict):
            return extracted
        return {}

    def _extract_with_regex(self, user_text: str) -> dict[str, Any]:
        text = user_text.lower().replace(",", "")
        parsed: dict[str, Any] = {}

        def _match(pattern: str) -> re.Match[str] | None:
            return re.search(pattern, text, flags=re.IGNORECASE)

        m = _match(r"(?:current\s*)?tread(?:\s*depth)?[^\d]{0,12}(-?\d+(?:\.\d+)?)\s*mm")
        if not m:
            m = _match(r"(-?\d+(?:\.\d+)?)\s*mm[^\n.]{0,20}tread")
        if m:
            parsed["current_tread_depth(mm)"] = float(m.group(1))

        m = _match(r"(?:standard|new|original)\s*tread(?:\s*depth)?[^\d]{0,12}(-?\d+(?:\.\d+)?)\s*mm")
        if m:
            parsed["Standard_tread_depth(mm)"] = float(m.group(1))

        m = _match(r"(?:driven|odometer|odo|mileage)[^\d]{0,20}(-?\d+(?:\.\d+)?)\s*(km|kilometers|mi|miles)")
        if m:
            value = float(m.group(1))
            unit = m.group(2).lower()
            parsed["kilometers_driven(km)"] = value * 1.60934 if unit in {"mi", "miles"} else value

        m = _match(r"(?:average\s*)?(?:inflation\s*)?pressure[^\d]{0,12}(-?\d+(?:\.\d+)?)\s*psi")
        if m:
            parsed["average_inflation_pressure(psi)"] = float(m.group(1))
        if "average_inflation_pressure(psi)" not in parsed:
            m = _match(r"(?:psi|pressure)[^\d-]{0,8}(-?\d+(?:\.\d+)?)")
            if m:
                parsed["average_inflation_pressure(psi)"] = float(m.group(1))
        if "average_inflation_pressure(psi)" not in parsed:
            m = _match(r"(-?\d+(?:\.\d+)?)\s*psi")
            if m:
                parsed["average_inflation_pressure(psi)"] = float(m.group(1))

        m = _match(r"(?:recommended|target)\s*(?:inflation\s*)?pressure[^\d]{0,12}(-?\d+(?:\.\d+)?)\s*psi")
        if m:
            parsed["recommended_inflation_pressure(psi)"] = float(m.group(1))

        m = _match(r"(?:tyre|tire)?\s*age[^\d]{0,12}(-?\d+(?:\.\d+)?)\s*(?:years?|yrs?)")
        if not m:
            m = _match(r"(-?\d+(?:\.\d+)?)\s*(?:years?|yrs?)\s*(?:old)?")
        if m:
            parsed["tyre_age(years)"] = float(m.group(1))

        m = _match(r"(-?\d+(?:\.\d+)?)\s*(?:punctures?|flats?)")
        if m:
            parsed["number_of_punctures"] = float(m.group(1))

        if "retreaded yes" in text or "retreaded: yes" in text or "retreaded true" in text:
            parsed["retreaded"] = 1
        if "retreaded no" in text or "retreaded: no" in text or "retreaded false" in text:
            parsed["retreaded"] = 0

        road_tokens = {
            "smooth": "smooth",
            "rough": "rough",
            "mixed": "mixed",
            "wet": "wet",
        }
        for token, canonical in road_tokens.items():
            if token in text:
                parsed["road_condition"] = canonical
                break

        weather_tokens = {
            "dry": "dry",
            "rain": "rainy",
            "snow": "snowy",
            "humid": "tropical and humid",
            "mixed": "mixed conditions",
        }
        for token, canonical in weather_tokens.items():
            if token in text:
                parsed["weather_condition"] = canonical
                break

        if "driven axle" in text or "driven wheels" in text:
            parsed["axle_type(driven/dead)"] = "driven"
        if "dead axle" in text or "non driven axle" in text:
            parsed["axle_type(driven/dead)"] = "dead"

        vehicle_match = _match(
            r"(?:i\s+have|my\s+car\s+is|my\s+car\s+model\s+is|vehicle\s+is|car\s+is|model\s+is)\s+(?:a|an)?\s*([a-z0-9][a-z0-9 +\-]{1,30}?)(?=\s+and|\s*,|$)"
        )
        if vehicle_match:
            parsed["vehicle_model"] = vehicle_match.group(1).strip()

        def _clean_brand_candidate(raw: str) -> str | None:
            candidate = re.sub(r"\s+", " ", raw.strip(" .,:;!?")).lower()
            if not candidate or len(candidate) < 3:
                return None
            if candidate.startswith(("and ", "but ", "or ")):
                return None
            if any(token in candidate for token in ("psi", "pressure", "mm", "year", "km", "mile")):
                return None
            if any(token in candidate for token in ("dont know", "don't know", "unknown", "not sure")):
                return None
            if candidate in {"and", "but", "i", "ive", "i've", "it", "its", "it's"}:
                return None
            return candidate

        generic_brand = _match(r"(?:generic|unknown)[^\n.]{0,14}(?:tyre|tire)\s*brand")
        if generic_brand:
            parsed["tyre_brand"] = "generic / unknown brand"

        unknown_brand = _match(r"(?:dont know|don't know|not sure)[^\n.]{0,20}(?:tyre|tire)\s*brand")
        if unknown_brand and "tyre_brand" not in parsed:
            parsed["tyre_brand"] = "unknown brand"

        if "tyre_brand" not in parsed:
            explicit_brand = _match(
                r"(?:tyre|tire)\s*brand(?:\s*is|:)?\s*([a-z0-9 +/\-]{3,40}?)(?=\s+(?:and|but|with|its|it's|i|i've)\b|$)"
            )
            if explicit_brand:
                candidate = _clean_brand_candidate(explicit_brand.group(1))
                if candidate:
                    parsed["tyre_brand"] = candidate

        if "tyre_brand" not in parsed:
            brand_match = _match(r"(?:tyre|tire)\s*brand[^\w]{0,4}([a-z0-9 +/\-]{3,40})")
            if brand_match:
                candidate = _clean_brand_candidate(brand_match.group(1))
                if candidate:
                    parsed["tyre_brand"] = candidate

        return parsed

    def extract_features(self, user_text: str) -> tuple[dict[str, Any], str]:
        extracted: dict[str, Any] = {}
        extractor = "regex"

        try:
            gemini = self._extract_with_gemini(user_text)
            if gemini:
                extracted = gemini
                extractor = "gemini"
        except Exception:
            extracted = {}
            extractor = "regex"

        if not extracted:
            extracted = self._extract_with_regex(user_text)

        # Keep only values that came from user extraction (not defaults)
        cleaned: dict[str, Any] = {}
        for key, value in extracted.items():
            canonical = self._normalize_feature_key(key)
            if not canonical:
                continue
            coerced = self._coerce_value(canonical, value)
            if coerced is not None:
                cleaned[canonical] = coerced

        # If a parsed value contributes to engineered fields, include the derived fields.
        if cleaned:
            with_derived = dict(cleaned)
            merged = dict(self.defaults)
            merged.update(cleaned)
            self._derive_engineered_features(merged)
            for col in (
                "tread_depth_used_mm",
                "tread_depth_used_pct",
                "pressure_gap_psi",
                "pressure_gap_pct",
                "km_used_ratio_vs_expected",
            ):
                with_derived[col] = merged[col]
            cleaned = with_derived

        return cleaned, extractor

    def get_or_create_session(self, session_id: str | None) -> tuple[str, dict[str, Any]]:
        if session_id and session_id in self.sessions:
            return session_id, self.sessions[session_id]

        new_id = session_id or str(uuid.uuid4())
        if new_id not in self.sessions:
            self.sessions[new_id] = {
                "features": {},
                "model": DEFAULT_MODEL,
                "has_prediction": False,
                "followup_count": 0,
                "noinfo_count": 0,
                "history": [],
            }
        return new_id, self.sessions[new_id]

    def should_use_defaults(self, user_text: str, force_predict: bool) -> bool:
        if force_predict:
            return True
        lowered = user_text.lower()
        return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in DEFAULT_INTENT_TOKENS)

    def user_signals_done_sharing(self, user_text: str) -> bool:
        lowered = user_text.lower()
        return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in DONE_SHARING_TOKENS)

    def user_signals_no_valid_response(self, user_text: str) -> bool:
        lowered = user_text.lower()
        return any(re.search(rf"\b{re.escape(token)}\b", lowered) for token in NO_VALID_RESPONSE_TOKENS)

    @staticmethod
    def _normalize_text(user_text: str) -> str:
        return re.sub(r"\s+", " ", user_text.lower()).strip()

    def is_smalltalk_turn(self, user_text: str) -> bool:
        normalized = self._normalize_text(user_text).rstrip(".!?")
        return normalized in SMALL_TALK_TOKENS

    @staticmethod
    def _format_feature_value(value: Any) -> str:
        if isinstance(value, float):
            rounded = round(value, 3)
            if rounded.is_integer():
                return str(int(rounded))
            return str(rounded)
        return str(value)

    @staticmethod
    def _has_value(record: dict[str, Any], field: str) -> bool:
        return field in record and record[field] not in ("", None)

    def _friendly_name(self, field: str) -> str:
        return FRIENDLY_FIELD_NAMES.get(field, field)

    def _summarize_parsed(self, parsed_features: dict[str, Any]) -> str:
        interesting = [
            "vehicle_model",
            "tyre_brand",
            "average_inflation_pressure(psi)",
            "tyre_age(years)",
            "current_tread_depth(mm)",
            "kilometers_driven(km)",
        ]
        bits: list[str] = []
        for field in interesting:
            if field not in parsed_features:
                continue
            value = parsed_features[field]
            pretty_field = self._friendly_name(field)
            if field == "average_inflation_pressure(psi)":
                bits.append(f"{pretty_field}: {self._format_feature_value(value)} psi")
            elif field == "current_tread_depth(mm)":
                bits.append(f"{pretty_field}: {self._format_feature_value(value)} mm")
            elif field == "kilometers_driven(km)":
                bits.append(f"{pretty_field}: {self._format_feature_value(value)} km")
            elif field == "tyre_age(years)":
                bits.append(f"{pretty_field}: {self._format_feature_value(value)} years")
            else:
                bits.append(f"{pretty_field}: {self._format_feature_value(value)}")
        if not bits:
            return ""
        return "I captured: " + ", ".join(bits) + "."

    def _friendly_missing(self, missing_fields: list[str]) -> list[str]:
        return [self._friendly_name(field) for field in missing_fields]

    @staticmethod
    def _history_tail(history: list[dict[str, str]], max_items: int = 6) -> list[dict[str, str]]:
        if not history:
            return []
        clipped = history[-max_items:]
        return [item for item in clipped if item.get("text", "").strip()]

    def _append_history(self, session: dict[str, Any], role: str, text: str) -> None:
        if "history" not in session:
            session["history"] = []
        session["history"].append({"role": role, "text": text})
        if len(session["history"]) > 16:
            session["history"] = session["history"][-16:]

    def _conversational_response(
        self,
        *,
        session: dict[str, Any],
        user_message: str,
        deterministic_message: str,
        needs_follow_up: bool,
        missing_fields: list[str],
        suggest_autofill: bool,
        parsed_features: dict[str, Any],
        prediction: PredictResponse | None,
    ) -> str:
        api_key = self._gemini_api_key()
        if not api_key:
            return self._local_conversational_response(
                user_message=user_message,
                deterministic_message=deterministic_message,
                needs_follow_up=needs_follow_up,
                missing_fields=missing_fields,
                suggest_autofill=suggest_autofill,
                parsed_features=parsed_features,
                prediction=prediction,
                followup_count=int(session.get("followup_count", 0)),
            )

        prompt_payload = {
            "goal": "Respond like a helpful chatbot for tyre-life estimation.",
            "style_rules": [
                "Natural and conversational (not robotic).",
                "Acknowledge user info you understood.",
                "If follow-up needed, ask only 1-2 short questions.",
                "Do not invent numeric values.",
                "Keep reply under 140 words.",
            ],
            "needs_follow_up": needs_follow_up,
            "missing_fields_friendly": self._friendly_missing(missing_fields),
            "suggest_autofill": suggest_autofill,
            "parsed_features": parsed_features,
            "prediction": prediction.model_dump() if prediction else None,
            "fallback_message": deterministic_message,
            "followup_count": int(session.get("followup_count", 0)),
            "recent_history": self._history_tail(session.get("history", [])),
            "latest_user_message": user_message,
        }

        system_prompt = "\n".join(
            [
                "You are TireLife assistant for tyre remaining-life estimation.",
                "Be warm and conversational, like a modern chat assistant.",
                "Acknowledge captured details before asking follow-up questions.",
                "Ask at most one focused follow-up question when possible.",
                "Do not invent user values.",
                "If suggest_autofill=false, do not mention defaults.",
                "If suggest_autofill=true, mention defaults as an option in one short sentence.",
            ]
        )

        history_contents: list[dict[str, Any]] = []
        for turn in self._history_tail(session.get("history", []), max_items=8):
            role = "user" if turn.get("role") == "user" else "model"
            history_contents.append({"role": role, "parts": [{"text": turn["text"]}]})
        history_contents.append(
            {
                "role": "user",
                "parts": [
                    {
                        "text": (
                            "Conversation state:\n"
                            + json.dumps(prompt_payload, ensure_ascii=True)
                            + "\n\nWrite the next assistant response."
                        )
                    }
                ],
            }
        )

        body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": history_contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "responseMimeType": "text/plain",
            },
        }

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._chat_model_name()}:generateContent"

        try:
            with httpx.Client(timeout=20.0) as client:
                resp = client.post(url, headers=self._gemini_headers(api_key), json=body)
                resp.raise_for_status()
                payload = resp.json()
            text = (
                payload.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
                .strip()
            )
            if text:
                return text
        except Exception:
            return self._local_conversational_response(
                user_message=user_message,
                deterministic_message=deterministic_message,
                needs_follow_up=needs_follow_up,
                missing_fields=missing_fields,
                suggest_autofill=suggest_autofill,
                parsed_features=parsed_features,
                prediction=prediction,
                followup_count=int(session.get("followup_count", 0)),
            )

        return self._local_conversational_response(
            user_message=user_message,
            deterministic_message=deterministic_message,
            needs_follow_up=needs_follow_up,
            missing_fields=missing_fields,
            suggest_autofill=suggest_autofill,
            parsed_features=parsed_features,
            prediction=prediction,
            followup_count=int(session.get("followup_count", 0)),
        )

    def _local_conversational_response(
        self,
        *,
        user_message: str,
        deterministic_message: str,
        needs_follow_up: bool,
        missing_fields: list[str],
        suggest_autofill: bool,
        parsed_features: dict[str, Any],
        prediction: PredictResponse | None,
        followup_count: int,
    ) -> str:
        recap = self._summarize_parsed(parsed_features)

        if not needs_follow_up and prediction is not None:
            if deterministic_message and not deterministic_message.startswith("Estimated remaining tyre life:"):
                return deterministic_message
            message = (
                f"Thanks, based on what you shared I estimate about {prediction.predicted_rul_km:,.0f} km "
                f"({prediction.predicted_rul_miles:,.0f} miles) remaining."
            )
            if prediction.defaults_used_count:
                message += (
                    f" I still had to assume {prediction.defaults_used_count} field(s), "
                    "so we can refine this if you share more details."
                )
            return message

        if not needs_follow_up:
            return deterministic_message

        lines: list[str] = []
        if recap:
            lines.append(recap)
        elif followup_count >= 2:
            lines.append("Thanks, we can still do this with partial info.")

        needs_wear = any(field in missing_fields for field in WEAR_SIGNAL_FIELDS)
        needs_context = any(field in missing_fields for field in CONTEXT_SIGNAL_FIELDS)
        lowered = user_message.lower()
        user_said_no_more = self.user_signals_done_sharing(user_message)

        if needs_wear and needs_context:
            lines.append(
                "To make this estimate meaningful, could you share one wear detail "
                "(tread depth in mm or distance driven) and one context detail "
                "(pressure or tire age)?"
            )
        elif needs_wear:
            if "year" in lowered and "drive" in lowered:
                lines.append(
                    "I captured tire age, which helps for context. For wear, do you know either "
                    "tread depth (mm) or rough distance on these tires?"
                )
            else:
                lines.append(
                    "Great, that helps. Could you add one wear detail: either current tread depth (mm) "
                    "or roughly how far these tires have been driven?"
                )
        elif needs_context:
            lines.append(
                "Nice, we’re close. Could you share one context detail: either current pressure (psi) "
                "or tire age in years?"
            )
        elif missing_fields:
            friendly = self._friendly_missing(missing_fields)
            ask = ", ".join(friendly[:2])
            lines.append(f"Could you share {ask} so I can tighten the estimate?")
        else:
            lines.append("Could you share one more detail so I can refine the estimate?")

        if suggest_autofill or user_said_no_more:
            lines.append("If that’s all you know right now, just tell me and I’ll estimate automatically with defaults.")

        return "\n\n".join(lines).strip()

    def chat(self, request: ChatRequest) -> ChatResponse:
        session_id, session = self.get_or_create_session(request.session_id)
        if "features" not in session:
            session["features"] = {}
        if "has_prediction" not in session:
            session["has_prediction"] = False
        if "followup_count" not in session:
            session["followup_count"] = 0
        if "noinfo_count" not in session:
            session["noinfo_count"] = 0

        session["model"] = request.model
        self._append_history(session, "user", request.message)

        user_accepts_defaults = self.should_use_defaults(request.message, request.force_predict)
        user_is_done_sharing = self.user_signals_done_sharing(request.message)
        user_no_valid_response = self.user_signals_no_valid_response(request.message)
        smalltalk_turn = self.is_smalltalk_turn(request.message)
        parsed_features, extractor = self.extract_features(request.message)
        if parsed_features:
            user_no_valid_response = False
        required_in_turn = [field for field in REQUIRED_CHAT_FIELDS if field in parsed_features]
        if parsed_features:
            session["noinfo_count"] = 0
        elif not smalltalk_turn:
            session["noinfo_count"] = int(session.get("noinfo_count", 0)) + 1
        session["features"].update(parsed_features)
        inferred_profile_features = self._apply_vehicle_profile_inference(
            session_features=session["features"],
            parsed_features=parsed_features,
        )
        if inferred_profile_features:
            session["features"].update(inferred_profile_features)

        wear_present = [f for f in WEAR_SIGNAL_FIELDS if self._has_value(session["features"], f)]
        context_present = [f for f in CONTEXT_SIGNAL_FIELDS if self._has_value(session["features"], f)]

        if smalltalk_turn and not parsed_features and not user_accepts_defaults and not user_is_done_sharing:
            if session.get("has_prediction"):
                prediction = self.predict(features=session["features"], model_name=session["model"])
                deterministic_message = (
                    f"Hey! Your latest estimate is about {prediction.predicted_rul_km:,.0f} km "
                    f"({prediction.predicted_rul_miles:,.0f} miles) remaining. "
                    "Share any new tire detail if you want me to refine it, or press New Case to start fresh."
                )
                assistant_message = self._conversational_response(
                    session=session,
                    user_message=request.message,
                    deterministic_message=deterministic_message,
                    needs_follow_up=False,
                    missing_fields=[],
                    suggest_autofill=False,
                    parsed_features={},
                    prediction=prediction,
                )
                self._append_history(session, "assistant", assistant_message)
                return ChatResponse(
                    session_id=session_id,
                    needs_follow_up=False,
                    assistant_message=assistant_message,
                    missing_fields=[],
                    suggest_autofill=False,
                    extractor=extractor,
                    parsed_features=parsed_features,
                    prediction=prediction,
                )

            deterministic_message = (
                "Hey! Share whatever tire details you know, and I’ll ask follow-up questions "
                "until we can make a solid estimate."
            )
            assistant_message = self._conversational_response(
                session=session,
                user_message=request.message,
                deterministic_message=deterministic_message,
                needs_follow_up=True,
                missing_fields=REQUIRED_CHAT_FIELDS,
                suggest_autofill=False,
                parsed_features={},
                prediction=None,
            )
            self._append_history(session, "assistant", assistant_message)
            return ChatResponse(
                session_id=session_id,
                needs_follow_up=True,
                assistant_message=assistant_message,
                missing_fields=REQUIRED_CHAT_FIELDS,
                suggest_autofill=False,
                extractor=extractor,
                parsed_features=parsed_features,
                prediction=None,
            )

        if (not wear_present or not context_present) and not user_accepts_defaults:
            recap = self._summarize_parsed(parsed_features)
            missing_fields: list[str] = []
            prompt_lines: list[str] = []
            if not wear_present:
                missing_fields.extend(WEAR_SIGNAL_FIELDS)
                prompt_lines.append(
                    "- Please share either `current tread depth (mm)` or `distance driven on these tires (km/miles)`."
                )
            if not context_present:
                missing_fields.extend(CONTEXT_SIGNAL_FIELDS)
                prompt_lines.append(
                    "- Please share either `tire pressure (psi)` or `tire age (years)`."
                )

            session["followup_count"] = int(session.get("followup_count", 0)) + 1
            should_auto_predict = bool(
                user_is_done_sharing
                or user_accepts_defaults
                or (
                    (
                        user_no_valid_response
                        or int(session.get("noinfo_count", 0)) >= AUTO_PREDICT_AFTER_NOINFO_TURNS
                    )
                    and int(session.get("followup_count", 0)) >= 2
                )
            )
            suggest_autofill = bool(
                not should_auto_predict and session["followup_count"] >= AUTOFILL_OFFER_AFTER_FOLLOWUPS
            )

            if should_auto_predict:
                prediction = self.predict(features=session["features"], model_name=session["model"])
                session["has_prediction"] = True
                session["followup_count"] = 0
                session["noinfo_count"] = 0
                deterministic_message = (
                    f"I’ll continue with defaults for missing fields and estimate now: "
                    f"{prediction.predicted_rul_km:,.0f} km ({prediction.predicted_rul_miles:,.0f} miles) remaining "
                    f"using {prediction.model}."
                )
                assistant_message = self._conversational_response(
                    session=session,
                    user_message=request.message,
                    deterministic_message=deterministic_message,
                    needs_follow_up=False,
                    missing_fields=sorted(set(missing_fields)),
                    suggest_autofill=False,
                    parsed_features=parsed_features,
                    prediction=prediction,
                )
                self._append_history(session, "assistant", assistant_message)
                return ChatResponse(
                    session_id=session_id,
                    needs_follow_up=False,
                    assistant_message=assistant_message,
                    missing_fields=[],
                    suggest_autofill=False,
                    extractor=extractor,
                    parsed_features=parsed_features,
                    prediction=prediction,
                )

            assistant_message = (
                (recap + "\n\n") if recap else ""
            ) + "I need one wear signal plus one context signal for a solid estimate.\n" + "\n".join(prompt_lines)
            if suggest_autofill:
                assistant_message += (
                    "\n\nIf that’s all you know, tell me and I’ll estimate automatically with defaults."
                )

            assistant_message = self._conversational_response(
                session=session,
                user_message=request.message,
                deterministic_message=assistant_message,
                needs_follow_up=True,
                missing_fields=sorted(set(missing_fields)),
                suggest_autofill=suggest_autofill,
                parsed_features=parsed_features,
                prediction=None,
            )
            self._append_history(session, "assistant", assistant_message)

            return ChatResponse(
                session_id=session_id,
                needs_follow_up=True,
                assistant_message=assistant_message,
                missing_fields=sorted(set(missing_fields)),
                suggest_autofill=suggest_autofill,
                extractor=extractor,
                parsed_features=parsed_features,
                prediction=None,
            )

        # Prevent stale context reuse after a prior prediction. If the current
        # turn only updates a tiny subset, ask for fresh core values first.
        if (
            session.get("has_prediction")
            and len(required_in_turn) < MIN_REQUIRED_FIELDS_UPDATED_PER_TURN
            and not user_accepts_defaults
            and not user_is_done_sharing
            and not user_no_valid_response
            and not smalltalk_turn
        ):
            missing_fresh_required = [
                field for field in REQUIRED_CHAT_FIELDS if field not in required_in_turn
            ]
            stored_core = [
                f"- {field}: {self._format_feature_value(session['features'].get(field))}"
                for field in REQUIRED_CHAT_FIELDS
                if field in session["features"]
            ]
            fresh_hints = [
                f"- {REQUIRED_CHAT_FIELD_HINTS[field]}" for field in missing_fresh_required
            ]
            assistant_message = (
                "I want to avoid reusing old values from a previous estimate.\n"
                "Current stored core values in this chat:\n"
                + "\n".join(stored_core)
                + "\n\nFor a fresh prediction, please update these:\n"
                + "\n".join(fresh_hints)
            )
            session["followup_count"] = int(session.get("followup_count", 0)) + 1
            suggest_autofill = bool(
                user_is_done_sharing or session["followup_count"] >= AUTOFILL_OFFER_AFTER_FOLLOWUPS
            )
            if suggest_autofill:
                assistant_message += (
                    "\n\nIf you'd rather keep the old values, reply with `use defaults`."
                )

            assistant_message = self._conversational_response(
                session=session,
                user_message=request.message,
                deterministic_message=assistant_message,
                needs_follow_up=True,
                missing_fields=missing_fresh_required,
                suggest_autofill=suggest_autofill,
                parsed_features=parsed_features,
                prediction=None,
            )
            self._append_history(session, "assistant", assistant_message)
            return ChatResponse(
                session_id=session_id,
                needs_follow_up=True,
                assistant_message=assistant_message,
                missing_fields=missing_fresh_required,
                suggest_autofill=suggest_autofill,
                extractor=extractor,
                parsed_features=parsed_features,
                prediction=None,
            )

        prediction = self.predict(features=session["features"], model_name=session["model"])
        session["has_prediction"] = True
        session["followup_count"] = 0

        critical_defaulted = [
            field
            for field in (REQUIRED_CHAT_FIELDS + ACCURACY_BOOST_CHAT_FIELDS)
            if field in prediction.defaults_used
        ]

        if len(critical_defaulted) >= ACCURACY_FOLLOW_UP_TRIGGER_COUNT and not user_accepts_defaults:
            if user_is_done_sharing or user_no_valid_response:
                session["followup_count"] = 0
                session["noinfo_count"] = 0
            else:
                session["followup_count"] = int(session.get("followup_count", 0)) + 1

            if user_is_done_sharing or user_no_valid_response:
                assistant_message = (
                    f"Estimated remaining tyre life: {prediction.predicted_rul_km:,.0f} km "
                    f"({prediction.predicted_rul_miles:,.0f} miles) using {prediction.model}."
                )
                if prediction.defaults_used_count:
                    assistant_message += (
                        f" I filled {prediction.defaults_used_count} missing feature(s) with defaults."
                    )
                assistant_message = self._conversational_response(
                    session=session,
                    user_message=request.message,
                    deterministic_message=assistant_message,
                    needs_follow_up=False,
                    missing_fields=[],
                    suggest_autofill=False,
                    parsed_features=parsed_features,
                    prediction=prediction,
                )
                self._append_history(session, "assistant", assistant_message)
                return ChatResponse(
                    session_id=session_id,
                    needs_follow_up=False,
                    assistant_message=assistant_message,
                    missing_fields=[],
                    suggest_autofill=False,
                    extractor=extractor,
                    parsed_features=parsed_features,
                    prediction=prediction,
                )

            missing_for_accuracy = [
                field for field in ACCURACY_BOOST_CHAT_FIELDS if field in critical_defaulted
            ][:4]

            followup_lines = [
                f"- {ACCURACY_BOOST_FIELD_HINTS[field]}" for field in missing_for_accuracy
            ]
            assistant_message = (
                "I can improve this estimate with a couple of extra details.\n"
                "If you know any of these, please share:\n"
                + "\n".join(followup_lines)
            )
            suggest_autofill = bool(
                session["followup_count"] >= AUTOFILL_OFFER_AFTER_FOLLOWUPS
            )
            if suggest_autofill:
                assistant_message += (
                    "\n\nIf that’s all you know, say that and I’ll keep estimating with defaults."
                )

            assistant_message = self._conversational_response(
                session=session,
                user_message=request.message,
                deterministic_message=assistant_message,
                needs_follow_up=True,
                missing_fields=missing_for_accuracy,
                suggest_autofill=suggest_autofill,
                parsed_features=parsed_features,
                prediction=prediction,
            )
            self._append_history(session, "assistant", assistant_message)
            return ChatResponse(
                session_id=session_id,
                needs_follow_up=True,
                assistant_message=assistant_message,
                missing_fields=missing_for_accuracy,
                suggest_autofill=suggest_autofill,
                extractor=extractor,
                parsed_features=parsed_features,
                prediction=None,
            )

        assistant_message = (
            f"Estimated remaining tyre life: {prediction.predicted_rul_km:,.0f} km "
            f"({prediction.predicted_rul_miles:,.0f} miles) using {prediction.model}."
        )
        if prediction.defaults_used_count:
            assistant_message += (
                f" I auto-filled {prediction.defaults_used_count} feature(s) with dataset defaults."
            )

        assistant_message = self._conversational_response(
            session=session,
            user_message=request.message,
            deterministic_message=assistant_message,
            needs_follow_up=False,
            missing_fields=[],
            suggest_autofill=False,
            parsed_features=parsed_features,
            prediction=prediction,
        )
        self._append_history(session, "assistant", assistant_message)

        return ChatResponse(
            session_id=session_id,
            needs_follow_up=False,
            assistant_message=assistant_message,
            missing_fields=[],
            suggest_autofill=False,
            extractor=extractor,
            parsed_features=parsed_features,
            prediction=prediction,
        )


service = TireRULService(data_path=DATA_PATH, model_dir=MODEL_DIR)

app = FastAPI(title="TireLife FastAPI Backend", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    service.startup()


@app.get("/api/health", response_model=HealthResponse)
def health() -> HealthResponse:
    has_gemini = bool(service._gemini_api_key())
    return HealthResponse(
        status="ok",
        dataset_path=str(service.data_path),
        artifacts_dir=str(ARTIFACTS_DIR),
        model_dir=str(service.model_dir),
        frontend_dist_dir=str(FRONTEND_DIST_DIR),
        frontend_static_enabled=(FRONTEND_DIST_DIR / "index.html").exists(),
        models_available=sorted(service.models.keys()),
        required_chat_fields=REQUIRED_CHAT_FIELDS,
        gemini_extraction_enabled=has_gemini,
        gemini_chat_enabled=has_gemini,
        gemini_model=service._chat_model_name() if has_gemini else None,
        gemini_key_source=service._gemini_key_source(),
    )


@app.post("/api/config/gemini", response_model=GeminiConfigResponse)
def configure_gemini(request: GeminiConfigRequest) -> GeminiConfigResponse:
    try:
        return service.configure_gemini(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/predict", response_model=PredictResponse)
def predict(request: PredictRequest) -> PredictResponse:
    try:
        return service.predict(features=request.features, model_name=request.model)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        return service.chat(request)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _register_frontend_routes() -> None:
    index_path = FRONTEND_DIST_DIR / "index.html"
    if not index_path.exists():
        return

    @app.get("/", include_in_schema=False)
    def serve_frontend_index() -> FileResponse:
        return FileResponse(index_path)

    @app.get("/{full_path:path}", include_in_schema=False)
    def serve_frontend_asset_or_index(full_path: str) -> FileResponse:
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404, detail="API route not found")

        requested_path = (FRONTEND_DIST_DIR / full_path).resolve()
        try:
            requested_path.relative_to(FRONTEND_DIST_DIR)
        except ValueError:
            return FileResponse(index_path)

        if requested_path.is_file():
            return FileResponse(requested_path)

        return FileResponse(index_path)


_register_frontend_routes()
