"""
headline_app.py — نسخه‌ی کوانتومی «هد لاین» (v4.2-QUANTUM, Ultimate UI)
"""
from __future__ import annotations

import cmath
import hashlib
import logging
import math
import os
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Iterable

import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, Response, g, jsonify, request
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

# ============================================================
# 1) Config
# ============================================================
@dataclass(frozen=True)
class Config:
    debug: bool = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    host: str = os.getenv("APP_HOST", "0.0.0.0")
    port: int = int(os.getenv("APP_PORT", "5000"))
    max_content_length: int = int(os.getenv("MAX_CONTENT_LENGTH", str(10 * 1024 * 1024)))
    min_image_side: int = int(os.getenv("MIN_IMAGE_SIDE", "240"))
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()
    rate_limit_per_minute: int = int(os.getenv("RATE_LIMIT_PER_MINUTE", "60"))
    enable_cors: bool = os.getenv("ENABLE_CORS", "false").lower() == "true"
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")
    cache_size: int = int(os.getenv("CACHE_SIZE", "256"))
    cache_ttl_seconds: int = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
    api_version: str = "v1"
    n_mc_samples: int = int(os.getenv("N_MC_SAMPLES", "20"))
    mc_noise_std: float = float(os.getenv("MC_NOISE_STD", "0.015"))
    prior_strength: float = float(os.getenv("PRIOR_STRENGTH", "0.12"))
    engine_version: str = "quantum-v4.2"

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)-8s | %(name)s | rid=%(request_id)s | %(message)s",
)

class _RidFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = getattr(g, "request_id", "-") if hasattr(g, "request_id") else "-"
        return True

logger = logging.getLogger("headline")
for _h in logger.handlers:
    _h.addFilter(_RidFilter())

# ============================================================
# 2) Landmarks + quantum constants
# ============================================================
class LandmarkIndex:
    FOREHEAD_TOP   = 10
    CHIN_BOTTOM    = 152
    FOREHEAD_LEFT  = 54
    FOREHEAD_RIGHT = 284
    TEMPLE_LEFT    = 127
    TEMPLE_RIGHT   = 356
    ZYGOMATIC_LEFT  = 234
    ZYGOMATIC_RIGHT = 454
    JAW_LEFT  = 58
    JAW_RIGHT = 288
    GONIAL_LEFT  = 172
    GONIAL_RIGHT = 397
    BROW_LEFT   = 105
    BROW_RIGHT  = 334
    NOSE_TIP    = 1
    NOSE_BRIDGE = 168
    EYE_L_OUT   = 133
    EYE_L_IN    = 33
    EYE_R_IN    = 362
    EYE_R_OUT   = 263
    MOUTH_L     = 291
    MOUTH_R     = 61
    JAW_MID_L   = 365
    JAW_MID_R   = 136

SHAPES: tuple[str, ...] = ("oval", "round", "square", "heart", "oblong", "diamond", "triangle")
SHAPE_TO_IDX: dict[str, int] = {s: i for i, s in enumerate(SHAPES)}

SHAPE_LABELS_FA: dict[str, str] = {
    "oval": "بیضی", "round": "گرد", "square": "مربعی",
    "heart": "قلبی", "oblong": "کشیده", "diamond": "الماس",
    "triangle": "مثلثی",
}

QUANTUM_TARGETS: dict[str, dict[str, np.ndarray]] = {
    "oval":     {"geo": np.array([1.45, 1.12, 1.15, 1.05]), "curv": np.array([0.55, 0.60, 0.50]), "sym":  np.array([0.02, 0.05]), "ang":  np.array([0.55, 0.60])},
    "round":    {"geo": np.array([1.10, 1.00, 1.05, 1.25]), "curv": np.array([0.85, 0.70, 0.65]), "sym":  np.array([0.03, 0.03]), "ang":  np.array([0.70, 0.75])},
    "square":   {"geo": np.array([1.10, 1.00, 1.00, 1.55]), "curv": np.array([0.20, 0.35, 0.30]), "sym":  np.array([0.03, 0.04]), "ang":  np.array([0.90, 0.85])},
    "heart":    {"geo": np.array([1.45, 1.35, 0.90, 1.10]), "curv": np.array([0.65, 0.80, 0.75]), "sym":  np.array([0.04, 0.08]), "ang":  np.array([0.50, 0.45])},
    "oblong":   {"geo": np.array([1.75, 1.00, 1.05, 1.20]), "curv": np.array([0.45, 0.40, 0.50]), "sym":  np.array([0.02, 0.06]), "ang":  np.array([0.65, 0.55])},
    "diamond":  {"geo": np.array([1.55, 0.85, 1.20, 1.15]), "curv": np.array([0.60, 0.35, 0.90]), "sym":  np.array([0.03, 0.07]), "ang":  np.array([0.60, 0.40])},
    "triangle": {"geo": np.array([1.30, 0.78, 1.10, 1.35]), "curv": np.array([0.50, 0.30, 0.75]), "sym":  np.array([0.03, 0.10]), "ang":  np.array([0.75, 0.80])},
}

QUANTUM_SIGMA: dict[str, np.ndarray] = {
    "geo":  np.array([0.18, 0.14, 0.14, 0.20]),
    "curv": np.array([0.20, 0.22, 0.22]),
    "sym":  np.array([0.04, 0.06]),
    "ang":  np.array([0.15, 0.15]),
}

OBSERVER_WEIGHTS: dict[str, float] = {"geo": 0.42, "curv": 0.24, "sym": 0.14, "ang": 0.20}
OBSERVER_PHASES: dict[str, float] = {"geo": 0.0, "curv": math.pi / 3, "sym": 2 * math.pi / 3, "ang": math.pi}

_PRIOR_RAW = np.array([0.22, 0.18, 0.16, 0.12, 0.10, 0.12, 0.10], dtype=float)
SHAPE_PRIOR = _PRIOR_RAW / _PRIOR_RAW.sum()

# ============================================================
# 3) Quantum Face Analyzer
# ============================================================
class QuantumFaceAnalyzer:
    _MAGIC = {b"\xff\xd8\xff": "jpeg", b"\x89PNG\r\n\x1a\n": "png", b"RIFF": "webp"}

    def __init__(self, cfg: Config | None = None) -> None:
        self.cfg = cfg or Config()
        self._lock = threading.Lock()
        self._mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5)

    @staticmethod
    def _sniff(data: bytes) -> bool:
        for magic, kind in QuantumFaceAnalyzer._MAGIC.items():
            if data.startswith(magic):
                if kind == "webp" and b"WEBP" not in data[:16]: return False
                return True
        return False

    def _decode(self, data: bytes) -> np.ndarray | None:
        if not data or not self._sniff(data): return None
        arr = np.frombuffer(data, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return img if img is not None and img.size else None

    @staticmethod
    def _geom_helpers(lm, w: int, h: int):
        def d(i: int, j: int) -> float:
            a, b = lm[i], lm[j]
            return float(np.hypot((a.x - b.x) * w, (a.y - b.y) * h))
        def angle(i: int, j: int, k: int) -> float:
            a, b, c = lm[i], lm[j], lm[k]
            v1 = np.array([(a.x - b.x) * w, (a.y - b.y) * h])
            v2 = np.array([(c.x - b.x) * w, (c.y - b.y) * h])
            n1 = np.linalg.norm(v1) + 1e-9
            n2 = np.linalg.norm(v2) + 1e-9
            return float(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1)))
        return d, angle

    def _observables(self, lm, w: int, h: int) -> dict[str, np.ndarray] | None:
        I = LandmarkIndex
        d, angle = self._geom_helpers(lm, w, h)
        face_len   = d(I.FOREHEAD_TOP, I.CHIN_BOTTOM)
        forehead_w = d(I.FOREHEAD_LEFT, I.FOREHEAD_RIGHT)
        cheek_w    = d(I.ZYGOMATIC_LEFT, I.ZYGOMATIC_RIGHT)
        jaw_w      = d(I.JAW_LEFT, I.JAW_RIGHT)
        if cheek_w < 1e-6 or jaw_w < 1e-6 or forehead_w < 1e-6: return None
        geo = np.array([face_len / cheek_w, forehead_w / jaw_w, cheek_w / forehead_w, (face_len * 0.5) / max(jaw_w * 0.5, 1e-6)])
        mid_jaw_y = (lm[I.JAW_MID_L].y + lm[I.JAW_MID_R].y) * 0.5
        gonial_y  = (lm[I.GONIAL_LEFT].y + lm[I.GONIAL_RIGHT].y) * 0.5
        chin_y    = lm[I.CHIN_BOTTOM].y
        jaw_round = float(np.clip(abs(mid_jaw_y - gonial_y) / (abs(chin_y - gonial_y) + 1e-9), 0, 1))
        fore_curve = float(np.clip(angle(I.BROW_LEFT, I.FOREHEAD_TOP, I.BROW_RIGHT) / math.pi, 0, 1))
        cheek_prom = float(np.clip((cheek_w - 0.5 * (forehead_w + jaw_w)) / (cheek_w + 1e-9) + 0.5, 0, 1))
        curv = np.array([jaw_round, fore_curve, cheek_prom])
        lr_asym = abs(d(I.ZYGOMATIC_LEFT, I.CHIN_BOTTOM) - d(I.ZYGOMATIC_RIGHT, I.CHIN_BOTTOM)) / (cheek_w + 1e-9)
        ul_bal  = abs(face_len * 0.5 - d(I.FOREHEAD_TOP, I.NOSE_TIP)) / (face_len + 1e-9)
        sym = np.array([float(np.clip(lr_asym, 0, 0.2)), float(np.clip(ul_bal, 0, 0.2))])
        chin_ang  = angle(I.JAW_LEFT, I.CHIN_BOTTOM, I.JAW_RIGHT) / math.pi
        cheek_ang = angle(I.ZYGOMATIC_LEFT, I.NOSE_TIP, I.ZYGOMATIC_RIGHT) / math.pi
        ang = np.array([float(chin_ang), float(cheek_ang)])
        return {"geo": geo, "curv": curv, "sym": sym, "ang": ang}

    @staticmethod
    def _fidelities(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, x in obs.items():
            mu = np.stack([QUANTUM_TARGETS[s][name] for s in SHAPES])
            out[name] = np.exp(-0.5 * np.sum(((x - mu) / QUANTUM_SIGMA[name]) ** 2, axis=1))
        return out

    @staticmethod
    def _signed_residuals(obs: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for name, x in obs.items():
            mu = np.stack([QUANTUM_TARGETS[s][name] for s in SHAPES])
            out[name] = ((x - mu) / QUANTUM_SIGMA[name]).mean(axis=1)
        return out

    def _wavefunction(self, obs: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        F, R = self._fidelities(obs), self._signed_residuals(obs)
        psi = np.zeros(len(SHAPES), dtype=complex)
        for name, f in F.items():
            psi += OBSERVER_WEIGHTS[name] * np.sqrt(f) * np.exp(1j * (OBSERVER_PHASES[name] + np.arctan2(R[name], 1.0)))
        prob = np.abs(psi) ** 2
        return psi, prob / (prob.sum() + 1e-12)

    @staticmethod
    def _density_matrix(psi: np.ndarray) -> np.ndarray: return np.outer(psi, np.conj(psi))

    @classmethod
    def _purity(cls, psi: np.ndarray) -> float:
        rho = cls._density_matrix(psi)
        return float(np.real(np.trace(rho @ rho)))

    @classmethod
    def _coherence(cls, psi: np.ndarray) -> float:
        rho = cls._density_matrix(psi)
        n = len(psi)
        if n <= 1: return 0.0
        off = np.sum(np.abs(rho)) - np.sum(np.abs(np.diag(rho)))
        return float(off / (n * (n - 1)))

    def _bayesian_update(self, prob: np.ndarray) -> np.ndarray:
        logp = np.log(prob + 1e-12) + self.cfg.prior_strength * np.log(SHAPE_PRIOR + 1e-12)
        logp -= logp.max()
        post = np.exp(logp)
        return post / (post.sum() + 1e-12)

    @staticmethod
    def _von_neumann_entropy(p: np.ndarray) -> float:
        return float(-np.sum(p * np.log(p + 1e-12)) / math.log(len(SHAPES)))

    def _monte_carlo(self, lm, w: int, h: int) -> tuple[float, np.ndarray]:
        rng = np.random.default_rng(42)
        samples: list[np.ndarray] = []
        for _ in range(self.cfg.n_mc_samples):
            noisy = [SimpleNamespace(x=l.x + rng.normal(0, self.cfg.mc_noise_std), y=l.y + rng.normal(0, self.cfg.mc_noise_std), z=getattr(l, "z", 0.0)) for l in lm]
            try:
                obs = self._observables(noisy, w, h)
                if obs is None: continue
                _, p = self._wavefunction(obs)
                samples.append(self._bayesian_update(p))
            except Exception: continue
        if not samples: return 1.0, np.zeros(len(SHAPES))
        arr = np.stack(samples)
        return float(np.clip(1.0 - arr.std(axis=0).mean(), 0.0, 1.0)), arr.mean(axis=0)

    @staticmethod
    def _image_quality(img: np.ndarray) -> dict[str, Any]:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return {"sharpness": round(float(np.clip(cv2.Laplacian(gray, cv2.CV_64F).var() / 500.0, 0, 1)), 3),
                "brightness": round(float(np.clip(gray.mean() / 255.0, 0, 1)), 3),
                "width": int(img.shape[1]), "height": int(img.shape[0])}

    def analyze(self, image_bytes: bytes, min_side: int = 240) -> dict[str, Any]:
        t0 = time.perf_counter()
        img = self._decode(image_bytes)
        if img is None: return {"error": "invalid_image"}
        h, w = img.shape[:2]
        if min(h, w) < min_side: return {"error": "image_too_small", "min_side": min_side}
        quality = self._image_quality(img)
        warnings_list = [w for w, cond in [("blurry_image", quality["sharpness"] < 0.30), ("too_dark", quality["brightness"] < 0.20), ("too_bright", quality["brightness"] > 0.90)] if cond]
        with self._lock:
            results = self._mesh.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        if not results.multi_face_landmarks: return {"error": "no_face", "quality": quality, "warnings": warnings_list}
        lm = results.multi_face_landmarks[0].landmark
        obs = self._observables(lm, w, h)
        if obs is None: return {"error": "feature_failed", "quality": quality, "warnings": warnings_list}
        
        psi, prob = self._wavefunction(obs)
        posterior = self._bayesian_update(prob)
        entropy = self._von_neumann_entropy(posterior)
        stability, _ = self._monte_carlo(lm, w, h)
        if stability < 0.5: warnings_list.append("low_stability")
        if entropy > 0.80: warnings_list.append("high_entropy")
        
        top_idx = int(np.argmax(posterior))
        top_shape = SHAPES[top_idx]
        confidence = float(posterior[top_idx])
        margin = float((1.0 - stability) * 0.5)
        F = self._fidelities(obs)
        contribs = {name: round(float(OBSERVER_WEIGHTS[name] * math.sqrt(F[name][top_idx])) / (sum(OBSERVER_WEIGHTS[n] * math.sqrt(F[n][top_idx]) for n in OBSERVER_WEIGHTS) or 1.0), 4) for name in OBSERVER_WEIGHTS}
        
        d, _ = self._geom_helpers(lm, w, h)
        I = LandmarkIndex
        return {
            "shape": top_shape, "shape_fa": SHAPE_LABELS_FA[top_shape],
            "confidence": round(confidence, 3), "confidence_interval": [round(max(0.0, confidence - margin), 3), round(min(1.0, confidence + margin), 3)],
            "uncertainty_entropy": round(entropy, 3), "purity": round(self._purity(psi), 3), "coherence": round(self._coherence(psi), 4),
            "mc_stability": round(stability, 3), "scores": {SHAPES[i]: round(float(posterior[i]), 3) for i in range(len(SHAPES))},
            "observable_contributions": contribs,
            "ratios": {"len_cheek": round(float(obs["geo"][0]), 3), "fore_jaw": round(float(obs["geo"][1]), 3), "cheek_fore": round(float(obs["geo"][2]), 3), "jaw_angle": round(float(obs["geo"][3]), 3)},
            "measurements_px": {"face_length": round(d(I.FOREHEAD_TOP, I.CHIN_BOTTOM), 1), "forehead_width": round(d(I.FOREHEAD_LEFT, I.FOREHEAD_RIGHT), 1), "cheekbone_width": round(d(I.ZYGOMATIC_LEFT, I.ZYGOMATIC_RIGHT), 1), "jaw_width": round(d(I.JAW_LEFT, I.JAW_RIGHT), 1), "temple_width": round(d(I.TEMPLE_LEFT, I.TEMPLE_RIGHT), 1), "image_width": w, "image_height": h},
            "image_quality": quality, "warnings": warnings_list, "engine": self.cfg.engine_version, "elapsed_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

# ============================================================
# 4) Hairstyle registry
# ============================================================
@dataclass(frozen=True, slots=True)
class HairStyle:
    id: str
    name_fa: str
    name_en: str
    description_fa: str
    why_fa: str
    length: str
    textures: tuple[str, ...]
    vibe: str
    maintenance: str
    difficulty: int
    popularity: int
    face_shapes: tuple[str, ...]
    gender: str = "masculine"
    styling_minutes: int = 5
    products: tuple[str, ...] = ()
    thickness: tuple[str, ...] = ("fine", "medium", "thick")
    seasons: tuple[str, ...] = ("all",)
    occasions: tuple[str, ...] = ("casual",)
    tags: tuple[str, ...] = ()
    receding_friendly: bool = False

    def __post_init__(self) -> None:
        assert self.length in ("short", "medium", "long")
        assert self.vibe in ("classic", "modern", "edgy", "elegant")
        assert self.maintenance in ("low", "medium", "high")
        assert 1 <= self.difficulty <= 5
        assert 1 <= self.popularity <= 10
        assert all(s in SHAPE_TO_IDX for s in self.face_shapes)
        assert all(t in ("straight", "wavy", "curly") for t in self.textures)
        assert all(t in ("fine", "medium", "thick") for t in self.thickness)

def _S(_id, fa, en, desc, why, length, textures, vibe, maint, diff, pop, *shapes, **kw):
    tex = textures if isinstance(textures, tuple) else (textures,)
    return HairStyle(_id, fa, en, desc, why, length, tex, vibe, maint, diff, pop, shapes, **kw)

# (لیست ۲۳۴ مدل مو)
HAIRSTYLES: tuple[HairStyle, ...] = (
    _S("ov-classic-taper","تیپر کلاسیک","Classic Taper","کوتاه تمیز با تیپر نرم.","متعادل‌کننده بیضی.","short",("straight","wavy"),"classic","low",2,9,"oval",products=("wax",)),
    _S("ov-pompadour","پامپادور","Pompadour","حجم بالا و کناره‌های کوتاه.","شیکی کلاسیک.","short",("straight",),"classic","medium",4,8,"oval",products=("pomade","blow-dryer")),
    _S("ov-quiff","کویف","Quiff","حجم جلو با انتهای بافت‌دار.","پیشانی متعادل را برجسته می‌کند.","short",("straight","wavy"),"modern","medium",3,9,"oval",products=("clay","wax")),
    _S("ov-side-part","فرق کج","Side Part","فرق واضح با شانه‌ی مرتب.","همه‌کاره و رسمی.","short",("straight","wavy"),"classic","low",2,9,"oval",products=("pomade",)),
    _S("ov-crew","کرو کات","Crew Cut","کوتاه یکدست با کمی حجم.","خطوط ساده بیضی.","short",("straight",),"classic","low",1,8,"oval",products=("wax",)),
    _S("ov-buzz","باز کات","Buzz Cut","خیلی کوتاه یکنواخت.","مینیمال.","short",("straight",),"edgy","low",1,7,"oval",occasions=("sport",)),
    _S("ov-textured-crop","کراپ بافت‌دار","Textured Crop","کوتاه با بافت و چتری کوتاه.","مدرن.","short",("straight","wavy"),"modern","low",2,9,"oval",products=("clay",)),
    _S("ov-slick-back","اسلیک بک","Slick Back","شانه به عقب با ژل.","رسمی و شیک.","short",("straight",),"elegant","medium",3,7,"oval",products=("pomade","gel")),
    _S("ov-french-crop","کراپ فرانسوی","French Crop","کراپ با چتری صاف کوتاه.","سادگی.","short",("straight",),"modern","low",2,8,"oval",products=("wax",)),
    _S("ov-ivy-league","آیوی لیگ","Ivy League","کرو با فرق کنار.","دانشگاهی.","short",("straight",),"classic","low",2,7,"oval",products=("pomade",)),
    _S("ov-undercut","آندرکات مدرن","Modern Undercut","کناره‌های تراشیده با رویه‌ی بلند.","کنتراست جذاب.","short",("straight","wavy"),"edgy","medium",3,9,"oval",products=("clay",)),
    _S("ov-bro-flow","برو فلو","Bro Flow","موی متوسط شلخته با جریان طبیعی.","راحت و اسپرت.","medium",("wavy",),"modern","low",2,8,"oval",products=("sea-salt-spray",)),
    _S("ov-medium-side-part","فرق متوسط","Medium Side Part","فرق کنار با طول متوسط.","کلاسیک مدرن.","medium",("straight","wavy"),"classic","medium",3,8,"oval",products=("pomade",)),
    _S("ov-modern-pompadour","پامپ مدرن","Modern Pompadour","پامپ با کناره‌های تیپر.","شیک امروزی.","short",("straight","wavy"),"modern","medium",3,9,"oval",products=("pomade","blow-dryer")),
    _S("ov-tousled-top","رویه شلخته","Tousled Top","رویه بافت‌دار شلخته.","حالت طبیعی.","short",("wavy",),"modern","low",2,8,"oval",products=("clay",)),
    _S("ov-curly-taper","فر تیپر","Curly Taper","فر با تیپر نرم.","حجم کناری.","short",("curly",),"modern","medium",2,8,"oval",products=("curl-cream",)),
    _S("ov-wavy-slick","اسلیک موج","Wavy Slick","اسلیک با موج ملایم.","شیک و نرم.","short",("wavy",),"elegant","medium",3,7,"oval",products=("pomade",)),
    _S("ov-pompadour-fade","پامپ فید","Pompadour Fade","پامپ با فید.","حجم بالا + کناره تمیز.","short",("straight","wavy"),"modern","medium",4,9,"oval",products=("pomade","blow-dryer")),
    _S("ov-caesar","سزار","Caesar","کوتاه با چتری کوتاه.","کلاسیک.","short",("straight",),"classic","low",1,8,"oval",products=("wax",)),
    _S("ov-brush-up","براش آپ","Brush Up","موها به بالا.","ارتفاع کنترل‌شده.","short",("straight","wavy"),"modern","medium",2,8,"oval",products=("wax",)),
    _S("ov-messy-quiff","کویف شلخته","Messy Quiff","کویف شلخته بافت‌دار.","مدرن شهری.","short",("wavy",),"modern","medium",3,8,"oval",products=("clay",)),
    _S("ov-short-shag","شگ کوتاه","Short Shag","لایه‌های کوتاه.","بافت و حرکت.","short",("wavy","curly"),"edgy","medium",3,7,"oval",products=("sea-salt-spray",)),
    _S("ov-medium-layer","لایه متوسط","Medium Layers","موی متوسط لایه‌دار.","نرمی.","medium",("straight","wavy"),"classic","medium",3,8,"oval",products=("cream",)),
    _S("ov-curtain","پرده‌ای","Curtain Hair","فرق وسط با پرده.","مدرن.","medium",("straight","wavy"),"modern","medium",2,9,"oval",products=("cream",)),
    _S("ov-mohawk-soft","موهاک نرم","Soft Mohawk","موهاک نرم بدون سیخ.","جسورانه.","short",("straight",),"edgy","medium",3,6,"oval",products=("clay",)),
    _S("ov-man-bun","مان بان","Man Bun","موی متوسط به بالا گره.","اسپرت و شیک.","medium",("straight","wavy"),"modern","medium",3,7,"oval",products=("hair-tie",)),
    _S("ov-faux-hawk","فو هاوک","Faux Hawk","حجم مرکزی.","جسورانه.","short",("straight",),"edgy","medium",3,7,"oval",products=("gel",)),
    _S("ov-low-fade","فید پایین","Low Fade","فید پایین با رویه متوسط.","تمیز.","short",("straight","wavy"),"modern","low",2,9,"oval",products=("wax",)),
    _S("ov-mid-fade","فید متوسط","Mid Fade","فید متوسط.","تعادل.","short",("straight",),"modern","low",2,9,"oval",products=("wax",)),
    _S("ov-drop-fade","دراپ فید","Drop Fade","فید قوسی.","جذاب.","short",("straight",),"modern","low",3,8,"oval",products=("wax",)),
    _S("ov-taper-fade","تیپر فید","Taper Fade","تیپر با فید.","تمیز کلاسیک.","short",("straight",),"classic","low",2,9,"oval",products=("wax",)),
    _S("ov-textured-spiky","سیخ بافت‌دار","Textured Spiky","سیخ‌های بافت‌دار.","مدرن.","short",("straight",),"edgy","medium",2,7,"oval",products=("gel",)),
    _S("ov-natural-waves","موج طبیعی","Natural Waves","موج طبیعی متوسط.","شیک.","medium",("wavy",),"elegant","medium",3,8,"oval",products=("sea-salt-spray",)),
    _S("ov-brush-back","براش بک","Brush Back","شانه به عقب.","کلاسیک.","short",("straight",),"classic","medium",3,8,"oval",products=("pomade",)),
    _S("ov-comb-over","کامب اور","Modern Comb-Over","شانه با فرق کنار.","شیک.","short",("straight","wavy"),"classic","low",2,8,"oval",products=("pomade",)),
    _S("ov-skin-fade","اسکین فید","Skin Fade","فید تا پوست.","تمیز و مدرن.","short",("straight",),"modern","low",3,9,"oval",products=("wax",)),
    _S("rd-high-fade-vol","فید بلند حجم","High Fade Volume","فید بلند با حجم.","کشیده‌تر.","short",("straight","wavy"),"modern","medium",3,9,"round",products=("wax","blow-dryer")),
    _S("rd-pomp-fade","پامپ فید","Pompadour Fade","پامپ با فید.","گردی را می‌شکند.","short",("straight",),"classic","medium",4,9,"round",products=("pomade",)),
    _S("rd-quiff-taper","کویف تیپر","Quiff Taper","کویف با تیپر.","نسبت عمودی.","short",("straight","wavy"),"modern","medium",3,8,"round",products=("clay",)),
    _S("rd-faux-hawk","فو هاوک","Faux Hawk","حجم مرکزی.","خط عمودی.","short",("straight",),"edgy","medium",3,7,"round",products=("gel",)),
    _S("rd-spiky","سیخ","Spiky Hair","سیخ‌های مرتب.","ارتفاع.","short",("straight",),"edgy","medium",2,7,"round",products=("gel",)),
    _S("rd-textured-crop-fade","کراپ بافت‌دار فید","Textured Crop Fade","کراپ با فید.","تاپ حجم.","short",("straight","wavy"),"modern","low",2,9,"round",products=("clay",)),
    _S("rd-undercut-vol","آندرکات حجم","Undercut Volume","آندرکات با رویه پرحجم.","کشیدگی.","short",("straight","wavy"),"modern","medium",3,8,"round",products=("clay",)),
    _S("rd-asym-side","فرق نامتقارن","Asymmetric Side","فرق عمیق.","گردی می‌شکند.","short",("straight",),"modern","low",2,8,"round",products=("pomade",)),
    _S("rd-mohawk-fade","موهاک فید","Mohawk Fade","نوار مرکزی.","عمودی قوی.","short",("straight",),"edgy","medium",4,6,"round",products=("gel",)),
    _S("rd-angular-fringe","چتری زاویه‌دار","Angular Fringe","چتری زاویه‌دار.","خط تیز.","short",("straight",),"modern","medium",3,8,"round",products=("wax",)),
    _S("rd-layered-top","لایه‌دار","Layered Top","لایه‌های بالای سر.","ارتفاع.","short",("wavy",),"modern","medium",3,8,"round",products=("clay",)),
    _S("rd-top-knot-fade","توپ‌نات فید","Top Knot Fade","فید + گره.","حجم بالا.","medium",("straight","wavy"),"modern","medium",3,7,"round",products=("hair-tie",)),
    _S("rd-high-pompadour","پامپ بلند","High Pompadour","پامپ با حجم زیاد.","کشیدگی.","short",("straight",),"classic","high",4,8,"round",products=("pomade","blow-dryer")),
    _S("rd-vertical-quiff","کویف عمودی","Vertical Quiff","کویف ارتفاع بالا.","عمودی.","short",("straight",),"modern","high",4,8,"round",products=("clay",)),
    _S("rd-brush-up-fade","براش فید","Brush Up Fade","براش با فید.","تمیز.","short",("straight",),"modern","medium",2,8,"round",products=("wax",)),
    _S("rd-textured-spiky-fade","سیخ فید","Spiky Fade","سیخ با فید.","ارتفاع.","short",("straight",),"edgy","medium",3,7,"round",products=("gel",)),
    _S("rd-high-fade-pomp","فید پامپ","High Fade Pomp","پامپ فید بلند.","عمودی.","short",("straight",),"modern","medium",4,8,"round",products=("pomade",)),
    _S("rd-pompadour-taper","پامپ تیپر","Pompadour Taper","پامپ تیپر.","کشیدگی.","short",("straight",),"classic","medium",3,8,"round",products=("pomade",)),
    _S("rd-skin-fade-vol","اسکین فید حجم","Skin Fade Volume","اسکین فید با حجم.","تعادل.","short",("straight",),"modern","medium",3,8,"round",products=("wax",)),
    _S("rd-curly-fade","فر فید","Curly Fade","فر با فید.","کنتراست.","short",("curly",),"modern","medium",3,8,"round",products=("curl-cream",)),
    _S("rd-curly-taper","فر تیپر","Curly Taper","فر تیپر.","کنتراست.","short",("curly",),"modern","medium",2,8,"round",products=("curl-cream",)),
    _S("rd-curly-crop","کراپ فر","Curly Crop","کراپ فر.","حجم کنترل‌شده.","short",("curly",),"modern","low",2,8,"round",products=("curl-cream",)),
    _S("rd-curly-mohawk","موهاک فر","Curly Mohawk","موهاک فر.","جسورانه.","short",("curly",),"edgy","medium",4,6,"round",products=("gel",)),
    _S("rd-curly-top","فر بالای سر","Curly Top","فر پرحجم بالا.","حجم بالا.","short",("curly",),"modern","medium",3,8,"round",products=("curl-cream",)),
    _S("rd-curly-short","فر کوتاه","Short Curls","فر کوتاه.","طبیعی.","short",("curly",),"modern","low",2,8,"round",products=("cream",)),
    _S("rd-curly-medium","فر متوسط","Medium Curls","فر متوسط.","گردی نرم.","medium",("curly",),"modern","medium",3,8,"round",products=("curl-cream",)),
    _S("rd-messy-spiky","سیخ شلخته","Messy Spiky","سیخ شلخته.","جذاب.","short",("straight","wavy"),"edgy","medium",3,7,"round",products=("clay",)),
    _S("rd-angular-pomp","پامپ زاویه‌دار","Angular Pompadour","پامپ زاویه‌دار.","کشیدگی.","short",("straight",),"modern","medium",3,7,"round",products=("pomade",)),
    _S("rd-side-quiff","کویف کناری","Side Quiff","کویف کنار.","کشیدگی.","short",("straight","wavy"),"modern","medium",3,8,"round",products=("clay",)),
    _S("rd-brush-back-vol","براش حجم","Volume Brush Back","براش حجم‌دار.","عمودی.","short",("straight","wavy"),"modern","medium",3,8,"round",products=("pomade",)),
    _S("rd-textured-fringe","چتری بافت","Textured Fringe","چتری بافت.","گردی می‌شکند.","short",("straight","wavy"),"modern","medium",3,8,"round",products=("clay",)),
    _S("rd-fade-crew","فید کرو","Fade Crew","کرو با فید.","تمیز.","short",("straight",),"classic","low",2,8,"round",products=("wax",)),
    _S("rd-buzz-beard","باز با ریش","Buzz with Beard","باز + ریش.","تعادل.","short",("straight",),"edgy","low",2,7,"round",products=()),
    _S("rd-buzz-fade","باز فید","Buzz Fade","باز با فید.","تمیز.","short",("straight",),"modern","low",2,8,"round",products=()),
    _S("sq-soft-side","فرق نرم","Soft Side Part","فرق کنار با موج نرم.","نرم‌کننده فک.","short",("wavy",),"classic","low",2,9,"square",products=("pomade",)),
    _S("sq-textured-quiff","کویف بافت","Textured Quiff","کویف بافت شلخته.","خطوط صاف را می‌شکند.","short",("wavy",),"modern","medium",3,9,"square",products=("clay",)),
    _S("sq-messy-fringe","چتری شلخته","Messy Fringe","چتری نامرتب.","نرمی.","short",("wavy",),"modern","low",2,8,"square",products=("clay",)),
    _S("sq-wavy-medium","موج متوسط","Wavy Medium","موج طبیعی متوسط.","نرمی.","medium",("wavy",),"elegant","medium",2,8,"square",products=("cream",)),
    _S("sq-crew","کرو کلاسیک","Classic Crew","کرو تمیز.","اقتدار.","short",("straight",),"classic","low",1,8,"square",products=("wax",)),
    _S("sq-bro-flow","برو فلو","Bro Flow","جریان متوسط.","انعطاف.","medium",("wavy",),"modern","low",2,8,"square",products=("sea-salt-spray",)),
    _S("sq-curly-fade","فر فید","Curly Top Fade","فر با فید.","گردی فر نرم.","short",("curly",),"modern","medium",2,8,"square",products=("curl-cream",)),
    _S("sq-curtain","پرده‌ای","Curtain Hair","فرق وسط پرده‌ای.","نرمی وسط.","medium",("straight","wavy"),"modern","medium",2,9,"square",products=("cream",)),
    _S("sq-layered-taper","تیپر لایه‌دار","Layered Taper","تیپر لایه‌دار.","انتقال نرم.","short",("straight","wavy"),"modern","low",2,8,"square",products=("wax",)),
    _S("sq-medium-shag","شگ متوسط","Medium Shag","شگ بافت‌دار.","حجم اطراف فک.","medium",("wavy","curly"),"edgy","medium",3,7,"square",products=("sea-salt-spray",)),
    _S("sq-wavy-slick","اسلیک موج","Wavy Slick","اسلیک با موج.","شیک.","short",("wavy",),"elegant","medium",3,7,"square",products=("pomade",)),
    _S("sq-tousled-medium","شلخته متوسط","Tousled Medium","موی متوسط شلخته.","نرمی.","medium",("wavy",),"modern","low",2,8,"square",products=("sea-salt-spray",)),
    _S("sq-pompadour","پامپادور","Pompadour","پامپ کلاسیک.","شیک.","short",("straight",),"classic","medium",4,8,"square",products=("pomade",)),
    _S("sq-modern-quiff","کویف مدرن","Modern Quiff","کویف مدرن.","نرمی.","short",("wavy",),"modern","medium",3,9,"square",products=("clay",)),
    _S("sq-side-sweep","سویپ کناری","Side Sweep","سویپ کناری.","نرمی.","short",("straight","wavy"),"classic","low",2,8,"square",products=("cream",)),
    _S("sq-soft-crew","کرو نرم","Soft Crew","کرو با طول نرم.","نرمی.","short",("straight","wavy"),"classic","low",2,8,"square",products=("wax",)),
    _S("sq-curly-top","فر بالای سر","Curly Top","فر بالای سر.","گردی.","short",("curly",),"modern","medium",2,8,"square",products=("curl-cream",)),
    _S("sq-curly-medium","فر متوسط","Curly Medium","فر متوسط.","حجم.","medium",("curly",),"modern","medium",3,8,"square",products=("curl-cream",)),
    _S("sq-curly-crop","کراپ فر","Curly Crop","کراپ فر.","تمیز.","short",("curly",),"modern","low",2,8,"square",products=("curl-cream",)),
    _S("sq-wavy-shag","شگ موج","Wavy Shag","شگ موج‌دار.","بافت.","medium",("wavy",),"edgy","medium",3,7,"square",products=("sea-salt-spray",)),
    _S("sq-medium-layers","لایه‌های متوسط","Medium Layers","موی متوسط لایه‌دار.","نرمی.","medium",("straight","wavy"),"classic","medium",3,8,"square",products=("cream",)),
    _S("sq-textured-side","فرق بافت","Textured Side Part","فرق با بافت.","نرمی.","short",("wavy",),"modern","medium",3,8,"square",products=("clay",)),
    _S("sq-fringe-textured","چتری بافت","Textured Fringe","چتری بافت.","نرمی.","short",("wavy",),"modern","medium",3,8,"square",products=("clay",)),
    _S("sq-undercut","آندرکات","Undercut","کناره تراشیده.","کنتراست.","short",("straight","wavy"),"edgy","medium",3,8,"square",products=("clay",)),
    _S("sq-undercut-medium","آندرکات متوسط","Undercut Medium","آندرکات متوسط.","نرمی.","medium",("straight","wavy"),"modern","medium",3,8,"square",products=("clay",)),
    _S("sq-slick-back","اسلیک بک","Slick Back","اسلیک به عقب.","رسمی.","short",("straight",),"elegant","medium",3,7,"square",products=("pomade",)),
    _S("sq-pompadour-taper","پامپ تیپر","Pompadour Taper","پامپ تیپر.","نرمی.","short",("straight",),"classic","medium",3,8,"square",products=("pomade",)),
    _S("sq-taper-soft","تیپر نرم","Soft Taper","تیپر نرم.","نرمی.","short",("straight","wavy"),"classic","low",2,8,"square",products=("wax",)),
    _S("sq-side-part-classic","فرق کلاسیک","Classic Side Part","فرق کلاسیک.","شیک.","short",("straight",),"classic","low",2,9,"square",products=("pomade",)),
    _S("sq-quiff-soft","کویف نرم","Soft Quiff","کویف نرم.","نرمی.","short",("straight","wavy"),"modern","medium",3,8,"square",products=("clay",)),
    _S("sq-crew-taper","کرو تیپر","Crew Taper","کرو با تیپر.","تمیز.","short",("straight",),"classic","low",2,8,"square",products=("wax",)),
    _S("sq-fade-side-part","فرق فید","Fade Side Part","فرق با فید.","شیک.","short",("straight",),"classic","low",2,8,"square",products=("pomade",)),
    _S("sq-french-crop","کراپ فرانسوی","French Crop","کراپ صاف.","تمیز.","short",("straight",),"modern","low",2,8,"square",products=("wax",)),
    _S("sq-buzz-fade","باز فید","Buzz Fade","باز فید.","تمیز.","short",("straight",),"modern","low",2,8,"square",products=()),
    _S("he-medium-side","فرق متوسط","Medium Side Part","فرق با طول متوسط.","پیشانی متعادل.","medium",("straight","wavy"),"classic","medium",3,9,"heart",products=("pomade",)),
    _S("he-tousled-fringe","چتری شلخته","Tousled Fringe","چتری شلخته.","پوشش پیشانی.","short",("wavy",),"modern","low",2,9,"heart",products=("clay",)),
    _S("he-curtain","پرده‌ای","Curtain Hair","فرق وسط پرده.","پیشانی پوشیده.","medium",("straight","wavy"),"modern","medium",2,9,"heart",products=("cream",)),
    _S("he-chin-layers","لایه چانه","Chin Length Layers","لایه تا چانه.","حجم فک.","medium",("wavy",),"elegant","medium",3,8,"heart",products=("cream",)),
    _S("he-bro-flow","برو فلو","Bro Flow","جریان متوسط.","پایین صورت.","medium",("wavy",),"modern","low",2,8,"heart",products=("sea-salt-spray",)),
    _S("he-messy-mid","متوسط شلخته","Messy Mid Length","شلخته متوسط.","عدم تقارن.","medium",("wavy",),"modern","low",2,7,"heart",products=("sea-salt-spray",)),
    _S("he-wavy-shag","شگ موج","Wavy Shag","شگ موج.","حجم.","medium",("wavy",),"edgy","medium",3,8,"heart",products=("sea-salt-spray",)),
    _S("he-low-fade-len","فید کم طول","Low Fade Length","فید پایین با طول.","تعادل.","medium",("straight","wavy"),"modern","medium",3,8,"heart",products=("wax",)),
    _S("he-tapered-mullet","مولت","Tapered Mullet","کوتاه جلو، بلند عقب.","حجم عقب.","medium",("wavy",),"edgy","medium",3,7,"heart",products=("sea-salt-spray",)),
    _S("he-slick-beard","اسلیک ریش","Slick with Beard","اسلیک + ریش.","پایین پر.","short",("straight",),"classic","medium",3,8,"heart",products=("pomade",)),
    _S("he-long-wavy","بلند موج","Long Wavy","موج بلند.","جریان طولی.","long",("wavy",),"elegant","high",3,7,"heart",products=("sea-salt-spray",)),
    _S("he-fringe-medium","چتری متوسط","Medium Fringe","چتری متوسط.","پیشانی پوشیده.","medium",("straight","wavy"),"modern","medium",3,8,"heart",products=("cream",)),
    _S("he-side-fringe","چتری کناری","Side Fringe","چتری سویپ کناری.","پیشانی پوشیده.","medium",("straight",),"classic","medium",3,8,"heart",products=("cream",)),
    _S("he-bangs-curtain","چتری پرده","Curtain Bangs","چتری پرده وسط.","پیشانی پوشیده.","medium",("straight","wavy"),"modern","medium",2,8,"heart",products=("cream",)),
    _S("he-swept-medium","سویپ متوسط","Swept Medium","سویپ متوسط.","پیشانی.","medium",("straight","wavy"),"classic","medium",3,8,"heart",products=("cream",)),
    _S("he-textured-layer","لایه بافت","Textured Layers","لایه بافت‌دار.","حجم.","medium",("wavy",),"modern","medium",3,8,"heart",products=("clay",)),
    _S("he-soft-pompadour","پامپ نرم","Soft Pompadour","پامپ نرم.","تعادل.","short",("straight",),"classic","medium",3,8,"heart",products=("pomade",)),
    _S("he-modern-quiff","کویف مدرن","Modern Quiff","کویف مدرن.","تعادل.","short",("straight","wavy"),"modern","medium",3,8,"heart",products=("clay",)),
    _S("he-crew-classic","کرو کلاسیک","Classic Crew","کرو کلاسیک.","تمیز.","short",("straight",),"classic","low",2,8,"heart",products=("wax",)),
    _S("he-side-part-soft","فرق نرم","Soft Side Part","فرق نرم.","نرمی.","short",("straight","wavy"),"classic","low",2,8,"heart",products=("pomade",)),
    _S("he-long-straight","بلند صاف","Long Straight","موی بلند صاف.","تعادل.","long",("straight",),"elegant","high",3,7,"heart",products=("cream",)),
    _S("he-man-bun","مان بان","Man Bun","مان بان.","اسپرت.","long",("straight","wavy"),"modern","medium",3,7,"heart",products=("hair-tie",)),
    _S("he-half-up","نیمه بالا","Half Up","نیمه بالا.","مدرن.","long",("straight","wavy"),"modern","medium",3,7,"heart",products=("hair-tie",)),
    _S("he-tousled-fringe-fade","چتری شلخته فید","Tousled Fringe Fade","چتری شلخته با فید.","پیشانی پوشیده.","short",("wavy",),"modern","medium",3,8,"heart",products=("clay",)),
    _S("he-medium-waves","موج متوسط","Medium Waves","موج متوسط.","نرمی.","medium",("wavy",),"elegant","medium",3,8,"heart",products=("sea-salt-spray",)),
    _S("he-long-layers","لایه‌های بلند","Long Layers","لایه بلند.","جریان.","long",("straight","wavy"),"elegant","high",3,7,"heart",products=("cream",)),
    _S("he-natural-flow","جریان طبیعی","Natural Flow","جریان طبیعی.","نرمی.","medium",("wavy",),"modern","low",2,7,"heart",products=("sea-salt-spray",)),
    _S("he-pompadour-medium","پامپ متوسط","Pompadour Medium","پامپ متوسط.","شیک.","medium",("straight",),"classic","medium",4,8,"heart",products=("pomade",)),
    _S("he-curly-mid","فر متوسط","Curly Mid","فر متوسط.","حجم.","medium",("curly",),"modern","medium",3,7,"heart",products=("curl-cream",)),
    _S("he-curly-fringe","فر چتری","Curly Fringe","چتری فر.","پیشانی پوشیده.","medium",("curly",),"modern","medium",3,7,"heart",products=("curl-cream",)),
    _S("he-textured-fringe","چتری بافت","Textured Fringe","چتری بافت.","پیشانی پوشیده.","short",("wavy",),"modern","medium",3,8,"heart",products=("clay",)),
    _S("he-fringe-curtain","پرده چتری","Fringe Curtain","پرده چتری.","تعادل.","medium",("straight","wavy"),"modern","medium",3,8,"heart",products=("cream",)),
    _S("ob-side-part-width","فرق عرض","Side Part Width","فرق با حجم کناری.","عرض.","short",("straight","wavy"),"classic","low",2,9,"oblong",products=("pomade",)),
    _S("ob-messy-fringe","چتری شلخته","Messy Fringe","چتری شلخته.","طول پیشانی.","short",("straight","wavy"),"modern","low",2,9,"oblong",products=("clay",)),
    _S("ob-caesar","سزار","Short Caesar","کوتاه یکدست.","کوتاهی افقی.","short",("straight",),"classic","low",1,8,"oblong",products=("wax",)),
    _S("ob-buzz-beard","باز ریش","Buzz Beard","باز + ریش پر.","عرض پایین.","short",("straight",),"edgy","low",1,8,"oblong",products=()),
    _S("ob-wavy-side","فرق موج","Wavy Side","موج کناری.","عرض.","short",("wavy",),"classic","low",2,8,"oblong",products=("pomade",)),
    _S("ob-low-quiff","کویف کم","Low Quiff","کویف کم حجم.","بدون ارتفاع.","short",("straight",),"modern","medium",3,8,"oblong",products=("clay",)),
    _S("ob-layered-fringe","چتری لایه","Layered Fringe","چتری لایه پهن.","عرض پیشانی.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-curly-crop","کراپ فر","Curly Crop","کراپ فر.","حجم کناری.","short",("curly",),"modern","low",2,8,"oblong",products=("curl-cream",)),
    _S("ob-horizontal-fringe","چتری افقی","Horizontal Fringe","چتری صاف افقی.","خطوط افقی.","short",("straight",),"classic","low",2,7,"oblong",products=("wax",)),
    _S("ob-mid-layers","لایه متوسط","Mid Layers","لایه متوسط.","عرض.","medium",("wavy",),"modern","medium",3,8,"oblong",products=("cream",)),
    _S("ob-textured-sweep","سویپ بافت","Textured Sweep","سویپ بافت.","حرکت افقی.","short",("wavy",),"modern","medium",2,8,"oblong",products=("clay",)),
    _S("ob-bowl-modern","کاسه مدرن","Modern Bowl","کاسه با تیپر.","خط افقی چتری.","short",("straight",),"edgy","medium",3,7,"oblong",products=("wax",)),
    _S("ob-crew-classic","کرو کلاسیک","Classic Crew","کرو.","تمیز.","short",("straight",),"classic","low",2,8,"oblong",products=("wax",)),
    _S("ob-crew-fade","کرو فید","Crew Fade","کرو فید.","تمیز.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-buzz-fade","باز فید","Buzz Fade","باز فید.","تمیز.","short",("straight",),"modern","low",2,8,"oblong",products=()),
    _S("ob-fade-short","فید کوتاه","Short Fade","فید کوتاه.","تمیز.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-taper-short","تیپر کوتاه","Short Taper","تیپر کوتاه.","تمیز.","short",("straight",),"classic","low",2,8,"oblong",products=("wax",)),
    _S("ob-caesar-fade","سزار فید","Caesar Fade","سزار فید.","تمیز.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-textured-crop","کراپ بافت","Textured Crop","کراپ بافت.","تمیز.","short",("straight","wavy"),"modern","low",2,8,"oblong",products=("clay",)),
    _S("ob-french-crop","کراپ فرانسوی","French Crop","کراپ صاف.","تمیز.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-side-part-soft","فرق نرم","Soft Side Part","فرق نرم.","تمیز.","short",("straight","wavy"),"classic","low",2,8,"oblong",products=("pomade",)),
    _S("ob-side-part-classic","فرق کلاسیک","Classic Side Part","فرق کلاسیک.","شیک.","short",("straight",),"classic","low",2,9,"oblong",products=("pomade",)),
    _S("ob-modern-comb-over","کامب اور","Modern Comb-Over","شانه با فرق.","شیک.","short",("straight","wavy"),"classic","low",2,8,"oblong",products=("pomade",)),
    _S("ob-slick-back","اسلیک بک","Slick Back","اسلیک.","رسمی.","short",("straight",),"elegant","medium",3,7,"oblong",products=("pomade",)),
    _S("ob-pompadour-low","پامپ کم","Low Pompadour","پامپ کم.","تعادل.","short",("straight",),"classic","medium",3,8,"oblong",products=("pomade",)),
    _S("ob-quiff-short","کویف کوتاه","Short Quiff","کویف کوتاه.","تعادل.","short",("straight",),"modern","medium",3,8,"oblong",products=("clay",)),
    _S("ob-tousled-top","رویه شلخته","Tousled Top","رویه شلخته.","عرض.","short",("wavy",),"modern","low",2,8,"oblong",products=("clay",)),
    _S("ob-wavy-crop","کراپ موج","Wavy Crop","کراپ موج.","تمیز.","short",("wavy",),"modern","low",2,8,"oblong",products=("clay",)),
    _S("ob-curly-top","فر بالا","Curly Top","فر بالا.","حجم.","short",("curly",),"modern","medium",2,8,"oblong",products=("curl-cream",)),
    _S("ob-curly-fringe","فر چتری","Curly Fringe","چتری فر.","پیشانی.","short",("curly",),"modern","medium",3,7,"oblong",products=("curl-cream",)),
    _S("ob-buzz-beard-long","باز ریش بلند","Buzz Long Beard","باز + ریش بلند.","عرض پایین.","short",("straight",),"edgy","medium",2,7,"oblong",products=()),
    _S("ob-side-sweep","سویپ کناری","Side Sweep","سویپ کناری.","عرض.","short",("straight","wavy"),"classic","low",2,8,"oblong",products=("pomade",)),
    _S("ob-brush-up-low","براش کم","Low Brush Up","براش کم.","تعادل.","short",("straight",),"modern","low",2,8,"oblong",products=("wax",)),
    _S("ob-crew-buzz","کرو باز","Crew Buzz","کرو باز.","تمیز.","short",("straight",),"classic","low",1,8,"oblong",products=()),
    _S("di-medium-taper","تیپر متوسط","Medium Taper","تیپر متوسط.","گونه‌ها.","medium",("straight","wavy"),"classic","low",2,9,"diamond",products=("wax",)),
    _S("di-tousled-mid","شلخته متوسط","Tousled Medium","شلخته متوسط.","حجم کناری.","medium",("wavy",),"modern","low",2,8,"diamond",products=("sea-salt-spray",)),
    _S("di-curtain-bangs","چتری پرده","Curtain Bangs","پرده وسط.","پیشانی پهن.","medium",("straight","wavy"),"modern","medium",2,9,"diamond",products=("cream",)),
    _S("di-messy-fringe","چتری شلخته","Messy Fringe","چتری شلخته.","پیشانی پر.","short",("straight","wavy"),"modern","low",2,8,"diamond",products=("clay",)),
    _S("di-side-swept","سویپ کناری","Side Swept","سویپ کناری.","قرینگی نرم.","medium",("straight",),"classic","low",2,8,"diamond",products=("pomade",)),
    _S("di-bro-flow","برو فلو","Bro Flow","جریان متوسط.","حجم کناری.","medium",("wavy",),"modern","low",2,9,"diamond",products=("sea-salt-spray",)),
    _S("di-wavy-mid","موج متوسط","Wavy Medium","موج متوسط.","عرض.","medium",("wavy",),"elegant","medium",2,8,"diamond",products=("cream",)),
    _S("di-layered-side","فرق لایه","Layered Side","فرق لایه‌دار.","پیشانی.","medium",("wavy",),"classic","medium",3,8,"diamond",products=("cream",)),
    _S("di-textured-pomp","پامپ بافت","Textured Pomp","پامپ بافت.","تعادل بالا.","short",("straight","wavy"),"modern","medium",3,8,"diamond",products=("pomade",)),
    _S("di-long-side-fringe","چتری بلند کناری","Long Side Fringe","چتری بلند کناری.","تعادل.","medium",("straight","wavy"),"modern","medium",3,8,"diamond",products=("cream",)),
    _S("di-curly-mid","فر متوسط","Curly Medium","فر متوسط.","عرض.","medium",("curly",),"edgy","medium",3,7,"diamond",products=("curl-cream",)),
    _S("di-fringe-medium","چتری متوسط","Medium Fringe","چتری متوسط.","پیشانی.","medium",("straight","wavy"),"modern","medium",3,8,"diamond",products=("cream",)),
    _S("di-swept-bangs","چتری سویپ","Swept Bangs","چتری سویپ.","پیشانی.","medium",("straight",),"modern","medium",3,8,"diamond",products=("cream",)),
    _S("di-textured-medium","متوسط بافت","Textured Medium","بافت متوسط.","حجم.","medium",("wavy",),"modern","medium",3,8,"diamond",products=("clay",)),
    _S("di-layered-medium","لایه متوسط","Layered Medium","لایه متوسط.","حجم.","medium",("straight","wavy"),"classic","medium",3,8,"diamond",products=("cream",)),
    _S("di-crew-soft","کرو نرم","Soft Crew","کرو نرم.","تمیز.","short",("straight","wavy"),"classic","low",2,8,"diamond",products=("wax",)),
    _S("di-side-part-classic","فرق کلاسیک","Classic Side Part","فرق کلاسیک.","تمیز.","short",("straight",),"classic","low",2,8,"diamond",products=("pomade",)),
    _S("di-quiff-soft","کویف نرم","Soft Quiff","کویف نرم.","تعادل.","short",("straight","wavy"),"modern","medium",3,8,"diamond",products=("clay",)),
    _S("di-pompadour-medium","پامپ متوسط","Medium Pompadour","پامپ متوسط.","شیک.","medium",("straight",),"classic","medium",4,8,"diamond",products=("pomade",)),
    _S("di-side-sweep","سویپ کناری","Side Sweep","سویپ کناری.","پیشانی.","short",("straight","wavy"),"classic","low",2,8,"diamond",products=("pomade",)),
    _S("di-crew-fade","کرو فید","Crew Fade","کرو فید.","تمیز.","short",("straight",),"modern","low",2,8,"diamond",products=("wax",)),
    _S("di-french-crop","کراپ فرانسوی","French Crop","کراپ.","تمیز.","short",("straight",),"modern","low",2,8,"diamond",products=("wax",)),
    _S("di-curly-crop","کراپ فر","Curly Crop","کراپ فر.","حجم.","short",("curly",),"modern","low",2,7,"diamond",products=("curl-cream",)),
    _S("di-curly-fringe","فر چتری","Curly Fringe","چتری فر.","پیشانی.","medium",("curly",),"modern","medium",3,7,"diamond",products=("curl-cream",)),
    _S("di-curly-side","فر کناری","Curly Side","فر کناری.","کناری.","medium",("curly",),"modern","medium",3,7,"diamond",products=("curl-cream",)),
    _S("di-curly-long","فر بلند","Curly Long","فر بلند.","جریان.","long",("curly",),"elegant","high",3,6,"diamond",products=("curl-cream",)),
    _S("di-long-wavy","بلند موج","Long Wavy","موج بلند.","جریان.","long",("wavy",),"elegant","high",3,6,"diamond",products=("sea-salt-spray",)),
    _S("di-medium-flow","جریان متوسط","Medium Flow","جریان متوسط.","نرمی.","medium",("wavy",),"modern","medium",3,8,"diamond",products=("sea-salt-spray",)),
    _S("di-messy-top","رویه شلخته","Messy Top","رویه شلخته.","تعادل.","short",("wavy",),"modern","low",2,8,"diamond",products=("clay",)),
    _S("di-mid-part","فرق وسط","Mid Part","فرق وسط.","تعادل.","medium",("straight","wavy"),"modern","medium",2,8,"diamond",products=("cream",)),
    _S("di-side-fringe-soft","چتری نرم کناری","Soft Side Fringe","چتری نرم کناری.","تعادل.","short",("straight","wavy"),"classic","low",2,8,"diamond",products=("cream",)),
    _S("di-brush-up-medium","براش متوسط","Medium Brush Up","براش متوسط.","حجم.","medium",("straight","wavy"),"modern","medium",3,8,"diamond",products=("wax",)),
    _S("tr-high-pomp","پامپ بلند","High Pompadour","پامپ با حجم.","پیشانی پهن.","short",("straight",),"classic","high",4,8,"triangle",products=("pomade","blow-dryer")),
    _S("tr-quiff-height","کویف بلند","Quiff Height","کویف ارتفاع.","تعادل بالا.","short",("straight","wavy"),"modern","medium",3,9,"triangle",products=("clay",)),
    _S("tr-faux-hawk","فو هاوک","Faux Hawk","حجم مرکزی.","بالا پر.","short",("straight",),"edgy","medium",3,8,"triangle",products=("gel",)),
    _S("tr-spiky-fade","سیخ فید","Spiky Fade","سیخ با فید.","ارتفاع.","short",("straight",),"edgy","medium",2,7,"triangle",products=("gel",)),
    _S("tr-high-top","های تاپ","High Top Fade","رویه بلند عمودی.","بالا بلند.","short",("curly","straight"),"edgy","medium",3,7,"triangle",products=("gel",)),
    _S("tr-textured-vol","حجم بافت","Textured Volume","رویه حجم‌دار.","عرض بالا.","short",("wavy",),"modern","medium",3,8,"triangle",products=("clay",)),
    _S("tr-undercut-vol","آندرکات حجم","Undercut Volume","آندرکات حجم.","تعادل.","short",("straight","wavy"),"edgy","medium",3,8,"triangle",products=("clay",)),
    _S("tr-mohawk","موهاک","Mohawk","نوار مرکزی.","حداکثر ارتفاع.","short",("straight",),"edgy","high",4,6,"triangle",products=("gel",)),
    _S("tr-vertical-fringe","چتری عمودی","Vertical Fringe","چتری عمودی.","بالا.","short",("straight",),"modern","medium",3,7,"triangle",products=("clay",)),
    _S("tr-tall-quiff","کویف بلند","Tall Quiff","کویف بلند.","تعادل.","short",("straight",),"modern","high",4,7,"triangle",products=("clay",)),
    _S("tr-crown-vol","حجم تاج","Crown Volume","حجم تاج.","بالا پر.","short",("wavy","curly"),"modern","medium",3,8,"triangle",products=("clay",)),
    _S("tr-pomp-fade","پامپ فید","Pomp Fade","پامپ فید.","ارتفاع.","short",("straight",),"modern","medium",4,8,"triangle",products=("pomade",)),
    _S("tr-quiff-fade","کویف فید","Quiff Fade","کویف فید.","تعادل.","short",("straight","wavy"),"modern","medium",3,8,"triangle",products=("clay",)),
    _S("tr-textured-high","بافت بلند","Textured High","بافت بلند.","بالا.","short",("wavy",),"modern","medium",3,8,"triangle",products=("clay",)),
    _S("tr-vertical-spiky","سیخ عمودی","Vertical Spiky","سیخ عمودی.","بالا.","short",("straight",),"edgy","medium",3,7,"triangle",products=("gel",)),
    _S("tr-brush-up-high","براش بلند","High Brush Up","براش بلند.","ارتفاع.","short",("straight",),"modern","medium",3,7,"triangle",products=("wax",)),
    _S("tr-slick-back-high","اسلیک بلند","High Slick","اسلیک با حجم بالا.","ارتفاع.","short",("straight",),"classic","medium",3,7,"triangle",products=("pomade",)),
    _S("tr-pompadour-classic","پامپ کلاسیک","Classic Pompadour","پامپ کلاسیک.","شیک.","short",("straight",),"classic","high",4,8,"triangle",products=("pomade",)),
    _S("tr-curly-top-high","فر بالا بلند","High Curly Top","فر بالا.","بالا.","short",("curly",),"modern","medium",3,7,"triangle",products=("curl-cream",)),
    _S("tr-curly-fade","فر فید","Curly Fade","فر فید.","بالا.","short",("curly",),"modern","medium",3,8,"triangle",products=("curl-cream",)),
    _S("tr-curly-side","فر کناری","Curly Side","فر کناری.","کناری.","short",("curly",),"modern","medium",3,7,"triangle",products=("curl-cream",)),
    _S("tr-curly-medium","فر متوسط","Curly Medium","فر متوسط.","بالا.","medium",("curly",),"modern","medium",3,7,"triangle",products=("curl-cream",)),
    _S("tr-side-quiff-high","کویف کناری بلند","High Side Quiff","کویف کناری بلند.","تعادل.","short",("straight","wavy"),"modern","medium",3,8,"triangle",products=("clay",)),
    _S("tr-brush-back-high","براش بلند","Brush High","براش بلند.","ارتفاع.","short",("straight",),"modern","medium",3,7,"triangle",products=("pomade",)),
    _S("tr-side-part-high","فرق بلند","High Side Part","فرق بلند.","تعادل.","short",("straight",),"classic","medium",3,8,"triangle",products=("pomade",)),
    _S("tr-crew-taper","کرو تیپر","Crew Taper","کرو تیپر.","تمیز.","short",("straight",),"classic","low",2,8,"triangle",products=("wax",)),
    _S("tr-crew-fade","کرو فید","Crew Fade","کرو فید.","تمیز.","short",("straight",),"modern","low",2,8,"triangle",products=("wax",)),
    _S("tr-french-crop","کراپ فرانسوی","French Crop","کراپ.","تمیز.","short",("straight",),"modern","low",2,8,"triangle",products=("wax",)),
    _S("tr-textured-crop","کراپ بافت","Textured Crop","کراپ بافت.","تمیز.","short",("straight","wavy"),"modern","low",2,8,"triangle",products=("clay",)),
    _S("tr-side-sweep","سویپ کناری","Side Sweep","سویپ کناری.","تعادل.","short",("straight","wavy"),"classic","low",2,8,"triangle",products=("pomade",)),
    _S("tr-tousled-top","رویه شلخته","Tousled Top","رویه شلخته.","حجم.","short",("wavy",),"modern","low",2,8,"triangle",products=("clay",)),
    _S("tr-quiff-mid","کویف متوسط","Mid Quiff","کویف متوسط.","تعادل.","short",("straight",),"modern","medium",3,8,"triangle",products=("clay",)),
)

class HairstyleRegistry:
    @staticmethod
    def to_dict(h: HairStyle) -> dict[str, Any]:
        return {
            "id": h.id, "name_fa": h.name_fa, "name_en": h.name_en,
            "description_fa": h.description_fa, "why_fa": h.why_fa,
            "length": h.length, "textures": list(h.textures), "vibe": h.vibe,
            "maintenance": h.maintenance, "difficulty": h.difficulty,
            "popularity": h.popularity, "face_shapes": list(h.face_shapes),
            "styling_minutes": h.styling_minutes, "products": list(h.products),
            "thickness": list(h.thickness), "seasons": list(h.seasons),
            "occasions": list(h.occasions), "tags": list(h.tags),
            "receding_friendly": h.receding_friendly,
        }

    @staticmethod
    def for_shape(shape: str) -> list[dict[str, Any]]:
        return [HairstyleRegistry.to_dict(h) for h in HAIRSTYLES if shape in h.face_shapes]

# ============================================================
# 5) RateLimiter + TTL LRU cache + Metrics
# ============================================================
class InMemoryRateLimiter:
    def __init__(self, limit_per_minute: int) -> None:
        self.limit = limit_per_minute
        self._buckets: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            ts = [t for t in self._buckets.get(key, []) if t >= now - 60.0]
            if len(ts) >= self.limit:
                self._buckets[key] = ts
                return False
            ts.append(now)
            self._buckets[key] = ts
            return True

class LRUCache:
    def __init__(self, maxsize: int = 128, ttl: int = 3600) -> None:
        self.maxsize = maxsize
        self.ttl = ttl
        self._data: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        with self._lock:
            item = self._data.get(key)
            if item is None: return None
            ts, val = item
            if time.time() - ts > self.ttl:
                del self._data[key]
                return None
            self._data.move_to_end(key)
            return val

    def put(self, key: str, value: dict[str, Any]) -> None:
        with self._lock:
            self._data[key] = (time.time(), value)
            self._data.move_to_end(key)
            while len(self._data) > self.maxsize: self._data.popitem(last=False)

_METRICS: dict[str, float] = {
    "requests_total": 0, "analyses_total": 0, "cache_hits": 0,
    "cache_misses": 0, "errors_total": 0, "rate_limited_total": 0,
    "total_latency_ms": 0.0,
}
_METRICS_LOCK = threading.Lock()
def _inc(key: str, by: float = 1.0) -> None:
    with _METRICS_LOCK: _METRICS[key] = _METRICS.get(key, 0.0) + by

# ============================================================
# 6) Frontend (HTML/CSS/JS)
# ============================================================
INDEX_HTML = r"""<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#8b6f4e">
<title>هد لاین — مشاور کوانتومی مدل مو</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@400;700&display=swap');
  
  :root{
    --p:#8b6f4e;--pd:#6d5640;--pl:#a98a68;--bg:#f7f5f2;--sf:#fff;
    --tx:#2c2c2c;--mu:#888;--bd:#e5e0d8;--er:#c0392b;--erb:#fdecea;
    --ok:#2e7d32;--qu:#5b3fa8;--qub:#e8e3f7;--r:12px;--sh:0 4px 12px rgba(139,111,78,.12)
  }
  *,*::before,*::after{box-sizing:border-box}
  body{margin:0;min-height:100dvh;background-color:var(--bg);color:var(--tx);
       font-family:"Vazirmatn",Tahoma,sans-serif;line-height:1.6;padding:20px;
       background-image: linear-gradient(rgba(247, 245, 242, 0.92), rgba(247, 245, 242, 0.95)), url('https://images.unsplash.com/photo-1585747860715-2ba37e788b70?q=80&w=2074&auto=format&fit=crop');
       background-size: cover; background-position: center; background-attachment: fixed;}
  .container { max-width: 520px; margin: 0 auto; position: relative; }
  
  .header { text-align: center; margin-bottom: 24px; }
  h1 { margin:0; font-size:2rem; font-weight:700; color:var(--p); }
  .h1-en { display: block; font-size: 1rem; font-weight: 400; color: var(--mu); letter-spacing: 2px; margin-top: 4px;}
  .badge{display:inline-block;background:var(--qub);color:var(--qu);font-size:.7rem;
         padding:4px 10px;border-radius:999px;font-weight:600;margin-top:8px;}
  
  .instruction{padding:12px 16px;background:rgba(255,255,255,0.85);backdrop-filter: blur(10px);border-right:4px solid var(--pl);
               border-radius:8px;font-size:.9rem;margin-bottom:16px;box-shadow:var(--sh)}
  .btn{display:flex;align-items:center;justify-content:center;gap:8px;width:100%;padding:14px;
       border:none;border-radius:var(--r);font-family:inherit;font-size:1rem;font-weight:600;
       cursor:pointer;transition:background .2s,transform .2s;-webkit-tap-highlight-color:transparent}
  .btn:active:not(:disabled){transform:scale(.98)}
  .btn:disabled{background:var(--bd);color:var(--mu);cursor:not-allowed}
  .btn--primary{background:var(--p);color:#fff}.btn--primary:hover:not(:disabled){background:var(--pd)}
  .btn--secondary{background:rgba(255,255,255,0.9);color:var(--p);border:1.5px solid var(--p)}
  .btn--ghost{background:transparent;color:var(--mu)}
  .actions{display:flex;flex-direction:column;gap:10px}
  .vh{position:absolute!important;width:1px;height:1px;overflow:hidden;clip:rect(0,0,0,0);white-space:nowrap}
  
  .preview{display:none;position:relative;border-radius:var(--r);overflow:hidden;margin-bottom:16px;box-shadow: var(--sh);}
  .preview.is-visible{display:block}
  .preview img{display:block;width:100%;max-height:320px;object-fit:cover}
  .preview__remove{position:absolute;top:8px;inset-inline-start:8px;width:36px;height:36px;padding:0;
       border:none;border-radius:50%;background:rgba(0,0,0,.6);color:#fff;font-size:18px;cursor:pointer;
       display:grid;place-items:center}
  
  .loading{display:none;text-align:center;padding:24px;background:rgba(255,255,255,0.9);backdrop-filter: blur(10px);border-radius:var(--r)}
  .loading.is-visible{display:block}
  .spinner{width:40px;height:40px;margin:0 auto 12px;border:3px solid var(--bd);border-top-color:var(--qu);
       border-radius:50%;animation:spin .8s linear infinite}
  @keyframes spin{to{transform:rotate(360deg)}}
  
  .alert{display:none;padding:12px 16px;border-radius:8px;font-size:.9rem;margin-top:12px}
  .alert.is-visible{display:block}
  .alert--error{background:var(--erb);color:var(--er);border-right:4px solid var(--er)}
  
  .result{display:none;margin-top:16px}
  .result.is-visible{display:block}
  .result__card{background:rgba(255,255,255,0.95);backdrop-filter: blur(10px);border-radius:var(--r);padding:20px;box-shadow:var(--sh)}
  .result__shape{text-align:center;padding-bottom:16px;margin-bottom:16px;border-bottom:1px solid var(--bd)}
  .result__shape-label{display:block;font-size:.85rem;color:var(--mu)}
  .result__shape-value{display:block;font-size:1.7rem;font-weight:700;color:var(--p);margin-top:4px}
  
  .quantum-box{background:linear-gradient(135deg,rgba(244,240,255,0.9) 0%,rgba(255,255,255,0.9) 100%);border:1px solid var(--qub);
       border-radius:10px;padding:12px 14px;margin:12px 0}
  .quantum-box__title{font-size:.75rem;color:var(--qu);font-weight:700; margin-bottom:8px;letter-spacing:.02em}
  .quantum-metrics{display:grid;grid-template-columns:1fr 1fr;gap:8px 14px}
  .qm{display:flex;flex-direction:column;gap:2px}
  .qm__label{font-size:.7rem;color:var(--mu)}
  .qm__val{font-size:.95rem;font-weight:700;color:var(--tx);font-feature-settings:"tnum"}
  .qm__val--qu{color:var(--qu)}
  
  .conf{display:flex;align-items:center;gap:8px;justify-content:center;margin:10px 0 14px}
  .conf__bar{flex:0 0 140px;height:6px;background:var(--bd);border-radius:3px;overflow:hidden;position:relative}
  .conf__ci{position:absolute;top:0;height:100%;background:rgba(91,63,168,.25)}
  .conf__fill{position:absolute;top:0;height:100%;background:var(--p);border-radius:3px;transition:width .5s}
  .conf__pct{font-size:.8rem;color:var(--mu)}
  
  .contribs{margin:10px 0 4px}
  .contrib{margin-bottom:6px}
  .contrib__row{display:flex;justify-content:space-between;font-size:.72rem;color:var(--mu);margin-bottom:2px}
  .contrib__bar{height:4px;background:var(--bd);border-radius:2px;overflow:hidden}
  .contrib__fill{height:100%;background:var(--qu);border-radius:2px}
  
  .warn-box{background:#fff8e1;border:1px solid #ffe082;border-radius:8px;padding:8px 12px;
       font-size:.78rem;color:#8a6d00;margin:8px 0}
       
  .prefs{margin:16px 0}
  .prefs__row{display:flex;flex-direction:column;gap:6px;margin-bottom:10px}
  .prefs__label{font-size:.85rem;color:var(--mu);font-weight:600}
  .chips{display:flex;flex-wrap:wrap;gap:6px}
  .chip{padding:6px 12px;border:1.5px solid var(--bd);background:rgba(255,255,255,0.9);border-radius:999px;
        font-family:inherit;font-size:.8rem;color:var(--tx);cursor:pointer;transition:all .15s}
  .chip:hover{border-color:var(--pl)}
  .chip--active{background:var(--p);border-color:var(--p);color:#fff}
  
  .result__title{margin:16px 0 12px;font-weight:700;display:flex;justify-content:space-between;align-items:center}
  .result__count{font-size:.8rem;color:var(--mu);font-weight:400}
  
  .style-card{display:flex;gap:12px;padding:12px;background:rgba(247,245,242,0.8);border-radius:8px;margin-bottom:10px}
  .style-num{width:26px;height:26px;border-radius:50%;background:var(--p);color:#fff;font-size:.85rem;
       display:grid;place-items:center;flex-shrink:0}
  .style-body{flex:1;min-width:0}
  .style-title{font-size:.95rem}
  .style-title-en{font-size:.75rem;color:var(--mu);font-weight:400;margin-inline-start:4px}
  .style-desc{font-size:.82rem;color:var(--mu);margin:4px 0 0}
  .style-why{font-size:.78rem;color:var(--p);margin:4px 0 0}
  .style-tags{display:flex;flex-wrap:wrap;gap:4px;margin-top:6px}
  .tag{font-size:.7rem;padding:2px 8px;background:var(--sf);border:1px solid var(--bd);border-radius:999px;color:var(--mu)}
  .tag--diff{color:#b45309;border-color:#fde68a;background:#fffbeb}
  .tag--maint{color:#166534;border-color:#bbf7d0;background:#f0fdf4}
  .tag--img{background:#e8f0fe;color:#1a73e8;border-color:#a8c7fa;text-decoration:none;display:inline-block;cursor:pointer}
  .tag--img:hover{background:#d2e3fc}
  
  .empty{padding:24px;text-align:center;color:var(--mu);font-size:.9rem}
  .footer { text-align: center; margin-top: 40px; font-size: 0.75rem; color: var(--mu); padding-bottom: 20px;}
  .footer strong { color: var(--p); }
  
  [hidden]{display:none!important}
  @media (prefers-reduced-motion:reduce){*,*::before,*::after{animation-duration:.01ms!important;transition-duration:.01ms!important}}
</style>
</head>
<body>
<div class="container">
  <header class="header">
    <h1>هد لاین <span class="h1-en">HEADLINE</span></h1>
    <span class="badge">v4.2 موتور کوانتومی</span>
  </header>

  <section id="step-upload">
    <p class="instruction">عکس <strong>روبه‌رو</strong>، با نور خوب، بدون لبخند، مو عقب.</p>
    <div id="preview" class="preview" aria-live="polite">
      <img id="preview-img" alt="پیش‌نمایش">
      <button id="btn-remove" class="preview__remove" type="button" aria-label="حذف">×</button>
    </div>
    <div class="actions">
      <label for="camera-input" class="btn btn--primary" role="button" tabindex="0">📸 دوربین</label>
      <input type="file" id="camera-input" class="vh" accept="image/*" capture="user">
      <label for="gallery-input" class="btn btn--secondary" role="button" tabindex="0">🖼️ گالری</label>
      <input type="file" id="gallery-input" class="vh" accept="image/jpeg,image/png,image/webp">
      <button id="btn-analyze" class="btn btn--primary" type="button" disabled>✨ تحلیل کوانتومی چهره</button>
    </div>
    <div id="error-box" class="alert alert--error" role="alert"></div>
  </section>

  <section id="loading" class="loading" aria-live="polite" aria-busy="true">
    <div class="spinner" aria-hidden="true"></div>
    <p style="margin:0;color:var(--qu);font-weight:600">در حال اندازه‌گیری کوانتومی...</p>
  </section>

  <section id="step-result" class="result" aria-live="polite">
    <div class="result__card">
      <div id="result-header"></div>
      <div id="prefs" class="prefs" aria-label="سلیقه">
        <div class="prefs__row"><span class="prefs__label">طول</span>
          <div class="chips" data-pref="length">
            <button class="chip chip--active" data-val="all" type="button">همه</button>
            <button class="chip" data-val="short" type="button">کوتاه</button>
            <button class="chip" data-val="medium" type="button">متوسط</button>
            <button class="chip" data-val="long" type="button">بلند</button>
          </div>
        </div>
        <div class="prefs__row"><span class="prefs__label">بافت</span>
          <div class="chips" data-pref="texture">
            <button class="chip chip--active" data-val="all" type="button">همه</button>
            <button class="chip" data-val="straight" type="button">صاف</button>
            <button class="chip" data-val="wavy" type="button">موج</button>
            <button class="chip" data-val="curly" type="button">فر</button>
          </div>
        </div>
        <div class="prefs__row"><span class="prefs__label">سبک</span>
          <div class="chips" data-pref="vibe">
            <button class="chip chip--active" data-val="all" type="button">همه</button>
            <button class="chip" data-val="classic" type="button">کلاسیک</button>
            <button class="chip" data-val="modern" type="button">مدرن</button>
            <button class="chip" data-val="edgy" type="button">جسورانه</button>
            <button class="chip" data-val="elegant" type="button">شیک</button>
          </div>
        </div>
        <div class="prefs__row"><span class="prefs__label">نگهداری</span>
          <div class="chips" data-pref="maintenance">
            <button class="chip chip--active" data-val="all" type="button">همه</button>
            <button class="chip" data-val="low" type="button">کم</button>
            <button class="chip" data-val="medium" type="button">متوسط</button>
            <button class="chip" data-val="high" type="button">زیاد</button>
          </div>
        </div>
      </div>
      <div class="result__title">
        <span>مدل‌های پیشنهادی</span>
        <span class="result__count" id="style-count"></span>
      </div>
      <div id="styles-list"></div>
    </div>
    <button id="btn-restart" class="btn btn--ghost" type="button" style="margin-top:12px">↺ شروع دوباره</button>
  </section>

  <footer class="footer">
    توسعه‌دهنده: <strong>امین راد</strong>
  </footer>
</div>

<script>
(() => {
  "use strict";
  const CONFIG = Object.freeze({
    API_PATH: "/api/v1/analyze",
    TIMEOUT_MS: 45000, MAX_FILE_MB: 10, MIN_SIDE: 240,
    ALLOWED_TYPES: ["image/jpeg", "image/png", "image/webp"],
  });
  const SHAPE_FA = {oval:"بیضی",round:"گرد",square:"مربعی",heart:"قلبی",oblong:"کشیده",diamond:"الماس",triangle:"مثلثی"};
  const LEN_FA   = {short:"کوتاه",medium:"متوسط",long:"بلند"};
  const TEX_FA   = {straight:"صاف",wavy:"موج‌دار",curly:"فر"};
  const VIBE_FA  = {classic:"کلاسیک",modern:"مدرن",edgy:"جسورانه",elegant:"شیک"};
  const MAINT_FA = {low:"نگهداری کم",medium:"نگهداری متوسط",high:"نگهداری زیاد"};
  const OBS_FA   = {geo:"هندسی",curv:"انحنا",sym:"تقارن",ang:"زاویه"};
  const WARN_FA  = {
    blurry_image:"عکس تار است — کیفیت تحلیل ممکن است پایین باشد.",
    too_dark:"عکس خیلی تاریک است.",
    too_bright:"عکس خیلی روشن است.",
    low_stability:"پایداری تحلیل پایین است — عکس واضح‌تری امتحان کنید.",
    high_entropy:"عدم‌قطعیت بالا — عکس روبه‌رو با نور خوب بگیرید.",
  };
  const ERROR_FA = {
    no_face:"چهره‌ای پیدا نشد. عکس روبه‌رو و با نور خوب بگیرید.",
    invalid_image:"تصویر قابل خواندن نیست. JPEG/PNG/WEBP.",
    image_too_small:`حداقل ${CONFIG.MIN_SIDE}×${CONFIG.MIN_SIDE} پیکسل.`,
    feature_failed:"محاسبه‌ی ویژگی‌ها ناموفق بود.",
    file_too_large:"حجم بیش از حد مجاز.",
    rate_limit_exceeded:"تعداد درخواست زیاد است.",
    internal_server_error:"خطای سرور.",
  };
  const $ = (id) => document.getElementById(id);
  const els = {
    upload:$("step-upload"),loading:$("loading"),result:$("step-result"),
    resultHeader:$("result-header"),stylesList:$("styles-list"),
    styleCount:$("style-count"),prefs:$("prefs"),
    preview:$("preview"),previewImg:$("preview-img"),
    cameraInput:$("camera-input"),galleryInput:$("gallery-input"),
    analyzeBtn:$("btn-analyze"),removeBtn:$("btn-remove"),
    restartBtn:$("btn-restart"),errorBox:$("error-box"),
  };
  if (!els.upload) return;

  let selectedFile = null, objectUrl = null, lastResult = null;
  let prefs = {length:"all",texture:"all",vibe:"all",maintenance:"all"};

  const showError = (m) => { els.errorBox.textContent = m; els.errorBox.classList.add("is-visible"); };
  const clearError = () => { els.errorBox.textContent = ""; els.errorBox.classList.remove("is-visible"); };
  const setState = (s) => {
    els.upload.hidden = s !== "upload";
    els.loading.classList.toggle("is-visible", s === "loading");
    els.result.classList.toggle("is-visible", s === "result");
  };
  function validateFile(f) {
    if (!f) return "فایلی انتخاب نشد.";
    if (!CONFIG.ALLOWED_TYPES.includes(f.type)) return "فرمت پشتیبانی نمی‌شود.";
    if (f.size > CONFIG.MAX_FILE_MB * 1024 * 1024) return `حداکثر ${CONFIG.MAX_FILE_MB}MB.`;
    return null;
  }
  function setFile(f) {
    clearError(); if (!f) return;
    const err = validateFile(f);
    if (err) { showError(err); resetFile(); return; }
    selectedFile = f;
    if (objectUrl) URL.revokeObjectURL(objectUrl);
    objectUrl = URL.createObjectURL(f);
    els.previewImg.src = objectUrl;
    els.preview.classList.add("is-visible");
    els.analyzeBtn.disabled = false;
  }
  function resetFile() {
    selectedFile = null; lastResult = null;
    if (objectUrl) { URL.revokeObjectURL(objectUrl); objectUrl = null; }
    els.previewImg.removeAttribute("src");
    els.preview.classList.remove("is-visible");
    els.analyzeBtn.disabled = true;
    els.cameraInput.value = ""; if (els.galleryInput) els.galleryInput.value = "";
  }
  async function requestAnalysis(file) {
    const form = new FormData(); form.append("image", file);
    const ctrl = new AbortController();
    const tid = setTimeout(() => ctrl.abort(), CONFIG.TIMEOUT_MS);
    try {
      const res = await fetch(CONFIG.API_PATH, { method:"POST", body:form, signal:ctrl.signal });
      const data = await res.json().catch(() => ({}));
      if (!res.ok || data.error) throw new Error(ERROR_FA[data.error] || data.message_fa || `خطای ${res.status}`);
      return data;
    } finally { clearTimeout(tid); }
  }
  function renderHeader(data) {
    const shapeFa = data.shape_fa || SHAPE_FA[data.shape] || data.shape;
    const pct = Math.round((data.confidence || 0) * 100);
    const [ciLo, ciHi] = data.confidence_interval || [0, 1];
    const ciPctLo = Math.round(ciLo * 100), ciPctHi = Math.round(ciHi * 100);
    const entropy = data.uncertainty_entropy ?? null;
    const purity = data.purity ?? null;
    const coherence = data.coherence ?? null;
    const stab = data.mc_stability ?? null;
    const quality = data.image_quality || {};
    const warnings = data.warnings || [];
    const cont = data.observable_contributions || {};

    const wrapper = document.createElement("div");
    wrapper.innerHTML = `
      <div class="result__shape">
        <span class="result__shape-label">فرم صورت (اندازه‌گیری بورن)</span>
        <span class="result__shape-value">${shapeFa}</span>
      </div>
      ${warnings.length ? `<div class="warn-box">⚠️ ${warnings.map(w => WARN_FA[w] || w).join(" • ")}</div>` : ""}
      <div class="conf">
        <span class="conf__bar">
          <span class="conf__ci" style="left:${ciPctLo}%;width:${ciPctHi-ciPctLo}%"></span>
          <span class="conf__fill" style="left:0;width:${pct}%"></span>
        </span>
        <span class="conf__pct">${pct}% (CI ${ciPctLo}–${ciPctHi}%)</span>
      </div>
      <div class="quantum-box">
        <div class="quantum-box__title">⚛️ سنجه‌های کوانتومی</div>
        <div class="quantum-metrics">
          <div class="qm"><span class="qm__label">آنتروپی فون‌نویمان</span>
            <span class="qm__val qm__val--qu">${entropy !== null ? entropy.toFixed(3) : "—"}</span></div>
          <div class="qm"><span class="qm__label">خلوص Tr(ρ²)</span>
            <span class="qm__val qm__val--qu">${purity !== null ? purity.toFixed(3) : "—"}</span></div>
          <div class="qm"><span class="qm__label">همدوسی ℓ₁</span>
            <span class="qm__val qm__val--qu">${coherence !== null ? coherence.toFixed(3) : "—"}</span></div>
          <div class="qm"><span class="qm__label">پایداری MC</span>
            <span class="qm__val qm__val--qu">${stab !== null ? (stab*100).toFixed(1)+"%" : "—"}</span></div>
          <div class="qm"><span class="qm__label">شارپنس تصویر</span>
            <span class="qm__val">${quality.sharpness !== undefined ? (quality.sharpness*100).toFixed(0)+"%" : "—"}</span></div>
          <div class="qm"><span class="qm__label">روشنایی</span>
            <span class="qm__val">${quality.brightness !== undefined ? (quality.brightness*100).toFixed(0)+"%" : "—"}</span></div>
        </div>
        <div class="contribs">
          <div class="quantum-box__title" style="margin-top:8px">سهم مشاهده‌گرها</div>
          ${Object.entries(cont).map(([k,v]) => `
            <div class="contrib">
              <div class="contrib__row"><span>${OBS_FA[k]||k}</span><span>${(v*100).toFixed(0)}%</span></div>
              <div class="contrib__bar"><div class="contrib__fill" style="width:${(v*100).toFixed(0)}%"></div></div>
            </div>`).join("")}
        </div>
      </div>`;
    els.resultHeader.replaceChildren(wrapper);
  }
  function styleMatches(s) {
    if (prefs.length !== "all" && s.length !== prefs.length) return false;
    if (prefs.texture !== "all" && !s.textures.includes(prefs.texture)) return false;
    if (prefs.vibe !== "all" && s.vibe !== prefs.vibe) return false;
    if (prefs.maintenance !== "all" && s.maintenance !== prefs.maintenance) return false;
    return true;
  }
  function renderStyles() {
    if (!lastResult) return;
    const list = lastResult.hairstyles.filter(styleMatches);
    els.styleCount.textContent = `${list.length} از ${lastResult.hairstyles.length}`;
    if (list.length === 0) {
      const e = document.createElement("div");
      e.className = "empty"; e.textContent = "با این فیلترها مدلی پیدا نشد.";
      els.stylesList.replaceChildren(e); return;
    }
    const fragment = document.createDocumentFragment();
    list.forEach((s, i) => {
      const card = document.createElement("div");
      card.className = "style-card";
      const num = document.createElement("span");
      num.className = "style-num"; num.textContent = String(i + 1);
      const body = document.createElement("div");
      body.className = "style-body";
      const title = document.createElement("strong");
      title.className = "style-title"; title.textContent = s.name_fa;
      const en = document.createElement("span");
      en.className = "style-title-en"; en.textContent = s.name_en;
      title.appendChild(en); body.appendChild(title);
      const desc = document.createElement("p");
      desc.className = "style-desc"; desc.textContent = s.description_fa;
      body.appendChild(desc);
      if (s.why_fa) {
        const why = document.createElement("p");
        why.className = "style-why"; why.textContent = "💡 " + s.why_fa;
        body.appendChild(why);
      }
      const tags = document.createElement("div");
      tags.className = "style-tags";
      const defs = [
        [LEN_FA[s.length], ""], [VIBE_FA[s.vibe], ""],
        [MAINT_FA[s.maintenance], "tag--maint"],
        [`سختی ${s.difficulty}/5`, "tag--diff"],
        ...s.textures.map(t => [TEX_FA[t], ""]),
      ].filter(([t]) => t);
      defs.forEach(([t, cls]) => {
        const tag = document.createElement("span");
        tag.className = "tag " + (cls || "");
        tag.textContent = t;
        tags.appendChild(tag);
      });
      
      // دکمه دیدن عکس مدل
      const imgLink = document.createElement("a");
      imgLink.href = "https://www.google.com/search?tbm=isch&q=" + encodeURIComponent(s.name_en + " haircut");
      imgLink.target = "_blank";
      imgLink.className = "tag tag--img";
      imgLink.textContent = "🔍 دیدن عکس";
      tags.appendChild(imgLink);

      body.appendChild(tags);
      card.append(num, body);
      fragment.appendChild(card);
    });
    els.stylesList.replaceChildren(fragment);
  }
  function renderResult(data) {
    lastResult = data; renderHeader(data); renderStyles(); setState("result");
  }
  async function handleAnalyze() {
    if (!selectedFile) return;
    clearError(); els.analyzeBtn.disabled = true; setState("loading");
    try { renderResult(await requestAnalysis(selectedFile)); }
    catch (err) {
      setState("upload"); els.analyzeBtn.disabled = !selectedFile;
      if (err.name === "AbortError") showError("زمان درخواست تمام شد.");
      else if (err.message === "Failed to fetch") showError("اتصال برقرار نشد.");
      else showError(err.message || "خطای نامشخص.");
    }
  }
  els.prefs.querySelectorAll(".chips").forEach(chipsEl => {
    const key = chipsEl.dataset.pref;
    chipsEl.addEventListener("click", (e) => {
      const btn = e.target.closest(".chip");
      if (!btn) return;
      chipsEl.querySelectorAll(".chip").forEach(c => c.classList.remove("chip--active"));
      btn.classList.add("chip--active");
      prefs[key] = btn.dataset.val;
      renderStyles();
    });
  });
  els.cameraInput.addEventListener("change", (e) => setFile(e.target.files?.[0]));
  if (els.galleryInput) els.galleryInput.addEventListener("change", (e) => setFile(e.target.files?.[0]));
  els.analyzeBtn.addEventListener("click", handleAnalyze);
  if (els.removeBtn) els.removeBtn.addEventListener("click", () => { resetFile(); clearError(); });
  if (els.restartBtn) els.restartBtn.addEventListener("click", () => {
    resetFile(); clearError(); setState("upload");
  });
  document.querySelectorAll('label[role="button"]').forEach(l => {
    l.addEventListener("keydown", (e) => {
      if (e.key === "Enter" || e.key === " ") { e.preventDefault(); l.click(); }
    });
  });
  setState("upload");
})();
</script>
</body>
</html>
"""

# ============================================================
# 7) Flask app
# ============================================================
def create_app(cfg: Config | None = None) -> Flask:
    cfg = cfg or Config()
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = cfg.max_content_length
    app.config["JSON_AS_ASCII"] = False

    analyzer = QuantumFaceAnalyzer(cfg)
    app.extensions["analyzer"] = analyzer
    limiter = InMemoryRateLimiter(cfg.rate_limit_per_minute)
    cache = LRUCache(cfg.cache_size, cfg.cache_ttl_seconds)

    if cfg.enable_cors:
        from flask_cors import CORS
        origins = "*" if cfg.cors_origins.strip() == "*" else [o.strip() for o in cfg.cors_origins.split(",")]
        CORS(app, resources={r"/*": {"origins": origins}})
        logger.info("CORS enabled: %s", origins)

    @app.before_request
    def _assign_rid():
        g.request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:12]
        g.t0 = time.perf_counter()
        _inc("requests_total")

    @app.after_request
    def _sec(resp: Response) -> Response:
        resp.headers["X-Content-Type-Options"] = "nosniff"
        resp.headers["X-Frame-Options"] = "SAMEORIGIN"
        resp.headers["Referrer-Policy"] = "no-referrer"
        resp.headers["X-Request-ID"] = getattr(g, "request_id", "-")
        elapsed_ms = (time.perf_counter() - getattr(g, "t0", time.perf_counter())) * 1000
        _inc("total_latency_ms", elapsed_ms)
        resp.headers["X-Response-Time"] = f"{elapsed_ms:.1f}ms"
        return resp

    @app.errorhandler(RequestEntityTooLarge)
    def _large(_e):
        _inc("errors_total")
        return jsonify({"error": "file_too_large", "message_fa": "حجم فایل بیش از حد مجاز است."}), 413

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        _inc("errors_total")
        return jsonify({"error": e.name.lower().replace(" ", "_"),
                        "message_fa": e.description or e.name}), e.code or 500

    @app.errorhandler(Exception)
    def _unhandled(e: Exception):
        _inc("errors_total")
        logger.exception("Unhandled: %s", e)
        return jsonify({"error": "internal_server_error",
                        "message_fa": "خطای سرور."}), 500

    @app.get("/")
    def index() -> Response:
        return Response(INDEX_HTML, mimetype="text/html; charset=utf-8")

    @app.get("/health")
    def health():
        return jsonify({
            "status": "ok",
            "engine": cfg.engine_version,
            "version": cfg.api_version,
            "cache_size": cfg.cache_size,
            "cache_ttl_seconds": cfg.cache_ttl_seconds,
            "n_mc_samples": cfg.n_mc_samples,
            "n_hairstyles": len(HAIRSTYLES),
            "n_shapes": len(SHAPES),
        })

    @app.get("/metrics")
    def metrics():
        with _METRICS_LOCK:
            snap = dict(_METRICS)
        if snap.get("requests_total", 0) > 0:
            snap["avg_latency_ms"] = round(
                snap.get("total_latency_ms", 0) / snap["requests_total"], 2)
        return jsonify(snap)

    @app.get(f"/api/{cfg.api_version}/version")
    def version():
        return jsonify({
            "engine": cfg.engine_version,
            "api_version": cfg.api_version,
            "n_hairstyles": len(HAIRSTYLES),
            "n_shapes": len(SHAPES),
        })

    @app.get(f"/api/{cfg.api_version}/shapes")
    def shapes():
        return jsonify({
            "shapes": [
                {"key": s, "label_fa": SHAPE_LABELS_FA[s],
                 "prior": float(SHAPE_PRIOR[i]),
                 "n_hairstyles": sum(1 for h in HAIRSTYLES if s in h.face_shapes)}
                for i, s in enumerate(SHAPES)
            ]
        })

    @app.get(f"/api/{cfg.api_version}/styles")
    def list_styles():
        shape = request.args.get("shape")
        length = request.args.get("length")
        texture = request.args.get("texture")
        vibe = request.args.get("vibe")
        maint = request.args.get("maintenance")
        q = (request.args.get("q") or "").lower()
        try:
            limit = max(1, min(int(request.args.get("limit", "100")), 500))
            offset = max(0, int(request.args.get("offset", "0")))
        except ValueError:
            return jsonify({"error": "invalid_pagination"}), 400

        styles = HAIRSTYLES
        if shape: styles = [s for s in styles if shape in s.face_shapes]
        if length: styles = [s for s in styles if s.length == length]
        if texture: styles = [s for s in styles if texture in s.textures]
        if vibe: styles = [s for s in styles if s.vibe == vibe]
        if maint: styles = [s for s in styles if s.maintenance == maint]
        if q:
            styles = [s for s in styles if q in s.name_fa.lower()
                      or q in s.name_en.lower()
                      or q in s.description_fa.lower()
                      or any(q in t.lower() for t in s.tags)]

        total = len(styles)
        page = styles[offset:offset + limit]
        return jsonify({
            "total": total, "offset": offset, "limit": limit,
            "styles": [HairstyleRegistry.to_dict(s) for s in page],
        })

    def _do_analyze() -> tuple[Response, int]:
        ip = (request.headers.get("X-Forwarded-For", request.remote_addr or "")
              .split(",")[0]).strip() or "unknown"
        if not limiter.is_allowed(ip):
            _inc("rate_limited_total")
            return jsonify({"error": "rate_limit_exceeded",
                            "message_fa": "تعداد درخواست‌ها زیاد است."}), 429

        if "image" not in request.files:
            return jsonify({"error": "no_file", "message_fa": "فایلی ارسال نشده."}), 400
        f = request.files["image"]
        if not f.filename:
            return jsonify({"error": "empty_filename"}), 400
        data = f.read()
        if not data:
            return jsonify({"error": "empty_file"}), 400

        digest = hashlib.sha256(data).hexdigest()
        etag = f'"{digest[:16]}"'

        if request.headers.get("If-None-Match") == etag:
            return Response(status=304, headers={"ETag": etag})

        cached = cache.get(digest)
        if cached is not None:
            _inc("cache_hits")
            resp = jsonify({**cached, "cached": True})
            resp.headers["ETag"] = etag
            return resp, 200

        _inc("cache_misses")
        result = analyzer.analyze(data, min_side=cfg.min_image_side)
        if "error" in result:
            _inc("errors_total")
            return jsonify(result), 400

        _inc("analyses_total")
        payload = {
            **result,
            "hairstyles": HairstyleRegistry.for_shape(result["shape"]),
            "cached": False,
        }
        cache.put(digest, payload)
        resp = jsonify(payload)
        resp.headers["ETag"] = etag
        return resp, 200

    @app.post(f"/api/{cfg.api_version}/analyze")
    def analyze_v1():
        return _do_analyze()

    @app.post("/analyze")
    def analyze_legacy():
        return _do_analyze()

    logger.info(
        "App ready | engine=%s | version=%s | styles=%d | shapes=%d | mc=%d | "
        "debug=%s | cors=%s | rate=%d/min | cache=%d | ttl=%ds",
        cfg.engine_version, cfg.api_version, len(HAIRSTYLES), len(SHAPES),
        cfg.n_mc_samples, cfg.debug, cfg.enable_cors,
        cfg.rate_limit_per_minute, cfg.cache_size, cfg.cache_ttl_seconds,
    )
    return app

app = create_app()

if __name__ == "__main__":
    _cfg = Config()
    port = int(os.getenv("PORT", _cfg.port))
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
