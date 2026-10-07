"""EMA modelini yükleme, donanım algılama, metin parçalama ve stream sarmalayıcı.

Not: ema_lightning API'si (stream/say/lightning imzaları, chunk dtype/şekli)
kurulum sonrası kaynak koddan doğrulanmıştır.
"""
from __future__ import annotations

import contextlib
import logging
import os
import random
import re
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("tts")

SAMPLE_RATE = 48000  # model kartının varsayılanı (48000/24000/16000/8000)
MIN_PART = 70  # bundan kısa parçalar bir sonrakiyle birleştirilir
MAX_PART = 300  # bundan uzun parçalar virgül/noktalı virgül/kelime sınırından bölünür
PAUSE_SECONDS = 0.2  # parçalar arası doğal duraklama (200 ms)
LIGHTNING_BUDGET_S = 60  # hızlı mod için azami bekleme (saniye); aşılırsa hazır modda devam

# CUDA graph ("hızlı mod") kurulumu bu uygulamada her açılışta ~2 dakika sürüyor
# (40 graph: torch.compile + cuDNN autotune + doğrulama) ve tek akışlı okumada
# düz GPU yolu zaten hızlı (ilk ses ~1 sn, sonrakiler ~80 ms). Bu yüzden
# varsayılan olarak KAPALIDIR; EMA_LIGHTNING=1 ile açılır.
LIGHTNING_ENABLED = os.environ.get("EMA_LIGHTNING", "").strip().lower() in ("1", "true", "yes", "on")
# torch.compile önbelleğini /tmp'te bırakma: yeniden başlatmada/temizlikte
# yeniden derleme (dakikalarca) yaşanmasın diye kalıcı dizin kullan.
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(Path.home() / ".cache" / "torchinductor"))


# ---------------------------------------------------------------- yükleme

def load_engine():
    """EMA nesnesini uygulama ömrü için bir kez kurar (yalnızca model yükleme).

    Dönen değer: (tts, mod, arayüz_etiketi, ayrıntı_metası).
    Mod yalnızca `cpu` veya `gpu` olur: `.lightning()` yalnızca
    LIGHTNING_ENABLED ise `start_lightning()` ile arka planda kurulur,
    bitince durum `gpu_fast`'e yükselir. Hata uygulamayı düşürmez.
    """
    import torch
    from ema_lightning import EMA

    log.info("Model ağırlıkları hazırlanıyor (ilk çalıştırmada ema.pt/decoder.pt iner)...")
    tts = EMA()  # ağırlıkları Hugging Face'ten indirir, GPU varsa onu kullanır

    if not torch.cuda.is_available():
        log.info("GPU bulunamadı; CPU modunda çalışılıyor")
        return tts, "cpu", "Hazır (CPU)", None

    if not LIGHTNING_ENABLED:
        log.info("Hızlı mod kapalı; düz GPU yolu kullanılıyor (açmak için EMA_LIGHTNING=1)")
        return tts, "gpu", "Hazır (GPU)", None

    log.info("GPU bulundu; hızlı mod arka planda hazırlanacak (bütçe %s sn)", LIGHTNING_BUDGET_S)
    return tts, "gpu", "Hazır (GPU)", "Hızlı mod hazırlanıyor…"


def start_lightning(tts, status) -> None:
    """`.lightning(batch_size=8)` işini ARKA PLANDA kurar, 60 sn bütçe uygular.

    Bütçe aşılırsa okuma hazır modda başlar; iş bitince (başarı ya da hata)
    `status` sözlüğü güncellenir. Hata hiçbir koşulda uygulamayı düşürmez,
    yalnızca log'a ve duruma yazılır.
    """
    finished = threading.Event()

    def _update(**fields):
        with contextlib.suppress(Exception):
            status.update(fields)  # istemci /api/status ile okur

    def _work():
        try:
            # Ölçüldü (RTX 3060 12 GB): batch=8 → 40 graph, ~127 sn; batch=1 → 10 graph,
            # ~48 sn. Süre torch.compile + cuDNN autotune + graph kaydı + doğrulamadan gelir.
            # Bu yüzden hızlı mod varsayılan olarak kapalıdır (EMA_LIGHTNING=1 ile açılır).
            tts.lightning(batch_size=8)
        except Exception as error:  # düz GPU moduyla devam, uygulama yaşasın
            log.exception("lightning modu kurulamadı; düz GPU moduyla devam ediliyor")
            import torch
            with contextlib.suppress(Exception):
                torch.cuda.empty_cache()  # yarım kalmış graph belleğini bırak
            detail = describe_error(error)
            if detail.startswith("Üretim sırasında hata: "):
                detail = detail.removeprefix("Üretim sırasında hata: ")
            _update(label="Hazır (GPU)", mode="gpu", detail=f"Hızlı mod kurulamadı: {detail}")
        else:
            log.info("lightning (hızlı) mod aktif")
            _update(label="Hazır (GPU, hızlı mod)", mode="gpu_fast", detail=None)
        finally:
            finished.set()
            with contextlib.suppress(Exception):
                timer.cancel()  # iş bittiyse bütçe zamanlayıcısını bekleme

    def _deadline():
        if not finished.is_set():
            _update(label="Hazır (GPU)", mode="gpu",
                    detail=f"Hızlı mod {LIGHTNING_BUDGET_S} sn'de tamamlanmadı; okuma hazır "
                           "modda çalışıyor (arka planda tamamlanabilir).")

    timer = threading.Timer(LIGHTNING_BUDGET_S, _deadline)
    timer.daemon = True
    threading.Thread(target=_work, daemon=True, name="ema-lightning").start()
    timer.start()


def describe_error(error: BaseException) -> str:
    """Üretim/yükleme hatasını kullanıcıya anlaşılır Türkçe metne çevirir."""
    name = type(error).__name__
    if "OutOfMemory" in name or "out of memory" in str(error).lower():
        return ("GPU bellek yetersiz (CUDA out of memory); CPU'ya düşülmedi. "
                "Metni kısaltıp tekrar deneyin; başka GPU süreçleri belleği "
                "dolduruyor olabilir.")
    if isinstance(error, ValueError):
        return f"Geçersiz ayar: {error}"
    if isinstance(error, FileNotFoundError):
        return f"Model dosyası bulunamadı: {error}"
    return f"Üretim sırasında hata: {error or name}"


# ---------------------------------------------------------------- parçalama

_SENTENCE = re.compile(r"(?<=[.!?…])\s+")
_BREAKERS = re.compile(r"(?<=[,;])\s+")


def split_text(text: str) -> list[str]:
    """Metni cümle/paragraf sınırlarından makul uzunlukta parçalara böler.

    Çok kısa cümleleri birleştirir, çok uzun cümleleri virgül/noktalı
    virgül ve kelime sınırlarından böler.
    """
    parts: list[str] = []
    for para in re.split(r"\n\s*\n", text.strip()):
        para = " ".join(para.split())
        if not para:
            continue
        sentences = [s.strip() for s in _SENTENCE.split(para) if s.strip()]
        # her paragraf kendi içinde toplanır: paragraf sınırı korunur
        group: list[str] = []
        for sentence in sentences:
            for piece in _split_long(sentence):
                # kısa parçaları birleştir, ama MAX_PART'i aşma
                mergeable = len(piece) < MIN_PART or (group and len(group[-1]) < MIN_PART)
                if group and mergeable and len(group[-1]) + 1 + len(piece) <= MAX_PART:
                    group[-1] = f"{group[-1]} {piece}"
                else:
                    group.append(piece)
        parts.extend(group)
    return parts


def _split_long(sentence: str) -> list[str]:
    """Uzun bir cümleyi önce virgül/noktalı virgülde, olmazsa kelimede böler."""
    if len(sentence) <= MAX_PART:
        return [sentence]

    rough: list[str] = []
    buf = ""
    for seg in _BREAKERS.split(sentence):
        if buf and len(buf) + 1 + len(seg) > MAX_PART:
            rough.append(buf)
            buf = seg
        else:
            buf = f"{buf} {seg}" if buf else seg
    if buf:
        rough.append(buf)

    out: list[str] = []
    for part in rough:
        if len(part) <= MAX_PART:
            out.append(part)
            continue
        cur = ""
        for word in part.split():
            while len(word) > MAX_PART:  # limiti aşan tek kelime: sert böl
                if cur:
                    out.append(cur)
                    cur = ""
                out.append(word[:MAX_PART])
                word = word[MAX_PART:]
            if not word:
                continue
            if cur and len(cur) + 1 + len(word) > MAX_PART:
                out.append(cur)
                cur = word
            else:
                cur = f"{cur} {word}" if cur else word
        if cur:
            out.append(cur)
    return out


def resolve_seed(seed) -> int | None:
    """İstemciden gelen seed alanını doğrular; boşsa rastgele seçer."""
    if seed is None or seed == "":
        return None
    value = int(seed)
    if value < 0:
        raise ValueError("seed negatif olamaz")
    return value


# ---------------------------------------------------------------- üretici

def pause_chunk(sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Parçalar arası kısa sessizlik (~200 ms)."""
    return np.zeros(int(PAUSE_SECONDS * sample_rate), dtype=np.float32)


def stream_parts(tts, parts, speed, seed, sample_rate, cancel):
    """Parçaları sırayla stream() ile üretir: (parça_indeksi, float32 chunk).

    Üretici thread'inde çağrılır. cancel set edilirse döngüden çıkılır ve
    generator.close() ile kalan iş deterministik olarak iptal edilir.
    Parçalar arasında ~200 ms sessizlik ekler.
    """
    for index, part in enumerate(parts):
        if cancel.is_set():
            return
        gen = tts.stream(part, speed=speed, seed=seed, sample_rate=sample_rate)
        try:
            for chunk in gen:
                if cancel.is_set():
                    return
                yield index, chunk
        finally:
            gen.close()  # erken çıkınca o metnin kalan işi iptal olur
        if index < len(parts) - 1 and not cancel.is_set():
            yield index, pause_chunk(sample_rate)


def to_bytes(chunk: np.ndarray) -> bytes:
    """Chunk'ı WebSocket için float32 mono ham baytlara çevirir."""
    return np.ascontiguousarray(chunk, dtype=np.float32).tobytes()


def random_seed() -> int:
    """Seed boş bırakıldığında kullanılacak rastgele tohum."""
    return random.SystemRandom().randrange(2**31)
