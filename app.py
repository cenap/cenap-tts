"""FastAPI sunucusu: arayüz, model durumu ve WebSocket ile ses akışı.

Sunucu yalnızca 127.0.0.1'de dinler. EMA nesnesi lifespan içinde tek kez
oluşturulur; her okuma ayrı bir üretim thread'inde çalışır.
"""
from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import queue
import tempfile
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
from pydantic import BaseModel

import tts_engine as engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("app")

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"


class Reading:
    """Tek bir okuma işi: üretici thread + kuyruk + iptal bayrağı."""

    def __init__(self, tts, text: str, speed: float, seed: int | None):
        self.cancel = threading.Event()
        self.finished = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=32)  # GPU önden fazla çalışmasın
        self.parts = engine.split_text(text)
        self.speed = speed
        self.seed = seed if seed is not None else engine.random_seed()
        self.sample_rate = engine.SAMPLE_RATE
        self.thread = threading.Thread(target=self._run, args=(tts,), daemon=True)
        self.thread.start()

    def _run(self, tts) -> None:
        try:
            for index, chunk in engine.stream_parts(
                tts, self.parts, self.speed, self.seed, self.sample_rate, self.cancel
            ):
                self._put(("audio", index, chunk))
            if not self.cancel.is_set():
                self._put(("done",))
        except Exception as error:  # üretim hatasını istemciye bildir, sunucu yaşasın
            log.exception("üretim hatası")
            self._put(("error", engine.describe_error(error)))
        finally:
            self.finished.set()

    def _put(self, item) -> None:
        # iptal edilirse bloklamadan çık
        while not self.cancel.is_set():
            try:
                self.q.put(item, timeout=0.2)
                return
            except queue.Full:
                continue


async def _pump(ws: WebSocket, reading: Reading) -> None:
    """Kuyruktaki chunk'ları WebSocket'e aktarır (JSON konfig + ikili ses)."""
    last_part = None
    try:
        while True:
            try:
                item = await asyncio.to_thread(reading.q.get, True, 0.5)
            except queue.Empty:
                if reading.finished.is_set() and reading.q.empty():
                    if not reading.cancel.is_set():
                        await ws.send_json({"type": "done"})
                        await ws.close(code=1000)
                    return
                continue
            kind = item[0]
            if kind == "audio":
                index, chunk = item[1], item[2]
                if index != last_part:  # parça başlangıcı: vurgu için bildir
                    last_part = index
                    await ws.send_json({"type": "part", "index": index})
                await ws.send_bytes(engine.to_bytes(chunk))
            elif kind == "done":
                await ws.send_json({"type": "done"})
                await ws.close(code=1000)
                return
            elif kind == "error":
                await ws.send_json({"type": "error", "message": item[1]})
                await ws.close(code=1000)
                return
    except (WebSocketDisconnect, RuntimeError):
        return  # istemci gitti; kapanış handler'da ele alınır
    except Exception:
        log.exception("akış aktarımında hata")


async def _close(ws: WebSocket, code: int = 1000) -> None:
    with contextlib.suppress(Exception):
        await ws.close(code=code)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Başlangıçta modeli bir kez yükle; indirme sürerken sunucu ayakta kalsın."""
    status = {
        "state": "loading",
        "label": "Model yükleniyor…",
        "mode": None,
        "started": time.monotonic(),
        "detail": "İlk çalıştırmada model ağırlıkları (~34 MB) Hugging Face'ten indirilir.",
    }
    app.state.status = status
    app.state.tts = None
    app.state.active = None  # eşzamanlı tek okuma

    def _load() -> None:
        try:
            tts, mode, label, detail = engine.load_engine()
            app.state.tts = tts
            status.update(state="ready", label=label, mode=mode, detail=detail)
            log.info("model hazır: %s", label)
            if mode == "gpu" and engine.LIGHTNING_ENABLED:
                # hızlı mod 60 sn bütçeyle ARKA PLANDA kurulur; okuma beklemez
                engine.start_lightning(tts, status)
        except Exception as error:
            log.exception("model yüklenemedi")
            status.update(state="error", label="Model yüklenemedi",
                          detail=engine.describe_error(error))

    threading.Thread(target=_load, daemon=True, name="ema-load").start()
    yield
    if app.state.active is not None:
        app.state.active.cancel.set()


app = FastAPI(title="Türkçe Metin Okuma", lifespan=lifespan)


@app.get("/api/status")
def status():
    """Arayüzün model durumunu sorguladığı uç."""
    data = dict(app.state.status)
    if data.get("state") == "loading":
        elapsed = int(time.monotonic() - data.get("started", time.monotonic()))
        # uzun yüklemelerde (indirme) kullanıcıya süre göster; 60 sn'yi
        # aşarsa açıkça bildir — bekleme asla sessizce uzamaz
        data["label"] = f"Model yükleniyor… ({elapsed} sn)"
        if elapsed > engine.LIGHTNING_BUDGET_S:
            data["detail"] = (f"{engine.LIGHTNING_BUDGET_S} sn'yi aştı; indirme/devam ediyor. "
                              "Sunucu ayakta, durum sayfayı yenileyerek izleyin.")
    return data


class WavRequest(BaseModel):
    text: str
    speed: float = 1.0
    seed: int | None = None


@app.post("/api/wav")
def wav(req: WavRequest):
    """Aynı metni `say()` ile dosyaya üretir (opsiyonel: WAV indirme)."""
    if app.state.status["state"] != "ready":
        return JSONResponse({"error": "Model henüz hazır değil"}, status_code=503)
    text = req.text.strip()
    if not text:
        return JSONResponse({"error": "Metin boş"}, status_code=400)
    try:
        if not 0.25 <= req.speed <= 4:
            raise ValueError("speed 0.25 ile 4 arasında olmalı")
        seed = engine.resolve_seed(req.seed)
    except (ValueError, TypeError) as error:
        return JSONResponse({"error": f"Geçersiz ayar: {error}"}, status_code=400)

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    try:
        app.state.tts.say(text, speed=req.speed, seed=seed,
                          sample_rate=engine.SAMPLE_RATE, path=path)
    except Exception as error:
        with contextlib.suppress(OSError):
            os.unlink(path)
        return JSONResponse({"error": engine.describe_error(error)}, status_code=500)
    return FileResponse(path, media_type="audio/wav", filename="okuma.wav",
                        background=BackgroundTask(_remove, path))


def _remove(path: str) -> None:
    with contextlib.suppress(OSError):
        os.unlink(path)


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    """Okuma oturuumu: start mesajı alır, sesi ikili chunk'larla yollar."""
    await ws.accept()
    try:
        start = await ws.receive_json()
    except (WebSocketDisconnect, RuntimeError, ValueError):
        return
    if not isinstance(start, dict) or start.get("type") != "start":
        await _close(ws, code=1008)
        return

    state = app.state.status["state"]
    if state != "ready":
        label = app.state.status["label"]
        detail = app.state.status.get("detail") or ""
        await ws.send_json({"type": "error",
                            "message": f"Model henüz hazır değil: {label} {detail}".strip()})
        await _close(ws)
        return

    text = str(start.get("text") or "")
    if not text.strip():
        await ws.send_json({"type": "error", "message": "Metin boş; okunacak bir metin girin."})
        await _close(ws)
        return
    try:
        speed = float(start.get("speed", 1.0))
        if not 0.25 <= speed <= 4:
            raise ValueError("hız 0.25 ile 4 arasında olmalı")
        seed = engine.resolve_seed(start.get("seed"))
    except (ValueError, TypeError) as error:
        await ws.send_json({"type": "error", "message": f"Geçersiz ayar: {error}"})
        await _close(ws)
        return

    # eşzamanlı tek okuma: yenisi gelirse öncekini iptal et
    previous = app.state.active
    if previous is not None:
        log.info("yeni okuma: önceki iptal ediliyor")
        previous.cancel.set()

    reading = Reading(app.state.tts, text, speed, seed)
    app.state.active = reading
    log.info("okuma başladı: %d parça, hız=%s, seed=%s", len(reading.parts), speed, reading.seed)

    pump = asyncio.create_task(_pump(ws, reading))
    try:
        await ws.send_json({
            "type": "config",
            "sample_rate": reading.sample_rate,
            "speed": reading.speed,
            "seed": reading.seed,
            "parts": reading.parts,
        })
        while True:
            msg = await ws.receive_json()  # istemci "stop" gönderir veya bağlantı kopar
            if isinstance(msg, dict) and msg.get("type") == "stop":
                log.info("istemci durdurdu; üretim iptal ediliyor")
                reading.cancel.set()
                with contextlib.suppress(Exception):
                    await ws.send_json({"type": "stopped"})
                break
    except (WebSocketDisconnect, RuntimeError, ValueError):
        log.info("istemci bağlantısı kapandı; üretim iptal ediliyor")
        reading.cancel.set()
    finally:
        reading.cancel.set()
        pump.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await pump
        if app.state.active is reading:
            app.state.active = None
        log.info("okuma oturumu kapandı (bitti mi: %s)", reading.finished.is_set())


# statik arayüz en sonda mount edilir ki API/WS yolları öncelikli olsun
app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
