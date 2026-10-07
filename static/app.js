"use strict";

// ------------------------------------------------------------ yardımcılar
const el = (id) => document.getElementById(id);
const ui = {
  text: el("text"),
  count: el("count"),
  seedInfo: el("seedInfo"),
  read: el("readBtn"),
  stop: el("stopBtn"),
  pause: el("pauseBtn"),
  download: el("downloadBtn"),
  status: el("status"),
  statusText: el("statusText"),
  notice: el("notice"),
  reading: el("readingView"),
  speed: el("speed"),
  speedVal: el("speedVal"),
  seed: el("seed"),
};

// uygulama durumu
const st = {
  ready: false,       // model hazır mı
  reading: false,     // okuma oturumu açık mı
  paused: false,
  finishing: false,   // üretim bitti, kalan ses çalınıyor mu
  userStopped: false,
  ws: null,
  ctx: null,          // AudioContext
  sources: new Set(), // planlanmış AudioBufferSourceNode'lar
  nextStart: 0,       // sürekli ilerleyen gapless zaman çizelgesi
  sampleRate: 48000,
  parts: [],          // {text, span, startTime}
  pendingPart: null,  // sıradaki sesin ait olduğu parça
  currentPart: -1,
  ticker: null,
};

function setStatus(kind, text, detail) {
  ui.status.className = "status " + kind;
  ui.statusText.textContent = detail ? text + " — " + detail : text;
}

function showNotice(message, kind) {
  if (!message) {
    ui.notice.hidden = true;
    ui.notice.textContent = "";
    return;
  }
  ui.notice.hidden = false;
  ui.notice.textContent = message;
  ui.notice.className = "notice" + (kind === true ? " error" : kind === "ok" ? " ok" : "");
}

function updateButtons() {
  ui.read.disabled = !st.ready || st.reading;
  ui.stop.disabled = !st.reading;
  ui.pause.disabled = !st.reading;
  ui.pause.textContent = st.paused ? "Devam" : "Duraklat";
  ui.download.disabled = !st.ready || st.reading;
}

// ------------------------------------------------------------ model durumu
function shouldKeepPolling(data) {
  // yükleniyor → bitene kadar izle; hazırsa hızlı mod kurulumu (arka plan)
  // bitene kadar izle; hata/kurulamadı durumunda dur
  if (data.state === "loading") return true;
  if (data.state !== "ready") return false;
  if (data.mode !== "gpu") return false; // cpu veya gpu_fast: geçiş yok
  return !(data.detail || "").includes("Hızlı mod kurulamadı");
}

async function pollStatus() {
  let data = null;
  try {
    const res = await fetch("/api/status", { cache: "no-store" });
    data = await res.json();
  } catch (e) {
    // sunucuya ulaşılamadı, tekrar dene
  }
  if (!data) {
    setTimeout(pollStatus, 1500);
    return;
  }
  st.ready = data.state === "ready";
  if (data.state === "loading") {
    setStatus("loading", data.label || "Model yükleniyor…", data.detail);
  } else if (data.state === "ready") {
    setStatus("ready", data.label, data.detail);
  } else {
    setStatus("error", data.label, data.detail);
    showNotice("Model yüklenemedi: " + (data.detail || "bilinmeyen hata"), true);
  }
  updateButtons();
  if (shouldKeepPolling(data)) {
    // ör. hızlı mod arka planda kuruluyor: durumu izlemeye devam et
    setTimeout(pollStatus, data.state === "loading" ? 700 : 2000);
  }
}

// ------------------------------------------------------------ ses çalma
async function ensureContext() {
  const Ctx = window.AudioContext || window.webkitAudioContext;
  if (!st.ctx) st.ctx = new Ctx({ sampleRate: st.sampleRate });
  if (st.ctx.state === "suspended") await st.ctx.resume();
}

function clearAudio() {
  for (const src of st.sources) {
    try {
      src.onended = null;
      src.stop();
    } catch (e) {
      // zaten durmuş olabilir
    }
  }
  st.sources.clear();
  st.nextStart = 0;
}

function playChunk(arrayBuffer) {
  const data = new Float32Array(arrayBuffer);
  if (!data.length || !st.ctx) return;
  const buffer = st.ctx.createBuffer(1, data.length, st.sampleRate);
  buffer.copyToChannel(data, 0);

  const src = st.ctx.createBufferSource();
  src.buffer = buffer;
  src.connect(st.ctx.destination);

  const now = st.ctx.currentTime;
  // boşluksuz planlama: her zaman bir sonraki başlangıç anında kaldığımız yerden
  if (st.nextStart < now) st.nextStart = now + 0.08; // sunucu geride kaldıysa küçük tampon
  if (st.pendingPart !== null && st.parts[st.pendingPart]) {
    st.parts[st.pendingPart].startTime = st.nextStart;
    st.pendingPart = null;
  }
  src.start(st.nextStart);
  st.nextStart += buffer.duration;
  st.sources.add(src);
  src.onended = () => {
    st.sources.delete(src);
    if (st.finishing && st.sources.size === 0) finishReading();
  };
}

// ------------------------------------------------------------ vurgu
function renderParts(parts) {
  ui.reading.innerHTML = "";
  st.parts = parts.map((text) => {
    const span = document.createElement("span");
    span.className = "part";
    span.textContent = text + " ";
    ui.reading.appendChild(span);
    return { text, span, startTime: null };
  });
  ui.reading.hidden = false;
  ui.reading.scrollTop = 0;
  st.currentPart = -1;
}

function setCurrentPart(index) {
  if (st.currentPart >= 0 && st.parts[st.currentPart]) {
    st.parts[st.currentPart].span.classList.remove("current");
  }
  st.currentPart = index;
  const part = st.parts[index];
  if (!part) return;
  part.span.classList.add("current");
  part.span.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function startTicker() {
  stopTicker();
  st.ticker = setInterval(() => {
    if (!st.reading || !st.ctx) return;
    const time = st.ctx.currentTime;
    let current = -1;
    for (let i = 0; i < st.parts.length; i++) {
      const start = st.parts[i].startTime;
      if (start === null) break; // henüz planlanmadı
      if (start <= time) current = i;
    }
    if (current >= 0 && current !== st.currentPart) setCurrentPart(current);
  }, 200);
}

function stopTicker() {
  if (st.ticker) clearInterval(st.ticker);
  st.ticker = null;
}

// ------------------------------------------------------------ okuma akışı
function currentSettings() {
  const seedRaw = ui.seed.value.trim();
  const seed = seedRaw === "" ? null : Number.parseInt(seedRaw, 10);
  if (seed !== null && (!Number.isInteger(seed) || seed < 0)) {
    showNotice("Tohum 0 veya daha büyük bir tamsayı olmalı.", true);
    return null;
  }
  return { speed: Number(ui.speed.value), seed };
}

function closeWs(askStop) {
  const ws = st.ws;
  st.ws = null;
  if (!ws) return;
  try {
    if (askStop && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ type: "stop" }));
    }
  } catch (e) {
    // gönderilemediyse zaten kopuk
  }
  try {
    ws.close();
  } catch (e) {
    // yoksay
  }
}

function finishReading() {
  st.reading = false;
  st.finishing = false;
  st.paused = false;
  stopTicker();
  updateButtons();
}

function abortReading(message, isError) {
  clearAudio(); // planlı tüm sesi derhal durdur
  closeWs(true);
  st.reading = false;
  st.finishing = false;
  st.paused = false;
  st.pendingPart = null;
  stopTicker();
  updateButtons();
  if (message) showNotice(message, isError);
}

async function startReading() {
  const text = ui.text.value;
  if (!text.trim()) {
    showNotice("Okunacak metin boş.", true);
    ui.text.focus();
    return;
  }
  if (!st.ready) {
    showNotice("Model henüz hazır değil, lütfen bekleyin.", true);
    return;
  }
  const settings = currentSettings();
  if (!settings) return;
  if (st.reading) abortReading(null, false); // önceki okumayı kes

  try {
    await ensureContext();
  } catch (e) {
    showNotice("Ses başlatılamadı: " + e.message, true);
    return;
  }

  st.userStopped = false;
  st.finishing = false;
  st.paused = false;
  st.pendingPart = null;
  st.sampleRate = 48000;
  clearAudio();
  showNotice("");
  ui.seedInfo.textContent = "";
  st.reading = true;
  updateButtons();
  startTicker();

  const scheme = location.protocol === "https:" ? "wss://" : "ws://";
  const ws = new WebSocket(scheme + location.host + "/ws");
  ws.binaryType = "arraybuffer";
  st.ws = ws;

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: "start",
      text,
      speed: settings.speed,
      seed: settings.seed,
    }));
  };

  ws.onmessage = (ev) => {
    if (typeof ev.data === "string") handleServer(JSON.parse(ev.data));
    else playChunk(ev.data);
  };

  ws.onclose = () => {
    if (st.ws !== ws) return;
    st.ws = null;
    if (st.finishing) return; // üretim bitti, kalan ses çalıyor
    if (st.reading) {
      abortReading(
        st.userStopped ? "Okuma durduruldu." : "Sunucu bağlantısı kapandı.",
        !st.userStopped
      );
    }
  };

  ws.onerror = () => showNotice("Sunucu bağlantısında hata.", true);
}

function handleServer(msg) {
  switch (msg.type) {
    case "config":
      st.sampleRate = msg.sample_rate || 48000;
      renderParts(msg.parts || []);
      ui.seedInfo.textContent = "Tohum: " + msg.seed;
      showNotice("Okunuyor… (" + (msg.parts || []).length + " parça)");
      break;
    case "part":
      st.pendingPart = msg.index;
      break;
    case "done":
      st.finishing = true;
      showNotice("Okuma tamamlandı.", "ok");
      closeWs(false);
      if (st.sources.size === 0) finishReading();
      break;
    case "error":
      abortReading(msg.message, true);
      break;
    case "stopped":
      break;
    default:
      break;
  }
}

// ------------------------------------------------------------ kontroller
function stopReading() {
  if (!st.reading) return;
  st.userStopped = true;
  abortReading("Okuma durduruldu.", false);
}

async function togglePause() {
  if (!st.reading || !st.ctx) return;
  try {
    if (st.paused) {
      await st.ctx.resume();
      st.paused = false;
      showNotice("Devam ediyor.");
    } else {
      await st.ctx.suspend();
      st.paused = true;
      showNotice("Duraklatıldı.");
    }
  } catch (e) {
    showNotice("Duraklatma başarısız: " + e.message, true);
  }
  updateButtons();
}

async function downloadWav() {
  const text = ui.text.value;
  if (!text.trim()) {
    showNotice("Okunacak metin boş.", true);
    return;
  }
  const settings = currentSettings();
  if (!settings) return;
  ui.download.disabled = true;
  showNotice("WAV üretiliyor, lütfen bekleyin…");
  try {
    const res = await fetch("/api/wav", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, speed: settings.speed, seed: settings.seed }),
    });
    if (!res.ok) {
      let detail = "HTTP " + res.status;
      try {
        detail = (await res.json()).error || detail;
      } catch (e) {
        // gövde JSON değil
      }
      throw new Error(detail);
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "okuma.wav";
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 5000);
    showNotice("WAV dosyası indirildi.", "ok");
  } catch (e) {
    showNotice("WAV üretilemedi: " + e.message, true);
  } finally {
    updateButtons();
  }
}

// ------------------------------------------------------------ olaylar
ui.text.addEventListener("input", () => {
  ui.count.textContent = ui.text.value.length + " karakter";
});
ui.speed.addEventListener("input", () => {
  ui.speedVal.textContent = Number(ui.speed.value).toFixed(2) + "×";
});
ui.read.addEventListener("click", startReading);
ui.stop.addEventListener("click", stopReading);
ui.pause.addEventListener("click", togglePause);
ui.download.addEventListener("click", downloadWav);

// .txt dosyasını sürükle-bırak ile yükleme
ui.text.addEventListener("dragover", (e) => e.preventDefault());
ui.text.addEventListener("drop", (e) => {
  e.preventDefault();
  const file = e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files[0];
  if (!file) return;
  if (!/\.txt$/i.test(file.name)) {
    showNotice("Yalnızca .txt dosyaları desteklenir.", true);
    return;
  }
  file.text().then((content) => {
    ui.text.value = content;
    ui.text.dispatchEvent(new Event("input"));
    showNotice(file.name + " yüklendi.", "ok");
  });
});

// okuma sırasında sekme kapanırsa uyar
window.addEventListener("beforeunload", (e) => {
  if (st.reading && !st.finishing) {
    e.preventDefault();
    e.returnValue = "";
  }
});

pollStatus();
