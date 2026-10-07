# Türkçe Metin Okuma (TTS)

Yerel çalışan, tarayıcı arayüzlü Türkçe metin okuma uygulaması.
Model: [canberkkkkkk/ema-lightning](https://huggingface.co/canberkkkkkk/ema-lightning)
(Apache 2.0, ~8.6M parametre, ~34 MB, yalnızca Türkçe).

Metni yapıştırıp **Oku**'ya basarsınız; ses üretilir üretilmez **hemen çalmaya
başlar** (WebSocket + Web Audio ile boşluksuz akış). İstediğiniz an
**Durdur**'a basabilirsiniz.

## Kurulum

Python 3.10+ gerekir:

```bash
pip install -r requirements.txt
```

### CUDA'lı PyTorch notu

`ema-lightning` PyTorch'a bağlıdır. GPU kullanmak istiyorsanız **CUDA'lı
PyTorch** kurulmuş olmalıdır (varsayılan `pip install torch` Linux'ta genelde
CUDA'lı wheel getirir). CUDA sürücünüzün kurulu olduğundan emin olun:

```bash
python -c "import torch; print(torch.cuda.is_available())"  # True olmalı
```

CPU'da da çalışır; yalnızca daha yavaştır.

## Çalıştırma

```bash
python app.py
```

Sunucu **yalnızca** `127.0.0.1`'de dinler. Tarayıcıda açın:

```
http://127.0.0.1:8000
```

## İlk çalıştırma: model indirme

Model ağırlıkları (`ema.pt`, `decoder.pt`) **ilk çalıştırmada** Hugging
Face'ten otomatik indirilir (~34 MB). Sonraki açılışlarda model ~5 saniyede
hazır olur ve **Oku** hemen kullanılabilir.

### CUDA graph "hızlı mod" (isteğe bağlı)

`.lightning()` (torch.compile + CUDA graph kaydı) her açılışta **~2 dakika**
sürer (40 graph: torch.compile + cuDNN autotune + graph kaydı + doğrulama;
RTX 3060 12 GB'da ölçüldü). Tek akışlı okumada düz GPU yolu zaten hızlı
olduğundan (ilk ses ~1 sn, sonrakiler ~0.1 sn) bu mod **varsayılan olarak
kapalıdır**. Açmak için:

```bash
EMA_LIGHTNING=1 python app.py
```

`torch.compile` önbelleği `~/.cache/torchinductor` altında tutulur (eskiden
`/tmp` idi; `/tmp` temizlenince yeniden derleme dakikalar sürüyordu).

Durum değişimi:

| Durum | Anlam |
|---|---|
| `Hazır (GPU)` | NVIDIA GPU; düz GPU yolu (varsayılan) |
| `Hazır (GPU, hızlı mod)` | `EMA_LIGHTNING=1` ile hızlı mod kuruldu |
| `Hazır (CPU)` | GPU yok; CPU yolu |

Hızlı mod açıkken kurulum başarısız olursa uygulama **çökmez**, log'a
yazılır, düz GPU yoluyla devam eder ve arayüzde uyarı görünür. CUDA bellek
hatasında CPU'ya düşülmez, hata açıkça bildirilir.

**Not (batch boyutu):** Hızlı mod açıkken `lightning(batch_size=8)` çağrılır
(40 graph). Kütüphanenin önbelleğe aldığı batch=128, 170 graph → 12 GB
bellek ve 30+ dakika (sonunda CUDA OOM) demektir; bu yüzden kullanılmaz.


## Kullanım

- **Oku**: okumayı başlatır (üretim başlar başlamaz ses çalar).
- **Durdur**: üretimi ve çalmayı tamamen keser; sonra yeniden okuyabilirsiniz.
- **Duraklat / Devam**: sesi askıya alır/Devam ettirir (`AudioContext`).
- **Gelişmiş ayarlar**: hız (0.25–4, adım 0.05) ve isteğe bağlı seed
  (boş bırakılırsa rastgele seçilir, seçilen değer arayüzde gösterilir).
- **WAV olarak indir**: aynı metni `say()` ile dosya olarak üretir.
- Uzun metinler cümle/paragraf sınırlarından parçalara bölünür; parçalar
  arası ~200 ms doğal duraklama eklenir ve parçalar arasında boşluk olmaz.
- Okuma sırasında o an okunan parça metinde vurgulanır.
- `.txt` dosyasını metin kutusuna sürükleyip bırakabilirsiniz.

İlk ses chunk'ı geliş süreleri (ölçüldü): düz GPU'da ~1 sn (sonraki
okumalarda ~0.1 sn), `EMA_LIGHTNING=1` hızlı modda ~0.2 sn, CPU'da ~18 sn.

Not: **Bu ses yapay zekâ ile üretilmiştir.** (Model kartının istediği etiket
arayüzde gösterilir.)

## Proje yapısı

```
app.py            FastAPI + WebSocket + statik sunum
tts_engine.py     model yükleme, donanım algılama, metin parçalama, stream sarmalayıcı
static/index.html arayüz
static/app.js     Web Audio gapless çalma, WebSocket istemcisi
static/style.css  açık/koyu tema, mobil düzen
requirements.txt
```

## Kullanılan bileşenler

| Bileşen | Kullanım |
|---|---|
| Python 3.10+ | Sunucu ve metin/ses üretim akışı |
| FastAPI | Durum ve WAV HTTP uçları; WebSocket oturumu |
| Uvicorn | Yerel ASGI sunucusu |
| `ema-lightning` | Türkçe konuşma sentezi modeli ve akışı |
| PyTorch | Model çalıştırma; CUDA varsa NVIDIA GPU desteği |
| NumPy | PCM ses parçalarının işlenmesi ve WebSocket'e aktarılması |
| HTML, CSS, JavaScript | Tarayıcı arayüzü; ek bir frontend derleme adımı yoktur |
| Web Audio API | Gelen ses parçalarını tarayıcıda boşluksuz çalma |

Doğrudan Python bağımlılıkları `requirements.txt` içinde listelenir. Modelin
PyTorch ve diğer çalışma zamanı bağımlılıkları paket yöneticisi tarafından
kurulur.

## HTTP API

- `GET /api/status`: Modelin yüklenme/hazır olma durumunu, çalışma modunu ve
  varsa açıklama mesajını JSON olarak döndürür.
- `POST /api/wav`: WAV dosyası üretir. JSON gövdesi `text` (zorunlu), `speed`
  (0.25–4, varsayılan 1) ve `seed` (isteğe bağlı, negatif olmayan tamsayı)
  alanlarını kabul eder. Model hazır değilse `503`, boş metin veya geçersiz
  ayarlarda `400` döner.
- `WS /ws`: Akışlı okuma ve iptal için WebSocket bağlantısı açar.

## WebSocket protokolü

1. İstemci, `start` türünde bir JSON mesajı gönderir: `text` zorunlu;
   `speed` (0.25–4) ve `seed` isteğe bağlıdır.
2. Sunucu `config` mesajında `sample_rate` (48.000 Hz), seçilen `seed` ve
   metin parçalarını (`parts`) bildirir.
3. Her metin parçası başlamadan önce `part` mesajı (`index`) gelir. Ses,
   48 kHz mono float32 PCM içeren ikili WebSocket mesajlarıyla aktarılır.
4. Akış `done` mesajıyla tamamlanır; hatalar `error` mesajıyla bildirilir.
5. İstemci `stop` göndererek okumayı kesebilir. Bağlantı kopması da üretimi
   iptal eder. Aynı anda yalnızca bir okuma yürütülür; yeni okuma öncekinin
   yerini alır.

## Gizlilik ve güvenlik

- Sunucu varsayılan olarak yalnızca `127.0.0.1:8000` üzerinde dinler; ağdan
  erişime açık bir servis olarak yapılandırılmamıştır.
- Metin, bu yerel uygulama içinde işlenir. Model ağırlıkları ilk çalıştırmada
  Hugging Face'ten indirilir; bu indirme dışında uygulama metni harici bir
  sentez servisine göndermez.
- Sunucuyu ağ üzerinden erişilebilir hâle getirmek kimlik doğrulama ve
  güvenli dağıtım yapılandırması gerektirir. Mevcut uygulamayı doğrudan
  internete açmayın.
- Üretilen ses yapay zekâ çıktısıdır; arayüzde bu bilgiyi belirten not
  gösterilir.

## Geliştirme

Sanal ortamla yerel kurulum:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Windows PowerShell'de etkinleştirme komutu:

```powershell
.venv\Scripts\Activate.ps1
```

Depoda şu anda otomatik test paketi bulunmuyor. Python dosyaları için temel
sözdizimi kontrolü şu komutla yapılabilir:

```bash
python -m compileall app.py tts_engine.py
```

Davranış değişikliklerinde arayüzden akışlı okuma, durdurma ve WAV indirme
işlemlerini de kontrol edin; GPU'ya özel değişiklikler için CUDA ortamında
doğrulama yapın.

## Katkıda bulunma

Hata bildirimlerinde yeniden üretme adımlarını, beklenen ve gerçekleşen
sonucu, işletim sistemi/Python sürümünü ve ilgili GPU/CUDA bilgisini paylaşın.
Kullanıcı metni veya kişisel bilgi içerebilecek kayıtları göndermeden önce
temizleyin.

Değişiklik katkısı için:

1. Büyük değişiklikleri uygulamadan önce bir issue açıp yaklaşımı tartışın.
2. Değişikliği odaklı tutun ve kullanıcıya görünen davranışı açıklayın.
3. İlgili kontrolleri çalıştırıp sonuçlarını pull request açıklamasında
   belirtin. Yeni davranış için test eklenebiliyorsa testi de katkıya ekleyin.

## Lisanslar

Bu depoda henüz bir proje lisans dosyası (`LICENSE`) bulunmadığından, proje
kaynak kodunun lisansı belirtilmemiştir. Model kartında belirtilen
[`canberkkkkkk/ema-lightning` lisansı](https://huggingface.co/canberkkkkkk/ema-lightning)
Apache-2.0'dır; bu, bu depodaki kodun lisansını belirlemez. Yeniden kullanım
ve dağıtım koşullarını netleştirmek için proje sahibinin ayrıca bir lisans
seçip depoya eklemesi gerekir.

Aynı anda yalnızca tek okuma yapılır; yeni "Oku" gelirse önceki iptal olur.
