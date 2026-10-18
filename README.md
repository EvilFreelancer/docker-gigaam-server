# GigaAM API Server in Docker

Распознавание русской речи на [GigaAM v3](https://github.com/salute-developers/GigaAM)
(Sber, MIT) за API, совместимым с OpenAI, так что любой клиент `openai` или обычный `curl`
работает без адаптеров. Два образа из одних исходников, для видеокарт NVIDIA (CUDA) и Intel (Arc A и
B, Arc Pro, через torch XPU).

- модель по умолчанию `v3_e2e_rnnt`, отдаётся как `gigaam-v3-e2e-rnnt`, текст сразу с
  пунктуацией, заглавными буквами и нормализованными числами;
- псевдонимы `gigaam`, `whisper-1`, `default` и голое имя чекпойнта `v3_e2e_rnnt` ведут на
  модель по умолчанию, поэтому клиенты, где зашит `whisper-1`, работают без правок;
- длинное аудио режется по паузам через silero-vad, без закрытых моделей pyannote и без
  токена HuggingFace.

## API

| Метод | Путь | Что делает |
| --- | --- | --- |
| `POST` | `/v1/audio/transcriptions` | multipart; `response_format` = `json` (по умолчанию), `text`, `srt`, `vtt`, `verbose_json`; `timestamp_granularities[]` = `segment`, `word`; `stream=true` отдаёт SSE |
| `POST` | `/v1/audio/translations` | отклоняется ошибкой 400 в формате OpenAI, GigaAM не переводит |
| `GET` | `/v1/models`, `/v1/models/{id}` | объекты моделей в формате OpenAI |
| `GET` | `/health` | `{"status":"ok","models":[...],"device":"cuda"}` |
| `GET` | `/docs` | Swagger UI от FastAPI |

Ошибки приходят в конверте OpenAI (`{"error": {"message", "type", "param", "code"}}`),
поэтому `NotFoundError` и `BadRequestError` в SDK ведут себя так же, как с api.openai.com.

## Запуск

Папка проекта самодостаточна, веса лежат рядом в `models/` и подключаются по относительному
пути, так что её можно целиком переносить с машины на машину.

NVIDIA.

```bash
cp docker-compose.dist.yml docker-compose.yml
cp .env.dist .env
docker compose pull      # опубликованный образ evilfreelancer/gigaam-server:latest-cuda
docker compose up -d     # или docker compose up -d --build, чтобы собрать самому
```

Intel (Arc A770, B580, Arc Pro B60 и другие).

```bash
cp docker-compose.intel.dist.yml docker-compose.yml
cp .env.dist .env
docker compose pull      # evilfreelancer/gigaam-server:latest-xpu
docker compose up -d
```

Хосту с Intel нужен только драйвер ядра (i915 или xe) и узлы `/dev/dri`, пользовательский
драйвер и torch уже в образе. Карта выбирается через `GPU_INDEX` в порядке Level Zero, он
совпадает с порядком PCI в `/dev/dri/by-path`.

При первом старте сервер скачивает чекпойнт с CDN Sber в `./models` (около 450 МБ на
модель), дальше он берётся оттуда. `docker compose logs -f gigaam` пишет
`models ready: [...]`, когда эндпойнт готов.

## Примеры

```bash
curl -s http://localhost:8770/v1/audio/transcriptions \
  -F file=@audio.mp3 \
  -F model=gigaam-v3-e2e-rnnt
# {"text":"...","usage":{"type":"duration","seconds":11}}
```

Проверка без своего файла, Sber публикует два русских примера.

```bash
curl -sSL https://cdn.chatwm.opensmodel.sberdevices.ru/GigaAM/example.wav \
| curl -sS http://localhost:8770/v1/audio/transcriptions \
    -F "file=@-;filename=example.wav;type=audio/wav" \
    -F model=gigaam-v3-e2e-rnnt
```

`long_example.wav` на том же CDN длится 71 секунду и проверяет нарезку по паузам.

Субтитры и временные метки слов.

```bash
curl -s http://localhost:8770/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=gigaam-v3-e2e-rnnt -F response_format=srt

curl -s http://localhost:8770/v1/audio/transcriptions \
  -F file=@audio.mp3 -F model=gigaam-v3-e2e-rnnt \
  -F response_format=verbose_json \
  -F 'timestamp_granularities[]=segment' \
  -F 'timestamp_granularities[]=word'
```

Официальный клиент на Python.

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8770/v1", api_key="not-needed")

with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="gigaam-v3-e2e-rnnt").text)

# потоковый режим: события transcript.text.delta по сегментам, затем transcript.text.done
with open("audio.mp3", "rb") as f:
    for event in client.audio.transcriptions.create(
        file=f, model="gigaam-v3-e2e-rnnt", stream=True
    ):
        print(event.type, getattr(event, "delta", ""))
```

Принимается любой формат, который декодирует ffmpeg (wav, mp3, ogg, m4a, webm, mp4, flac),
внутри всё приводится к 16 кГц моно.

## Длинное аудио

GigaAM не распознаёт за один проход больше 25 секунд. Апстримный `transcribe_longform` режет
аудио через pyannote, чей чекпойнт сегментации закрыт на HuggingFace и требует токена, поэтому
`asr.py` режет через silero-vad (MIT, веса внутри пакета):

1. аудио до `GIGAAM_MAX_CHUNK_S` (22 с) уходит в модель целиком;
2. более длинное делится по границам речи, соседние куски склеиваются, пока кусок короче
   предела, а пауза между ними короче `GIGAAM_VAD_MAX_GAP_S`;
3. куски декодируются пачками по `GIGAAM_BATCH_SIZE`, временные метки слов сдвигаются обратно
   во время файла.

Если VAD не нашёл речи совсем, файл всё равно обрабатывается окнами фиксированной длины.
Запросы идут по очереди под одной блокировкой, потому что модель silero хранит внутреннее
состояние RNN и два потока внутри VAD роняют процесс.

## Настройки

Всё в `.env` (шаблон `.env.dist`).

| Переменная | По умолчанию | Значение |
| --- | --- | --- |
| `GPU_INDEX` | `0` | номер карты: порядок `nvidia-smi` для NVIDIA, порядок Level Zero для Intel |
| `GIGAAM_PORT` | `8770` | порт на хосте |
| `GIGAAM_IMAGE_TAG` | `latest` | тег образа на Docker Hub |
| `GIGAAM_MODELS` | `v3_e2e_rnnt` | чекпойнты через запятую, каждый отдаётся как `gigaam-<имя через дефисы>` |
| `GIGAAM_FP16` | `1` | энкодер в fp16 |
| `GIGAAM_LANGUAGE` | `ru` | язык в ответе `verbose_json` |
| `GIGAAM_BATCH_SIZE` | `8` | кусков на проход |
| `GIGAAM_MAX_CHUNK_S` | `22` | предел длины куска, меньше 25 |
| `GIGAAM_VAD_MAX_GAP_S` | `0.6` | допустимая пауза при склейке кусков |
| `GIGAAM_MAX_FILE_MB` | `1024` | предел размера файла |
| `GIGAAM_API_KEY` | пусто | Bearer-токен, пусто значит без авторизации |

Есть и другие чекпойнты, это `v3_rnnt` и `v3_ctc` (сырой текст без пунктуации), `v3_e2e_ctc`
(быстрее, чуть хуже), `multilingual_ctc` и `multilingual_large_ctc` (больше 70 языков). Их можно
перечислить в `GIGAAM_MODELS` и отдавать сразу несколько, модель выбирается полем `model`.

## Скорость

Модель `v3_e2e_rnnt`, энкодер в fp16, один клиент, время с отправкой файла по локальной сети.

| Аудио | RTX 5060 Ti | Arc A770 (xe) |
| --- | --- | --- |
| 71 с, wav | 2.0 с | 1.7 с |
| 712 с, mp3, 40 кусков | 13.1 с | 13.0 с |

Тексты на обеих картах совпали слово в слово. Память видеокарты около 1.2 ГБ вместе с пачкой.
Узкое место в жадном декодировании RNNT, а не в размере пачки, так что `GIGAAM_BATCH_SIZE`
больше 8 почти ничего не даёт.

## За LiteLLM

Сервис подключается к LiteLLM как модель с `mode: audio_transcription` и `api_base`
`http://<host>:8770/v1`. Три особенности этого пути:

- `usage.seconds` в ответе целое число, LiteLLM разбирает его как `int` и отвечает 500 на
  дробное, поэтому сервер его округляет (`duration` остаётся точным);
- если попросить через LiteLLM `timestamp_granularities[]=segment` и `=word` одним запросом,
  придут слова с пустым `segments`, по одной гранулярности за запрос работает;
- LiteLLM заворачивает любой `response_format` в свой JSON, `srt` и `vtt` приходят текстом в
  поле `text`, а `stream=true` он буферизует целиком. За готовым `.srt` и настоящим потоком
  обращайтесь к сервису напрямую.

## Заметки

- `gigaam` на PyPI застрял на версии 0.1.0 и не знает чекпойнтов v3, поэтому образы ставят
  апстрим на закреплённом коммите (`GIGAAM_REF` в Dockerfile). `asr.py` пользуется
  `AudioDataset` и `model._decode`, теми же внутренностями, что и апстримный longform, так что
  коммит стоит поднимать осознанно;
- образ CUDA берёт torch с индекса CUDA 12.8, в нём есть ядра для Blackwell (sm_120) и всех
  карт старше;
- образ XPU собран на `intel/compute-runtime` (драйверы Level Zero и OpenCL на Ubuntu 24.04),
  torch для XPU приносит свой рантайм SYCL колёсами. Сам GigaAM от устройства не зависит,
  энкодер идёт под fp16 autocast так же, как на CUDA;
- `language`, `prompt` и `temperature` принимаются и игнорируются, у GigaAM v3 фиксированный
  декодер, а параметры оставлены, чтобы не ломать клиентов OpenAI;
- `logprobs` не считаются, `avg_logprob` и `no_speech_prob` в `verbose_json` заполнены
  нейтральными значениями ради совместимости со схемой whisper.

## Ссылки

- https://github.com/salute-developers/GigaAM
- https://habr.com/ru/companies/sberdevices/articles/805569/
