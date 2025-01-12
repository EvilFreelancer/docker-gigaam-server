# GigaAM API Webserver in Docker

GigaAM HTTP-сервер для автоматического распознавания речи (ASR), предоставляющий API,
совместимое с OpenAI-клиентами, и обеспечивающий гибкую настройку распределённой
обработки аудиофайлов. Система оптимизирована для развертывания и маршрутизации
запросов между несколькими Docker-контейнерами, основанными
на [GigaAM](https://github.com/salute-developers/GigaAM).

Если планируется обучение моделей с использованием Flash Attention, то надо будет выполнить ещё и:

```shell
pip install setuptools psutil torch flash-attn --no-build-isolation
```

https://github.com/salute-developers/GigaAM

https://habr.com/ru/companies/sberdevices/articles/805569/

* Generate [Hugging Face API token](https://huggingface.co/docs/hub/security-tokens)
* Accept the conditions to access:
    * [pyannote/voice-activity-detection](https://huggingface.co/pyannote/voice-activity-detection)
    * [pyannote/segmentation](https://huggingface.co/pyannote/segmentation)
