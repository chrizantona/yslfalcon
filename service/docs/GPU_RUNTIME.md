# Контракт подключения GPU runtime

Этот документ описывает только границу между сервисом и готовой моделью. Модель, её SDK,
предобработка, веса, калибровка и benchmark принадлежат модельной команде. Обычный запуск Falcon
остается в режиме `preview` и не требует GPU.

## Что предоставляет сервис

- отдельный контейнер `runtime` во внутренней сети; API и worker обращаются только к нему;
- bearer-аутентификацию внутренних запросов и бинарный ограниченный формат признаков;
- `docker-compose.gpu.yml` с NVIDIA reservation, read-only `/models`, настраиваемым `/dev/shm` и
  консервативной конкуренцией `1`;
- проверку полного bundle до старта: пути, отсутствие symlink/path traversal, размеры, SHA-256,
  отсутствие неучтённых файлов и суммарный вес всех `role=weight` не более 2 000 000 000 байт;
- pin образа строго по `@sha256` digest, точный ожидаемый `model_id`, сумму байтов весов и
  fail-closed startup;
- честную передачу в API/UI фактических `device`, `precision`, `weights_bytes` и признака подключения
  настоящей модели.

Сервис не скачивает runtime или веса, не импортирует PyTorch/ONNX/TensorRT в API и не считает
отсутствие весов выполнением конкурсного лимита.

## Что должен предоставить runtime-образ

Образ должен быть локально доступен под тем же immutable reference, который записан в
`runtime-bundle.json`. Он запускается наследуемой командой
`uvicorn falcon_service.model_runtime:app --host 0.0.0.0 --port 8001`; следовательно, модельная
сборка расширяет существующий runtime adapter, а не API/worker. Образ обязан:

1. Работать без исходящего интернета и иметь контейнерный healthcheck `/health/live`.
2. Понимать `FALCON_ENCODER=production`, `FALCON_MODEL_MANIFEST=/models/runtime-bundle.json`,
   `FALCON_MODEL_DEVICE=cuda`, `FALCON_RUNTIME_MAX_CONCURRENCY` и `FALCON_RUNTIME_TOKEN`.
3. После полной загрузки отвечать на bearer-защищённый `GET /internal/v1/capabilities`:

```json
{
  "mode": "production",
  "model_id": "exact-serving-model-id",
  "dimension": 1024,
  "rerank_window": 20,
  "calibration_policy": null,
  "real_model_connected": true,
  "device": "cuda:0",
  "precision": "float16",
  "weights_bytes": 1800000000
}
```

`weights_bytes` включает основную, вспомогательные модели и реранкер и не превышает
2 000 000 000. Если калибровка ещё не передана, значение остаётся `null`, а сервис честно возвращает
`calibration_unavailable`.

4. Реализовать `POST /internal/v1/encode` и `/internal/v1/rerank` в точности по локальному OpenAPI
   runtime. Вход — только текущая фотография/BBox и выбранные кандидаты одной закреплённой галереи.
   Runtime не читает другие query, camera ID, порядок CSV, image ID, номера или сетевые источники.
5. Возвращать L2-нормализованный конечный `float32` embedding заявленной размерности и неизменный
   `model_id`. Rerank обязан соблюдать score-permutation contract сервиса.
6. Завершать startup с ошибкой при отсутствии/повреждении файлов, несовместимой CUDA или OOM. Нельзя
   молча переходить на preview/CPU и при этом заявлять production.

## Bundle

Внешняя директория содержит `runtime-bundle.json` и каждый необходимый локальный файл. Пример схемы
лежит в `models/runtime-bundle.example.json`. В `files` перечисляются все файлы кроме самого
манифеста; допустимые роли:

- `weight` — любые encoder/projection/reranker/auxiliary weights, учитываются в лимите;
- `calibration` — политика отказа;
- `metadata` — конфигурация и словари без весов.

Нельзя хранить дополнительные незаявленные файлы. Bundle монтируется только для чтения и не входит
в Git или обычный service image.

## Запуск и остановка

```bash
cd service
export FALCON_RUNTIME_IMAGE='registry.example/falcon-runtime@sha256:<64-hex-digest>'
export FALCON_MODEL_BUNDLE_DIR='/absolute/path/to/falcon-model-bundle'
export FALCON_EXPECTED_MODEL_ID='exact-serving-model-id'
./scripts/start_gpu.sh
```

Скрипт выполняет preflight, передаёт API/worker проверенную сумму байтов весов, проверяет наличие
локального образа, валидирует итоговый Compose и
запускает `--no-build --pull never`. Это исключает случайную сборку preview под тегом production и
сетевое скачивание во время старта. API readiness проходит только после handshake с точным
`model_id` и `weights_bytes`.

Остановка сохраняет БД и артефакты:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml down
```

Перед активацией новой модели требуется новая полная версия галереи с тем же `model_id`; уже
сохранённые результаты и старые версии не пересчитываются задним числом.

## Приёмка передачи

Минимальный сервисный gate:

1. `python3 scripts/gpu_preflight.py` проходит на фактическом bundle.
2. Все контейнеры healthy, `/api/v1/health/ready` сообщает production mode.
3. `/api/v1/capabilities` и экран «Модель и качество» показывают фактические model/device/
   precision/weight bytes без секретов.
4. Эталонные изображения дают идентичные embeddings standalone и через runtime с заданным допуском.
5. Параллельная нагрузка не вызывает OOM; перегрузка возвращает `429`, а не повреждённый результат.
6. После отключения исходящей сети проходят import, search, replay и export.
7. Отдельно модельная команда прикладывает официальный `reid.benchmark` и отчёт `val_v2`; сервисные
   p50/p95 preview не используются как конкурсные цифры.
