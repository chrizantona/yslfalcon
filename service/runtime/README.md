# Модель внутри runtime-образа

GPU runtime с обученной моделью собирается `deploy/runtime.Dockerfile` из корня репозитория. Своей копии модели у
сервиса нет: образ берёт те же файлы, что и образ сдачи. Сервисный код модель не импортирует: адаптер
`RepositoryEncoder` подгружает её только внутри runtime-образа, а API, worker и web остаются без PyTorch.

| Файл в репозитории | Где в образе | Что это |
|---|---|---|
| `src/reid/` | `/app/runtime/reid/` | код модели: вырезка по рамке, башня SigLIP 2, fp16, голова ракурса |
| `configs/final_model.yaml` | `/app/runtime/final_model.yaml` | сдаваемая конфигурация: архитектура, веса, порог отказа, δ поправки на ракурс |
| `configs/weights.json` | `/app/runtime/weights.json` | откуда брать веса и их sha256 |
| `requirements-inference.txt` | — | зависимости модели, torch с CUDA 12.6 |
| `service/runtime/fetch_weights.py` | `/app/runtime/fetch_weights.py` | запекает веса в `/models` при сборке и пишет `runtime-bundle.json` |

Порог, чекпоинт или голова ракурса меняются в `configs/` один раз — и в сабмите, и в сервисе. Контекст сборки
ограничен этими файлами через `deploy/runtime.Dockerfile.dockerignore`.
