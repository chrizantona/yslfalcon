# Данные и веса: источники для воспроизведения

Всё, что нужно, чтобы с нуля повторить эксперименты из `experiments/`. На поде внешние файлы лежат в `/workspace/ext_data/`, датасет соревнования — в `/workspace/lct2026_data/`; пути задаёт `configs/data.yaml`. Скачать внешние данные и публичные веса с проверкой sha256: `KAGGLE_API_TOKEN=<токен> bash scripts/fetch_external.sh`.

## Датасет соревнования
| | |
|---|---|
| Что | ЛЦТ 2026, задача 7 «Фалькон Тех»: `images/`, `train.csv`, `test_query.csv`, `test_gallery.csv`, README |
| Откуда | архив `dataset.zip` из личного кабинета участника; публичной ссылки нет |
| Размер, sha256 | 7 218 674 831 байт, `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613` |
| Распаковка | `scripts/pod_setup.sh` → `/dev/shm/lct2026/dataset`; вырезки — `reid.crops` → `/dev/shm/lct2026/crops512` |
| Протокол валидации | `protocol/val_v1/`, `protocol/val_v2/` в репозитории (строятся `reid.protocol`) |

## Внешние датасеты
| Файл | Источник | Оригинал и лицензия | Размер, sha256 | Где используется |
|---|---|---|---|---|
| `kaggle/veriwild_test.zip` | Kaggle [`mrkdagods/veriwild-test`](https://www.kaggle.com/datasets/mrkdagods/veriwild-test), версия от 14.07.2026 | VERI-Wild, Lou et al., CVPR 2019, [github.com/PKU-IMRE/VERI-Wild](https://github.com/PKU-IMRE/VERI-Wild); некоммерческое исследовательское использование; номера замаскированы | 8 202 185 309 байт, `34f4ba6bf73302145ce85ab2477cb2761972d053706cdb5932e8ab44a39c3b56` | EXP-009 (претрейн) → EXP-010, EXP-015, EXP-016, EXP-019 |
| `kaggle/veri776.zip` | Kaggle [`abhyudaya12/veri-vehicle-re-identification-dataset`](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset), версия 1 | VeRi-776, Liu et al., ECCV 2016, [github.com/VehicleReId/VeRi](https://github.com/VehicleReId/VeRi); CC BY-NC 4.0 по странице Kaggle | 992 234 678 байт, `9d45b9ad779d04d58c14abc09a0f681427145464c0aa9faa3ac19b3fea4d0694` | пока нигде |

Kaggle-копии сверены с официальной статистикой: VERI-Wild test — 10 000 машин, 128 517 снимков галереи + 10 000 запросов; VeRi-776 — 37 778 / 11 579 / 1 678 снимков.

## Публичные веса
| Файл | Источник | Лицензия | sha256 | Где используется |
|---|---|---|---|---|
| CLIP ViT-B/16 `laion2b_s34b_b88k` | open_clip 3.3.0 скачивает `open_clip_model.safetensors` из [HF `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K), ревизия `7288da5a0d6f0b51c4a2b27c624837a9236d0112` | MIT | `3f25d29d3cc74e1d25d47e0593b4dd0864ced1e9b4d8e486a247ca4502f227f1` | все ViT-эксперименты кроме EXP-011…014, 017, 018 |
| CLIP ViT-L/14 `laion2b_s32b_b82k`, `weights/clip/exp020.safetensors` | [LAION/OpenCLIP, Ross Wightman](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), `open_clip_pytorch_model.safetensors`, ревизия `1627032197142fbe2a7cfec626f4ced3ae60d07a` | MIT | `7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019` | EXP-020…029, 036 |
| timm `resnet50.a1_in1k` | [HF `timm/resnet50.a1_in1k`](https://huggingface.co/timm/resnet50.a1_in1k), `model.safetensors`, ревизия `767268603ca0cb0bfe326fa87277f19c419566ef` | Apache-2.0 | `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb` | EXP-002 |
| `weights/fast-reid/veriwild_bot_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth), JDAI-CV | Apache-2.0 (обучен на VERI-Wild train) | `f591e4f7f31ce0be2f4079171cfc03fcbecab5ea5b40f9148c753b9c6f061e8e` | EXP-011, EXP-017, EXP-018 |
| `weights/fast-reid/vehicleid_bot_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth) | Apache-2.0 (обучен на VehicleID) | `48992dbfdb99b93c9a0c2f15dd27b7a738f8b8a7e49dd231391ed211f88d759a` | EXP-012 |
| `weights/fast-reid/veri_sbs_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth) | Apache-2.0 (обучен на VeRi-776) | `57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd` | пока нигде |
| `weights/clip-reid/VehicleID_clipreid_ViT-B-16_60.pth` | Google Drive `168BLegHHxNqatW5wx1YyL2REaThWoof5` из README [github.com/Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | MIT (обучен на VehicleID) | `538e8288741b45deb1dd8225e47b57fc0702fde79156ca673599e53c1905b27b` | EXP-013 |
| `weights/clip-reid/VeRi_clipreid_ViT-B-16_60.pth` | Google Drive `1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e` из README CLIP-ReID | MIT (обучен на VeRi-776) | `71f6f2723be4d563f1df201133daa53f57bee2a67266486559d5ce453a60f805` | EXP-014 |

## Наши веса
Чекпоинты лежат на поде в `/workspace/checkpoints/`, имена записаны в карточках и реестрах; в git их нет. Ранние эксперименты используют случайные имена, EXP-020…024 — имена по номеру EXP по просьбе пользователя. Старые эксперименты воспроизводятся обучением с тега (`git checkout exp-NNN`, команда из карточки); для ночной цепочки фактические код и конфиги с новыми именами сохранены в `artifacts/EXP-NNN/pod/`, переименование закоммичено в `75f0c12`. Цепочка текущего бейзлайна по качеству — EXP-022 → EXP-023 → EXP-024; историческая — EXP-009 → EXP-010 → EXP-016. Числа совпадут с точностью до недетерминизма обучения на GPU; инференс детерминирован (`reid.benchmark`).

## Как повторить с нуля
1. Положить `dataset.zip` в `/workspace/lct2026_data/` и сверить sha256.
2. `KAGGLE_API_TOKEN=<токен> bash scripts/fetch_external.sh` — внешние данные и публичные веса.
3. Эксперименты — в порядке номеров, командой из карточки с тега `exp-NNN`; пересчёт на текущий протокол — `scripts/pod_v2_job.sh`.

## Ночной прогон ViT-L
`bash scripts/fetch_external.sh clip_l` скачивает только закреплённые веса ViT-L, `bash scripts/fetch_external.sh veriwild` — только внешний датасет. Публичная версия 1 Kaggle доступна без токена; оба файла проверяются по sha256.

Окружение пода отдельное: `uv python install 3.13.7`; `uv venv --python 3.13.7 .venv`; `uv pip install --python .venv/bin/python -r requirements-pod.txt --index-url https://download.pytorch.org/whl/cu130`; `uv pip install --python .venv/bin/python -r requirements.txt pytest==9.1.1`. Системные пакеты сервисов не меняются.

По просьбе пользователя директории новых запусков названы `EXP-020`…`EXP-024`, наши чекпоинты — `exp020_seed0.pt` и т. п.; номера EXP сохраняются в реестре и метаданных.

Перед ночным прогоном полный архив передаётся с локальной машины. Для ускорения старта отдельно подготовлены все 11 416 вырезок кодом `crop_box`/`save_crop` из репозитория (OpenCV 5.0.0.93, max_side=512, JPEG quality=95) и переданы оригинальные JPEG фолда 0 и `test_query.csv`. Контрольные суммы каждой вырезки, каждого кадра и CSV сверяются на поде; `scripts/pod_preflight.py` сравнивает байты вырезок с повторной обработкой оригиналов на поде. CSV и фиксированные фолды не изменены. Полный архив — тот же файл с sha256 из раздела датасета соревнования.

Для исключения различий округления ARM/x86 тренировочные вырезки ночного прогона формируются повторно **на поде** штатным `save_crop`: локально JPEG области машин вырезаны без перекодирования `jpegtran` с MCU-выравниванием и полями вокруг bbox, перед resize на поде sha256 всех исходных BGR-пикселей области сверяется с оригиналом. Затем те же crop_max_side=512 и JPEG quality=95. Оригинальные кадры оценки переданы без изменений. Подготовка — `scripts/stage_train_jpegs.py` и `scripts/pod_prepare_crops.py`; все 5 725 тренировочных областей проверены, вырезки оценки пересозданы из оригинальных JPEG. Промежуточные файлы — вне git. Первоначальное указание отложить пуш отменено пользователем 17.09.2026: приватный GitHub разрешён для кода, артефактов и весов. Большие файлы передаются Release assets, исходные теги не перезаписываются.

Полный архив ЛЦТ на поде проверен по указанному sha256 и распакован. Цепочка завершилась 17.09.2026 в 01:26 МСК, контроллер вышел; `scripts/pod_names_job.py` удалил временные ссылки на прежние пути. Восемь чекпоинтов сохранены под именами EXP, sha256 — `artifacts/EXP-020/checkpoint_sha256.json`. Результаты всех пяти экспериментов и снимки кода/конфигов/протокола/логов забраны локально; проверены 87 файлов результатов и 165 копий файлов протокола, квитанция — `artifacts/EXP-020/archive_receipt.json`. Переименование закоммичено в `75f0c12`, теги этого прогона сохраняют исходные имена; соответствие — `artifacts/EXP-020/name_map.json`.

Завершённые папки запусков EXP-020…024 и собственные временные окружения `/tmp/exp1`, `/dev/shm/exp1` удалены после проверки архива; квитанция — `artifacts/EXP-020/cleanup_receipt.json`. Конкурентные сервисы, датасеты, публичные веса и наши чекпоинты сохранены. Фактическая venv этого прогона располагалась на исполняемой файловой системе `/tmp`, поскольку `/dev/shm` на поде смонтирован с `noexec`; первоначальное неисполняемое окружение тоже убрано.

## Передача EXP-022…024
[Приватный Release exp-024](https://github.com/chrizantona/yslfalcon/releases/tag/exp-024) содержит архивы результатов, AP по каждому запросу, эмбеддинги, логи и исходные снимки. SHA256SUMS / manifest.json сопровождают каждый файл; uploader сверяет GitHub digest.

| Наш файл | Автор, версия | sha256 | Использование |
|---|---|---|---|
| exp022_seed0.pt | этот проект, EXP-022, seed 0; trained visual tower + внешняя ID-голова | `47270c9fe75469ca19b7007480ecc1ea0f7a72b7c774ee466259bc8a08420368` | старт обучения EXP-023, 026…029, 036; не нужен для инференса супа |
| exp024.pt | этот проект, EXP-024; равномерный суп EXP-023 seed 0/1/2 | `b1a937546b14b693d35196d3dab23fcbae48b1e7657fe30743307a4a8b125bde` | инференс EXP-024, TTA EXP-025, Docker; один файл 1 218 807 919 байт |

Оба обученных файла основаны на публичном LAION/OpenCLIP и VERI-Wild, источники, авторы и версии — в таблицах выше. Для новых веток не требуются новые внешние данные или публичные веса. Имена EXP по прямому указанию пользователя имеют приоритет над прежними случайными именами. Docker скачивает обученный exp024.pt через asset API приватного Release с BuildKit secret и закреплённым SHA256. Docker-сборка и RTX A5000 пока не проверены.
