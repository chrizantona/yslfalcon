# Передача EXP-044…048: результаты и эмбеддинги

Все результаты третьего пода — в [приватном Release exp-047 репозитория yslfalcon](https://github.com/chrizantona/yslfalcon/releases/tag/exp-047). Доступ получают участники этой приватной репы; скачивать нужно под своим аккаунтом GitHub. После `git pull` открыть Release и скачать нужные assets. Большие файлы остаются Release assets этой же репы.

## Что скачать

| EXP | Архив | Массивов query/gallery | Байтов | SHA256 |
|---|---|---:|---:|---|
| EXP-044 | [EXP-044_results.tar.gz](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/EXP-044_results.tar.gz) | 2 | 4594129 | `0e117c6f38cd6e1b5b25845acefd1683989846fcdfc69e3320ebb5d5fdf48a28` |
| EXP-045 | [EXP-045_results.tar.gz](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/EXP-045_results.tar.gz) | 6 | 10144275 | `daf2a77e1063d6a3f3622656fc625c09abdfcd6513a486a8fa19ec3c6cc242b7` |
| EXP-046 | [EXP-046_results.tar.gz](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/EXP-046_results.tar.gz) | 6 | 10151362 | `f23ac1ba9fdda74fde9ccd6d60ee64f91e208bc954a9012860a211250bdc3151` |
| EXP-047 | [EXP-047_results.tar.gz](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/EXP-047_results.tar.gz) | 4 | 7327256 | `fbf0dcf374d1bdf9b1575da23977c3624c68be5a1b68c79fac08917b8244ef60` |
| EXP-048 | [EXP-048_results.tar.gz](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/EXP-048_results.tar.gz) | 6 | 10144152 | `5399cc4c5f3d345253f55a9959c210422d38277891b4af7974c92c0a40570c6e` |

Итого 24 массива, 1359 файлов в проверенных снимках. Каждому архиву соответствуют метрики, эмбеддинги и AP всех сидов этой EXP. Не нужно повторно обучать модели или вычислять forward для сравнения.

Дополнительные assets Release:

- `embeddings_index.json` — путь, SHA256, форма, dtype и размер каждого массива, CSV порядка query/gallery.
- `analysis.json` — исходные summaries, FP16-проверка EXP-047 и bootstrap EXP-046/047/048. Сравнения EXP-046/048 сделаны после создания исходных архивов и дополнительно находятся здесь.
- `manifest.json`, `SHA256SUMS`, `archive_receipts.json`, `exp047_archive_receipt.json` — контрольные суммы и квитанции. У всех загруженных assets проверены размер и серверный GitHub digest.
- `checkpoint_manifest.json`, `selected_members.json` — состав супа, источник, размеры и суммы весов.

## Как использовать

Скачать архивы и `SHA256SUMS` в `artifacts/handoff_night_044_048/`. Проверить сумму выбранного архива (на Linux `sha256sum`, на macOS `shasum -a 256`) по таблице или `SHA256SUMS`. Затем распаковать из корня репозитория:

```bash
tar -xzf artifacts/handoff_night_044_048/EXP-045_results.tar.gz
```

Пути после распаковки:

- EXP-044: `artifacts/EXP-044/seed0/`.
- EXP-045/046/048: `artifacts/EXP-NNN/seed0/`, `seed1/`, `seed2/`; общий AP по сидам — `per_query_ap_mean.csv` в корне EXP.
- EXP-047: `artifacts/EXP-047/val_fold0/` (FP32), `fp16_check/val_fold0/` (FP16).
- В каждом из этих каталогов: `query_emb.npy`, `gallery_emb.npy`, `per_query_ap.csv`, `metrics.json`, `threshold.json`, `refusal_curve.csv`.
- Снимок кода, конфигов, протокола и логов: `artifacts/EXP-NNN/pod/`. Корневые `logs/` и `status/` архива относятся к его прогону; для нескольких EXP распаковывать по отдельным временным каталогам, если нужны эти дополнительные копии.

Массивы валидации сохранены float16 независимо от точности forward; перед пересчётом привести к float32. Query — 1137×768, gallery — 789×768. Query-строки совпадают с `pod/protocol/val_v2/fold0_query.csv` и порядком `per_query_ap.csv` (это отдельно проверено), gallery — с `pod/protocol/val_v2/fold0_gallery.csv`. Не сортировать массивы или CSV независимо друг от друга. Формат `embeddings.npy` для сдачи по query/test-gallery — отдельный протокол; эти массивы относятся к валидации.

Сравнения выполнялись `reid.research_compare`: пары AP по `image_id` только для совмещения запросов, bootstrap по `vehicle_id`, 2000 повторов, seed 0. Метрики — единственный `evaluate_fold`, неизменённый val_v2/dev-фолд 0; обучение на 1–3, фолд 4 не оценивался. При пересчёте использовать `reid.recompute` с тем же протоколом.

## Итоги

| EXP | mAP@10 | Решение |
|---|---:|---|
| 044 | 0,716465 | reference, претрейн только VERI-Wild |
| 045 | 0,860054 ± 0,004518 | accepted относительно EXP-030 |
| 046 | 0,826543 ± 0,000058 | rejected относительно EXP-045, CI [−0,046023; −0,020214] |
| 047 | 0,863684; FP16 0,863603 | inconclusive к EXP-045, CI [−0,000929; +0,008245]; лучше EXP-037, CI [+0,002237; +0,038523] |
| 048 | 0,832100 ± 0,001246 | accepted относительно EXP-020, CI [+0,029830; +0,066494] |

Общий лидер остаётся EXP-043 (0,885639). H200 FP16 для EXP-047 — 18,22 мс / 420,7 FPS, 2139 МБ VRAM, один файл 1218,8 МБ. A5000 отложена пользователем; официальный `evaluate.py` ожидается. Отказ записан по текущему val_v2 вместе с проверкой F1_strict/score_strict; сертификация по официальному оценщику не заявляется.

## Веса

Копия выбранного супа EXP-047 — [398a6b20e0344621991d19cf5dc7f743.pt](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/398a6b20e0344621991d19cf5dc7f743.pt). Автор — этот проект, версия `exp-047`, размер 1218822583 байт, SHA256 `98b60ea55cc7cfd7627669a8969746178c97999621e471cfe9cff4965a80d035`. Оригинальные чекпоинты остаются в `/workspace/checkpoints/` на поде, имена не меняются. Публичные источники DFN-2B/VERI-Wild и версии — [DATA.md](../DATA.md).

Код четырёх обучений — `7511a8e` (`exp-044/045/046/048`); суп — `f768eda` (`exp-047`). Конфиги не менялись после соответствующих тегов; для безопасного запуска менялись только GPU IDs в двух очередях. Чужие задачи занимали GPU/CPU; журнал ночной нагрузки сохранён в `pod/status/hardware.jsonl`. Собственные обучения и замеры завершены.

[Квитанция всех 14 опубликованных assets и индекс эмбеддингов](HANDOFF_EXP-044-048.json) также доступны в git. GitHub подтвердил SHA256 и размер копии супа; asset API: `https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/571284418`.

Уборка выполнена 18.09.2026 в 01:33 МСК после проверки локальных архивов и всех 14 GitHub assets. Наших процессов обучения, оценки, очередей и мониторинга на поде нет. Папки EXP-044/045/046/047/048, queue_b, /tmp/EXP-044, /dev/shm/EXP-044 и лишние временные архивы удалены. Все 11 новых чекпоинтов, датасеты, публичные веса и сторонние процессы сохранены; квитанция — artifacts/handoff_night_044_048/cleanup_receipt.json и сводка в HANDOFF_EXP-044-048.json.
