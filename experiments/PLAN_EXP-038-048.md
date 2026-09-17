# EXP-038…048: план на два пода

Составлен 17.09.2026. Все конфиги, очереди и теги `exp-038`…`exp-048` указывают на один коммит; менять конфиги нельзя. Если что-то нужно поменять — новый номер EXP, коммит и тег.

## Зачем эта серия
- Лучший mAP@10 на фолде 0 сейчас у SigLIP 2 So400m/14 (EXP-032, 0,857, отказ 0,911), но он обучен по простому рецепту EXP-020. Для сдачи готов EXP-037 (LAION ViT-L/14 → претрейн VERI-Wild → ArcFace с imprinting → суп, 0,843).
- **Главный вопрос:** даёт ли та же цепочка прирост на лучших бэкбонах — SigLIP 2 и быстрый DFN-2B. Вклад претрейна и головы разделён на два эксперимента.
- **Дешёвые проверки:**
  - полезен ли VehicleID в претрейне — на ViT-B/16, прежде чем тратить ~3 часа на SigLIP 2;
  - годится ли ConvNeXt как бэкбон.

## Кто что запускает
| EXP | Что | Родитель | Под | Очередь | Память на процесс | Время |
|---|---|---|---|---|---|---|
| 038 | претрейн ViT-B/16: VERI-Wild test + VehicleID (reference) | EXP-009 | второй (rougenn) | `pod_a_vehicleid` | 10,5 ГБ | ~1–1,5 ч |
| 039 | ViT-B/16 от EXP-038, рецепт EXP-010 | EXP-010 | второй | `pod_a_vehicleid` | 10,5 ГБ | ~20 мин |
| 040 | претрейн SigLIP 2 на VERI-Wild (reference) | EXP-032 | второй | `pod_a_siglip2` | 42 ГБ | ~1 ч |
| 041 | SigLIP 2 от EXP-040, линейная голова | EXP-032 | второй | `pod_a_siglip2` | 42 ГБ × 3 | ~20 мин |
| 042 | SigLIP 2 от EXP-040, ArcFace с imprinting | EXP-041 | второй | `pod_a_siglip2` | 42 ГБ × 3 | ~20 мин |
| 043 | суп лучшего из EXP-041/042 | — | второй | конфиг после результатов | — | ~5 мин |
| **044** | **претрейн DFN-2B ViT-L/14 на VERI-Wild (reference)** | EXP-030 | **третий (Аланбек)** | `pod_b_dfn2b` | ~40 ГБ | ~40 мин |
| **045** | **DFN-2B от EXP-044, линейная голова** | EXP-030 | **третий** | `pod_b_dfn2b` | ~40 ГБ × 3 | ~15 мин |
| **046** | **DFN-2B от EXP-044, ArcFace с imprinting** | EXP-045 | **третий** | `pod_b_dfn2b` | ~40 ГБ × 3 | ~15 мин |
| 047 | суп лучшего из EXP-045/046 | — | третий или второй | конфиг после результатов | — | ~5 мин |
| **048** | **CLIP ConvNeXt-L-D, вход 320** | EXP-020 | **третий** | `pod_b_convnext` | **52 ГБ** | ~1 ч (сиды по очереди) |

Память — пик `torch.cuda.max_memory_allocated` на батче 128 в bf16 (предпроверка на H200); к ней добавьте 2–3 ГБ на контекст CUDA. Время — оценка для H200 без чужой нагрузки на CPU. Оценка каждой модели на фолде 0 и замер скорости идут в конце каждой задачи.

EXP-035 (уверенность для отказа по эмбеддингам EXP-037) в эту серию не входит. Он ждёт артефакты EXP-037 и `evaluate.py` организаторов.

## Третий под: пошагово
Нужно: 3–4 GPU, на каждой ≥ 45 ГБ свободной памяти (для EXP-048 — ≥ 55 ГБ на одной GPU); ~20 ГБ RAM в `/dev/shm` (датасет, вырезки, VERI-Wild); ~15 ГБ диска под веса и ~6 ГБ на каждый чекпоинт-сид; `uv` и доступ в интернет.

1. **Код.**
   ```bash
   mkdir -p /workspace/yslfalcon_runs && cd /workspace/yslfalcon_runs
   git clone git@github.com:chrizantona/yslfalcon.git queue_b && cd queue_b
   git checkout exp-044        # тот же коммит, что exp-045, exp-046, exp-048
   mkdir -p status logs
   ```
2. **Датасет соревнования.** Положите `dataset.zip` (из личного кабинета) в `/workspace/lct2026_data/`. sha256 `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613`. Распаковку и вырезки задача сделает сама.
3. **Внешние данные и веса** — с проверкой sha256; ссылки и версии в [DATA.md](../DATA.md).
   ```bash
   bash scripts/fetch_external.sh veriwild     # VERI-Wild test, публичная версия 1 на Kaggle (~8,2 ГБ)
   bash scripts/fetch_external.sh backbones    # DFN-2B, EVA02, SigLIP 2, ConvNeXt-L-D (~8,6 ГБ)
   ```
   Если Kaggle требует токен: `KAGGLE_API_TOKEN=<токен> bash scripts/fetch_external.sh veriwild`. Токен никуда не записывать.
4. **Окружение — один раз** (Python 3.13.7, torch 2.10.0+cu130; нужен драйвер NVIDIA для CUDA 13, у нас 580.95; ~15 мин):
   ```bash
   RUNTIME_NAME=EXP-044 GPU=0 bash scripts/pod_env_job.sh
   ls status/env_DONE          # venv в /tmp/EXP-044/.venv, тесты пройдены
   ```
5. **GPU.** Посмотрите `nvidia-smi` и впишите номера свободных GPU в `scripts/queues/pod_b_dfn2b.txt` и `scripts/queues/pod_b_convnext.txt` (формат описан в файлах). Номера GPU — единственное, что можно править; конфиги трогать нельзя. EXP-048 нужна отдельная GPU с ≥ 55 ГБ свободной памяти; если такой нет — запускайте её после DFN-2B на освободившейся GPU.
6. **Запуск** — отвязано от ssh, обе очереди можно параллельно:
   ```bash
   QUEUE=scripts/queues/pod_b_dfn2b.txt PREPARED_ENV=/tmp/EXP-044/.venv \
     setsid nohup bash scripts/pod_queue_job.sh > status/queue_pod_b_dfn2b.log 2>&1 < /dev/null &
   QUEUE=scripts/queues/pod_b_convnext.txt PREPARED_ENV=/tmp/EXP-044/.venv \
     setsid nohup bash scripts/pod_queue_job.sh > status/queue_pod_b_convnext.log 2>&1 < /dev/null &
   ```
7. **Как следить.**
   - `cat status/queue_*_stage` — какая EXP идёт.
   - `cat /workspace/yslfalcon_runs/EXP-044/status/stage` — этап внутри EXP: dataset → venv → tests → crops → external → train → aggregate → slices → benchmark → done.
   - `tail -1 /workspace/yslfalcon_runs/EXP-044/logs/train_seed0.log` — эпоха, loss, `dev.mAP10` на 10/20/30-й эпохе.
   - Итог: `status/queue_<имя>_DONE` или `_FAILED`; внутри EXP — `status/DONE` или `status/FAILED`, логи в `logs/`.
8. **Если упало.** Посмотрите `status/FAILED` и последний лог в `logs/` и напишите нам.
   - Не правьте конфиги и не перезапускайте с другими параметрами.
   - Если после исправления кода эксперимент не меняется (например, упал загрузчик), перезапустите ту же EXP из новой строки очереди: папку `/workspace/yslfalcon_runs/EXP-NNN` перед этим переименуйте.
   - Чекпоинт с тем же именем повторно не запишется. Удалите частичный файл из `/workspace/checkpoints/`, только если обучение этой EXP не дошло до конца.
9. **Когда очереди закончились.** Для каждой из EXP-044, 045, 046, 048:
   ```bash
   cd /workspace/yslfalcon_runs/EXP-NNN
   tar -czf /workspace/EXP-NNN_results.tar.gz artifacts/EXP-NNN logs status
   sha256sum /workspace/EXP-NNN_results.tar.gz
   ```
   - Передайте архивы и их sha256 — через приватный GitHub Release, как для EXP-037, или как договоримся. Карточки, реестр и README заполним мы.
   - Чекпоинты остаются на поде в `/workspace/checkpoints/`. Имена берутся из конфигов (`5efa9f751a93443288075459a01447d5.pt` у EXP-044) и из `artifacts/EXP-NNN/seed*/metrics.json`. Файлы не переименовывать.
   - Напишите, были ли GPU и CPU заняты чужими задачами во время замера скорости (этап benchmark).
10. **Уборка.** После того как мы подтвердим получение архивов, удалите папки `/workspace/yslfalcon_runs/EXP-0NN`, `/workspace/yslfalcon_runs/queue_b`, `/tmp/EXP-044` и кэш `/dev/shm/EXP-044`. Датасеты, публичные веса и чекпоинты оставьте.

**Правила** (полностью — [AGENTS.md](../AGENTS.md)):
- фолд 4 не использовать;
- номера на снимках не использовать;
- данные, веса, артефакты и токены не коммитить;
- публичные веса брать только через `fetch_external.sh` с проверкой sha256;
- теги не переставлять.

## Второй под (rougenn)
- VehicleID распакован `scripts/prepare_vehicleid.sh` в `/dev/shm/ext/vehicleid`: 221 567 снимков с меткой, 26 328 машин.
- Очереди `pod_a_siglip2` (GPU 2, затем 1/3/2) и `pod_a_vehicleid` (GPU 0) идут параллельно. Окружение — готовая venv прошлой серии.

## Решения после серии
- **EXP-039 против EXP-010** (парный bootstrap, AP EXP-010 лежат локально). Если CI выше нуля и прирост от +0,01 — отдельный EXP: SigLIP 2 с претрейном на VERI-Wild + VehicleID (~3 ч).
- **SigLIP 2:**
  - EXP-041 против EXP-032, EXP-042 против EXP-041;
  - по 45·mAP@10 + 10·отказ выбрать голову;
  - суп EXP-043 → сравнение с EXP-037 и замер fp16.
- **DFN-2B:**
  - EXP-045 против EXP-030, EXP-046 против EXP-045;
  - суп EXP-047 → сравнение с EXP-037.
- **ConvNeXt, EXP-048 против EXP-020.** Если принят, дальше по одному:
  - ConvNeXt V2-L;
  - DINOv3 ConvNeXt-L;
  - прямоугольный вход;
  - last stride 1.
- **Финальный кандидат:** по баллам и скорости на RTX A5000. Затем одна проверка на фолде 4 и переобучение на фолдах 0–3.
