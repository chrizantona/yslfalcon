# EXP-070…078: план на два пода

Составлен 19.09.2026. Все конфиги, очереди и теги `exp-070`…`exp-078` указывают на один коммит; менять конфиги нельзя. Если что-то нужно поменять — новый номер EXP, коммит и тег.

## Зачем эта серия
- Лучший кандидат — **EXP-054** (0,8939, отказ 0,912): SigLIP 2 So400m/14 → претрейн VERI-Wild (EXP-040) → ArcFace с imprinting + SupCon (EXP-053, 0,8895 ± 0,0017) → суп трёх сидов.
- Рецепт с triplet не реагировал на отступ, сглаживание и lr (EXP-049…052). SupCon дал +0,006, но взят с настройками по умолчанию — здесь проверяем именно его, по одному изменению от EXP-053.
- Родитель всех дообучений — EXP-053; сравнение — парный bootstrap по машинам с EXP-053 (его AP — в Release `exp-054`), принятый вариант затем идёт в суп против EXP-054.

## Кто что запускает
| EXP | Что меняем | Родитель | Под | Очередь | Память на сид | Время |
|---|---|---|---|---|---|---|
| 070 | температура SupCon 0,1 | EXP-053 | второй (rougenn) | `pod_a_supcon_round2` | 45 ГБ | ~15 мин |
| 071 | температура SupCon 0,05 | EXP-053 | второй | `pod_a_supcon_round2` | 45 ГБ | ~15 мин |
| 072 | сиды 3–5 рецепта EXP-053 | EXP-053 | второй | `pod_a_supcon_round2` | 45 ГБ | ~15 мин |
| 073 | суп шести сидов (EXP-053 + EXP-072) против EXP-054 | EXP-072 | второй | вручную после 072 | — | ~10 мин |
| 074 | уверенность для отказа на эмбеддингах EXP-054, без обучения | EXP-054 | второй | вручную | CPU | минуты |
| **075** | **8 снимков на машину (PK 32×8, батч 256)** | EXP-053 | **третий (товарищ)** | `pod_b_supcon_variants` | **82 ГБ** | ~30 мин |
| **076** | **SupCon + batch-hard triplet** | EXP-053 | **третий** | `pod_b_supcon_variants` | 45 ГБ | ~15 мин |
| **077** | **претрейн VERI-Wild с ArcFace (imprinting) + SupCon (reference)** | EXP-040 | **третий** | `pod_b_supcon_pretrain` | ~45 ГБ | ~1 ч |
| **078** | **рецепт EXP-053 от претрейна EXP-077** | EXP-053 | **третий** | `pod_b_supcon_pretrain` | 45 ГБ | ~15 мин |

Память — пик `torch.cuda.memory_reserved` одного шага в bf16 на H200 (предпроверка 19.09.2026: EXP-053 — 45,2 ГБ и 0,27 с на шаг, EXP-075 — 81,6 ГБ и 0,53 с, EXP-076 — 44,9 ГБ и 0,27 с); к ней добавьте 2–3 ГБ на контекст CUDA. Время — для H200 без чужой нагрузки на CPU; оценка на фолде 0 и замер скорости идут в конце каждой задачи.

## Третий под: пошагово
Нужно: GPU H200; для EXP-075 — по одному сиду на GPU с ≥ 90 ГБ свободной памяти, для остальных ≥ 50 ГБ на сид; ~20 ГБ RAM в `/dev/shm` (датасет, вырезки, VERI-Wild); ~2 ГБ диска на каждый чекпоинт-сид; `uv`, `gh` с доступом к приватным релизам и интернет.

1. **Код.**
   ```bash
   mkdir -p /workspace/yslfalcon_runs && cd /workspace/yslfalcon_runs
   git clone git@github.com:chrizantona/yslfalcon.git queue_b070 && cd queue_b070
   git checkout exp-075        # тот же коммит, что exp-076, exp-077, exp-078
   mkdir -p status logs
   ```
2. **Датасет соревнования** — как в прошлых сериях: `dataset.zip` в `/workspace/lct2026_data/`, sha256 `a17950796be648c086b6d313e5d2508447e194f4ab4140bc37143d1fbba47613`. Распаковку и вырезки задача сделает сама.
3. **Внешние данные и публичные веса** (с проверкой sha256; уже скачанные файлы пропускаются):
   ```bash
   bash scripts/fetch_external.sh veriwild     # VERI-Wild test — нужен для EXP-077
   bash scripts/fetch_external.sh backbones    # SigLIP 2 So400m/14
   ```
4. **Наш претрейн EXP-040** — старт EXP-075 и EXP-076 (EXP-078 стартует с EXP-077, который очередь сделает сама):
   ```bash
   mkdir -p /workspace/checkpoints
   gh release download exp-040 --repo chrizantona/yslfalcon --pattern b155ab652c084d75bf3054d188e01313.pt --dir /workspace/checkpoints
   echo "bf49a96c7d4fd9908103270bd0472f8048905d633c870a472ddeccc30df9b041  /workspace/checkpoints/b155ab652c084d75bf3054d188e01313.pt" | sha256sum -c
   ```
   Путь к чекпоинтам задаёт `checkpoint_dir` в `configs/data.yaml` (`/workspace/checkpoints`). Имя файла не менять.
5. **Окружение.** Годится venv прошлой серии (например, `/tmp/EXP-044/.venv`: Python 3.13.7, torch 2.10.0+cu130). Если её нет — один раз, ~15 мин:
   ```bash
   RUNTIME_NAME=EXP-075 GPU=0 bash scripts/pod_env_job.sh
   ls status/env_DONE          # venv в /tmp/EXP-075/.venv, тесты пройдены
   ```
6. **GPU.** По `nvidia-smi` впишите номера свободных GPU в `scripts/queues/pod_b_supcon_variants.txt` и `scripts/queues/pod_b_supcon_pretrain.txt` (формат описан в файлах). Номера GPU — единственное, что можно править. Сиды EXP-075 — на разные GPU с ≥ 90 ГБ; если таких нет — поставьте `0` в последнюю колонку (сиды по очереди на одной GPU, ~30 мин).
7. **Запуск** — отвязано от ssh, обе очереди можно параллельно на разных GPU:
   ```bash
   PREPARED_ENV=/tmp/EXP-044/.venv   # или /tmp/EXP-075/.venv
   QUEUE=scripts/queues/pod_b_supcon_variants.txt PREPARED_ENV=$PREPARED_ENV \
     setsid nohup bash scripts/pod_queue_job.sh > status/queue_pod_b_supcon_variants.log 2>&1 < /dev/null &
   QUEUE=scripts/queues/pod_b_supcon_pretrain.txt PREPARED_ENV=$PREPARED_ENV \
     setsid nohup bash scripts/pod_queue_job.sh > status/queue_pod_b_supcon_pretrain.log 2>&1 < /dev/null &
   ```
8. **Как следить.**
   - `cat status/queue_*_stage` — какая EXP идёт.
   - `cat /workspace/yslfalcon_runs/EXP-075/status/stage` — этап внутри EXP: dataset → venv → tests → crops → (external) → train → aggregate → slices → benchmark → done.
   - `tail -1 /workspace/yslfalcon_runs/EXP-075/artifacts/EXP-075/seed0/train_log.jsonl` — эпоха, лоссы, `dev.mAP10` на 20-й и 30-й эпохе.
   - Итог: `status/queue_<имя>_DONE` или `_FAILED`; внутри EXP — `status/DONE` или `status/FAILED`, логи в `logs/`.
9. **Если упало.** Посмотрите `status/FAILED` и последний лог в `logs/` и напишите нам. Не правьте конфиги и не перезапускайте с другими параметрами. Чекпоинт с тем же именем повторно не запишется: удалите частичный файл из `/workspace/checkpoints/`, только если обучение этой EXP не дошло до конца.
10. **Сравнение с EXP-053** — для EXP-075, 076, 078:
    ```bash
    mkdir -p /tmp/cmp053 && cd /tmp/cmp053
    gh release download exp-054 --repo chrizantona/yslfalcon --pattern EXP-053_results.tar.gz --pattern SHA256SUMS
    grep EXP-053_results.tar.gz SHA256SUMS | sha256sum -c && tar xzf EXP-053_results.tar.gz
    cd /workspace/yslfalcon_runs/EXP-075
    PYTHONPATH=src $PREPARED_ENV/bin/python -m reid.research_compare --base /tmp/cmp053/EXP-053 \
      --new artifacts/EXP-075 --out artifacts/EXP-075/bootstrap_vs_EXP-053.json
    ```
    Решение пишется в поле `decision` (accepted — CI выше нуля или прирост больше разброса сидов).
11. **Результаты.** Для каждой из EXP-075, 076, 077, 078:
    ```bash
    cd /workspace/yslfalcon_runs/EXP-NNN
    tar -czf /workspace/EXP-NNN_results.tar.gz artifacts/EXP-NNN logs status
    sha256sum /workspace/EXP-NNN_results.tar.gz
    ```
    - Выложите архивы и `SHA256SUMS` в приватный Release (например, `exp-078`), как для EXP-067. Карточки, реестр и README заполним мы.
    - Чекпоинты остаются на поде в `/workspace/checkpoints/`. Имена заданы в конфигах: `1fb351a2de9a47a6a0f4f84c5983e460_seed{0,1,2}.pt` (EXP-075), `35e8f18893674694b040bff746c2fa69_seed{0,1,2}.pt` (EXP-076), `53b74390daf24d159ad6a628742be673.pt` (EXP-077), `37d4f18809ed439393725db947acd69c_seed{0,1,2}.pt` (EXP-078). Файлы не переименовывать.
    - Если вариант принят, напишите — конфиг супа подготовим мы (имена чекпоинтов уже известны); веса супа тогда понадобится выложить в Release.
    - Напишите, были ли GPU и CPU заняты чужими задачами во время замера скорости (этап benchmark).
12. **Уборка.** После того как мы подтвердим получение архивов, удалите `/workspace/yslfalcon_runs/EXP-07*`, `/workspace/yslfalcon_runs/queue_b070`, `/tmp/cmp053` и временную venv, если создавали. Датасеты, публичные веса и чекпоинты оставьте.

**Правила** (полностью — [AGENTS.md](../AGENTS.md)): фолд 4 не использовать; номера на снимках не использовать; данные, веса, артефакты и токены не коммитить; публичные веса брать только через `fetch_external.sh` с проверкой sha256; теги не переставлять.

## Итог третьего пода (19.09.2026)

- EXP-075 — `inconclusive`: 0,888017 ± 0,001154; против EXP-053 −0,001467, CI [−0,005389; +0,002414].
- EXP-076 — `accepted`: 0,893253 ± 0,000670; против EXP-053 +0,003769, CI [+0,000164; +0,007551].
- EXP-077 — `reference`: 0,274025; новый рецепт претрейна не переносится.
- EXP-078 — `rejected`: 0,533423 ± 0,078357; против EXP-053 −0,356062, CI [−0,389831; −0,323117].
- EXP-082 — суп EXP-076, `inconclusive`: 0,895844; против EXP-054 +0,001947, CI [−0,001941; +0,006553], против EXP-079 −0,000291, CI [−0,004508; +0,003873].
- Все 24 массива эмбеддингов, per-query AP, логи и 11 checkpoint опубликованы в [приватном Release `exp-082`](https://github.com/chrizantona/yslfalcon/releases/tag/exp-082) с SHA256. Фолд 4 не использовался; прямого замера RTX A5000 нет.

## Второй под (rougenn)
- Очередь `pod_a_supcon_round2`: EXP-070 → 071 → 072, по три сида параллельно на GPU 1/2/3; GPU 0 оставлена сервисам пода.
- После EXP-072 — суп EXP-073 (`scripts/pod_soup_job.sh`, команда в карточке) и сравнение с EXP-054.
- EXP-074 — `reid.refusal_confidence` на эмбеддингах EXP-054, CPU.

## Решения после серии
- EXP-070, 071, 075, 076, 078 — каждый против EXP-053; принятые — в суп и против EXP-054; если приняты два — следующий EXP сочетает их.
- EXP-073 против EXP-054: если суп из шести лучше — берём его вместо EXP-054.
- EXP-074: другая уверенность для отказа — только при CI выше нуля.
- Финал (переобучение на фолдах 0–3 и одна проверка на фолде 4) — по лучшему рецепту после этой серии.
