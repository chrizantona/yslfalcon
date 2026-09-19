# Аудит реализации OT-GFAM по оригинальной статье

Источник: H. Ergin, A. S. Keçeli, [*Adaptive vehicle re-identification via operational attention and multi-part embedding*](https://www.sciencedirect.com/science/article/pii/S1077314226001104), CVIU 267 (2026), 104743, DOI `10.1016/j.cviu.2026.104743`. Проверен также [официальный репозиторий](https://github.com/hidayetergn/v-reid), встроенный ссылкой в PDF, commit `cb3ac1819e332383f75d9d110c16576b93ccc240`.

## Вывод

Текущий `ConvNeXtGLPE` — не точное воспроизведение опубликованной модели. Совпадает общая идея, но расходятся backbone/output channels, нормализация голов, источник gate во втором блоке, residual-формула и triplet mining. Эти различия достаточно велики, чтобы не переносить на текущую реализацию заявленные авторами результаты.

## Что совпадает

| Узел | Статья / авторский код | Текущий код |
|---|---|---|
| Вход и нормализация | 224×224, ImageNet statistics (Sec. 3, 4.2) | 224×224, ImageNet statistics |
| Макроструктура | ConvNeXt-L → один W-GFAM → один SW-GFAM → global/2-part/3-part (Fig. 1–4, Sec. 3.2) | Та же последовательность и шесть part-векторов |
| Окна | 7×7; shift −3/−3 и обратный +3/+3 (Eq. 14–17) | 7×7; shift 3 |
| Gate | GAP → MLP → sigmoid; поканально умножает токены до Q/K/V (Eq. 6, 12, 15) | Та же операция |
| Operational Q/K/V | Отдельные ONN для Q/K/V, сумма обучаемых преобразований степеней входа, tanh (Eq. 7–12) | Отдельные polynomial ONN Q/K/V с tanh |
| Supervision | Отдельные CE/triplet losses для global, двух половин и трёх полос; веса 1,0 и 0,5 (Algorithm 1, Eq. 2) | Шесть отдельных голов, сумма loss с весами 1,0 и 0,5 |
| Batch | 32, четыре кадра одной машины (Sec. 4.2) | PK 8×4 = 32 |
| Optimizer | Adam, weight decay 5e-4 (Sec. 4.2) | Adam, weight decay 5e-4 |

## Подтверждённые расхождения

1. **Backbone урезан на один stage.** Текущий код берёт `stem + stages[:3]`, то есть карту 768×14×14. Доступный авторский inference-код использует полный `forward_features`, сохраняет последний stage, меняя его stride на 1, и получает 1536 каналов. Его итоговый concat имеет размер 6×1536; наш — 6×768. Параметры статьи (ConvNeXt-L baseline 197,6M, Table 7) также соответствуют полному ConvNeXt-L, а не усечённому.
2. **Нет авторских BNNeck каждой части.** В `models/convnext_part_model.py` официального репозитория после каждого GeM стоит отдельный `BatchNorm1d`, затем L2. У нас перед OT-GFAM добавлен общий `BatchNorm2d(768)`, которого нет ни в статье, ни в опубликованном inference-коде, а после pooling BNNeck отсутствуют.
3. **SW-GFAM получает неправильный gate input.** Eq. 6 строит gate W-GFAM из входа первого блока; Eq. 15 строит gate SW-GFAM из `F_cnn^l`, то есть выхода W-блока. Текущий цикл передаёт обоим юнитам один исходный `block_input`.
4. **В unit добавлен неподтверждённый residual.** Eq. 3–4 и Fig. 2 задают `LN(GFAM(LN(x)) + MLP(LN(x)))`. Текущий код считает `LN(x + GFAM(LN(x)) + MLP(LN(x)))`. В Discussion авторы упоминают residual design, поэтому статья внутренне противоречива; однако явные формулы и схема не содержат `+x`.
5. **Triplet mining другой.** Fig. 1 и Algorithm 1 прямо называют batch-all Triplet Loss. `paper_loss` вызывает `batch_hard_triplet`, выбирающий один самый дальний positive и один ближайший negative для каждого anchor.
6. **Classifier получает другой feature.** Eq. 18–20 задают logits `l = Wv` от pooled vector и отдельно `e = v / ||v||`. Текущий код подаёт в classifier уже L2-normalized embedding. Fig. 1 дополнительно показывает Layer Normalization перед FC; это ещё одно противоречие статьи, но ни один вариант не совпадает с текущим.
7. **Большой претрейн использует другой датасет.** Статья обучает и оценивает модели на стандартных split VeRi-776 / VehicleID / VRU. EXP-126…129 используют размеченный VERI-Wild test как замену отсутствующему VehicleID, после чего планируется LCT fine-tune. Это полезный transfer-эксперимент, но не воспроизведение чисел статьи.
8. **Рецепт lr частично выдуман.** Статья сообщает Adam, base lr 3,5e-4, 120 эпох и warm-up 10 эпох, но не публикует milestones и коэффициент warm-up. `[40, 70]` заимствованы из другой ветки. Differential lr `1e-5 / 3,5e-4` отсутствует в Sec. 4.2, хотя в Limitations сами авторы рекомендуют меньший lr для polynomial terms.
9. **Не опубликованы важные параметры.** Статья не задаёт Taylor order `Q`, число attention heads, размер скрытого MLP, triplet margin, label-smoothing epsilon и наличие Swin boundary mask. Наши `Q=3`, 8 heads, MLP ratio 4, margin 0,3, smoothing 0,1 и mask — обоснованные, но непроверяемые предположения.

## Противоречия самой статьи и официального релиза

- На стр. 4 текст требует concat шести L2-векторов; Eq. 21 на стр. 7 требует их average. Официальный README и `ConvNeXtPartModel.forward` используют concat, как текущий код.
- Algorithm 1 и Eq. 1 называют GeM; Eq. 18–20 описывают GAP/обычное среднее. Официальный код использует GeM с `clamp(eps)`.
- Sec. 4.2 требует 224×224; официальный `test.yaml` задаёт 256×256.
- PDF говорит, что test code и weights опубликованы, но официальный репозиторий не содержит OT-GFAM, training loop или metric implementation. Он содержит только ConvNeXt + GLPE inference/demo. README всё ещё обещает раскрыть OT-GFAM «после принятия статьи».
- Доступный checkpoint скачан и проверен: 2 441 210 739 байт, sha256 `35ddfd1ea7b73b90a66ef41d652bf6600cef1ce9260f1d7aa8f2f84ac3f90206`. В нём epoch 60, optimizer/scheduler и 813,6 МБ модельных тензоров. Модель содержит только ConvNeXt-L + GLPE; ключей OT-GFAM/ONN нет. Встроенный config задаёт VeRi-776, 256×256 и 60 эпох, в отличие от 224×224 и 120 эпох в статье. После удаления training state inference-веса укладываются в лимит 2 ГБ.
- В Table 1 для VRU Rank-1 `99,12` выше Rank-5 `97,62`, что математически невозможно и указывает на ошибку таблицы.

## Что считать первыми исправлениями

1. Воспроизвести публично проверяемый GLPE: полный ConvNeXt-L с stride последнего stage = 1, шесть GeM, отдельные BNNeck, L2 и concat.
2. Отдельно заменить batch-hard на batch-all.
3. Отдельно исправить SW-gate на вход второго блока.
4. Отдельно проверить формулу без `+x`, потому что здесь статья противоречит собственному Discussion.
5. Не смешивать эти ablation с выбором датасета и lr; для каждого изменения нужен отдельный EXP и сравнение на одном `val_v2`.

Заявленные `90,10%` VehicleID без re-ranking и `90,27%` с k-reciprocal re-ranking нельзя сравнивать напрямую с LCT mAP@10: датасет, gallery/query protocol и метрика отличаются, а query-query re-ranking запрещён правилами ЛЦТ.
