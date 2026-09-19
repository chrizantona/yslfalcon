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
| `kaggle/veri776.zip` | Kaggle [`abhyudaya12/veri-vehicle-re-identification-dataset`](https://www.kaggle.com/datasets/abhyudaya12/veri-vehicle-re-identification-dataset), версия 1 | VeRi-776, Liu et al., ECCV 2016, [github.com/VehicleReId/VeRi](https://github.com/VehicleReId/VeRi); CC BY-NC 4.0 по странице Kaggle | 992 234 678 байт, `9d45b9ad779d04d58c14abc09a0f681427145464c0aa9faa3ac19b3fea4d0694` | EXP-033 (претрейн вместе с VERI-Wild test: `image_train` и `image_test`, без повторяющего их `image_query`) → EXP-034 |
| `vehicleid/VehicleID_V1.0.zip` | PKU IDM, выдаётся авторами по запросу (личная ссылка и пароли не публикуются) | VehicleID, Liu et al., CVPR 2016, [pkuml.org/resources/pku-vehicleid.html](https://www.pkuml.org/resources/pku-vehicleid.html); исследовательское использование по соглашению; номера закрыты чёрными прямоугольниками, на части снимков впечатан текст камеры | 7 568 990 734 байта, `75a3c8ba0b1cfb4ac4a088c4d0c3c0fceb0ae45abc17a15513e9915531739952`; zip зашифрован (ZipCrypto), 222 628 снимков, из них 221 567 с меткой машины, меток камер нет | EXP-038 (претрейн ViT-B/16 вместе с VERI-Wild test) → EXP-039. Распаковка — `scripts/prepare_vehicleid.sh` в `/dev/shm/ext/vehicleid` официальным статическим 7-Zip 26.03 (`7z2603-linux-x64.tar.xz` из [github.com/ip7z/7zip](https://github.com/ip7z/7zip/releases/tag/26.03), sha256 `dc99eff5008f1ab79bd7084c68513701547a808a89502bf4133683535ab3c695`); пароль — в файле вне git |

Kaggle-копии сверены с официальной статистикой: VERI-Wild test — 10 000 машин, 128 517 снимков галереи + 10 000 запросов; VeRi-776 — 37 778 / 11 579 / 1 678 снимков.

## Публичные веса
| Файл | Источник | Лицензия | sha256 | Где используется |
|---|---|---|---|---|
| CLIP ViT-B/16 `laion2b_s34b_b88k` | open_clip 3.3.0 скачивает `open_clip_model.safetensors` из [HF `laion/CLIP-ViT-B-16-laion2B-s34B-b88K`](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K), ревизия `7288da5a0d6f0b51c4a2b27c624837a9236d0112` | MIT | `3f25d29d3cc74e1d25d47e0593b4dd0864ced1e9b4d8e486a247ca4502f227f1` | ViT-B-эксперименты EXP-000…010, 015, 016, 019, EXP-038, EXP-039 |
| CLIP ViT-L/14 `laion2b_s32b_b82k`, `weights/clip/exp020.safetensors` | [LAION/OpenCLIP, Ross Wightman](https://huggingface.co/laion/CLIP-ViT-L-14-laion2B-s32B-b82K), `open_clip_pytorch_model.safetensors`, ревизия `1627032197142fbe2a7cfec626f4ced3ae60d07a` | MIT | `7d129ed747e0ed53e82dfcc140382b51be66b56e6a9bdc3258afd2846e3bb019` | EXP-020…029, 033, 034, 036, 037 |
| CLIP ViT-L/14 DFN-2B (`ViT-L-14-quickgelu`, `dfn2b`), `weights/clip/dfn2b_vitl14.bin` | [HF `apple/DFN2B-CLIP-ViT-L-14`](https://huggingface.co/apple/DFN2B-CLIP-ViT-L-14), `open_clip_pytorch_model.bin`, ревизия `13d68e40771108ec1812f566404e8c6f7de85f67`, Apple | apple-amlr | `c02c84198ede77ad80df0803224022f1e9e95ce42551baa91a1b5d87e229e2bb` | EXP-030, EXP-044…047 |
| EVA02-L/14 CLIP (`EVA02-L-14`, `merged2b_s4b_b131k`), `weights/clip/eva02_l14_merged2b.safetensors` | [HF `timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k`](https://huggingface.co/timm/eva02_large_patch14_clip_224.merged2b_s4b_b131k), `open_clip_model.safetensors`, ревизия `bf4190eb65dd5204ffb03e980108beb1200e0873`, BAAI (EVA) / timm | MIT | `00af04296f09f24dcc69559440a80b7a44daf4855a72827e016067e6e571b851` | EXP-031 |
| SigLIP 2 So400m/14 (`ViT-SO400M-14-SigLIP2`, `webli`), `weights/clip/siglip2_so400m14_webli.safetensors` | [HF `timm/ViT-SO400M-14-SigLIP2`](https://huggingface.co/timm/ViT-SO400M-14-SigLIP2), `open_clip_model.safetensors`, ревизия `f9e2bfdc4807cd0fbadf67301e45193d6f330c20`, Google / timm | Apache-2.0 | `bd5884118e72b5cc24c37cff23b8d202e2c3dac723b8a150901b022785235626` | EXP-032, EXP-040…042 |
| CLIP ConvNeXt-L-D 320 (`convnext_large_d_320`, `laion2b_s29b_b131k_ft_soup`), `weights/clip/convnext_large_d_320_ft_soup.safetensors` | [HF `laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup`](https://huggingface.co/laion/CLIP-convnext_large_d_320.laion2B-s29B-b131K-ft-soup), `open_clip_model.safetensors`, ревизия `654d0f80ff73c58e7281a3ca7dc425589049e2e1`, LAION | MIT | `1a8e47bdefd6d36202ec66a8975edda9e3c89560bcd1ecefe6f676311637a671` | EXP-048 |
| timm `resnet50.a1_in1k` | [HF `timm/resnet50.a1_in1k`](https://huggingface.co/timm/resnet50.a1_in1k), `model.safetensors`, ревизия `767268603ca0cb0bfe326fa87277f19c419566ef` | Apache-2.0 | `773525d5821de224f8f30c33377b7a795d7863e08522698200d3217d3f2a41bb` | EXP-002 |
| `weights/fast-reid/veriwild_bot_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veriwild_bot_R50-ibn.pth), JDAI-CV | Apache-2.0 (обучен на VERI-Wild train) | `f591e4f7f31ce0be2f4079171cfc03fcbecab5ea5b40f9148c753b9c6f061e8e` | EXP-011, EXP-017, EXP-018 |
| `weights/fast-reid/vehicleid_bot_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/vehicleid_bot_R50-ibn.pth) | Apache-2.0 (обучен на VehicleID) | `48992dbfdb99b93c9a0c2f15dd27b7a738f8b8a7e49dd231391ed211f88d759a` | EXP-012 |
| `weights/fast-reid/veri_sbs_R50-ibn.pth` | [FastReID release v0.1.1](https://github.com/JDAI-CV/fast-reid/releases/download/v0.1.1/veri_sbs_R50-ibn.pth) | Apache-2.0 (обучен на VeRi-776) | `57fb9c17d88911ea64390bf5427f43511435e7f88f6eed9dbc969d4b611e53cd` | пока нигде |
| `weights/clip-reid/VehicleID_clipreid_ViT-B-16_60.pth` | Google Drive `168BLegHHxNqatW5wx1YyL2REaThWoof5` из README [github.com/Syliz517/CLIP-ReID](https://github.com/Syliz517/CLIP-ReID) | MIT (обучен на VehicleID) | `538e8288741b45deb1dd8225e47b57fc0702fde79156ca673599e53c1905b27b` | EXP-013 |
| `weights/clip-reid/VeRi_clipreid_ViT-B-16_60.pth` | Google Drive `1RyfHdOBI2pan_wIGSim5-l6cM4S2WN8e` из README CLIP-ReID | MIT (обучен на VeRi-776) | `71f6f2723be4d563f1df201133daa53f57bee2a67266486559d5ce453a60f805` | EXP-014 |
| ResNeXt101-IBN-a, ImageNet, `weights/paper/resnext101_ibn_a-6ace051d.pth` | [IBN-Net release v1.0](https://github.com/XingangPan/IBN-Net/releases/download/v1.0/resnext101_ibn_a-6ace051d.pth), Xingang Pan; код архитектуры — `src/reid/ibn_resnext.py` из коммита `d1673389b36c1180cf9bc35ea8260d84046da915` | MIT | `6ace051dc2d2de2aace20cc327d9f2f64892696eaf13d8b1bbf0accf45684ac8` | EXP-055, EXP-056 |
| TinyCLIP ViT-61M/32 (LAION-400M), `weights/paper/tinyclip_vit61m_32_laion400m.safetensors` и `…_config.json` | [HF `wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M`](https://huggingface.co/wkcn/TinyCLIP-ViT-61M-32-Text-29M-LAION400M), `model.safetensors`, ревизия `94cbbea5c7949cfe7bdafde64bcea5e403f59852`, Microsoft (TinyCLIP); берётся только визуальная часть | MIT | `0c0670dfa6fbf32c3158ef57b10772d5c9668093bda676d78e1452268585d230`; config `40a59f3db33a1df844687e94f25b186bf187bec1d8d1389f0f88b1ab013e233b` | EXP-055, EXP-056 |
| TinyCLIP ViT-40M/32 (LAION-400M), `weights/paper/tinyclip_vit40m_32_laion400m.safetensors` и `…_config.json` | [HF `wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M`](https://huggingface.co/wkcn/TinyCLIP-ViT-40M-32-Text-19M-LAION400M), ревизия `95ec8197b3f2fe7f747865c61ca556cf0768b2f7`, Microsoft (TinyCLIP); активная ViT-B/32-подобная модель в [официальном inference-примере](https://github.com/microsoft/Cream/blob/main/TinyCLIP/inference.py), берётся только визуальная часть | MIT | `1b08131ac0ac54429aab216c397c550527d31f5682afb5d21d35091ad39538ae`; config `b786ac49b5cdcb16496a33c8ab4884de036b2f64de27d0b73caf8391f94c9bee` | EXP-093 |
| ConvNeXt-L ImageNet-22k (`convnext_large.fb_in22k`), `weights/paper/convnext_large_fb_in22k.safetensors` | [HF `timm/convnext_large.fb_in22k`](https://huggingface.co/timm/convnext_large.fb_in22k), `model.safetensors`, ревизия `be6de86d9928c6cc2a8986963243142f403f382d`, Meta / timm | Apache-2.0 | `7e02ea57d6024d94f5523e9b58cdddd81a1fb4f2e5128b4a879b76f9847058bf` | EXP-057, EXP-058 |

## OSNet-AIN для EXP-065…069

- Автор: **Kaiyang Zhou**, OSNet-AIN x1.0, лицензия MIT.
- Код: [deep-person-reid/osnet_ain.py](https://github.com/KaiyangZhou/deep-person-reid/blob/f6ba78eceb38712266318e8c7eefde174af15606/torchreid/models/osnet_ain.py),
  ревизия `f6ba78eceb38712266318e8c7eefde174af15606`. Неизменённая архитектура и лицензия
  сохранены в `src/reid/vendor/`; используется feature map до авторских FC/classifier.
- Веса: **ImageNet**, [HF kaiyangzhou/osnet](https://huggingface.co/kaiyangzhou/osnet),
  ревизия `a5c5cc037c24235cda3b21085b93ad77c9616224`,
  [закреплённый URL osnet_ain_x1_0_imagenet.pth](https://huggingface.co/kaiyangzhou/osnet/resolve/a5c5cc037c24235cda3b21085b93ad77c9616224/osnet_ain_x1_0_imagenet.pth),
  10 929 757 байт, sha256 `b8a578eacc122c16496c8835c96a0157db62963b3de05b5a5a4b1ff5bad7546a`.
- Загрузка: `bash scripts/fetch_external.sh osnet`, SHA256 проверяется до загрузки модели.
  Локальный путь `/workspace/ext_data/weights/osnet/osnet_ain_x1_0_imagenet.pth`.
- Использование: CNN-ветвь EXP-067/068/069; отдельная ImageNet-нормализация.
  Специализированные на людях MSMT17/Market1501-веса не используются.
- Семантический старт EXP-065/066/068/069: наш обученный суп EXP-054
  `95aa27ed552c43a1ad9c3d853390fa18.pt`, sha256
  `9be056d0c85cd8a69019d25f29abdf45f93528043ec35963f063b510c85c1f12` (карточка EXP-054).
  Проверенный источник: [приватный Release exp-054](https://github.com/chrizantona/yslfalcon/releases/tag/exp-054), asset API `https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/573190938`, 1 715 118 617 байт. AP/эмбеддинги — `EXP-054_results.tar.gz`, SHA256 `2eb8d0d1b5837571e5165a8a8e5e7c65ce016f3127ee9944768658e1802b0693`.
  Нужны именно эти веса и конфиг внутри checkpoint; публичный SigLIP не заменяет этот старт.
  Новые внешние тренировочные данные в этой серии не добавляются.

## Наши веса
Чекпоинты лежат на поде в `/workspace/checkpoints/`, имена записаны в карточках и реестрах; в git их нет. Ранние эксперименты используют случайные имена, EXP-020…029, 036, 037 — имена по номеру EXP по просьбе пользователя, EXP-030…034 (второй под) — снова случайные; имя претрейна EXP-033 задано заранее в его конфиге, потому что с него стартует EXP-034. Старые эксперименты воспроизводятся обучением с тега (`git checkout exp-NNN`, команда из карточки); для ночной цепочки фактические код и конфиги с новыми именами сохранены в `artifacts/EXP-NNN/pod/`, переименование закоммичено в `75f0c12`. Цепочка LAION-ветки — EXP-022 → EXP-023 → EXP-027 → EXP-028 → EXP-037; EXP-027 отдельно отклонён, но дочерний imprinting EXP-028 также принят против исходного EXP-023. Предыдущий суп — EXP-024; историческая — EXP-009 → EXP-010 → EXP-016. Числа совпадут с точностью до недетерминизма обучения на GPU; инференс детерминирован (`reid.benchmark`).

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

Оба обученных файла основаны на публичном LAION/OpenCLIP и VERI-Wild, источники, авторы и версии — в таблицах выше. Для новых веток не требуются новые внешние данные или публичные веса. Имена EXP по прямому указанию пользователя имеют приоритет над прежними случайными именами. Предыдущий суп exp024.pt доступен в Release exp-024. Текущий Docker использует обученный exp037.pt, BuildKit secret для приватного asset API и закреплённый SHA256; образ собран с fdfad69, Linux amd64; загрузка encoder+BNNeck и forward проверены без сети на CPU. NVIDIA runtime контейнера не проверен. RTX A5000 отложена пользователем.

## Серия EXP-025…029, EXP-036 и EXP-037

Новых внешних данных или публичных весов нет: тот же публичный LAION/OpenCLIP и тот же VERI-Wild test, источники и SHA256 выше. Наши новые чекпоинты именуются по EXP. Текущий суп `exp037.pt`: автор — этот проект, равномерное среднее `exp028_seed0.pt`, `exp028_seed1.pt`, `exp028_seed2.pt`; SHA256 `ed735cad60ba42f79e2e2b933fd5565256da9327de7018155a0be22642ce3a29`, размер 1 218 807 919 байт. Инференс использует один обученный visual encoder с BNNeck, без ArcFace-головы, без TTA и ансамбля. Dev-фолд 0, обучение 1–3; фолд 4 нетронут.

Источник нашего обученного файла и результатов — [приватный Release exp-037](https://github.com/chrizantona/yslfalcon/releases/tag/exp-037), автор `chrizantona/yslfalcon`, версия `exp-037`. Закреплённый URL весов для Docker: `https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/570071465`. SHA256 файла указан выше и подтверждён digest GitHub; Release также содержит SHA256SUMS / manifest.json для всех архивов и сабмитов.

Результаты EXP-025…029/036/037 и сабмит EXP-037 скачаны; SHA256 архивов и файлов проверены. Папки завершённых запусков и их собственное окружение на поде удалены 17.09.2026 в 13:53 МСК после проверки отсутствия активных процессов. Все чекпоинты и данные сохранены.

## Серия третьего пода EXP-044…048

DFN-2B и ConvNeXt-L-D взяты из закреплённых публичных источников выше через fetch_external.sh с проверкой SHA256; новых внешних источников нет. Четыре обучения выполнены точным кодом `7511a8e`, конфиги неизменны; суп EXP-047 — тег `exp-047`, `f768eda`. Обучение на фолдах 1–3, оценка val_v2/dev-фолд 0; фолд 4 нетронут.

[Приватный Release exp-047](https://github.com/chrizantona/yslfalcon/releases/tag/exp-047) содержит пять архивов EXP-044/045/046/047/048, все 24 query/gallery массива, AP по запросам, метрики, отказ, срезы, benchmark, логи и исходные снимки. Для каждой копии проверены SHA256 архива и файлов после распаковки (1359 файлов); embeddings_index.json фиксирует порядок строк, dtype, форму, размер и суммы всех массивов. Post-hoc bootstrap EXP-046/048 — также в analysis.json. [Инструкция загрузки](experiments/HANDOFF_EXP-044-048.md).

Наш обученный суп DFN-2B `398a6b20e0344621991d19cf5dc7f743.pt`: автор `chrizantona/yslfalcon`, версия `exp-047`; равномерное среднее трёх сидов EXP-045 после VERI-Wild, линейная голова. SHA256 `98b60ea55cc7cfd7627669a8969746178c97999621e471cfe9cff4965a80d035`, размер 1218822583 байт. [Копия чекпоинта](https://github.com/chrizantona/yslfalcon/releases/download/exp-047/398a6b20e0344621991d19cf5dc7f743.pt). Один encoder с BNNeck на инференсе, без ансамбля/TTA; файл хранится FP32, вычисления проверены FP32/FP16. Прирост относительно EXP-045 inconclusive, относительно EXP-037 подтверждён; общий лучший кандидат остаётся EXP-043. Исходные чекпоинты всех новых экспериментов сохранены на поде без переименования.

GitHub подтвердил размер и SHA256 копии супа EXP-047; закреплённый asset API: `https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/571284418`. Квитанции всех 14 assets и индекс эмбеддингов — [HANDOFF_EXP-044-048.json](experiments/HANDOFF_EXP-044-048.json).

Уборка выполнена 18.09.2026 в 01:33 МСК после проверки локальных архивов и всех 14 GitHub assets. Наших процессов обучения, оценки, очередей и мониторинга на поде нет. Папки EXP-044/045/046/047/048, queue_b, /tmp/EXP-044, /dev/shm/EXP-044 и лишние временные архивы удалены. Все 11 новых чекпоинтов, датасеты, публичные веса и сторонние процессы сохранены; квитанция — artifacts/handoff_night_044_048/cleanup_receipt.json и сводка в HANDOFF_EXP-044-048.json.

Наш обученный суп SigLIP 2 `95aa27ed552c43a1ad9c3d853390fa18.pt` (EXP-054, текущий лучший кандидат): равномерное среднее трёх сидов EXP-053 (SigLIP 2 So400m/14 после VERI-Wild, ArcFace с imprinting, SupCon); полный state_dict ReIDModel, инференс берёт бэкбон и BNNeck. 1 715 118 617 байт, SHA256 `9be056d0c85cd8a69019d25f29abdf45f93528043ec35963f063b510c85c1f12` — совпадает с копией на поде. [Приватный Release exp-054](https://github.com/chrizantona/yslfalcon/releases/tag/exp-054), закреплённый asset API: `https://api.github.com/repos/chrizantona/yslfalcon/releases/assets/573190938`; там же архивы EXP-053 и EXP-054 (AP, эмбеддинги, срезы, bootstrap, логи) и `SHA256SUMS`.
