# Где разместить Falcon на две недели

Дата проверки: 23 сентября 2026. Горизонт расчёта: 14 суток = 336 часов. Цены — ориентир на дату
проверки; диск, публичный IP, router, backup, налоги/конвертация и исходящий трафик считаются отдельно,
если провайдер не включил их в конфигурацию.

## Решение

Для финального живого прототипа рекомендован **один Selectel Cloud Server с NVIDIA A5000 24 ГБ,
8 vCPU и 32 ГБ RAM**. Это обычная VM с root, публичным IP и готовым Ubuntu GPU + Docker образом:
существующий Compose разворачивается без перекройки, web/API/worker/runtime/PostgreSQL остаются на
одном узле, а GPU совпадает с моделью конкурсного стенда. Текущий калькулятор показывает
**80,05 ₽/ч**, то есть **26 896,80 ₽ за 14 суток** плюс сеть/диск/backup. Более дешёвый L4 24 ГБ —
69,52 ₽/ч или 23 358,72 ₽ за тот же период, но A5000 предпочтительнее для репетиции поведения на
целевом классе GPU.

Источники: [калькулятор Selectel](https://selectel.ru/services/cloud/computing-power/),
[список и характеристики GPU](https://docs.selectel.ru/cloud-servers/create/gpus/),
[Docker на GPU VM](https://docs.selectel.ru/cloud-servers/manage/run-cuda-docker-containers/),
[публичный доступ и фильтрация](https://docs.selectel.ru/cloud-servers/cloud-networks/configure-access-to-internet/).

## Сравнение вариантов

| Вариант | 14 дней GPU | Насколько подходит | Вердикт |
|---|---:|---|---|
| Selectel A5000 VM | 26 896,80 ₽ + extras | Полная VM, Compose, A5000, российская оплата/регион, публичный IP | **Финальный стенд** |
| Selectel L4 VM | 23 358,72 ₽ + extras | Та же эксплуатация, чуть дешевле; другой GPU | Резерв при отсутствии A5000 |
| RunPod Secure A5000 Pod | $90,72 + storage | Очень дёшево, custom container и HTTPS proxy, но Pod — не естественная VM для пяти контейнеров | GPU runtime/rehearsal |
| Timeweb Cloud GPU | Цена видна только при заказе | Root VM, NVMe, почасовая оплата; A5000/4090 заявлены | Запросить quote/наличие |
| Vast.ai | Динамический marketplace | Дёшево, но неоднородные хосты/сеть/доступность | Не для защиты |
| Yandex Cloud GPU | По калькулятору и квоте | T4/V100/A100; GPU-квота по умолчанию нулевая | Избыточно и медленнее старт |
| Cloud.ru | Существенно дороже | Enterprise GPU, но конфигурации крупнее задачи | Не брать на 2 недели |

RunPod публикует A5000 Secure Cloud по **$0,27/ч** и volume disk по $0,10/GB/month. Pod умеет
custom image, persistent volume и HTTP-порт вида `https://<pod>-<port>.proxy.runpod.net`, но при
остановке GPU освобождается и на старте его может уже не быть. Поэтому RunPod хорош для отдельного
runtime или дешёвой репетиции, но рискован как единственная сцена защиты.

Источники: [RunPod pricing](https://www.runpod.io/pricing),
[Pod/custom image/ports/volume](https://docs.runpod.io/runpodctl/reference/runpodctl-pod),
[риск потери GPU после stop](https://docs.runpod.io/pods/troubleshooting/zero-gpus).

Timeweb подтверждает root-доступ, GPU-линейку, NVMe и почасовую оплату, но публичная страница не даёт
проверяемой цены конкретной A5000-конфигурации; перед выбором нужен письменный quote и подтверждение
наличия на нужные даты. Источник: [Timeweb GPU](https://timeweb.cloud/services/gpu).

## Рекомендуемая схема Selectel

```text
Internet :443
    │
    ▼
Caddy/Nginx на VM ──► 127.0.0.1:8090 (Falcon web)
                           │
             internal Compose network
             api ─ runtime:NVIDIA A5000
              │       │
           worker   /models:ro
              │
          PostgreSQL + artifact volume
```

- Ubuntu 24.04 GPU Driver 580 Docker или другой provider image, совместимый с CUDA runtime модели;
- не менее 100 ГБ диска: образы, bundle, PostgreSQL, originals/exports и запас на rollback;
- наружу только 443; SSH 22 только с IP команды; 8090, 8000, 8001 и 5432 не публиковать;
- домен + Caddy/Let's Encrypt, `FALCON_COOKIE_SECURE=true`, точный HTTPS origin;
- ежедневный dump PostgreSQL и копия artifact volume во внешнее хранилище;
- один runtime image digest, bundle manifest и сохранённый предыдущий релиз;
- A5000 держать запущенной непрерывно минимум за 48 часов до защиты. Selectel тарифицирует созданную
  VM и в `SHUTOFF`, поэтому остановка не экономит compute. RunPod после stop может не вернуть GPU.

## Блокер до публикации

Текущий browser auto-login специально принимает только `localhost`; удалённой формы входа в UI нет.
На публичном домене нельзя включить общий admin auto-login или переписывать Origin/Host в proxy — это
ломает защитную модель сервиса.

Перед интернет-развёртыванием нужен небольшой отдельный этап:

1. вернуть минимальный token-login для удалённого профиля или выпустить ограниченного demo-user;
2. оставить `FALCON_LOCAL_AUTO_LOGIN=false`;
3. задать `FALCON_COOKIE_SECURE=true` и единственный HTTPS origin;
4. ограничить demo workspace и подготовить очистку/retention;
5. пройти реальную браузерную приёмку и внешний smoke по домену.

До этого безопасный вариант — только закрытая репетиция через SSH tunnel/VPN, но не публичная ссылка
для жюри.

## План на две недели

- **День −3:** запросить/проверить квоту и фактическую доступность A5000; поднять VM и DNS.
- **День −2:** установить release, восстановить только demo-данные, проверить GPU parity и offline.
- **День −1:** полный smoke, browser QA, snapshot/backup, репетиция с другого интернета.
- **Дни 1–10:** healthcheck и ежедневный backup; изменения только через новый immutable release.
- **Дни 11–14:** freeze, оставить GPU включённой, иметь локальную запись и offline-стенд как fallback.
- **После:** скачать БД/артефакты, проверить backup, удалить VM/IP/disks и отозвать токены.
