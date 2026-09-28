import { Badge, CopyButton, Icon, PageHeading, SectionTitle } from '../components/ui';

export function IntegrationPage() {
  const endpoint = `${window.location.origin}/api/v1`;
  const health = `curl '${endpoint}/health/ready'`;
  const search = JSON.stringify(
    {
      image: 'vehicle.jpg',
      bbox: [10, 20, 300, 180],
      gallery_id: 'YOUR_GALLERY_ID',
      top_n: 10,
    },
    null,
    2,
  );
  return (
    <>
      <PageHeading
        title="Интеграция"
        description="Тот же поиск и те же результаты — из вашей системы."
        action={
          <div className="inline-items">
            <a className="button button-default" href="/docs" target="_blank" rel="noreferrer">
              Описание API
            </a>
            <a
              className="button button-default"
              href="/api/v1/openapi.json"
              target="_blank"
              rel="noreferrer"
            >
              <Icon name="code" size={17} />
              OpenAPI JSON
            </a>
          </div>
        }
      />
      <section className="panel integration-endpoint">
        <span className="outline-icon">
          <Icon name="code" size={26} />
        </span>
        <div>
          <span className="field-label">Адрес API</span>
          <code>{endpoint}</code>
          <p>Форматы запросов, ответы и примеры для вашей системы.</p>
        </div>
        <CopyButton text={endpoint} />
      </section>
      <div className="two-column">
        <section className="panel code-panel">
          <SectionTitle
            title="1. Проверить подключение"
            caption="Готовность сервиса к обработке запросов"
            action={<CopyButton text={health} />}
          />
          <pre>{health}</pre>
          <p>Проверка показывает, доступны ли база данных, хранилище и вычислительный модуль.</p>
        </section>
        <section className="panel code-panel">
          <SectionTitle
            title="2. Передать изображение"
            caption="Поля multipart-запроса POST /searches"
            action={<CopyButton text={search} />}
          />
          <pre>{search}</pre>
          <p>
            Файл передаётся полем image; bbox — рамка в пикселях. Полный пример вызова находится в
            описании API. Повтор с тем же Idempotency-Key возвращает сохранённый результат.
          </p>
        </section>
      </div>
      <section className="panel">
        <SectionTitle
          title="Основные операции"
          caption="Полный контракт и структуры ошибок доступны в OpenAPI"
        />
        <div className="endpoint-list">
          {[
            ['GET', '/capabilities', 'Возможности и режим вычислительного модуля'],
            ['POST', '/galleries', 'Создать галерею наблюдений'],
            ['POST', '/galleries/{id}/imports', 'Подготовить версию из изображений'],
            ['POST', '/galleries/{id}/import-sessions', 'Начать загрузку полной галереи по CSV'],
            [
              'GET',
              '/galleries/{id}/import-sessions/{session}',
              'Проверить прогресс и недостающие файлы',
            ],
            [
              'POST',
              '/galleries/{id}/import-sessions/{session}/files',
              'Передать часть изображений',
            ],
            [
              'POST',
              '/galleries/{id}/import-sessions/{session}/finalize',
              'Подготовить одну полную версию',
            ],
            [
              'POST',
              '/galleries/{id}/versions/{version}/activate',
              'Опубликовать подготовленную версию',
            ],
            ['POST', '/searches', 'Выполнить и сохранить поиск'],
            ['GET', '/searches/{id}', 'Получить результат и заключения'],
            ['POST', '/searches/{id}/replays', 'Повторить поиск в сохранённом контексте'],
            ['POST', '/searches/{id}/exports', 'Создать PDF, JSON, CSV или offline-дело ZIP'],
            ['GET', '/jobs/{id}', 'Проверить статус фоновой операции'],
          ].map(([method, path, description]) => (
            <div key={path}>
              <Badge tone={method === 'POST' ? 'purple' : 'neutral'}>{method}</Badge>
              <code>{path}</code>
              <span>{description}</span>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
