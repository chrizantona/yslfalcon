import { Badge, CopyButton, Icon, Message, PageHeading, SectionTitle } from '../components/ui';
import { shortId } from '../domain';
import type { Session } from '../types';

export function IntegrationPage({ session }: { session: Session }) {
  const endpoint = `${window.location.origin}/api/v1`;
  const auth = `curl '${endpoint}/capabilities' \\\n  -H "Authorization: Bearer $FALCON_TOKEN"`;
  const search = `curl '${endpoint}/searches' \\\n  -H "Authorization: Bearer $FALCON_TOKEN" \\\n  -H 'Idempotency-Key: replace-with-unique-request-id' \\\n  -F 'image=@vehicle.jpg' \\\n  -F 'bbox=[10,20,300,180]' \\\n  -F 'gallery_id=YOUR_GALLERY_ID' \\\n  -F 'top_n=10'`;
  return (
    <>
      <PageHeading
        eyebrow="ПОДКЛЮЧЕНИЕ"
        title="Интеграция"
        description="Тот же поиск и те же результаты — из вашей системы."
        action={
          <a
            className="button button-default"
            href="/api/v1/openapi.json"
            target="_blank"
            rel="noreferrer"
          >
            <Icon name="code" size={17} />
            OpenAPI JSON
          </a>
        }
      />
      <section className="panel integration-endpoint">
        <span className="outline-icon">
          <Icon name="code" size={26} />
        </span>
        <div>
          <span className="field-label">Адрес API</span>
          <code>{endpoint}</code>
          <p>
            Пространство: {session.workspace || shortId(session.workspace_id)} · запросы выполняются
            с правами токена
          </p>
        </div>
        <CopyButton text={endpoint} />
      </section>
      <div className="two-column">
        <section className="panel code-panel">
          <SectionTitle
            title="1. Проверить подключение"
            caption="Сохраните токен в переменной FALCON_TOKEN на своей машине"
            action={<CopyButton text={auth} />}
          />
          <pre>{auth}</pre>
          <p>
            Используйте личный токен доступа. Не добавляйте его в URL, код приложения или историю
            репозитория.
          </p>
        </section>
        <section className="panel code-panel">
          <SectionTitle
            title="2. Выполнить поиск"
            caption="Активная галерея, изображение и рамка в пикселях"
            action={<CopyButton text={search} />}
          />
          <pre>{search}</pre>
          <p>Повтор того же запроса с тем же Idempotency-Key возвращает сохранённый результат.</p>
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
            [
              'POST',
              '/galleries/{id}/versions/{version}/activate',
              'Опубликовать подготовленную версию',
            ],
            ['POST', '/searches', 'Выполнить и сохранить поиск'],
            ['GET', '/searches/{id}', 'Получить результат и заключения'],
            ['POST', '/searches/{id}/replays', 'Повторить поиск в сохранённом контексте'],
            ['POST', '/searches/{id}/exports', 'Создать PDF, JSON или CSV'],
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
      <Message>
        Браузер использует защищённую сессию и CSRF-токен. Внешние клиенты — заголовок
        Authorization. Загруженные изображения и результаты доступны только в пределах вашего
        рабочего пространства.
      </Message>
    </>
  );
}
