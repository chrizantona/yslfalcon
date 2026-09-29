import { useEffect } from 'react';
import { api } from '../api';
import {
  Badge,
  Button,
  EmptyState,
  Icon,
  JobProgress,
  Loading,
  Message,
  PageHeading,
  SectionTitle,
} from '../components/ui';
import { formatNumber, statusLabels } from '../domain';
import { usePagedResource, useResource } from '../hooks';

export function SystemPage() {
  const system = useResource(api.system);
  const jobs = usePagedResource(api.jobs);
  useEffect(() => {
    const timer = window.setInterval(() => {
      system.reload();
    }, 15000);
    return () => window.clearInterval(timer);
  }, [system.reload]);
  const refresh = () => {
    system.reload();
    jobs.reload();
  };
  return (
    <>
      <PageHeading
        title="Состояние сервиса"
        description="Доступность модулей, реальные задержки и фоновые операции."
        action={
          <Button icon="refresh" onClick={refresh} busy={system.loading}>
            Обновить
          </Button>
        }
      />
      {system.error && <Message tone="error">{system.error}</Message>}
      {system.loading && !system.data ? (
        <Loading />
      ) : (
        system.data && (
          <>
            <section className="panel">
              <SectionTitle
                title="Модули сервиса"
                caption="Состояние обновляется каждые 15 секунд"
                action={
                  <Badge
                    tone={
                      ['ok', 'healthy', 'ready'].includes(system.data.status)
                        ? 'success'
                        : 'warning'
                    }
                    dot
                  >
                    {statusLabels[system.data.status] || system.data.status}
                  </Badge>
                }
              />
              <div className="dependency-list">
                {Object.entries(system.data.dependencies).map(([key, dependency]) => (
                  <div key={key}>
                    <span className="dependency-icon">
                      <Icon
                        name={key === 'database' ? 'layers' : key === 'model' ? 'target' : 'shield'}
                        size={21}
                      />
                    </span>
                    <div>
                      <strong>
                        {{
                          database: 'База данных',
                          storage: 'Хранилище изображений',
                          model: 'Вычислительный модуль',
                          queue: 'Очередь заданий',
                        }[key] || key}
                      </strong>
                      <p>
                        {dependency.mode === 'preview'
                          ? 'Техническая заглушка · модель не подключена'
                          : dependency.model_id || 'Проверка доступности'}
                      </p>
                    </div>
                    <Badge
                      tone={
                        ['ok', 'ready', 'healthy', 'available'].includes(dependency.status)
                          ? 'success'
                          : 'warning'
                      }
                      dot
                    >
                      {statusLabels[dependency.status] || dependency.status}
                    </Badge>
                  </div>
                ))}
              </div>
            </section>
            <div className="two-column">
              <section className="panel">
                <SectionTitle title="Проход модели" caption="Измерения по выполненным поискам" />
                {system.data.timings.sample_count ? (
                  <div className="timing-stats">
                    <div>
                      <span>Медиана · p50</span>
                      <strong>
                        {formatNumber(system.data.timings.model_ms?.p50, 1)}
                        <small>мс</small>
                      </strong>
                    </div>
                    <div>
                      <span>95-й перцентиль</span>
                      <strong>
                        {formatNumber(system.data.timings.model_ms?.p95, 1)}
                        <small>мс</small>
                      </strong>
                    </div>
                    <p>
                      Только прогон модели на GPU — без приёма и подготовки кадра, поиска по галерее
                      и передачи по сети.
                    </p>
                  </div>
                ) : (
                  <div className="inline-empty">
                    <Icon name="activity" size={25} />
                    <p>После первых поисков здесь появятся измеренные задержки.</p>
                  </div>
                )}
              </section>
              <section className="panel">
                <SectionTitle title="Очередь заданий" caption="Подготовка данных и экспорт" />
                <dl className="queue-summary">
                  {['pending', 'running', 'completed', 'failed'].map((status) => (
                    <div key={status}>
                      <dt>{statusLabels[status]}</dt>
                      <dd>{system.data?.jobs[status] || 0}</dd>
                    </div>
                  ))}
                </dl>
              </section>
            </div>
          </>
        )
      )}
      <section className="panel">
        <SectionTitle title="Последние задания" />
        {jobs.error && <Message tone="error">{jobs.error}</Message>}
        {jobs.loading && !jobs.data ? (
          <Loading />
        ) : jobs.data?.items.length ? (
          <div className="jobs-list">
            {jobs.data.items.map((job) => (
              <JobProgress
                key={`${job.id}-${job.status}`}
                initial={job}
                onComplete={system.reload}
              />
            ))}
          </div>
        ) : (
          <EmptyState icon="activity" title="Очередь свободна">
            Задания появятся после импорта галереи или экспорта результата.
          </EmptyState>
        )}
        {jobs.data && (
          <div className="pagination-row">
            <span>
              Заданий: {jobs.data.items.length} из {jobs.data.total}
            </span>
            {jobs.data.next_offset != null && (
              <Button busy={jobs.loadingMore} onClick={jobs.loadMore}>
                Загрузить ещё
              </Button>
            )}
          </div>
        )}
      </section>
    </>
  );
}
