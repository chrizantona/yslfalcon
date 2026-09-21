import { useState } from 'react';
import { api } from '../api';
import { Badge, Button, EmptyState, Icon, Loading, Message, PageHeading } from '../components/ui';
import { decisionLabels, formatDate, formatNumber, shortId } from '../domain';
import { usePagedResource } from '../hooks';

export function HistoryPage({ navigate }: { navigate: (path: string) => void }) {
  const searches = usePagedResource(api.searches);
  const [filter, setFilter] = useState('all');
  const [text, setText] = useState('');
  const items =
    searches.data?.items.filter(
      (item) =>
        (filter === 'all' || item.decision === filter) &&
        (!text || item.id.includes(text) || item.gallery_id.includes(text)),
    ) || [];
  return (
    <>
      <PageHeading
        eyebrow="ЖУРНАЛ"
        title="История поиска"
        description="Результаты, версии данных и заключения сохраняются вместе."
        action={
          <Button icon="refresh" onClick={searches.reload}>
            Обновить
          </Button>
        }
      />
      <section className="panel">
        <div className="table-toolbar">
          <div className="input-icon">
            <Icon name="search" size={17} />
            <input
              placeholder="ID поиска или галереи"
              aria-label="Найти в истории"
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
          </div>
          <select
            aria-label="Фильтр решений"
            value={filter}
            onChange={(event) => setFilter(event.target.value)}
          >
            <option value="all">Все решения</option>
            <option value="candidates">Кандидат найден</option>
            <option value="no_confident_match">Отказы</option>
            <option value="calibration_unavailable">Без калибровки</option>
          </select>
        </div>
        {searches.error && <Message tone="error">{searches.error}</Message>}
        {searches.loading && !searches.data ? (
          <Loading />
        ) : items.length ? (
          <div className="history-list">
            {items.map((result) => (
              <button
                className="history-row"
                key={result.id}
                onClick={() => navigate(`search/${result.id}`)}
              >
                <img src={result.query.crop_url} alt="Изображение запроса" loading="lazy" />
                <div className="history-main">
                  <strong>{formatDate(result.created_at)}</strong>
                  <code>{shortId(result.id)}</code>
                </div>
                <div className="history-decision">
                  <Badge tone={result.decision === 'candidates' ? 'success' : 'neutral'}>
                    {decisionLabels[result.decision] || result.decision}
                  </Badge>
                  <span>
                    {result.ranked_candidates.length} кандидатов · {result.reviews.length}{' '}
                    заключений
                  </span>
                </div>
                <div className="history-time">
                  {formatNumber(result.timings.total_ms, 1)} <span>мс</span>
                </div>
                <Icon name="chevron" size={18} />
              </button>
            ))}
          </div>
        ) : (
          <EmptyState
            icon="history"
            title={
              text || filter !== 'all'
                ? 'Таких результатов пока нет'
                : 'Каждый поиск останется здесь'
            }
            action={
              !text &&
              filter === 'all' && (
                <Button icon="search" onClick={() => navigate('search')}>
                  Перейти к поиску
                </Button>
              )
            }
          >
            {text || filter !== 'all'
              ? 'Измените строку поиска или фильтр решения.'
              : 'Откройте сохранённый результат, повторите расчёт на той же версии галереи или экспортируйте паспорт.'}
          </EmptyState>
        )}
        {searches.data && (
          <div className="pagination-row">
            <span>
              Загружено {searches.data.items.length} из {searches.data.total}
              {text || filter !== 'all' ? ' · фильтр применяется к загруженным результатам' : ''}
            </span>
            {searches.data.next_offset != null && (
              <Button busy={searches.loadingMore} onClick={searches.loadMore}>
                Загрузить ещё
              </Button>
            )}
          </div>
        )}
      </section>
    </>
  );
}
