import { useEffect, useRef, useState } from 'react';
import { api } from '../api';
import { CropEditor, type ImageInput } from '../components/CropEditor';
import { Comparison } from '../components/Comparison';
import { ResultsSurface } from '../components/ResultsSurface';
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
import {
  decisionLabels,
  createOperationKey,
  errorMessage,
  formatDate,
  formatNumber,
  shortId,
  validBBox,
} from '../domain';
import { usePagedResource } from '../hooks';
import type { Candidate, Capabilities, Job, SearchResult, Session } from '../types';

export function SearchPage({
  searchId,
  capabilities,
  session,
  navigate,
}: {
  searchId?: string;
  capabilities: Capabilities | null;
  session: Session;
  navigate: (path: string) => void;
}) {
  const galleries = usePagedResource(api.galleries);
  const [galleryId, setGalleryId] = useState('');
  const [image, setImage] = useState<ImageInput | null>(null);
  const [topN, setTopN] = useState(10);
  const [result, setResult] = useState<SearchResult | null>(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [searching, setSearching] = useState(false);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [exportJob, setExportJob] = useState<Job | null>(null);
  const [exporting, setExporting] = useState(false);
  const [exportFormat, setExportFormat] = useState<'pdf' | 'json' | 'csv'>('pdf');
  const [replaying, setReplaying] = useState(false);
  const pendingSearch = useRef<{ signature: string; key: string } | null>(null);
  const activeGalleries =
    galleries.data?.items.filter((gallery) => gallery.active_version_id) || [];
  useEffect(() => {
    if (!galleryId && activeGalleries.length) setGalleryId(activeGalleries[0].id);
  }, [galleries.data, galleryId]);
  useEffect(() => {
    let alive = true;
    setError('');
    setSelected(null);
    setExportJob(null);
    if (!searchId) {
      setResult(null);
      setLoadingResult(false);
      return;
    }
    setLoadingResult(true);
    api
      .searchResult(searchId)
      .then((value) => {
        if (alive) setResult(value);
      })
      .catch((cause) => {
        if (alive) {
          setResult(null);
          setError(errorMessage(cause));
        }
      })
      .finally(() => {
        if (alive) setLoadingResult(false);
      });
    return () => {
      alive = false;
    };
  }, [searchId]);
  const search = async () => {
    if (!image || !galleryId) return;
    setSearching(true);
    setError('');
    const signature = JSON.stringify([image.url, image.bbox, galleryId, topN]);
    if (pendingSearch.current?.signature !== signature)
      pendingSearch.current = { signature, key: createOperationKey() };
    try {
      const saved = await api.search(
        image.file,
        image.bbox,
        galleryId,
        topN,
        pendingSearch.current.key,
      );
      pendingSearch.current = null;
      setImage(null);
      setResult(saved);
      navigate(`search/${saved.id}`);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSearching(false);
    }
  };
  const replay = async () => {
    if (!result) return;
    setReplaying(true);
    setError('');
    try {
      const saved = await api.replay(result.id);
      navigate(`search/${saved.id}`);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setReplaying(false);
    }
  };
  const exportResult = async () => {
    if (!result) return;
    setExporting(true);
    setError('');
    try {
      setExportJob(await api.export(result.id, exportFormat));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setExporting(false);
    }
  };
  const chosenGallery = galleries.data?.items.find((gallery) => gallery.id === result?.gallery_id);
  const canOperate = session.role !== 'viewer';
  return (
    <>
      <PageHeading
        eyebrow="ПОИСК ПО ИЗОБРАЖЕНИЮ"
        title="Визуальный поиск"
        description="Выберите кадр, выделите автомобиль и сравните детали."
        action={
          result && (
            <Button icon="plus" onClick={() => navigate('search')}>
              Новый поиск
            </Button>
          )
        }
      />
      <div className="search-layout">
        <section className="panel query-panel">
          <SectionTitle
            title={result ? 'Исходный запрос' : 'Изображение запроса'}
            caption={
              result ? formatDate(result.created_at) : 'Выберите снимок и область автомобиля'
            }
          />
          {result && !image ? (
            <div className="saved-query">
              <img src={result.query.crop_url} alt="Вырезка исходного запроса" />
              <div className="file-caption">
                <span>Рамка {result.query.bbox.join(' · ')}</span>
              </div>
              <a
                className="text-link"
                href={result.query.image_url}
                target="_blank"
                rel="noreferrer"
              >
                Открыть исходный кадр <Icon name="expand" size={14} />
              </a>
            </div>
          ) : (
            <CropEditor
              value={image}
              onChange={setImage}
              maxBytes={capabilities?.limits.max_upload_bytes}
            />
          )}
          {result && !image && (
            <Button icon="upload" className="full-width" onClick={() => navigate('search')}>
              Загрузить новый запрос
            </Button>
          )}
          <div className="query-settings">
            <label className="field-label" htmlFor="gallery-select">
              Галерея поиска
            </label>
            <select
              id="gallery-select"
              value={galleryId}
              onChange={(event) => setGalleryId(event.target.value)}
              disabled={galleries.loading || !activeGalleries.length}
            >
              <option value="">
                {galleries.loading ? 'Загружаем галереи…' : 'Выберите галерею'}
              </option>
              {activeGalleries.map((gallery) => (
                <option key={gallery.id} value={gallery.id}>
                  {gallery.name} · {gallery.observation_count} набл.
                </option>
              ))}
            </select>
            {galleries.error && (
              <Message tone="error" action={<Button onClick={galleries.reload}>Повторить</Button>}>
                {galleries.error}
              </Message>
            )}
            {galleries.data?.next_offset != null && (
              <Button tone="ghost" busy={galleries.loadingMore} onClick={galleries.loadMore}>
                Загрузить ещё галереи
              </Button>
            )}
            {!galleries.loading && !galleries.error && !activeGalleries.length && (
              <p className="field-hint">
                Сначала создайте галерею, импортируйте изображения и активируйте её версию.{' '}
                <button className="text-link" onClick={() => navigate('galleries')}>
                  Перейти к галереям
                </button>
              </p>
            )}
            <div className="topn-row">
              <label htmlFor="top-n">Кандидатов в выдаче</label>
              <select
                id="top-n"
                value={topN}
                onChange={(event) => setTopN(Number(event.target.value))}
              >
                {[5, 10, 20, 50].map((count) => (
                  <option key={count} value={count}>
                    {count}
                  </option>
                ))}
              </select>
            </div>
            <Button
              tone="primary"
              icon="search"
              className="full-width search-submit"
              busy={searching}
              disabled={
                !canOperate ||
                !image ||
                !galleryId ||
                !validBBox(image.bbox, image.width, image.height)
              }
              onClick={search}
            >
              {searching ? 'Обрабатываем запрос…' : 'Найти похожие'}
            </Button>
            {!canOperate && <p className="field-hint">Для нового поиска нужна роль оператора.</p>}
            <p className="privacy-note">
              <Icon name="shield" size={14} />
              Обработка внутри вашего контура
            </p>
          </div>
        </section>
        <ResultsSurface>
          {!result && (
            <div className="search-canvas-header">
              <div>
                <span className="canvas-indicator" />
                <strong>Результаты поиска</strong>
              </div>
              <small>Выдача по галерее</small>
            </div>
          )}
          {error && <Message tone="error">{error}</Message>}
          {loadingResult ? (
            <div className="panel">
              <Loading label="Открываем сохранённый результат…" />
            </div>
          ) : result ? (
            <>
              <div className="result-heading">
                <div>
                  <div className="inline-items">
                    <h2>Результаты поиска</h2>
                    <Badge>{result.ranked_candidates.length}</Badge>
                  </div>
                  <p>
                    {chosenGallery?.name || shortId(result.gallery_id)} <span>·</span> Версия{' '}
                    {shortId(result.gallery_version_id)}
                  </p>
                </div>
                <Button
                  tone="ghost"
                  icon="refresh"
                  onClick={replay}
                  busy={replaying}
                  disabled={!canOperate}
                >
                  Воспроизвести
                </Button>
              </div>
              <div
                className={`decision-card ${result.decision === 'no_confident_match' ? 'decision-refused' : result.decision === 'calibration_unavailable' ? 'decision-unavailable' : 'decision-accepted'}`}
              >
                <div className="decision-symbol">
                  <Icon
                    name={
                      result.decision === 'calibration_unavailable'
                        ? 'eye'
                        : result.decision === 'no_confident_match'
                          ? 'shield'
                          : 'check'
                    }
                    size={22}
                  />
                </div>
                <div>
                  <strong>{decisionLabels[result.decision] || result.decision}</strong>
                  <p>
                    {result.decision === 'calibration_unavailable'
                      ? 'Модель ещё не подключена. Ни один кандидат не считается подтверждённым совпадением.'
                      : result.decision === 'no_confident_match'
                        ? 'Порог не пройден. Изображения ниже доступны для дополнительной проверки.'
                        : 'Решение относится только к первому кандидату. Остальные изображения — альтернативы для проверки.'}
                  </p>
                </div>
              </div>
              {result.ranked_candidates.length ? (
                <div className="candidate-grid">
                  {result.ranked_candidates.map((candidate) => (
                    <button
                      className={`candidate-card ${candidate.accepted ? 'candidate-accepted' : ''}`}
                      key={candidate.observation_id}
                      onClick={() => setSelected(candidate)}
                    >
                      <div className="candidate-photo">
                        <img
                          src={candidate.crop_url}
                          alt={`Кандидат ${candidate.rank}`}
                          loading="lazy"
                        />
                        <span className="rank-number">
                          {String(candidate.rank).padStart(2, '0')}
                        </span>
                        <span className="candidate-open">
                          <Icon name="expand" size={17} />
                        </span>
                      </div>
                      <div className="candidate-details">
                        <div className="row-between">
                          <strong>
                            {candidate.external_id ||
                              `Наблюдение ${shortId(candidate.observation_id)}`}
                          </strong>
                          {candidate.accepted && <Icon name="check" size={16} />}
                        </div>
                        <div className="row-between">
                          <span>{candidate.accepted ? 'Принят алгоритмом' : 'Для проверки'}</span>
                          <code title="Косинус признаков, не вероятность совпадения">
                            {formatNumber(candidate.cosine, 4)}
                          </code>
                        </div>
                      </div>
                    </button>
                  ))}
                </div>
              ) : (
                <EmptyState title="Кандидаты отсутствуют">
                  Сохранённый результат не содержит изображений для сравнения.
                </EmptyState>
              )}
              <div className="result-footnote">
                <Icon name="eye" size={16} />
                <span>
                  Нажмите на изображение, чтобы сравнить детали и сохранить заключение. Числа —
                  косинус признаков, не проценты уверенности.
                </span>
              </div>
              <section className="panel result-passport">
                <SectionTitle
                  title="Паспорт результата"
                  caption="Сохранённый контекст поиска и независимое заключение"
                />
                <dl className="metadata-grid">
                  <div>
                    <dt>Поиск</dt>
                    <dd title={result.id}>{shortId(result.id)}</dd>
                  </div>
                  <div>
                    <dt>Модель</dt>
                    <dd title={result.model_id}>{shortId(result.model_id)}</dd>
                  </div>
                  <div>
                    <dt>Время обработки</dt>
                    <dd>{formatNumber(result.timings.total_ms, 1)} мс</dd>
                  </div>
                  <div>
                    <dt>Уверенность / порог</dt>
                    <dd>
                      {formatNumber(result.query_confidence)} / {formatNumber(result.threshold)}
                    </dd>
                  </div>
                </dl>
                <div className="passport-actions">
                  <span>
                    {result.reviews.length
                      ? `Заключений оператора: ${result.reviews.length}`
                      : 'Заключений оператора пока нет'}
                  </span>
                  <div className="inline-items">
                    <select
                      aria-label="Формат экспорта"
                      value={exportFormat}
                      onChange={(event) =>
                        setExportFormat(event.target.value as typeof exportFormat)
                      }
                    >
                      <option value="pdf">PDF</option>
                      <option value="json">JSON</option>
                      <option value="csv">CSV</option>
                    </select>
                    <Button
                      icon="download"
                      busy={exporting}
                      disabled={!canOperate}
                      onClick={exportResult}
                    >
                      Экспортировать
                    </Button>
                  </div>
                </div>
                {exportJob && <JobProgress initial={exportJob} />}
                <details className="technical-details">
                  <summary>Технические сведения</summary>
                  <pre>
                    {JSON.stringify(
                      {
                        decision: result.decision,
                        confidence_kind: result.confidence_kind,
                        calibration_reason: result.calibration_reason,
                        model_id: result.model_id,
                        gallery_version_id: result.gallery_version_id,
                        timings: result.timings,
                      },
                      null,
                      2,
                    )}
                  </pre>
                </details>
              </section>
            </>
          ) : (
            <div className="search-empty panel">
              <span className="canvas-kicker">ОТ КАДРА К ДЕТАЛЯМ</span>
              <div className="search-empty-art" aria-hidden="true">
                <div className="art-scan">
                  <svg viewBox="0 0 180 100">
                    <path d="m30 65 12-27c3-6 8-9 16-9h65c8 0 13 3 16 9l12 27v20H30V65Z" />
                    <path d="m42 59 8-20h77l10 20H42Z" />
                    <path d="M40 72h20m58 0h21M68 72h42" />
                    <path d="M38 86v7m104-7v7" />
                  </svg>
                  <i />
                  <i />
                  <i />
                  <i />
                </div>
                <div className="art-side side-one" />
                <div className="art-side side-two" />
                <div className="art-connector" />
              </div>
              <h2>
                Найдите автомобиль.
                <br />
                <span>Сравните детали.</span>
              </h2>
              <p>
                Добавьте изображение автомобиля и выберите галерею.
                <br />
                Здесь появятся кандидаты для внимательного сравнения.
              </p>
              <div className="search-steps">
                <span>
                  <b>1</b>Выберите область
                </span>
                <Icon name="chevron" size={15} />
                <span>
                  <b>2</b>Запустите поиск
                </span>
                <Icon name="chevron" size={15} />
                <span>
                  <b>3</b>Сравните детали
                </span>
              </div>
              <div className="empty-model-note">
                <Icon name="activity" size={17} />
                <span>
                  Каждый поиск сохраняется в истории.
                  <br />
                  <strong>К сравнению можно вернуться в любой момент.</strong>
                </span>
              </div>
            </div>
          )}
        </ResultsSurface>
      </div>
      {result && selected && (
        <Comparison
          result={result}
          candidate={selected}
          canReview={canOperate}
          onClose={() => setSelected(null)}
          onReview={(review) =>
            setResult((current) =>
              current?.id === review.search_id &&
              !current.reviews.some((existing) => existing.id === review.id)
                ? { ...current, reviews: [...current.reviews, review] }
                : current,
            )
          }
        />
      )}
    </>
  );
}
