import { useEffect, useRef, useState, type MouseEvent } from 'react';
import { api, explanationFromJob } from '../api';
import {
  errorMessage,
  formatDate,
  formatNumber,
  heatOpacity,
  shortId,
  synchronizedScrollOffset,
  verdictLabels,
} from '../domain';
import type {
  AnnotationKind,
  BBox,
  Candidate,
  Explanation,
  Job,
  Review,
  ReviewAnnotation,
  ReviewAnnotationDraft,
  SearchResult,
  Verdict,
} from '../types';
import { Badge, Button, Icon, JobProgress, Message, Modal, useToast } from './ui';

type Pan = { x: number; y: number; source: string };

function ImageViewer({
  imageRef,
  src,
  label,
  map,
  bbox,
  surface,
  zoom,
  onZoom,
  pan,
  onPan,
  annotations,
  annotationKind,
  onAnnotate,
}: {
  imageRef: string;
  src: string;
  label: string;
  map?: number[][];
  bbox: BBox;
  surface: 'crop' | 'original';
  zoom: number;
  onZoom: (value: number) => void;
  pan: Pan;
  onPan: (value: Pan) => void;
  annotations: (ReviewAnnotation | ReviewAnnotationDraft)[];
  annotationKind: AnnotationKind;
  onAnnotate?: (annotation: ReviewAnnotationDraft) => void;
}) {
  const [aspect, setAspect] = useState(1);
  const [naturalSize, setNaturalSize] = useState({ width: 1, height: 1 });
  const [viewportSize, setViewportSize] = useState({ width: 400, height: 320 });
  const viewport = useRef<HTMLDivElement>(null);
  const applyingPan = useRef(false);
  useEffect(() => {
    if (!viewport.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setViewportSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(viewport.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    const node = viewport.current;
    if (!node) return;
    applyingPan.current = true;
    node.scrollLeft = synchronizedScrollOffset(pan.x, node.scrollWidth, node.clientWidth);
    node.scrollTop = synchronizedScrollOffset(pan.y, node.scrollHeight, node.clientHeight);
    const frame = requestAnimationFrame(() => {
      applyingPan.current = false;
    });
    return () => cancelAnimationFrame(frame);
  }, [pan, viewportSize, zoom]);
  const maximum = map ? Math.max(...map.flat().map(Math.abs)) : 0;
  const addAnnotation = (event: MouseEvent<HTMLDivElement>) => {
    if (!onAnnotate) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const x = Math.min(1, Math.max(0, (event.clientX - bounds.left) / bounds.width));
    const y = Math.min(1, Math.max(0, (event.clientY - bounds.top) / bounds.height));
    onAnnotate({ image_ref: imageRef, surface, kind: annotationKind, x, y });
  };
  return (
    <div className="image-viewer">
      <div className="viewer-toolbar">
        <span>{label}</span>
        <div className="zoom-controls">
          <button
            aria-label={`Уменьшить: ${label}`}
            disabled={zoom <= 1}
            onClick={() => onZoom(Math.max(1, zoom - 0.5))}
          >
            −
          </button>
          <button aria-label="Исходный масштаб" onClick={() => onZoom(1)}>
            {zoom * 100}%
          </button>
          <button
            aria-label={`Увеличить: ${label}`}
            disabled={zoom >= 4}
            onClick={() => onZoom(Math.min(4, zoom + 0.5))}
          >
            +
          </button>
        </div>
      </div>
      <div
        className="viewer-viewport"
        ref={viewport}
        onScroll={(event) => {
          if (applyingPan.current) return;
          const node = event.currentTarget;
          onPan({
            x: node.scrollLeft / Math.max(1, node.scrollWidth - node.clientWidth),
            y: node.scrollTop / Math.max(1, node.scrollHeight - node.clientHeight),
            source: imageRef,
          });
        }}
      >
        <div
          className={`viewer-image ${onAnnotate ? 'annotation-active' : ''}`}
          style={{ width: Math.min(viewportSize.width, viewportSize.height * aspect) * zoom }}
          onClick={addAnnotation}
        >
          <img
            src={src}
            alt={label}
            onLoad={(event) => {
              const image = event.currentTarget;
              setAspect(image.naturalWidth / image.naturalHeight);
              setNaturalSize({ width: image.naturalWidth, height: image.naturalHeight });
            }}
          />
          {surface === 'original' && (
            <span
              className="board-bbox"
              style={{
                left: `${(bbox[0] / naturalSize.width) * 100}%`,
                top: `${(bbox[1] / naturalSize.height) * 100}%`,
                width: `${(bbox[2] / naturalSize.width) * 100}%`,
                height: `${(bbox[3] / naturalSize.height) * 100}%`,
              }}
            />
          )}
          {map && (
            <div
              className="heat-grid"
              style={{ gridTemplateColumns: `repeat(${map[0]?.length || 1}, 1fr)` }}
            >
              {map.flatMap((row, y) =>
                row.map((value, x) => (
                  <span
                    key={`${x}-${y}`}
                    title={`Δ cosine ${value.toFixed(5)}`}
                    style={{
                      background: value >= 0 ? 'var(--heat-positive)' : 'var(--heat-negative)',
                      opacity: heatOpacity(value, maximum),
                    }}
                  />
                )),
              )}
            </div>
          )}
          {annotations
            .filter((item) => item.image_ref === imageRef && item.surface === surface)
            .map((item, index) => (
              <span
                className={`evidence-marker evidence-${item.kind}`}
                key={
                  'id' in item
                    ? item.id
                    : `${item.image_ref}-${item.surface}-${item.x}-${item.y}-${index}`
                }
                style={{ left: `${item.x * 100}%`, top: `${item.y * 100}%` }}
                title={`${index + 1}. ${item.kind}`}
              >
                {index + 1}
              </span>
            ))}
        </div>
      </div>
    </div>
  );
}

const annotationLabels: Record<AnnotationKind, string> = {
  match: 'Совпадает',
  conflict: 'Противоречит',
  unclear: 'Неразличимо',
};

export function Comparison({
  result,
  candidate,
  canReview,
  onClose,
  onReview,
}: {
  result: SearchResult;
  candidate: Candidate;
  canReview: boolean;
  onClose: () => void;
  onReview: (review: Review) => void;
}) {
  const [job, setJob] = useState<Job | null>(null);
  const [explanation, setExplanation] = useState<Explanation | null>(null);
  const [showMap, setShowMap] = useState(true);
  const [originals, setOriginals] = useState(false);
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState<Pan>({ x: 0, y: 0, source: '' });
  const [annotationMode, setAnnotationMode] = useState(false);
  const [annotationKind, setAnnotationKind] = useState<AnnotationKind>('match');
  const [coordinateImage, setCoordinateImage] = useState('query');
  const [coordinateX, setCoordinateX] = useState(50);
  const [coordinateY, setCoordinateY] = useState(50);
  const [draftAnnotations, setDraftAnnotations] = useState<ReviewAnnotationDraft[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [verdict, setVerdict] = useState<Verdict>('uncertain');
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const toast = useToast();
  const competitor = result.ranked_candidates.find(
    (item) => item.observation_id !== candidate.observation_id,
  );
  const reviews = result.reviews.filter(
    (review) => review.observation_id === candidate.observation_id,
  );
  const savedAnnotations = reviews.flatMap((review) => review.annotations || []);
  const annotations = [...savedAnnotations, ...draftAnnotations];
  const surface = originals ? 'original' : 'crop';
  useEffect(() => {
    setExplanation(null);
    setJob(null);
    setError('');
    setDraftAnnotations([]);
    setCoordinateImage('query');
    setZoom(1);
    setPan({ x: 0, y: 0, source: '' });
  }, [candidate.observation_id]);
  const explain = async () => {
    setBusy(true);
    setError('');
    try {
      setJob(await api.explain(result.id, candidate.observation_id));
      setOriginals(false);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const save = async () => {
    setSaving(true);
    setError('');
    try {
      const review = await api.review(
        result.id,
        candidate.observation_id,
        verdict,
        note,
        draftAnnotations,
      );
      onReview(review);
      setNote('');
      setDraftAnnotations([]);
      setAnnotationMode(false);
      toast('Заключение и координатные пометки сохранены отдельно от решения алгоритма');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSaving(false);
    }
  };
  const viewer = (
    imageRef: string,
    label: string,
    original: string,
    crop: string,
    bbox: BBox,
    map?: number[][],
  ) => (
    <ImageViewer
      imageRef={imageRef}
      src={originals ? original : crop}
      label={label}
      bbox={bbox}
      surface={surface}
      map={!originals && showMap ? map : undefined}
      zoom={zoom}
      onZoom={setZoom}
      pan={pan}
      onPan={setPan}
      annotations={annotations}
      annotationKind={annotationKind}
      onAnnotate={
        annotationMode && canReview && draftAnnotations.length < 20
          ? (annotation) => setDraftAnnotations((current) => [...current, annotation].slice(0, 20))
          : undefined
      }
    />
  );
  return (
    <Modal wide title="Экспертная доска" onClose={onClose}>
      <div className="comparison-topline">
        <div className="inline-items">
          <Badge tone={candidate.accepted ? 'success' : 'neutral'}>
            {candidate.accepted ? 'Принят алгоритмом' : 'Кандидат для проверки'}
          </Badge>
          <span className="muted">
            Позиция {candidate.rank} · {shortId(candidate.observation_id)}
          </span>
        </div>
        <label className="switch-label">
          <input
            type="checkbox"
            checked={originals}
            onChange={(event) => {
              setOriginals(event.target.checked);
              setZoom(1);
              setPan({ x: 0, y: 0, source: '' });
            }}
          />
          Исходные кадры и BBox
        </label>
      </div>
      <div className={`comparison-images ${competitor ? 'comparison-images-three' : ''}`}>
        {viewer(
          'query',
          'Запрос',
          result.query.image_url,
          result.query.crop_url,
          result.query.bbox,
          explanation?.query_map,
        )}
        {viewer(
          candidate.observation_id,
          `Проверяемый кандидат № ${candidate.rank}`,
          candidate.image_url,
          candidate.crop_url,
          candidate.bbox,
          explanation?.candidate_map,
        )}
        {competitor &&
          viewer(
            competitor.observation_id,
            `Контрольный конкурент № ${competitor.rank}`,
            competitor.image_url,
            competitor.crop_url,
            competitor.bbox,
          )}
      </div>
      <div className="comparison-metrics">
        <span>
          Проверяемый кандидат <strong>{formatNumber(candidate.cosine, 5)}</strong>
        </span>
        {competitor && (
          <span>
            Контрольный конкурент <strong>{formatNumber(competitor.cosine, 5)}</strong>
          </span>
        )}
        <span>Синхронный zoom и pan · оценки не являются вероятностями</span>
      </div>
      <section className="annotation-tools">
        <div>
          <h3>Координатные пометки эксперта</h3>
          <p>
            Выберите смысл и нажмите на деталь. Пометки на конкуренте служат только контекстом для
            сравнения.
          </p>
        </div>
        <div className="annotation-actions">
          <div className="segmented-control" role="group" aria-label="Тип пометки">
            {(Object.entries(annotationLabels) as [AnnotationKind, string][]).map(
              ([value, label]) => (
                <button
                  key={value}
                  className={annotationKind === value ? 'selected' : ''}
                  aria-pressed={annotationKind === value}
                  onClick={() => setAnnotationKind(value)}
                >
                  {label}
                </button>
              ),
            )}
          </div>
          <Button
            tone={annotationMode ? 'primary' : 'default'}
            icon="target"
            disabled={!canReview || draftAnnotations.length >= 20}
            onClick={() => setAnnotationMode((value) => !value)}
          >
            {annotationMode ? 'Ставьте точки' : 'Добавить пометки'}
          </Button>
          {draftAnnotations.length > 0 && (
            <Button onClick={() => setDraftAnnotations((current) => current.slice(0, -1))}>
              Отменить последнюю
            </Button>
          )}
        </div>
        <span className="annotation-count">
          Новых пометок: {draftAnnotations.length}/20 · сохранённых: {savedAnnotations.length}
        </span>
        {canReview && (
          <details className="annotation-coordinate">
            <summary>Добавить точную координату с клавиатуры</summary>
            <div className="coordinate-row">
              <label>
                Изображение
                <select
                  value={coordinateImage}
                  onChange={(event) => setCoordinateImage(event.target.value)}
                >
                  <option value="query">Запрос</option>
                  <option value={candidate.observation_id}>
                    Проверяемый кандидат № {candidate.rank}
                  </option>
                  {competitor && (
                    <option value={competitor.observation_id}>
                      Контрольный конкурент № {competitor.rank}
                    </option>
                  )}
                </select>
              </label>
              <label>
                X, %
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={coordinateX}
                  onChange={(event) => setCoordinateX(Number(event.target.value))}
                />
              </label>
              <label>
                Y, %
                <input
                  type="number"
                  min="0"
                  max="100"
                  value={coordinateY}
                  onChange={(event) => setCoordinateY(Number(event.target.value))}
                />
              </label>
              <Button
                icon="plus"
                disabled={
                  !Number.isFinite(coordinateX) ||
                  !Number.isFinite(coordinateY) ||
                  coordinateX < 0 ||
                  coordinateX > 100 ||
                  coordinateY < 0 ||
                  coordinateY > 100 ||
                  draftAnnotations.length >= 20
                }
                onClick={() =>
                  setDraftAnnotations((current) => [
                    ...current,
                    {
                      image_ref: coordinateImage,
                      surface,
                      kind: annotationKind,
                      x: coordinateX / 100,
                      y: coordinateY / 100,
                    },
                  ])
                }
              >
                Добавить
              </Button>
            </div>
          </details>
        )}
      </section>
      <div className="explanation-controls">
        <div>
          <h3>Проверка областей изображения</h3>
          <p>
            {canReview
              ? 'Реальный расчёт изменения косинуса при перекрытии участков'
              : 'Для расчёта карты необходима роль оператора'}
          </p>
        </div>
        <Button
          icon="target"
          busy={busy}
          disabled={!canReview || job?.status === 'pending' || job?.status === 'running'}
          onClick={explain}
        >
          Рассчитать карту
        </Button>
      </div>
      {job && (
        <JobProgress
          initial={job}
          onUpdate={setJob}
          onComplete={(completed) => {
            const data = explanationFromJob(completed);
            if (data) {
              setExplanation(data);
              setShowMap(true);
            } else setError('Задание завершено, но сервер не вернул карту влияния.');
          }}
        />
      )}
      {explanation && (
        <div className="explanation-detail">
          <div className="row-between">
            <label className="switch-label">
              <input
                type="checkbox"
                checked={showMap}
                onChange={(event) => setShowMap(event.target.checked)}
                disabled={originals}
              />
              Показать карту на вырезках
            </label>
            <div className="heat-legend">
              <span className="negative" />
              Снижает оценку
              <span className="positive" />
              Поддерживает
            </div>
          </div>
          <p>
            {explanation.mode === 'preview'
              ? 'Демонстрация интерфейса объяснений: модель не подключена. Карта рассчитана для технического признака заглушки.'
              : 'Карта объясняет исходный косинус признаков, а не итоговое ранжирование.'}{' '}
            Перекрытие области меняет оценку; это не доказательство идентичности автомобиля.
          </p>
          <code>
            {explanation.method} · {explanation.grid} × {explanation.grid} ·{' '}
            {explanation.score_kind} {formatNumber(explanation.baseline_score, 5)}
          </code>
        </div>
      )}
      {error && <Message tone="error">{error}</Message>}
      <div className="review-section">
        <div>
          <h3>Заключение: запрос ↔ кандидат № {candidate.rank}</h3>
          <p>
            Вердикт относится только к проверяемому кандидату. Выдача и оценки алгоритма не
            меняются.
          </p>
        </div>
        {canReview ? (
          <>
            <div className="segmented-control" role="group" aria-label="Заключение оператора">
              {(Object.entries(verdictLabels) as [Verdict, string][]).map(([value, label]) => (
                <button
                  key={value}
                  className={verdict === value ? 'selected' : ''}
                  aria-pressed={verdict === value}
                  onClick={() => setVerdict(value)}
                >
                  {label}
                </button>
              ))}
            </div>
            <textarea
              value={note}
              maxLength={2000}
              placeholder="Какие детали помогли сделать вывод?"
              aria-label="Комментарий оператора"
              onChange={(event) => setNote(event.target.value)}
              rows={2}
            />
            <div className="align-right">
              <Button tone="primary" icon="check" busy={saving} onClick={save}>
                Сохранить по кандидату № {candidate.rank}
              </Button>
            </div>
          </>
        ) : (
          <p className="muted">Для заключений необходима роль оператора.</p>
        )}
        {reviews.length > 0 && (
          <div className="review-list">
            {reviews.map((review) => (
              <div key={review.id}>
                <div className="row-between">
                  <strong>{verdictLabels[review.verdict]}</strong>
                  <time>{formatDate(review.created_at)}</time>
                </div>
                {review.note && <p>{review.note}</p>}
                {!!review.annotations?.length && (
                  <p>Координатных пометок: {review.annotations.length}</p>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="comparison-footer">
        <Icon name="shield" size={16} />
        <span>
          Поиск {shortId(result.id)} · версия галереи {shortId(result.gallery_version_id)}
        </span>
      </div>
    </Modal>
  );
}
