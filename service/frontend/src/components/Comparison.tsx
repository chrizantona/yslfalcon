import { useEffect, useRef, useState } from 'react';
import { api, explanationFromJob } from '../api';
import {
  errorMessage,
  formatDate,
  formatNumber,
  heatOpacity,
  shortId,
  verdictLabels,
} from '../domain';
import type { Candidate, Explanation, Job, Review, SearchResult, Verdict } from '../types';
import { Badge, Button, Icon, JobProgress, Message, Modal, useToast } from './ui';

function ImageViewer({ src, label, map }: { src: string; label: string; map?: number[][] }) {
  const [zoom, setZoom] = useState(1);
  const [aspect, setAspect] = useState(1);
  const [viewportSize, setViewportSize] = useState({ width: 400, height: 320 });
  const viewport = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!viewport.current) return;
    const observer = new ResizeObserver(([entry]) =>
      setViewportSize({ width: entry.contentRect.width, height: entry.contentRect.height }),
    );
    observer.observe(viewport.current);
    return () => observer.disconnect();
  }, []);
  useEffect(() => {
    setZoom(1);
  }, [src]);
  const maximum = map ? Math.max(...map.flat().map(Math.abs)) : 0;
  return (
    <div className="image-viewer">
      <div className="viewer-toolbar">
        <span>{label}</span>
        <div className="zoom-controls">
          <button
            aria-label={`Уменьшить: ${label}`}
            disabled={zoom <= 1}
            onClick={() => setZoom((value) => Math.max(1, value - 0.5))}
          >
            −
          </button>
          <button aria-label="Исходный масштаб" onClick={() => setZoom(1)}>
            {zoom * 100}%
          </button>
          <button
            aria-label={`Увеличить: ${label}`}
            disabled={zoom >= 4}
            onClick={() => setZoom((value) => Math.min(4, value + 0.5))}
          >
            +
          </button>
        </div>
      </div>
      <div className="viewer-viewport" ref={viewport}>
        <div
          className="viewer-image"
          style={{ width: Math.min(viewportSize.width, viewportSize.height * aspect) * zoom }}
        >
          <img
            src={src}
            alt={label}
            onLoad={(event) =>
              setAspect(event.currentTarget.naturalWidth / event.currentTarget.naturalHeight)
            }
          />
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
        </div>
      </div>
    </div>
  );
}
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
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [verdict, setVerdict] = useState<Verdict>('uncertain');
  const [note, setNote] = useState('');
  const [saving, setSaving] = useState(false);
  const toast = useToast();
  const reviews = result.reviews.filter(
    (review) => review.observation_id === candidate.observation_id,
  );
  useEffect(() => {
    setExplanation(null);
    setJob(null);
    setError('');
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
      const review = await api.review(result.id, candidate.observation_id, verdict, note);
      onReview(review);
      setNote('');
      toast('Заключение сохранено отдельно от решения алгоритма');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSaving(false);
    }
  };
  return (
    <Modal wide title="Сравнение изображений" onClose={onClose}>
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
            onChange={(event) => setOriginals(event.target.checked)}
          />
          Исходные кадры
        </label>
      </div>
      <div className="comparison-images">
        <ImageViewer
          src={originals ? result.query.image_url : result.query.crop_url}
          label="Запрос"
          map={!originals && showMap ? explanation?.query_map : undefined}
        />
        <ImageViewer
          src={originals ? candidate.image_url : candidate.crop_url}
          label={`Кандидат № ${candidate.rank}`}
          map={!originals && showMap ? explanation?.candidate_map : undefined}
        />
      </div>
      <div className="comparison-metrics">
        <span>
          Косинус <strong>{formatNumber(candidate.cosine, 5)}</strong>
        </span>
        <span>
          Оценка ранжирования <strong>{formatNumber(candidate.score, 5)}</strong>
        </span>
        <span>Это числовые оценки, не вероятность совпадения</span>
      </div>
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
            Для расчёта используются уменьшенные копии вырезок; исходные изображения и результат
            поиска не меняются. Перекрытие области меняет оценку; это не доказательство идентичности
            автомобиля.
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
          <h3>Заключение оператора</h3>
          <p>Сохраняется отдельно. Выдача и оценки алгоритма остаются прежними.</p>
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
                Сохранить заключение
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
