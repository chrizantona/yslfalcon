import type { BBox } from './types.ts';

export const decisionLabels: Record<string, string> = {
  candidates: 'Кандидат найден',
  match: 'Кандидат найден',
  accepted: 'Кандидат найден',
  no_confident_match: 'Уверенное совпадение не найдено',
  calibration_unavailable: 'Ранжирование без решения о совпадении',
};
export const calibrationReasonLabels: Record<string, string> = {
  preview_descriptor_is_not_vehicle_reidentification: 'Модель ещё не подключена.',
  gallery_size_outside_validated_range: 'Размер галереи вне диапазона, на котором проверен порог.',
};
export const verdictLabels = {
  match: 'Совпадение',
  different: 'Другое ТС',
  uncertain: 'Недостаточно данных',
};
export const statusLabels: Record<string, string> = {
  pending: 'В очереди',
  running: 'Выполняется',
  completed: 'Завершено',
  failed: 'Ошибка',
  ready: 'Готова',
  active: 'Активна',
  retired: 'Архив',
  draft: 'Черновик',
  ingesting: 'Импорт',
  indexing: 'Индексация',
  unavailable: 'Недоступно',
  ok: 'Работает',
  healthy: 'Работает',
  degraded: 'Требует внимания',
  available: 'Доступно',
};
export const jobLabels: Record<string, string> = {
  import: 'Импорт галереи',
  gallery_import: 'Импорт галереи',
  explanation: 'Карта влияния',
  explain: 'Карта влияния',
  export: 'Экспорт результата',
};
export function formatDate(value: string): string {
  const date = new Date(value);
  return Number.isNaN(date.getTime())
    ? value
    : new Intl.DateTimeFormat('ru-RU', {
        day: '2-digit',
        month: 'short',
        hour: '2-digit',
        minute: '2-digit',
      }).format(date);
}
export function formatNumber(value: number | null | undefined, digits = 3): string {
  return value == null || !Number.isFinite(value)
    ? '—'
    : new Intl.NumberFormat('ru-RU', {
        maximumFractionDigits: digits,
      }).format(value);
}
export function formatBytes(value: number): string {
  return value >= 1024 * 1024
    ? `${formatNumber(value / 1024 / 1024, 1)} МБ`
    : `${formatNumber(value / 1024, 0)} КБ`;
}
export function shortId(value: string): string {
  return value.length > 16 ? `${value.slice(0, 8)}…${value.slice(-5)}` : value;
}
export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : 'Не удалось выполнить действие';
}

/** Bounding boxes use the original image coordinate system, never display pixels. */
export function bboxFromPoints(
  start: [number, number],
  end: [number, number],
  width: number,
  height: number,
): BBox {
  const x = Math.max(0, Math.min(width - 1, Math.floor(Math.min(start[0], end[0]))));
  const y = Math.max(0, Math.min(height - 1, Math.floor(Math.min(start[1], end[1]))));
  return [
    x,
    y,
    Math.max(1, Math.min(width - x, Math.ceil(Math.abs(end[0] - start[0])))),
    Math.max(1, Math.min(height - y, Math.ceil(Math.abs(end[1] - start[1])))),
  ];
}
export function validBBox(bbox: BBox, width: number, height: number): boolean {
  return (
    bbox.every(Number.isFinite) &&
    bbox.every(Number.isInteger) &&
    bbox[0] >= 0 &&
    bbox[1] >= 0 &&
    bbox[2] > 0 &&
    bbox[3] > 0 &&
    bbox[0] + bbox[2] <= width &&
    bbox[1] + bbox[3] <= height
  );
}
export function normalizeProgress(value: number): number {
  return Math.max(0, Math.min(100, value));
}
export function safeDownloadUrl(value: unknown): string | null {
  if (typeof value !== 'string') return null;
  // Artifacts stay within the current authenticated origin; reject schemes and protocol-relative URLs.
  return value.startsWith('/api/') && !value.includes('\\') ? value : null;
}
export function heatOpacity(value: number, maximum: number): number {
  return maximum > 0 ? Math.min(0.8, (Math.abs(value) / maximum) * 0.8) : 0;
}

/** Convert a shared normalized pan position back to pixels after zoom or resize. */
export function synchronizedScrollOffset(
  fraction: number,
  scrollSize: number,
  clientSize: number,
): number {
  return Math.min(1, Math.max(0, fraction)) * Math.max(0, scrollSize - clientSize);
}

export function createOperationKey(): string {
  // getRandomValues also works on an isolated HTTP appliance; randomUUID requires HTTPS.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  return Array.from(bytes, (byte) => byte.toString(16).padStart(2, '0')).join('');
}
