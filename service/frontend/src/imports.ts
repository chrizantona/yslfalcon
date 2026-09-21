import type { BBox, ImportSession } from './types';

export interface ManifestEntry {
  imageId: string;
  bbox: BBox;
}
export interface ImportIssue {
  message: string;
  row?: number;
  imageId?: string;
}
export interface ParsedManifest {
  entries: ManifestEntry[];
  issues: ImportIssue[];
  ignoredColumns: string[];
}

function csvRows(source: string): string[][] {
  const rows: string[][] = [];
  let row: string[] = [];
  let value = '';
  let quoted = false;
  let closed = false;
  const text = source.replace(/^\uFEFF/, '');
  const finishField = () => {
    row.push(value);
    value = '';
    closed = false;
  };
  const finishRow = () => {
    const hasContent = row.length > 0 || value !== '' || closed;
    finishField();
    if (hasContent) rows.push(row);
    row = [];
  };
  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        value += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
        closed = true;
      } else value += char;
    } else if (char === ',') finishField();
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[index + 1] === '\n') index += 1;
      finishRow();
    } else if (char === '"') {
      if (value || closed) throw new Error('Некорректные кавычки в CSV.');
      value = '';
      quoted = true;
    } else if (closed) {
      throw new Error('После закрывающей кавычки в CSV ожидается запятая или конец строки.');
    } else value += char;
  }
  if (quoted) throw new Error('В CSV не закрыта кавычка.');
  if (value || row.length || closed) finishRow();
  return rows;
}

const imageExtension = /\.(jpe?g|png)$/i;
const imageStem = (value: string) => value.replace(imageExtension, '');
const normalizedFilename = (value: string) =>
  value.replace(imageExtension, (extension) => extension.toLowerCase());

/** Checks structure only. The server remains authoritative for image dimensions and publication. */
export function parseGalleryCsv(source: string): ParsedManifest {
  let rows: string[][];
  try {
    rows = csvRows(source);
  } catch (cause) {
    return {
      entries: [],
      issues: [{ message: cause instanceof Error ? cause.message : 'Не удалось прочитать CSV.' }],
      ignoredColumns: [],
    };
  }
  const required = ['image_id', 'x', 'y', 'w', 'h'];
  const headers = rows[0] || [];
  const missing = required.filter((column) => !headers.includes(column));
  const ignoredColumns = headers.filter((column) => !required.includes(column));
  if (missing.length)
    return {
      entries: [],
      issues: [{ message: `В CSV не хватает колонок: ${missing.join(', ')}.` }],
      ignoredColumns,
    };
  if (new Set(headers).size !== headers.length)
    return {
      entries: [],
      issues: [{ message: 'Названия колонок CSV повторяются.' }],
      ignoredColumns,
    };
  const entries: ManifestEntry[] = [];
  const issues: ImportIssue[] = [];
  const aliases = new Set<string>();
  if (rows.length - 1 > 10_000)
    issues.push({ message: 'В одной сессии можно подготовить до 10 000 наблюдений.' });
  rows.slice(1).forEach((record, index) => {
    const row = index + 2;
    if (record.length !== headers.length) {
      issues.push({ row, message: 'Количество полей не совпадает с заголовком CSV.' });
      return;
    }
    const cell = (name: string) => record[headers.indexOf(name)].trim();
    const imageId = cell('image_id');
    if (
      !imageId ||
      imageId.length > 255 ||
      /[\/\\\u0000-\u001F]/.test(imageId) ||
      imageId === '.' ||
      imageId === '..'
    ) {
      issues.push({
        row,
        imageId,
        message: 'image_id должен быть именем файла или его идентификатором без пути.',
      });
      return;
    }
    const hasSuffix = imageId.lastIndexOf('.') > 0 && !imageId.endsWith('.');
    if (hasSuffix && !imageExtension.test(imageId)) {
      issues.push({
        row,
        imageId,
        message: 'Допустимы имена JPEG/PNG или идентификаторы без расширения.',
      });
      return;
    }
    const names = imageExtension.test(imageId)
      ? [normalizedFilename(imageId)]
      : ['.jpg', '.jpeg', '.png'].map((extension) => imageId + extension);
    if (names.some((name) => aliases.has(name))) {
      issues.push({ row, imageId, message: 'Повторяющийся или неоднозначный image_id.' });
      return;
    }
    names.forEach((name) => aliases.add(name));
    const raw = ['x', 'y', 'w', 'h'].map(cell);
    const bbox = raw.map(Number) as BBox;
    if (
      raw.some((value) => !/^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$/.test(value)) ||
      bbox.some((value) => !Number.isSafeInteger(value) || Math.abs(value) > 10_000_000) ||
      bbox[2] <= 0 ||
      bbox[3] <= 0
    ) {
      issues.push({
        row,
        imageId,
        message:
          'BBox: целые x, y и положительные w, h в пикселях; модуль координат не больше 10 000 000.',
      });
      return;
    }
    entries.push({ imageId, bbox });
  });
  if (rows.length < 2) issues.push({ message: 'В CSV нет наблюдений.' });
  return { entries, issues, ignoredColumns };
}

export interface UploadFile {
  name: string;
  size: number;
}
export function matchImportFiles<T extends UploadFile>(
  imageIds: string[],
  files: T[],
  maxUploadBytes: number,
) {
  const selected = new Map<string, T>();
  const missing: string[] = [];
  const issues: ImportIssue[] = [];
  const used = new Set<T>();
  const images = files.filter((file) => imageExtension.test(file.name));
  const byName = new Map<string, T[]>();
  const byStem = new Map<string, T[]>();
  for (const file of images) {
    const name = normalizedFilename(file.name);
    byName.set(name, [...(byName.get(name) || []), file]);
    const stem = imageStem(file.name);
    byStem.set(stem, [...(byStem.get(stem) || []), file]);
  }
  for (const imageId of imageIds) {
    const matches = imageExtension.test(imageId)
      ? byName.get(normalizedFilename(imageId)) || []
      : byStem.get(imageId) || [];
    if (!matches.length) missing.push(imageId);
    else if (matches.length > 1)
      issues.push({
        imageId,
        message: 'Найдено несколько файлов для одного image_id. Оставьте один.',
      });
    else {
      const file = matches[0];
      used.add(file);
      if (file.size > maxUploadBytes || file.size === 0)
        issues.push({
          imageId,
          message: file.size === 0 ? 'Файл пуст.' : 'Размер файла превышает ограничение сервиса.',
        });
      else selected.set(imageId, file);
    }
  }
  return { selected, missing, issues, unusedCount: files.filter((file) => !used.has(file)).length };
}

export function chunkImportFiles<T extends UploadFile>(
  files: T[],
  limits: ImportSession['limits'],
): T[][] {
  if (limits.max_chunk_files < 1 || limits.max_chunk_bytes < 1)
    throw new Error('Сервер вернул некорректные ограничения загрузки.');
  const byteBudget =
    limits.max_chunk_bytes - Math.min(1024 * 1024, Math.ceil(limits.max_chunk_bytes / 8));
  const chunks: T[][] = [];
  let chunk: T[] = [];
  let bytes = 0;
  for (const file of files) {
    const size = file.size + 1024 + file.name.length * 4;
    if (file.size > limits.max_upload_bytes || size > byteBudget)
      throw new Error(`Файл ${file.name} не помещается в разрешённый запрос.`);
    if (chunk.length >= limits.max_chunk_files || bytes + size > byteBudget) {
      chunks.push(chunk);
      chunk = [];
      bytes = 0;
    }
    chunk.push(file);
    bytes += size;
  }
  if (chunk.length) chunks.push(chunk);
  return chunks;
}

export async function transferImportFiles(
  files: File[],
  transport: {
    status: () => Promise<ImportSession>;
    upload: (files: File[]) => Promise<ImportSession>;
  },
  onUpdate: (session: ImportSession) => void,
  shouldPause: () => boolean = () => false,
): Promise<ImportSession> {
  let session = await transport.status();
  onUpdate(session);
  while (session.status === 'uploading' && session.missing_image_ids.length && !shouldPause()) {
    const matched = matchImportFiles(
      session.missing_image_ids,
      files,
      session.limits.max_upload_bytes,
    );
    if (matched.issues.length) throw new Error(matched.issues[0].message);
    const chunk = chunkImportFiles([...matched.selected.values()], session.limits)[0];
    if (!chunk) break;
    const before = session.uploaded_count;
    session = await transport.upload(chunk);
    onUpdate(session);
    if (session.uploaded_count <= before)
      throw new Error(
        'Сервер не подтвердил новые файлы. Обновите состояние загрузки перед повтором.',
      );
  }
  return session;
}
