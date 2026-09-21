import { useEffect, useMemo, useRef, useState } from 'react';
import { api } from '../api';
import { createOperationKey, errorMessage, formatBytes, formatDate, shortId } from '../domain';
import { useResource } from '../hooks';
import { matchImportFiles, parseGalleryCsv, transferImportFiles } from '../imports';
import type { Capabilities, ImportSession, Job } from '../types';
import { Badge, Button, Icon, JobProgress, Message, SectionTitle, useToast } from './ui';

export function StagedImport({
  galleryId,
  capabilities,
  onComplete,
}: {
  galleryId: string;
  capabilities: Capabilities | null;
  onComplete: () => void;
}) {
  const sessions = useResource(() => api.importSessions(galleryId), [galleryId]);
  const [session, setSession] = useState<ImportSession | null>(null);
  const [manifestText, setManifestText] = useState('');
  const [manifestName, setManifestName] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [job, setJob] = useState<Job | null>(null);
  const [phase, setPhase] = useState<'idle' | 'reading' | 'opening' | 'uploading' | 'finalizing'>(
    'idle',
  );
  const [error, setError] = useState('');
  const [paused, setPaused] = useState(false);
  const pause = useRef(false);
  const alive = useRef(true);
  const key = useRef('');
  if (!key.current) key.current = createOperationKey();
  const folderInput = useRef<HTMLInputElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const busy = phase !== 'idle';
  const finalized = session?.status === 'finalized';
  const parsed = useMemo(
    () => (manifestName ? parseGalleryCsv(manifestText) : null),
    [manifestText, manifestName],
  );
  const ids = session?.missing_image_ids || parsed?.entries.map((entry) => entry.imageId) || [];
  const maxFileBytes =
    session?.limits.max_upload_bytes || capabilities?.limits.max_upload_bytes || 20 * 1024 * 1024;
  const matched = useMemo(
    () => matchImportFiles(ids, files, maxFileBytes),
    [ids, files, maxFileBytes],
  );
  const total = session?.expected_count ?? parsed?.entries.length ?? 0;
  const uploaded = session?.uploaded_count ?? 0;
  const csvIssues = session ? [] : parsed?.issues || [];
  const hasValidManifest = !!session || (!!parsed && !parsed.issues.length);
  const canUpload =
    !finalized && !csvIssues.length && matched.selected.size > 0 && !matched.issues.length;

  useEffect(() => {
    alive.current = true;
    folderInput.current?.setAttribute('webkitdirectory', '');
    return () => {
      alive.current = false;
      pause.current = true;
    };
  }, []);

  const loadManifest = async (file?: File) => {
    if (!file) return;
    setError('');
    setManifestText('');
    setManifestName('');
    setPhase('reading');
    key.current = createOperationKey();
    try {
      if (file.size > 2 * 1024 * 1024) throw new Error('CSV должен быть не больше 2 МБ.');
      const text = await file.text();
      if (alive.current) {
        setManifestText(text);
        setManifestName(file.name);
      }
    } catch (cause) {
      if (alive.current) setError(errorMessage(cause));
    } finally {
      if (alive.current) setPhase('idle');
    }
  };

  const openSession = async (sessionId: string) => {
    setPhase('opening');
    setError('');
    setFiles([]);
    setJob(null);
    setPaused(false);
    try {
      const next = await api.importSession(galleryId, sessionId);
      const nextJob = next.job_id ? await api.job(next.job_id) : null;
      if (alive.current) {
        setSession(next);
        setJob(nextJob);
      }
    } catch (cause) {
      if (alive.current) setError(errorMessage(cause));
    } finally {
      if (alive.current) setPhase('idle');
    }
  };

  const upload = async () => {
    setPhase('uploading');
    setError('');
    setPaused(false);
    pause.current = false;
    let current = session;
    try {
      if (!current) {
        current = await api.createImportSession(galleryId, manifestText, key.current);
        if (alive.current) setSession(current);
      }
      const sessionId = current.id;
      await transferImportFiles(
        files,
        {
          status: () => api.importSession(galleryId, sessionId),
          upload: (chunk) => api.uploadImportFiles(galleryId, sessionId, chunk),
        },
        (next) => {
          if (alive.current) setSession(next);
        },
        () => pause.current,
      );
    } catch (cause) {
      // A response can be lost after a successful write. Read authoritative progress before retrying.
      if (current && alive.current) {
        try {
          setSession(await api.importSession(galleryId, current.id));
        } catch (refreshError) {
          if (alive.current) {
            setError(
              `${errorMessage(cause)} Не удалось обновить прогресс: ${errorMessage(refreshError)}. Следующий повтор сначала проверит состояние на сервере.`,
            );
          }
          return;
        }
      }
      if (alive.current) setError(errorMessage(cause));
    } finally {
      if (alive.current) {
        setPhase('idle');
        sessions.reload();
      }
    }
  };

  const finalize = async () => {
    if (!session) return;
    setPhase('finalizing');
    setError('');
    try {
      const nextJob = await api.finalizeImport(galleryId, session.id);
      if (alive.current) {
        setJob(nextJob);
        setSession({ ...session, status: 'finalized', job_id: nextJob.id });
        sessions.reload();
      }
    } catch (cause) {
      if (alive.current) setError(errorMessage(cause));
    } finally {
      if (alive.current) setPhase('idle');
    }
  };

  const startNew = () => {
    setSession(null);
    setFiles([]);
    setManifestText('');
    setManifestName('');
    setJob(null);
    setError('');
    setPaused(false);
    key.current = createOperationKey();
  };

  return (
    <section className="panel staged-import">
      <SectionTitle
        title="Загрузить галерею по CSV"
        caption="Один список наблюдений → загрузка частями → одна новая версия"
        action={<Badge tone="purple">JPEG / PNG + CSV</Badge>}
      />
      <p className="staged-lead">
        Подходит для полной выборки до 10 000 наблюдений, в том числе 750 изображений галереи.
        Загруженные части сохраняются на сервере; действующая галерея продолжает обслуживать поиск.
      </p>
      {sessions.error && (
        <Message
          tone="error"
          action={
            <Button icon="refresh" disabled={busy} onClick={sessions.reload}>
              Повторить
            </Button>
          }
        >
          {sessions.error}
        </Message>
      )}
      {sessions.data && sessions.data.items.length > 0 && (
        <div className="import-resume">
          <label>
            Сохранённые загрузки
            <select
              value={session?.id || ''}
              disabled={busy}
              onChange={(event) => {
                if (event.target.value) void openSession(event.target.value);
                else startNew();
              }}
            >
              <option value="">Новая загрузка</option>
              {sessions.data.items.map((item) => (
                <option key={item.id} value={item.id}>
                  {formatDate(item.created_at)} · {item.uploaded_count}/{item.expected_count} ·{' '}
                  {item.status === 'finalized' ? 'Передана на обработку' : 'Можно продолжить'} ·{' '}
                  {shortId(item.id)}
                </option>
              ))}
            </select>
          </label>
          <Button
            icon="refresh"
            tone="ghost"
            disabled={busy}
            onClick={sessions.reload}
            aria-label="Обновить сохранённые загрузки"
          />
        </div>
      )}
      <ol className="import-steps" aria-label="Этапы загрузки">
        <li className={hasValidManifest ? 'step-complete' : 'step-current'}>
          <span>1</span> Список наблюдений
        </li>
        <li
          className={
            session?.validation.ready ? 'step-complete' : hasValidManifest ? 'step-current' : ''
          }
        >
          <span>2</span> Изображения
        </li>
        <li
          className={finalized ? 'step-complete' : session?.validation.ready ? 'step-current' : ''}
        >
          <span>3</span> Подготовка версии
        </li>
      </ol>
      {!session && (
        <div className="staged-csv-field">
          <label htmlFor={`csv-${galleryId}`}>CSV с рамками автомобилей</label>
          <input
            id={`csv-${galleryId}`}
            type="file"
            accept=".csv,text/csv"
            disabled={busy}
            onChange={(event) => void loadManifest(event.target.files?.[0])}
          />
          <p className="field-hint">
            Колонки <code>image_id,x,y,w,h</code>, координаты в пикселях исходного кадра. image_id —
            имя файла или имя без расширения. CSV до 2 МБ.
          </p>
          {manifestName && <Badge>{manifestName}</Badge>}
          {!!parsed?.ignoredColumns.length && (
            <p className="field-hint">
              Дополнительные колонки не участвуют в поиске: {parsed.ignoredColumns.join(', ')}.
            </p>
          )}
        </div>
      )}
      {session && (
        <p className="import-session-id">
          Загрузка <code>{session.id}</code>
        </p>
      )}
      <input
        ref={fileInput}
        className="visually-hidden"
        type="file"
        multiple
        accept="image/jpeg,image/png"
        disabled={busy || finalized}
        aria-label="Файлы для загрузки по CSV"
        onChange={(event) => {
          setFiles(Array.from(event.target.files || []));
          setError('');
          event.target.value = '';
        }}
      />
      <input
        ref={folderInput}
        className="visually-hidden"
        type="file"
        multiple
        disabled={busy || finalized}
        aria-label="Папка изображений для CSV"
        onChange={(event) => {
          setFiles(Array.from(event.target.files || []));
          setError('');
          event.target.value = '';
        }}
      />
      {!finalized && (
        <div className="staged-file-picker">
          <div className="staged-file-icon">
            <Icon name="image" size={25} />
          </div>
          <div>
            <strong>
              {files.length ? `Выбрано файлов: ${files.length}` : 'Выберите папку с изображениями'}
            </strong>
            <p>
              {files.length
                ? `${formatBytes(files.reduce((sum, file) => sum + file.size, 0))} · будут отправлены только недостающие изображения из CSV`
                : session
                  ? 'После перезагрузки выберите ту же папку. Уже принятые файлы повторно не отправляются.'
                  : 'Можно выбрать весь каталог: файлы вне списка CSV не попадут в версию.'}
            </p>
          </div>
          <div className="staged-file-actions">
            <Button disabled={busy} icon="upload" onClick={() => folderInput.current?.click()}>
              Выбрать папку
            </Button>
            <Button disabled={busy} tone="ghost" onClick={() => fileInput.current?.click()}>
              Выбрать файлы
            </Button>
          </div>
        </div>
      )}
      {(parsed || session) && (
        <>
          <div className="import-counts" aria-live="polite">
            <div>
              <strong>{total}</strong>
              <span>наблюдений в CSV</span>
            </div>
            <div>
              <strong>{uploaded}</strong>
              <span>принято сервером</span>
            </div>
            <div>
              <strong>{matched.selected.size}</strong>
              <span>готово к отправке</span>
            </div>
            <div>
              <strong>{matched.missing.length}</strong>
              <span>нужно выбрать</span>
            </div>
          </div>
          {!!csvIssues.length && (
            <Message tone="error">
              <strong>Проверьте CSV: {csvIssues.length} замечаний</strong>
              <ul className="import-issues">
                {csvIssues.slice(0, 20).map((issue, index) => (
                  <li key={index}>
                    {issue.row && `Запись ${issue.row}: `}
                    {issue.imageId && `${issue.imageId} — `}
                    {issue.message}
                  </li>
                ))}
              </ul>
              {csvIssues.length > 20 && (
                <span>
                  И ещё {csvIssues.length - 20}. Исправьте список и выберите CSV повторно.
                </span>
              )}
            </Message>
          )}
          {!!matched.issues.length && (
            <Message tone="error">
              <ul className="import-issues">
                {matched.issues.slice(0, 20).map((issue, index) => (
                  <li key={index}>
                    {issue.imageId}: {issue.message}
                  </li>
                ))}
              </ul>
              {matched.issues.length > 20 && (
                <span>И ещё {matched.issues.length - 20} файлов.</span>
              )}
            </Message>
          )}
          {!!matched.missing.length && !finalized && (
            <details className="import-missing">
              <summary>Не выбраны изображения: {matched.missing.length}</summary>
              <p>
                {matched.missing.slice(0, 30).join(', ')}
                {matched.missing.length > 30 ? ` и ещё ${matched.missing.length - 30}` : ''}
              </p>
            </details>
          )}
          {files.length > 0 && matched.unusedCount > 0 && (
            <p className="field-hint">
              Не отправляются: {matched.unusedCount} файлов — вне оставшегося списка CSV или уже
              приняты сервером.
            </p>
          )}
          {!session && (
            <p className="field-hint">
              Это предварительная проверка списка. Формат изображения и попадание BBox в кадр сервер
              проверит при загрузке каждой части.
            </p>
          )}
        </>
      )}
      {session && (
        <div className="import-transfer" aria-live="polite">
          <div className="row-between">
            <strong>
              {finalized
                ? 'Файлы переданы на подготовку версии'
                : session.validation.ready
                  ? 'Все изображения приняты'
                  : 'Подтверждённый прогресс'}
            </strong>
            <span>
              {uploaded} / {total}
            </span>
          </div>
          <progress
            max={Math.max(total, 1)}
            value={uploaded}
            aria-label="Изображения, принятые сервером"
          />
          {session.validation.issues
            .filter((issue) => issue.code !== 'missing_files')
            .map((issue, index) => (
              <Message key={index} tone="error">
                {issue.image_id && `${issue.image_id}: `}
                {issue.message}
              </Message>
            ))}
          {!finalized && session.validation.ready && (
            <p>
              Теперь можно подготовить одну полную версию. Она станет доступна для поиска только
              после явной активации.
            </p>
          )}
          {paused && (
            <p>
              {busy
                ? 'Пауза после подтверждения текущей части…'
                : 'Загрузка приостановлена. Принятые файлы сохранены.'}
            </p>
          )}
        </div>
      )}
      {error && (
        <Message tone="error">
          {error} Принятые сервером части сохранены. Повтор продолжит загрузку после проверки
          состояния.
        </Message>
      )}
      <div className="import-footer staged-footer">
        <p>
          <Icon name="shield" size={15} /> Модель ReID не подключена. Подготовка использует
          заглушку.
        </p>
        <div className="inline-items">
          {phase === 'uploading' && (
            <Button
              disabled={paused}
              onClick={() => {
                pause.current = true;
                setPaused(true);
              }}
            >
              Приостановить
            </Button>
          )}
          {!finalized && !session?.validation.ready && (
            <Button
              tone="primary"
              icon="upload"
              busy={phase === 'uploading'}
              disabled={busy || !canUpload}
              onClick={() => void upload()}
            >
              {session ? 'Продолжить загрузку' : 'Загрузить по частям'}
            </Button>
          )}
          {!finalized && session?.validation.ready && (
            <Button
              tone="primary"
              icon="check"
              busy={phase === 'finalizing'}
              disabled={busy}
              onClick={() => void finalize()}
            >
              Подготовить новую версию
            </Button>
          )}
          {finalized && (
            <Button onClick={startNew} disabled={busy}>
              Новая загрузка
            </Button>
          )}
        </div>
      </div>
      {job && (
        <JobProgress
          initial={job}
          onUpdate={setJob}
          onComplete={() => {
            onComplete();
            sessions.reload();
            toast('Полная версия подготовлена. Проверьте её и активируйте в списке версий.');
          }}
        />
      )}
    </section>
  );
}
