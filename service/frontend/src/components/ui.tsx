import {
  createContext,
  useContext,
  useEffect,
  useId,
  useRef,
  useState,
  type ButtonHTMLAttributes,
  type ReactNode,
} from 'react';
import {
  errorMessage,
  formatDate,
  jobLabels,
  normalizeProgress,
  safeDownloadUrl,
  shortId,
  statusLabels,
} from '../domain';
import { useJob } from '../hooks';
import type { Job } from '../types';

export type IconName =
  | 'search'
  | 'layers'
  | 'history'
  | 'shield'
  | 'code'
  | 'activity'
  | 'arrow'
  | 'upload'
  | 'plus'
  | 'close'
  | 'check'
  | 'chevron'
  | 'refresh'
  | 'download'
  | 'image'
  | 'expand'
  | 'eye'
  | 'alert'
  | 'copy'
  | 'menu'
  | 'target'
  | 'spark';
const paths: Record<IconName, ReactNode> = {
  search: (
    <>
      <circle cx="10.5" cy="10.5" r="6.5" />
      <path d="m16 16 4.5 4.5" />
    </>
  ),
  layers: (
    <>
      <path d="m12 3 9 5-9 5-9-5 9-5Z" />
      <path d="m3 12 9 5 9-5M3 16l9 5 9-5" />
    </>
  ),
  history: (
    <>
      <path d="M3 11a9 9 0 1 1 2 7M3 4v7h7M12 7v5l3 2" />
    </>
  ),
  shield: (
    <>
      <path d="m12 3 8 3v6c0 4-5 8-8 9-3-1-8-5-8-9V6l8-3Z" />
      <path d="m8 12 3 3 5-6" />
    </>
  ),
  code: (
    <>
      <path d="m8 6-6 6 6 6m8-12 6 6-6 6m-3-16-2 20" />
    </>
  ),
  activity: (
    <>
      <path d="M2 12h5l3-8 4 16 3-8h5" />
    </>
  ),
  arrow: (
    <>
      <path d="M4 12h16m-6-6 6 6-6 6" />
    </>
  ),
  upload: (
    <>
      <path d="M12 16V3m-5 5 5-5 5 5M4 15v6h16v-6" />
    </>
  ),
  plus: <path d="M12 5v14M5 12h14" />,
  close: <path d="m6 6 12 12M6 18 18 6" />,
  check: <path d="m5 12 4 4L19 6" />,
  chevron: <path d="m9 5 7 7-7 7" />,
  refresh: (
    <>
      <path d="M3 10a9 9 0 0 1 16-5l2 3M21 14a9 9 0 0 1-16 5l-2-3M21 2v6h-6M3 22v-6h6" />
    </>
  ),
  download: (
    <>
      <path d="M12 3v13m-5-5 5 5 5-5M4 17v4h16v-4" />
    </>
  ),
  image: (
    <>
      <rect x="3" y="3" width="18" height="18" rx="3" />
      <circle cx="8" cy="8" r="1" />
      <path d="m3 17 5-5 4 4 3-3 6 6" />
    </>
  ),
  expand: (
    <>
      <path d="M3 9V3h6m6 0h6v6M3 15v6h6m6 0h6v-6" />
    </>
  ),
  eye: (
    <>
      <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12Z" />
      <circle cx="12" cy="12" r="3" />
    </>
  ),
  alert: (
    <>
      <path d="m12 3 10 18H2L12 3Z" />
      <path d="M12 9v5m0 3v1" />
    </>
  ),
  copy: (
    <>
      <rect x="8" y="8" width="13" height="13" rx="2" />
      <path d="M16 8V3H3v13h5" />
    </>
  ),
  menu: <path d="M4 6h16M4 12h16M4 18h16" />,
  target: (
    <>
      <circle cx="12" cy="12" r="8" />
      <circle cx="12" cy="12" r="3" />
      <path d="M12 1v3m0 16v3M1 12h3m16 0h3" />
    </>
  ),
  spark: (
    <>
      <path d="m12 3 2.5 6.5L21 12l-6.5 2.5L12 21l-2.5-6.5L3 12l6.5-2.5L12 3Z" />
    </>
  ),
};
export function Icon({
  name,
  size = 20,
  className = '',
}: {
  name: IconName;
  size?: number;
  className?: string;
}) {
  return (
    <svg
      className={className}
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}
export function Brand({ compact = false }: { compact?: boolean }) {
  return (
    <div className="brand">
      <span className="brand-mark">
        <svg width="29" height="30" viewBox="0 0 32 32" fill="none" aria-hidden="true">
          <path d="m3 5 12 4 14-4-10 11-4 13-3-12L3 5Z" fill="currentColor" />
          <path d="m13 12 8-2-6 9-2-7Z" fill="var(--canvas)" />
        </svg>
      </span>
      {!compact && (
        <span>
          ФАЛЬКОН<small>ВИЗУАЛЬНЫЙ ПОИСК</small>
        </span>
      )}
    </div>
  );
}
export function Button({
  children,
  tone = 'default',
  icon,
  busy,
  className = '',
  disabled,
  ...props
}: ButtonHTMLAttributes<HTMLButtonElement> & {
  tone?: 'primary' | 'default' | 'ghost' | 'danger';
  icon?: IconName;
  busy?: boolean;
}) {
  return (
    <button className={`button button-${tone} ${className}`} disabled={disabled || busy} {...props}>
      {busy ? <span className="spinner" /> : icon && <Icon name={icon} size={17} />}
      {children}
    </button>
  );
}
export function Badge({
  children,
  tone = 'neutral',
  dot = false,
}: {
  children: ReactNode;
  tone?: 'neutral' | 'success' | 'warning' | 'danger' | 'purple';
  dot?: boolean;
}) {
  return (
    <span className={`badge badge-${tone}`}>
      {dot && <i />}
      {children}
    </span>
  );
}
export function Message({
  children,
  tone = 'info',
  action,
}: {
  children: ReactNode;
  tone?: 'info' | 'error' | 'warning' | 'success';
  action?: ReactNode;
}) {
  return (
    <div className={`message message-${tone}`} role={tone === 'error' ? 'alert' : 'status'}>
      <Icon name={tone === 'success' ? 'check' : 'alert'} size={18} />
      <div>{children}</div>
      {action}
    </div>
  );
}
export function EmptyState({
  icon = 'search',
  title,
  children,
  action,
}: {
  icon?: IconName;
  title: string;
  children: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <Icon name={icon} size={30} />
      </div>
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Loading({ label = 'Загружаем данные…' }: { label?: string }) {
  return (
    <div className="loading" role="status">
      <span className="spinner" />
      {label}
    </div>
  );
}
export function PageHeading({
  title,
  description,
  action,
}: {
  title: string;
  description: string;
  action?: ReactNode;
}) {
  return (
    <div className="page-heading">
      <div>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {action && <div className="heading-action">{action}</div>}
    </div>
  );
}
export function SectionTitle({
  title,
  caption,
  action,
}: {
  title: string;
  caption?: string;
  action?: ReactNode;
}) {
  return (
    <div className="section-title">
      <div>
        <h2>{title}</h2>
        {caption && <p>{caption}</p>}
      </div>
      {action}
    </div>
  );
}
export function Modal({
  title,
  children,
  onClose,
  wide = false,
}: {
  title: string;
  children: ReactNode;
  onClose: () => void;
  wide?: boolean;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const titleId = useId();
  useEffect(() => {
    dialog.current?.showModal();
    return () => dialog.current?.close();
  }, []);
  return (
    <dialog
      ref={dialog}
      aria-labelledby={titleId}
      className={`modal ${wide ? 'modal-wide' : ''}`}
      onCancel={(event) => {
        event.preventDefault();
        onClose();
      }}
      onClick={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div className="modal-heading">
        <h2 id={titleId}>{title}</h2>
        <Button tone="ghost" icon="close" aria-label="Закрыть окно" onClick={onClose} />
      </div>
      {children}
    </dialog>
  );
}
export function JobProgress({
  initial,
  onComplete,
  onUpdate,
}: {
  initial: Job;
  onComplete?: (job: Job) => void;
  onUpdate?: (job: Job) => void;
}) {
  const { job, error, retry } = useJob(initial, onComplete, onUpdate);
  if (!job) return null;
  const value = normalizeProgress(job.progress);
  const download = safeDownloadUrl(job.result?.download_url);
  const failed = job.status === 'failed';
  return (
    <div className={`job-card ${failed ? 'job-failed' : ''}`} aria-live="polite">
      <div className="row-between">
        <strong>{jobLabels[job.kind] || job.kind}</strong>
        <Badge tone={failed ? 'danger' : job.status === 'completed' ? 'success' : 'purple'}>
          {statusLabels[job.status]}
        </Badge>
      </div>
      <div className="job-meta">
        <code>{shortId(job.id)}</code>
        <span>{formatDate(job.created_at)}</span>
      </div>
      {['pending', 'running'].includes(job.status) && (
        <progress max="100" value={value} aria-label="Прогресс задания" />
      )}
      {job.message && <p>{job.message}</p>}
      {failed && (
        <Message tone="error">
          {typeof job.error === 'object'
            ? job.error?.message
            : job.error || 'Задание завершилось с ошибкой'}
        </Message>
      )}
      {error && (
        <Message
          tone="error"
          action={
            <Button onClick={retry} icon="refresh">
              Повторить
            </Button>
          }
        >
          {error}
        </Message>
      )}
      {download && (
        <a className="button button-primary" href={download} download>
          <Icon name="download" size={17} />
          Скачать файл
        </a>
      )}
    </div>
  );
}
const ToastContext = createContext<(message: string, error?: boolean) => void>(() => {});
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toast, setToast] = useState<{ message: string; error: boolean } | null>(null);
  useEffect(() => {
    if (!toast) return;
    const timer = window.setTimeout(() => setToast(null), 6500);
    return () => window.clearTimeout(timer);
  }, [toast]);
  return (
    <ToastContext.Provider value={(message, error = false) => setToast({ message, error })}>
      {children}
      {toast && (
        <div
          className={`toast ${toast.error ? 'toast-error' : ''}`}
          role={toast.error ? 'alert' : 'status'}
        >
          <Icon name={toast.error ? 'alert' : 'check'} size={18} />
          <span>{toast.message}</span>
          <Button
            tone="ghost"
            icon="close"
            aria-label="Закрыть уведомление"
            onClick={() => setToast(null)}
          />
        </div>
      )}
    </ToastContext.Provider>
  );
}
export const useToast = () => useContext(ToastContext);
export function CopyButton({ text, label = 'Копировать' }: { text: string; label?: string }) {
  const toast = useToast();
  return (
    <Button
      icon="copy"
      onClick={async () => {
        try {
          await navigator.clipboard.writeText(text);
          toast('Скопировано');
        } catch (error) {
          toast(errorMessage(error), true);
        }
      }}
    >
      {label}
    </Button>
  );
}
