import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import {
  Brand,
  Button,
  Icon,
  Loading,
  Message,
  ToastProvider,
  type IconName,
} from './components/ui';
import { errorMessage } from './domain';
import { useResource } from './hooks';
import { GalleriesPage } from './pages/GalleriesPage';
import { HistoryPage } from './pages/HistoryPage';
import { IntegrationPage } from './pages/IntegrationPage';
import { QualityPage } from './pages/QualityPage';
import { SystemPage } from './pages/SystemPage';
import { SearchPage } from './pages/SearchPage';
import type { Page, Session } from './types';

const navigation: { id: Page; label: string; icon: IconName; group?: string }[] = [
  { id: 'search', label: 'Визуальный поиск', icon: 'search', group: 'ПОИСК И ДАННЫЕ' },
  { id: 'galleries', label: 'Галереи', icon: 'layers' },
  { id: 'history', label: 'История', icon: 'history' },
  { id: 'quality', label: 'Модель и качество', icon: 'target', group: 'УПРАВЛЕНИЕ' },
  { id: 'integration', label: 'Интеграция', icon: 'code' },
  { id: 'system', label: 'Состояние сервиса', icon: 'activity' },
];
function ConnectionState({
  checking,
  error,
  onRetry,
}: {
  checking: boolean;
  error: string | null;
  onRetry: () => void;
}) {
  return (
    <div className="startup-screen">
      <Brand />
      {checking ? (
        <Loading label="Подключаем сервис…" />
      ) : (
        <main className="connection-problem">
          <span className="outline-icon">
            <Icon name="activity" size={28} />
          </span>
          <h1>Не удалось подключиться</h1>
          <p>Проверьте, что сервис запущен, и повторите подключение.</p>
          {error && (
            <details>
              <summary>Подробности</summary>
              <p role="alert">{error}</p>
            </details>
          )}
          <Button tone="primary" icon="refresh" onClick={onRetry}>
            Повторить подключение
          </Button>
        </main>
      )}
    </div>
  );
}
function ServiceShell({ session }: { session: Session }) {
  const capabilities = useResource(api.capabilities);
  const [path, setPath] = useState(window.location.hash.slice(1) || 'search');
  const [mobileMenu, setMobileMenu] = useState(false);
  useEffect(() => {
    const update = () => {
      setPath(window.location.hash.slice(1) || 'search');
      setMobileMenu(false);
    };
    window.addEventListener('hashchange', update);
    return () => window.removeEventListener('hashchange', update);
  }, []);
  const navigate = (next: string) => {
    window.location.hash = next;
  };
  const page = navigation.some((item) => item.id === path.split('/')[0])
    ? (path.split('/')[0] as Page)
    : 'search';
  const searchId = page === 'search' ? path.split('/')[1] : undefined;
  return (
    <div className="app-shell">
      <a
        href="#main-content"
        className="skip-link"
        onClick={(event) => {
          event.preventDefault();
          document.getElementById('main-content')?.focus();
        }}
      >
        Перейти к содержимому
      </a>
      {mobileMenu && (
        <button
          className="sidebar-backdrop"
          aria-label="Закрыть меню"
          onClick={() => setMobileMenu(false)}
        />
      )}
      <aside className={`sidebar ${mobileMenu ? 'sidebar-open' : ''}`}>
        <div className="sidebar-brand">
          <Brand />
        </div>
        <nav aria-label="Главное меню">
          {navigation.map((item) => (
            <div key={item.id}>
              {item.group && <p className="nav-group">{item.group}</p>}
              <a
                href={`#${item.id}`}
                className={page === item.id ? 'nav-item active' : 'nav-item'}
                aria-current={page === item.id ? 'page' : undefined}
              >
                <Icon name={item.icon} size={19} />
                <span>{item.label}</span>
                {page === item.id && <i />}
              </a>
            </div>
          ))}
        </nav>
        <div className="sidebar-bottom">
          <div className="sidebar-model">
            <span className="status-dot dot-warning" />
            <div>
              <strong>Модель-заглушка</strong>
              <span>Ожидает подключения</span>
            </div>
            <a href="#quality" aria-label="Открыть состояние модели">
              <Icon name="chevron" size={15} />
            </a>
          </div>
        </div>
      </aside>
      <div className="service-main">
        <header className="topbar">
          <div className="inline-items">
            <Button
              className="mobile-menu"
              tone="ghost"
              icon="menu"
              aria-label="Открыть меню"
              onClick={() => setMobileMenu(true)}
            />
            <span className="breadcrumb">Фалькон</span>
            <Icon name="chevron" size={13} />
            <span>{navigation.find((item) => item.id === page)?.label}</span>
          </div>
          <div className="topbar-meta">
            <span className="local-status">
              <i />
              Локальный сервис
            </span>
            <span className="topbar-version">v{capabilities.data?.service_version || '—'}</span>
          </div>
        </header>
        <main id="main-content" tabIndex={-1}>
          <div className="model-banner">
            <Icon name="activity" size={17} />
            <p>
              <strong>Модель ещё не подключена.</strong> Сервис работает с технической заглушкой;
              результаты не подтверждают идентичность автомобилей.
            </p>
            <a href="#quality">
              Подробнее
              <Icon name="chevron" size={13} />
            </a>
          </div>
          {capabilities.error && (
            <Message tone="error" action={<Button onClick={capabilities.reload}>Повторить</Button>}>
              {capabilities.error}
            </Message>
          )}
          {page === 'search' && (
            <SearchPage
              searchId={searchId}
              capabilities={capabilities.data}
              session={session}
              navigate={navigate}
            />
          )}
          {page === 'galleries' && (
            <GalleriesPage session={session} capabilities={capabilities.data} />
          )}
          {page === 'history' && <HistoryPage navigate={navigate} />}
          {page === 'quality' && <QualityPage capabilities={capabilities.data} />}
          {page === 'integration' && <IntegrationPage />}
          {page === 'system' && <SystemPage />}
          <footer className="service-footer">
            <span>Фалькон · Визуальный поиск</span>
            <span>Изображения и результаты в вашем контуре</span>
          </footer>
        </main>
      </div>
    </div>
  );
}
export default function App() {
  const [session, setSession] = useState<Session | null>(null);
  const [checking, setChecking] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const active = useRef(false);
  const connecting = useRef(false);
  const canRecover = useRef(true);
  const connect = useCallback(async () => {
    if (connecting.current) return;
    connecting.current = true;
    setChecking(true);
    setError(null);
    setSession(null);
    try {
      const next = await api.bootstrapSession();
      if (active.current) setSession(next);
    } catch (cause) {
      if (active.current) setError(errorMessage(cause));
    } finally {
      connecting.current = false;
      if (active.current) setChecking(false);
    }
  }, []);
  useEffect(() => {
    active.current = true;
    void connect();
    const unauthorized = () => {
      if (connecting.current) return;
      if (!canRecover.current) {
        setSession(null);
        setError('Подключение не восстановилось. Повторите попытку после проверки сервиса.');
        return;
      }
      canRecover.current = false;
      void connect();
    };
    // Allow another automatic recovery only after a protected operation actually succeeds.
    const connected = () => {
      if (!connecting.current) canRecover.current = true;
    };
    window.addEventListener('falcon:unauthorized', unauthorized);
    window.addEventListener('falcon:connected', connected);
    return () => {
      active.current = false;
      window.removeEventListener('falcon:unauthorized', unauthorized);
      window.removeEventListener('falcon:connected', connected);
    };
  }, [connect]);
  const retry = () => {
    canRecover.current = true;
    void connect();
  };
  return (
    <ToastProvider>
      {session && !checking ? (
        <ServiceShell session={session} />
      ) : (
        <ConnectionState checking={checking} error={error} onRetry={retry} />
      )}
    </ToastProvider>
  );
}
