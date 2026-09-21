import { useEffect, useState, type FormEvent } from 'react';
import { api, ApiError } from './api';
import {
  Badge,
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
  { id: 'search', label: 'Визуальный поиск', icon: 'search', group: 'РАБОЧЕЕ ПРОСТРАНСТВО' },
  { id: 'galleries', label: 'Галереи', icon: 'layers' },
  { id: 'history', label: 'История', icon: 'history' },
  { id: 'quality', label: 'Модель и качество', icon: 'target', group: 'УПРАВЛЕНИЕ' },
  { id: 'integration', label: 'Интеграция', icon: 'code' },
  { id: 'system', label: 'Состояние сервиса', icon: 'activity' },
];
function Login({
  onLogin,
  connectionError,
}: {
  onLogin: (session: Session) => void;
  connectionError: string | null;
}) {
  const [token, setToken] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submit = async (event: FormEvent) => {
    event.preventDefault();
    setBusy(true);
    setError('');
    try {
      onLogin(await api.login(token.trim()));
      setToken('');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="login-shell">
      <header>
        <Brand />
        <span>РАБОЧЕЕ ПРОСТРАНСТВО</span>
      </header>
      <main className="login-main">
        <div className="login-art" aria-hidden="true">
          <div className="login-orbit orbit-one" />
          <div className="login-orbit orbit-two" />
          <div className="login-focus">
            <Icon name="target" size={42} />
          </div>
          <span className="orbit-point point-one" />
          <span className="orbit-point point-two" />
        </div>
        <p className="eyebrow">ВИЗУАЛЬНЫЙ ПОИСК АВТОМОБИЛЕЙ</p>
        <h1>
          Внимание к деталям.
          <br />
          <span>Уверенность в процессе.</span>
        </h1>
        <p className="login-description">
          Изображения, сравнения и решения
          <br />в одном защищённом пространстве.
        </p>
        <form className="login-form" onSubmit={submit}>
          <label htmlFor="access-token">Токен доступа</label>
          <input
            id="access-token"
            type="password"
            value={token}
            onChange={(event) => setToken(event.target.value)}
            autoComplete="current-password"
            spellCheck={false}
            placeholder="Введите токен рабочего пространства"
            required
          />
          {(error || connectionError) && <Message tone="error">{error || connectionError}</Message>}
          <Button tone="primary" busy={busy} disabled={!token.trim()} type="submit">
            Войти в пространство
            <Icon name="arrow" size={17} />
          </Button>
          <p>
            <Icon name="shield" size={14} />
            Токен выдаёт администратор сервиса
          </p>
        </form>
      </main>
      <footer>
        <span>Фалькон · Сервис визуального поиска</span>
        <Badge>Модель ещё не подключена</Badge>
      </footer>
    </div>
  );
}
function Workspace({ session, onLogout }: { session: Session; onLogout: () => void }) {
  const capabilities = useResource(api.capabilities);
  const [path, setPath] = useState(window.location.hash.slice(1) || 'search');
  const [mobileMenu, setMobileMenu] = useState(false);
  const [logoutError, setLogoutError] = useState('');
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
  const logout = async () => {
    try {
      await api.logout();
      onLogout();
    } catch (cause) {
      setLogoutError(errorMessage(cause));
    }
  };
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
        <div className="workspace-label">
          <span className="workspace-avatar">
            <Icon name="layers" size={17} />
          </span>
          <div>
            <strong>{session.workspace || 'Рабочее пространство'}</strong>
            <span>Локальный контур</span>
          </div>
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
          <div className="user-row">
            <span className="user-avatar">
              {session.role === 'admin' ? 'А' : session.role === 'operator' ? 'О' : 'Н'}
            </span>
            <div>
              <strong>
                {
                  { admin: 'Администратор', operator: 'Оператор', viewer: 'Наблюдатель' }[
                    session.role
                  ]
                }
              </strong>
              <span>Защищённая сессия</span>
            </div>
            <Button
              tone="ghost"
              icon="logout"
              aria-label="Выйти из рабочего пространства"
              onClick={logout}
            />
          </div>
        </div>
      </aside>
      <div className="workspace-main">
        <header className="topbar">
          <div className="inline-items">
            <Button
              className="mobile-menu"
              tone="ghost"
              icon="menu"
              aria-label="Открыть меню"
              onClick={() => setMobileMenu(true)}
            />
            <span className="breadcrumb">Рабочее пространство</span>
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
          {logoutError && <Message tone="error">{logoutError}</Message>}
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
          {page === 'integration' && <IntegrationPage session={session} />}
          {page === 'system' && <SystemPage />}
          <footer className="workspace-footer">
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
  useEffect(() => {
    api
      .session()
      .then(setSession)
      .catch((cause) => {
        if (!(cause instanceof ApiError && cause.status === 401)) setError(errorMessage(cause));
      })
      .finally(() => setChecking(false));
    const unauthorized = () => setSession(null);
    window.addEventListener('falcon:unauthorized', unauthorized);
    return () => window.removeEventListener('falcon:unauthorized', unauthorized);
  }, []);
  return (
    <ToastProvider>
      {checking ? (
        <div className="startup-screen">
          <Brand />
          <Loading label="Подключаем рабочее пространство…" />
        </div>
      ) : session ? (
        <Workspace session={session} onLogout={() => setSession(null)} />
      ) : (
        <Login
          onLogin={(value) => {
            setSession(value);
            setError(null);
          }}
          connectionError={error}
        />
      )}
    </ToastProvider>
  );
}
