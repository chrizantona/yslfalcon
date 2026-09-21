import { useEffect, useRef, useState, type FormEvent } from 'react';
import { api } from '../api';
import {
  Badge,
  Button,
  EmptyState,
  Icon,
  JobProgress,
  Loading,
  Message,
  Modal,
  PageHeading,
  SectionTitle,
  useToast,
} from '../components/ui';
import {
  errorMessage,
  formatBytes,
  formatDate,
  formatNumber,
  shortId,
  statusLabels,
} from '../domain';
import { usePagedResource, useResource } from '../hooks';
import type { Capabilities, Gallery, GalleryVersion, Job, Session } from '../types';

export function GalleriesPage({
  session,
  capabilities,
}: {
  session: Session;
  capabilities: Capabilities | null;
}) {
  const galleries = usePagedResource(api.galleries);
  const [selectedId, setSelectedId] = useState('');
  const [creating, setCreating] = useState(false);
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const toast = useToast();
  const canManage = session.role === 'admin';
  const selected =
    galleries.data?.items.find((gallery) => gallery.id === selectedId) || galleries.data?.items[0];
  const create = async (event: FormEvent) => {
    event.preventDefault();
    setSaving(true);
    setError('');
    try {
      const gallery = await api.createGallery(name.trim(), description.trim());
      setSelectedId(gallery.id);
      galleries.reload();
      setCreating(false);
      setName('');
      setDescription('');
      toast('Галерея создана. Теперь можно добавить изображения.');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setSaving(false);
    }
  };
  return (
    <>
      <PageHeading
        eyebrow="ДАННЫЕ"
        title="Галереи наблюдений"
        description="Подготовка, проверка и публикация изображений для поиска."
        action={
          <Button
            icon="plus"
            tone="primary"
            disabled={!canManage}
            onClick={() => {
              setCreating(true);
              setError('');
            }}
          >
            Создать галерею
          </Button>
        }
      />
      {galleries.error && (
        <Message
          tone="error"
          action={
            <Button onClick={galleries.reload} icon="refresh">
              Повторить
            </Button>
          }
        >
          {galleries.error}
        </Message>
      )}
      {galleries.loading && !galleries.data ? (
        <Loading />
      ) : !galleries.data?.items.length ? (
        <section className="panel">
          <EmptyState
            icon="layers"
            title="Первая галерея начинается здесь"
            action={
              canManage && (
                <Button icon="plus" onClick={() => setCreating(true)}>
                  Создать галерею
                </Button>
              )
            }
          >
            Объедините изображения для поиска. Импорт создаёт новую версию, которую вы проверяете
            перед публикацией.
          </EmptyState>
        </section>
      ) : (
        <div className="gallery-layout">
          <aside className="gallery-list">
            {galleries.data.items.map((gallery) => (
              <button
                className={`gallery-list-item ${selected?.id === gallery.id ? 'selected' : ''}`}
                key={gallery.id}
                onClick={() => setSelectedId(gallery.id)}
              >
                <span className="gallery-list-icon">
                  <Icon name="layers" size={21} />
                </span>
                <span>
                  <strong>{gallery.name}</strong>
                  <small>{formatNumber(gallery.observation_count, 0)} наблюдений</small>
                </span>
                <i className={`status-dot ${gallery.active_version_id ? 'dot-success' : ''}`} />
              </button>
            ))}
          </aside>
          {selected && (
            <GalleryDetail
              key={selected.id}
              gallery={selected}
              canManage={canManage}
              capabilities={capabilities}
              onChange={galleries.reload}
            />
          )}
        </div>
      )}
      {galleries.data && galleries.data.next_offset != null && (
        <div className="pagination-row">
          <span>
            Галерей: {galleries.data.items.length} из {galleries.data.total}
          </span>
          <Button busy={galleries.loadingMore} onClick={galleries.loadMore}>
            Загрузить ещё
          </Button>
        </div>
      )}
      {creating && (
        <Modal title="Новая галерея" onClose={() => setCreating(false)}>
          <form onSubmit={create} className="stack-form">
            <p className="muted">Галерея объединяет наблюдения, среди которых выполняется поиск.</p>
            <label>
              Название
              <input
                autoFocus
                value={name}
                onChange={(event) => setName(event.target.value)}
                maxLength={120}
                required
                placeholder="Например, контрольная выборка"
              />
            </label>
            <label>
              Описание <span className="optional">необязательно</span>
              <textarea
                value={description}
                onChange={(event) => setDescription(event.target.value)}
                rows={3}
                maxLength={1000}
                placeholder="Какие изображения будут храниться в галерее"
              />
            </label>
            {error && <Message tone="error">{error}</Message>}
            <div className="modal-actions">
              <Button onClick={() => setCreating(false)} type="button">
                Отмена
              </Button>
              <Button tone="primary" type="submit" busy={saving} disabled={!name.trim()}>
                Создать
              </Button>
            </div>
          </form>
        </Modal>
      )}
    </>
  );
}
function GalleryDetail({
  gallery,
  canManage,
  capabilities,
  onChange,
}: {
  gallery: Gallery;
  canManage: boolean;
  capabilities: Capabilities | null;
  onChange: () => void;
}) {
  const versions = useResource(() => api.versions(gallery.id), [gallery.id]);
  const [files, setFiles] = useState<File[]>([]);
  const [manifest, setManifest] = useState<File | undefined>();
  const [manifestEntries, setManifestEntries] = useState<number | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [activating, setActivating] = useState<string | null>(null);
  const [confirmVersion, setConfirmVersion] = useState<GalleryVersion | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const toast = useToast();
  const totalSize = files.reduce((sum, file) => sum + file.size, 0);
  useEffect(() => {
    setFiles([]);
    setJob(null);
  }, [gallery.id]);
  const chooseFiles = (incoming: File[]) => {
    setError('');
    if (incoming.some((file) => !/\.(png|jpe?g)$/i.test(file.name))) {
      setError('В галерею можно добавить только JPEG и PNG.');
      return;
    }
    if (capabilities && incoming.length > capabilities.limits.max_import_files) {
      setError(`Один импорт: не более ${capabilities.limits.max_import_files} файлов.`);
      return;
    }
    if (capabilities && incoming.some((file) => file.size > capabilities.limits.max_upload_bytes)) {
      setError(
        `Каждый файл должен быть не больше ${formatBytes(capabilities.limits.max_upload_bytes)}.`,
      );
      return;
    }
    if (new Set(incoming.map((file) => file.name)).size !== incoming.length) {
      setError('Имена файлов должны быть уникальны в пределах импорта.');
      return;
    }
    setFiles(incoming);
  };
  const loadManifest = async (file?: File) => {
    setError('');
    setManifest(undefined);
    setManifestEntries(null);
    if (!file) return;
    try {
      const data: unknown = JSON.parse(await file.text());
      if (!data || typeof data !== 'object' || Array.isArray(data))
        throw new Error('Манифест должен быть JSON-объектом: имя файла → параметры наблюдения.');
      setManifest(file);
      setManifestEntries(Object.keys(data).length);
    } catch (cause) {
      setError(errorMessage(cause));
    }
  };
  const upload = async () => {
    setBusy(true);
    setError('');
    try {
      setJob(await api.importGallery(gallery.id, files, manifest));
      setFiles([]);
      setManifest(undefined);
      setManifestEntries(null);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setBusy(false);
    }
  };
  const activate = async () => {
    if (!confirmVersion) return;
    setActivating(confirmVersion.id);
    setError('');
    try {
      await api.activate(gallery.id, confirmVersion.id, gallery.active_version_id);
      versions.reload();
      onChange();
      setConfirmVersion(null);
      toast('Версия галереи опубликована');
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setActivating(null);
    }
  };
  return (
    <div className="gallery-detail">
      <section className="panel">
        <div className="gallery-detail-header">
          <div>
            <Badge tone={gallery.active_version_id ? 'success' : 'neutral'} dot>
              {gallery.active_version_id ? 'Доступна для поиска' : 'Ожидает публикации'}
            </Badge>
            <h2>{gallery.name}</h2>
            {gallery.description && <p>{gallery.description}</p>}
          </div>
          <div className="observation-count">
            <strong>{formatNumber(gallery.observation_count, 0)}</strong>
            <span>наблюдений</span>
          </div>
        </div>
        <div className="gallery-id">
          <code>{gallery.id}</code>
          <span>Создана {formatDate(gallery.created_at)}</span>
        </div>
      </section>
      <section className="panel">
        <SectionTitle
          title="Версии галереи"
          caption="Каждый поиск закреплён за конкретной версией"
          action={
            <Button
              icon="refresh"
              tone="ghost"
              aria-label="Обновить версии"
              onClick={versions.reload}
            />
          }
        />
        {versions.error && <Message tone="error">{versions.error}</Message>}
        {versions.loading && !versions.data ? (
          <Loading />
        ) : versions.data?.items.length ? (
          <div className="table-scroll">
            <table>
              <thead>
                <tr>
                  <th>Версия</th>
                  <th>Наблюдения</th>
                  <th>Состояние</th>
                  <th>Создана</th>
                  <th>
                    <span className="visually-hidden">Действия</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {versions.data.items.map((version) => (
                  <tr key={version.id}>
                    <td>
                      <strong>v{version.number}</strong>
                      <small title={version.model_id}>{shortId(version.model_id)}</small>
                    </td>
                    <td>{formatNumber(version.count, 0)}</td>
                    <td>
                      <Badge
                        tone={
                          version.id === gallery.active_version_id
                            ? 'success'
                            : version.status === 'failed'
                              ? 'danger'
                              : 'neutral'
                        }
                      >
                        {version.id === gallery.active_version_id
                          ? 'Активна'
                          : statusLabels[version.status] || version.status}
                      </Badge>
                    </td>
                    <td>{formatDate(version.created_at)}</td>
                    <td>
                      {canManage &&
                        version.id !== gallery.active_version_id &&
                        ['ready', 'retired', 'active'].includes(version.status) && (
                          <Button onClick={() => setConfirmVersion(version)}>Активировать</Button>
                        )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="inline-empty">
            <Icon name="layers" size={23} />
            <p>После импорта здесь появится первая версия галереи.</p>
          </div>
        )}
      </section>
      {canManage && (
        <section className="panel">
          <SectionTitle
            title="Подготовить новую версию"
            caption="Импорт полностью заменяет состав галереи после активации"
          />
          <input
            ref={fileInput}
            className="visually-hidden"
            type="file"
            multiple
            accept="image/jpeg,image/png"
            aria-label="Изображения для импорта"
            onChange={(event) => {
              chooseFiles(Array.from(event.target.files || []));
              event.target.value = '';
            }}
          />
          <button
            className="import-dropzone"
            onClick={() => fileInput.current?.click()}
            onDragOver={(event) => event.preventDefault()}
            onDrop={(event) => {
              event.preventDefault();
              chooseFiles(Array.from(event.dataTransfer.files));
            }}
          >
            <Icon name="upload" size={26} />
            <strong>
              {files.length
                ? `Выбрано файлов: ${files.length}`
                : 'Выберите изображения для галереи'}
            </strong>
            <span>
              {files.length
                ? `${formatBytes(totalSize)} · нажмите, чтобы заменить набор`
                : 'JPEG и PNG · можно перетащить несколько файлов'}
            </span>
          </button>
          {files.length > 0 && (
            <div className="file-list">
              {files.slice(0, 8).map((file) => (
                <span key={file.name}>
                  <Icon name="image" size={13} />
                  {file.name}
                </span>
              ))}
              {files.length > 8 && <span>и ещё {files.length - 8}</span>}
            </div>
          )}
          <div className="manifest-field">
            <label htmlFor="manifest-file">
              Манифест рамок <span className="optional">необязательно</span>
            </label>
            <input
              id="manifest-file"
              type="file"
              accept="application/json,.json"
              onChange={(event) => {
                void loadManifest(event.target.files?.[0]);
              }}
            />
            {manifest && (
              <p className="field-hint">
                {manifest.name}: {manifestEntries} записей. Координаты и соответствие файлам
                проверит сервер.
              </p>
            )}
            <p className="field-hint">
              Без манифеста каждый файл используется целиком. Формат:{' '}
              <code>{'{"car.jpg":{"bbox":[10,20,300,180]}}'}</code>
            </p>
          </div>
          {error && <Message tone="error">{error}</Message>}
          <div className="import-footer">
            <p>
              <Icon name="shield" size={15} />
              Действующая версия продолжит обслуживать поиск
            </p>
            <Button
              tone="primary"
              icon="upload"
              busy={busy}
              disabled={!files.length || job?.status === 'pending' || job?.status === 'running'}
              onClick={upload}
            >
              Начать импорт
            </Button>
          </div>
        </section>
      )}
      {job && (
        <JobProgress
          initial={job}
          onUpdate={setJob}
          onComplete={() => {
            versions.reload();
            onChange();
            toast('Импорт завершён. Проверьте и активируйте новую версию.');
          }}
        />
      )}
      {confirmVersion && (
        <Modal
          title={`Активировать версию v${confirmVersion.number}?`}
          onClose={() => setConfirmVersion(null)}
        >
          <div className="stack-form">
            <p>
              Новые поиски по галерее «{gallery.name}» будут использовать {confirmVersion.count}{' '}
              наблюдений из этой версии. Сохранённые результаты останутся привязаны к прежним
              данным.
            </p>
            {error && <Message tone="error">{error}</Message>}
            <div className="modal-actions">
              <Button onClick={() => setConfirmVersion(null)}>Отмена</Button>
              <Button tone="primary" busy={activating === confirmVersion.id} onClick={activate}>
                Активировать версию
              </Button>
            </div>
          </div>
        </Modal>
      )}
    </div>
  );
}
