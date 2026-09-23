import { api } from '../api';
import {
  Badge,
  EmptyState,
  Icon,
  Loading,
  Message,
  PageHeading,
  SectionTitle,
} from '../components/ui';
import { useResource } from '../hooks';
import { formatBytes, statusLabels } from '../domain';
import type { Capabilities } from '../types';

export function QualityPage({ capabilities }: { capabilities: Capabilities | null }) {
  const models = useResource(api.models);
  const calibrations = useResource(api.calibrations);
  const preview = !capabilities || capabilities.engine.mode === 'preview';
  return (
    <>
      <PageHeading
        title="Модель и качество"
        description="Конфигурация алгоритма и основания для решения о совпадении."
      />
      <div className="quality-intro panel">
        <div className="quality-symbol">
          <Icon name="target" size={32} />
        </div>
        <div>
          <Badge tone="warning" dot>
            {preview ? 'Модель не подключена' : 'Конфигурация модели'}
          </Badge>
          <h2>
            {preview
              ? 'Интерфейс готов. Модель — следующий шаг.'
              : 'Проверяемая конфигурация поиска'}
          </h2>
          <p>
            {preview
              ? 'Сейчас работает техническая заглушка. Можно пройти полный сценарий: импортировать данные, выполнить поиск, сравнить изображения и сохранить отчёт. Порядок кандидатов не является результатом обученной ReID-модели.'
              : 'Здесь показаны фактически подключённые конфигурации. Показатели качества должны подтверждаться отдельным протоколом оценки.'}
          </p>
        </div>
      </div>
      <div className="two-column">
        <section className="panel">
          <SectionTitle title="Вычислительный модуль" caption="Данные из работающего сервиса" />
          {models.error && <Message tone="error">{models.error}</Message>}
          {models.loading && !models.data ? (
            <Loading />
          ) : models.data?.items.length ? (
            models.data.items.map((model) => (
              <div className="model-entry" key={model.id}>
                <div className="row-between">
                  <strong>{model.mode === 'preview' ? 'Модель-заглушка' : 'Модель поиска'}</strong>
                  <Badge tone={model.status === 'ready' ? 'success' : 'danger'} dot>
                    {model.status === 'ready'
                      ? 'Модуль доступен'
                      : statusLabels[model.status] || model.status}
                  </Badge>
                </div>
                <dl className="detail-list">
                  <div>
                    <dt>Идентификатор</dt>
                    <dd>
                      <code>{model.id}</code>
                    </dd>
                  </div>
                  <div>
                    <dt>Размерность признака</dt>
                    <dd>{model.dimension}</dd>
                  </div>
                  <div>
                    <dt>Режим</dt>
                    <dd>{model.mode}</dd>
                  </div>
                  {model.device && (
                    <div>
                      <dt>Устройство</dt>
                      <dd>{model.device}</dd>
                    </div>
                  )}
                  {model.precision && (
                    <div>
                      <dt>Точность вычислений</dt>
                      <dd>{model.precision}</dd>
                    </div>
                  )}
                  {model.weights_bytes != null && (
                    <div>
                      <dt>Вес файлов модели</dt>
                      <dd>{formatBytes(model.weights_bytes)}</dd>
                    </div>
                  )}
                </dl>
                {model.manifest && (
                  <details className="technical-details">
                    <summary>Манифест модуля</summary>
                    <pre>{JSON.stringify(model.manifest, null, 2)}</pre>
                  </details>
                )}
              </div>
            ))
          ) : (
            <EmptyState icon="target" title="Нет настроенных моделей">
              Сервис не сообщил о доступных вычислительных модулях.
            </EmptyState>
          )}
        </section>
        <section className="panel">
          <SectionTitle
            title="Политика отказа"
            caption="При каких условиях допустимо подтвердить первый результат"
          />
          {calibrations.error && <Message tone="error">{calibrations.error}</Message>}
          {calibrations.loading && !calibrations.data ? (
            <Loading />
          ) : calibrations.data?.items.length ? (
            calibrations.data.items.map((policy, index) => (
              <div className="policy-entry" key={index}>
                <Badge tone="purple">Настроенная политика {index + 1}</Badge>
                <pre>{JSON.stringify(policy, null, 2)}</pre>
              </div>
            ))
          ) : (
            <div className="calibration-empty">
              <span className="outline-icon">
                <Icon name="shield" size={25} />
              </span>
              <h3>Калибровка не настроена</h3>
              <p>
                Выдача доступна для проверки. Сервис не объявляет совпадения подтверждёнными и не
                подставляет произвольный порог.
              </p>
              <Badge>calibration_unavailable</Badge>
            </div>
          )}
        </section>
      </div>
      <section className="panel">
        <SectionTitle title="Как читать результат" />
        <div className="principle-list">
          <div>
            <span>01</span>
            <div>
              <h3>Ранжирование</h3>
              <p>
                Кандидаты расположены по рассчитанному сходству. Оценки не являются вероятностью
                идентичности автомобиля.
              </p>
            </div>
          </div>
          <div>
            <span>02</span>
            <div>
              <h3>Решение о совпадении</h3>
              <p>
                Допустимо только с подходящей калибровкой. Решение относится к первому кандидату;
                остальные остаются альтернативами.
              </p>
            </div>
          </div>
          <div>
            <span>03</span>
            <div>
              <h3>Заключение оператора</h3>
              <p>
                Сохраняется независимо от алгоритма. Ни оценка, ни порядок выдачи задним числом не
                меняются.
              </p>
            </div>
          </div>
        </div>
      </section>
    </>
  );
}
