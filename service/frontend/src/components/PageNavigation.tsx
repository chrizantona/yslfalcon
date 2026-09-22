import type { Page } from '../types';
import { Icon } from './ui';
import { mainPages } from '../navigation';

export function PageNavigation({ page }: { page: Page }) {
  const pageIndex = mainPages.findIndex((item) => item.id === page);
  const currentIndex = pageIndex === -1 ? 1 : pageIndex;
  const previous = mainPages[(currentIndex + mainPages.length - 1) % mainPages.length];
  const next = mainPages[(currentIndex + 1) % mainPages.length];

  return (
    <nav className="page-navigation" aria-label="Переход между разделами">
      <a href={`#${previous.id}`} className="page-navigation-link page-navigation-previous">
        <span className="page-navigation-arrow">
          <Icon name="arrow" size={36} />
        </span>
        <span className="page-navigation-label">{previous.label}</span>
      </a>
      <a href={`#${next.id}`} className="page-navigation-link page-navigation-next">
        <span className="page-navigation-arrow">
          <Icon name="arrow" size={36} />
        </span>
        <span className="page-navigation-label">{next.label}</span>
      </a>
    </nav>
  );
}
