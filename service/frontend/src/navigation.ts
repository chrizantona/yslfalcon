import type { Page } from './types';

export const mainPages: { id: Page; label: string }[] = [
  { id: 'galleries', label: 'Галереи' },
  { id: 'search', label: 'Поиск' },
  { id: 'history', label: 'История' },
];

export function pageRollDirection(fromPath: string, toPath: string): -1 | 0 | 1 {
  const from = mainPages.findIndex((page) => page.id === fromPath.split('/')[0]);
  const to = mainPages.findIndex((page) => page.id === toPath.split('/')[0]);
  if (from < 0 || to < 0 || from === to) return 0;
  return (from + 1) % mainPages.length === to ? 1 : -1;
}
