import { useEffect, useState, type RefObject } from 'react';
import { flushSync } from 'react-dom';
import { pageRollDirection } from './navigation';
import { createPageTransition } from './visuals/pageTransition';

const readPath = () => window.location.hash.slice(1) || 'search';

export function usePageRoute(workspace: RefObject<HTMLDivElement | null>) {
  const [path, setPath] = useState(readPath);

  useEffect(() => {
    const root = document.documentElement;
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const controller = createPageTransition({
      initialPath: readPath(),
      direction: pageRollDirection,
      allowed: () =>
        !preference.matches && !document.hidden && !document.querySelector('dialog[open]'),
      commit: (next) => {
        const focusedContent = workspace.current?.contains(document.activeElement);
        flushSync(() => setPath(next));
        if (focusedContent && document.activeElement === document.body)
          document.getElementById('main-content')?.focus({ preventScroll: true });
      },
      setDirection: (direction) => {
        if (direction === 0) {
          delete root.dataset.pageRoll;
          root.style.removeProperty('--page-roll-radius');
          root.style.removeProperty('--page-roll-origin');
          return;
        }
        const bounds = workspace.current?.getBoundingClientRect();
        const radius = Math.min(960, Math.max(240, (bounds?.width ?? 800) * 0.8));
        const origin = Math.max(0, window.innerHeight * 0.45 - (bounds?.top ?? 0));
        root.style.setProperty('--page-roll-radius', `${radius}px`);
        root.style.setProperty('--page-roll-origin', `${origin}px`);
        root.dataset.pageRoll = direction === 1 ? 'next' : 'previous';
      },
      start:
        typeof document.startViewTransition === 'function'
          ? (update) => {
              const transition = document.startViewTransition(update);
              // Skipping a superseded transition rejects ready but still runs its update.
              void transition.ready.catch((cause: unknown) => {
                if (!(cause instanceof DOMException && cause.name === 'AbortError'))
                  console.warn('Page transition snapshot unavailable', cause);
              });
              return transition;
            }
          : undefined,
    });
    const update = () => controller.navigate(readPath());
    const finish = () => controller.finish();
    const motionChange = () => {
      if (preference.matches) finish();
    };
    const visibilityChange = () => {
      if (document.hidden) finish();
    };
    window.addEventListener('hashchange', update);
    window.addEventListener('resize', finish);
    window.addEventListener('keydown', finish, true);
    window.addEventListener('wheel', finish, { passive: true });
    document.addEventListener('visibilitychange', visibilityChange);
    preference.addEventListener('change', motionChange);
    return () => {
      window.removeEventListener('hashchange', update);
      window.removeEventListener('resize', finish);
      window.removeEventListener('keydown', finish, true);
      window.removeEventListener('wheel', finish);
      document.removeEventListener('visibilitychange', visibilityChange);
      preference.removeEventListener('change', motionChange);
      controller.dispose();
    };
  }, [workspace]);

  return path;
}
