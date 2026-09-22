import { useEffect, useRef, useState, type ReactNode } from 'react';

const SPACING = 22;
const BASE_RADIUS = 1;
const RADIUS_GROWTH = 1.8;
const SIGMA = 90;

export function ResultsSurface({ children }: { children: ReactNode }) {
  const surface = useRef<HTMLElement>(null);
  const canvas = useRef<HTMLCanvasElement>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const element = surface.current;
    const backdrop = canvas.current;
    const context = backdrop?.getContext('2d');
    if (!element || !backdrop || !context) return;

    const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
    let pointer: { x: number; y: number } | null = null;
    let frame: number | null = null;
    let painted = false;

    const draw = () => {
      frame = null;
      const bounds = element.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      const width = Math.ceil(bounds.width * ratio);
      const height = Math.ceil(bounds.height * ratio);
      if (backdrop.width !== width) backdrop.width = width;
      if (backdrop.height !== height) backdrop.height = height;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);
      context.fillStyle = 'rgba(255, 255, 255, 0.17)';
      context.beginPath();
      for (let y = SPACING / 2; y < bounds.height; y += SPACING) {
        for (let x = SPACING / 2; x < bounds.width; x += SPACING) {
          const distanceSquared = pointer
            ? (x - (pointer.x - bounds.left)) ** 2 + (y - (pointer.y - bounds.top)) ** 2
            : Infinity;
          const radius =
            BASE_RADIUS + RADIUS_GROWTH * Math.exp(-distanceSquared / (2 * SIGMA ** 2));
          context.moveTo(x + radius, y);
          context.arc(x, y, radius, 0, 2 * Math.PI);
        }
      }
      context.fill();
      if (!painted) {
        painted = true;
        setReady(true);
      }
    };

    // Coalesce input events; an idle surface never schedules its own animation frames.
    const scheduleDraw = () => {
      if (frame === null) frame = window.requestAnimationFrame(draw);
    };
    const resetPointer = () => {
      pointer = null;
      scheduleDraw();
    };
    const movePointer = (event: PointerEvent) => {
      pointer =
        event.pointerType === 'mouse' && !motionPreference.matches
          ? { x: event.clientX, y: event.clientY }
          : null;
      scheduleDraw();
    };
    const onScroll = () => {
      if (pointer) scheduleDraw();
    };
    const observer =
      typeof ResizeObserver === 'undefined' ? null : new ResizeObserver(scheduleDraw);
    observer?.observe(element);
    element.addEventListener('pointermove', movePointer, { passive: true });
    element.addEventListener('pointerleave', resetPointer);
    element.addEventListener('pointercancel', resetPointer);
    window.addEventListener('resize', scheduleDraw, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true, capture: true });
    motionPreference.addEventListener('change', resetPointer);
    scheduleDraw();

    return () => {
      observer?.disconnect();
      element.removeEventListener('pointermove', movePointer);
      element.removeEventListener('pointerleave', resetPointer);
      element.removeEventListener('pointercancel', resetPointer);
      window.removeEventListener('resize', scheduleDraw);
      window.removeEventListener('scroll', onScroll, true);
      motionPreference.removeEventListener('change', resetPointer);
      if (frame !== null) window.cancelAnimationFrame(frame);
    };
  }, []);

  return (
    <section
      ref={surface}
      className={`results-area results-surface${ready ? ' results-backdrop-ready' : ''}`}
      aria-live="polite"
    >
      <canvas ref={canvas} className="results-backdrop" aria-hidden="true" />
      <div className="results-surface-content">{children}</div>
    </section>
  );
}
