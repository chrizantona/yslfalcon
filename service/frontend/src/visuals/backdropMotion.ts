const TURN_DURATION = 72_000;
const FRAME_INTERVAL = 1000 / 30;

/** Owns the decorative clock; suspension never advances the vehicle offscreen. */
export function startBackdropMotion(
  paint: (angle: number) => void,
  onMotionAvailable: (available: boolean) => void,
) {
  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  let angle = -0.35;
  let paused = false;
  let frame: number | null = null;
  let previous: number | null = null;
  let disposed = false;

  const stop = () => {
    if (frame !== null) window.cancelAnimationFrame(frame);
    frame = null;
    previous = null;
  };
  const tick = (time: number) => {
    frame = null;
    if (disposed || paused || document.hidden || preference.matches) return;
    if (previous === null) previous = time;
    const elapsed = time - previous;
    if (elapsed >= FRAME_INTERVAL) {
      angle = (angle + (Math.min(elapsed, 100) / TURN_DURATION) * Math.PI * 2) % (Math.PI * 2);
      previous = time;
      paint(angle);
    }
    if (!disposed) frame = window.requestAnimationFrame(tick);
  };
  const sync = () => {
    stop();
    if (disposed) return;
    onMotionAvailable(!preference.matches);
    if (!document.hidden) {
      paint(angle);
      if (!disposed && !paused && !document.hidden && !preference.matches)
        frame = window.requestAnimationFrame(tick);
    }
  };

  preference.addEventListener('change', sync);
  document.addEventListener('visibilitychange', sync);
  sync();
  return {
    setPaused(value: boolean) {
      paused = value;
      sync();
    },
    redraw() {
      if (!disposed && !document.hidden) paint(angle);
    },
    dispose() {
      disposed = true;
      stop();
      preference.removeEventListener('change', sync);
      document.removeEventListener('visibilitychange', sync);
    },
  };
}
