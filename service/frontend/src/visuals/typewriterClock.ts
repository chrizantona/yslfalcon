type TypewriterRun = {
  length: number;
  paint: (visible: number) => void;
  restore: () => void;
};

type Timer = {
  schedule: (callback: () => void, delay: number) => number;
  cancel: (id: number) => void;
};

/** One owner and one timer: changing targets immediately restores the previous label. */
export function createTypewriterClock(timer: Timer) {
  let active: TypewriterRun | null = null;
  let pending: number | null = null;
  let generation = 0;
  let disposed = false;

  const stop = () => {
    generation += 1;
    if (pending !== null) timer.cancel(pending);
    pending = null;
    const previous = active;
    active = null;
    previous?.restore();
  };

  return {
    start(run: TypewriterRun) {
      if (disposed) return;
      stop();
      if (disposed) return;
      active = run;
      const token = generation;
      const length = Math.max(0, Math.trunc(run.length));
      const interval = Math.max(16, Math.min(42, 900 / Math.max(length, 1)));
      let visible = 0;
      const ownsRun = () => !disposed && active === run && generation === token;
      const tick = () => {
        if (!ownsRun()) return;
        pending = null;
        visible += 1;
        run.paint(visible);
        if (ownsRun() && visible < length) pending = timer.schedule(tick, interval);
      };
      run.paint(0);
      if (ownsRun() && length > 0) pending = timer.schedule(tick, interval);
    },
    stop,
    dispose() {
      disposed = true;
      stop();
    },
  };
}
