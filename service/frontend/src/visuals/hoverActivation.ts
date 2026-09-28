type Timer = {
  schedule: (callback: () => void, delay: number) => number;
  cancel: (id: number) => void;
};

/** Keep the original hit target intact until the browser has resolved the click. */
export function createHoverActivationGuard(stop: () => void, timer: Timer) {
  let pointerActive = false;
  let pending: number | null = null;

  const cancelPending = () => {
    if (pending !== null) timer.cancel(pending);
    pending = null;
  };
  const finish = () => {
    cancelPending();
    pointerActive = false;
    stop();
  };

  return {
    pointerDown() {
      cancelPending();
      pointerActive = true;
    },
    pointerUp() {
      if (!pointerActive) return;
      cancelPending();
      pending = timer.schedule(() => {
        pending = null;
        if (pointerActive) finish();
      }, 0);
    },
    pointerCancel: finish,
    focusIn() {
      if (!pointerActive) stop();
    },
    activate: finish,
    dispose() {
      cancelPending();
      pointerActive = false;
    },
  };
}
