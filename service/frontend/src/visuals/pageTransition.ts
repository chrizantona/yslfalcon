type Direction = -1 | 0 | 1;

type Transition = {
  finished: Promise<void>;
  skipTransition(): void;
};

type Options = {
  initialPath: string;
  commit(path: string): void;
  direction(fromPath: string, toPath: string): Direction;
  allowed(): boolean;
  setDirection(value: Direction): void;
  start?: (update: () => void) => Transition;
};

type Request = {
  path: string;
  applied: boolean;
  commitFailed: boolean;
  transition?: Transition;
};

/** Native transitions decorate navigation; only the latest request may update the route. */
export function createPageTransition(options: Options) {
  let committedPath = options.initialPath;
  let active: Request | null = null;
  let disposed = false;
  let ownedDirection: Direction = 0;

  const clearDirection = () => {
    if (ownedDirection === 0) return;
    ownedDirection = 0;
    options.setDirection(0);
  };
  const owns = (request: Request) => !disposed && active === request;
  const apply = (request: Request) => {
    if (request.applied) return;
    // Mark the attempt first: a navigation exception must never cause a second commit.
    request.applied = true;
    try {
      options.commit(request.path);
      committedPath = request.path;
    } catch (error) {
      request.commitFailed = true;
      throw error;
    }
  };
  const release = (request: Request) => {
    if (active !== request) return;
    active = null;
    clearDirection();
  };
  const settle = (request: Request) => {
    if (!owns(request)) return;
    try {
      apply(request);
    } finally {
      release(request);
    }
  };
  const cancel = () => {
    const previous = active;
    // skipTransition still runs queued updates, so invalidate ownership first.
    active = null;
    previous?.transition?.skipTransition();
    clearDirection();
    return previous;
  };

  return {
    navigate(path: string) {
      if (disposed) return;
      cancel();
      const request: Request = { path, applied: false, commitFailed: false };
      active = request;
      const direction = options.direction(committedPath, path);
      if (!options.start || !options.allowed() || direction === 0) {
        settle(request);
        return;
      }
      ownedDirection = direction;
      options.setDirection(direction);
      try {
        const transition = options.start(() => {
          if (owns(request)) apply(request);
        });
        request.transition = transition;
        void transition.finished.then(
          () => settle(request),
          (error: unknown) => {
            if (!owns(request)) return;
            console.warn('Page transition failed', error);
            settle(request);
          },
        );
        if (!owns(request)) transition.skipTransition();
      } catch (error) {
        if (request.commitFailed) {
          release(request);
          throw error;
        }
        if (!owns(request)) return;
        console.warn('Page transition failed', error);
        settle(request);
      }
    },
    finish() {
      const previous = cancel();
      if (previous && !disposed) apply(previous);
    },
    dispose() {
      disposed = true;
      cancel();
    },
  };
}
