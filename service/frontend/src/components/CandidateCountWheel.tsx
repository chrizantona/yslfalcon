import { useEffect, useRef, type CSSProperties, type KeyboardEvent } from 'react';
import './CandidateCountWheel.css';

const MIN_COUNT = 1;
const MAX_COUNT = 50;
const COUNTS = Array.from({ length: MAX_COUNT }, (_, index) => index + MIN_COUNT);

export function CandidateCountWheel({
  id,
  value,
  onChange,
  disabled = false,
}: {
  id: string;
  value: number;
  onChange: (value: number) => void;
  disabled?: boolean;
}) {
  const wheel = useRef<HTMLDivElement>(null);
  const gesture = useRef({ delta: 0, lastEvent: 0, lastStep: -Infinity });

  useEffect(() => {
    const element = wheel.current;
    if (!element || disabled) return;

    const resetGesture = () => {
      gesture.current.delta = 0;
    };
    const onWheel = (event: WheelEvent) => {
      // Leave zoom gestures, horizontal scrolling and the range edges to the page.
      if (event.ctrlKey || event.metaKey || Math.abs(event.deltaX) >= Math.abs(event.deltaY))
        return;
      const direction = Math.sign(event.deltaY);
      const next = value + direction;
      if (next < MIN_COUNT || next > MAX_COUNT) {
        resetGesture();
        return;
      }

      // Keep an active gesture on the drum while small trackpad deltas accumulate.
      event.preventDefault();
      const now = performance.now();
      const current = gesture.current;
      if (now - current.lastEvent > 220 || Math.sign(current.delta) !== direction)
        current.delta = 0;
      current.lastEvent = now;
      // Trackpads accumulate small movements; even a large mouse-wheel tick changes one number.
      const unit = event.deltaMode === 1 ? 20 : event.deltaMode === 2 ? 100 : 1;
      current.delta += event.deltaY * unit;
      if (Math.abs(current.delta) < 36 || now - current.lastStep < 120) return;
      current.delta = 0;
      current.lastStep = now;
      onChange(next);
    };

    element.addEventListener('wheel', onWheel, { passive: false });
    element.addEventListener('pointerleave', resetGesture);
    return () => {
      element.removeEventListener('wheel', onWheel);
      element.removeEventListener('pointerleave', resetGesture);
    };
  }, [value, onChange, disabled]);

  const change = (next: number) => {
    if (!disabled) onChange(Math.max(MIN_COUNT, Math.min(MAX_COUNT, next)));
  };
  const onKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    if (disabled || event.altKey || event.ctrlKey || event.metaKey) return;
    const next = {
      ArrowUp: value + 1,
      ArrowDown: value - 1,
      PageUp: value + 5,
      PageDown: value - 5,
      Home: MIN_COUNT,
      End: MAX_COUNT,
    }[event.key];
    if (next === undefined) return;
    event.preventDefault();
    change(next);
  };

  return (
    <div className="topn-row candidate-count">
      <div className="candidate-count-copy">
        <span id={`${id}-label`}>Кандидатов в выдаче</span>
        <span id={`${id}-help`} className="candidate-count-hint">
          Колесо или клавиши ↑ ↓
        </span>
      </div>
      <div className="candidate-count-control" data-disabled={disabled || undefined}>
        <div
          ref={wheel}
          id={id}
          className="candidate-count-wheel"
          role="spinbutton"
          tabIndex={disabled ? -1 : 0}
          aria-labelledby={`${id}-label`}
          aria-describedby={`${id}-help`}
          aria-valuemin={MIN_COUNT}
          aria-valuemax={MAX_COUNT}
          aria-valuenow={value}
          aria-disabled={disabled || undefined}
          onKeyDown={onKeyDown}
        >
          <div className="candidate-count-drum" aria-hidden="true">
            {COUNTS.map((count) => {
              const offset = count - value;
              return (
                <span
                  key={count}
                  className="candidate-count-number"
                  data-current={offset === 0 || undefined}
                  style={
                    {
                      '--count-offset': offset,
                      opacity:
                        Math.abs(offset) > 2
                          ? 0
                          : offset === 0
                            ? 1
                            : Math.abs(offset) === 1
                              ? 0.35
                              : 0.12,
                    } as CSSProperties
                  }
                >
                  {count}
                </span>
              );
            })}
          </div>
        </div>
        <div className="candidate-count-buttons">
          <button
            type="button"
            aria-label="Уменьшить количество кандидатов"
            aria-controls={id}
            disabled={disabled || value === MIN_COUNT}
            onClick={() => change(value - 1)}
          >
            −
          </button>
          <button
            type="button"
            aria-label="Увеличить количество кандидатов"
            aria-controls={id}
            disabled={disabled || value === MAX_COUNT}
            onClick={() => change(value + 1)}
          >
            +
          </button>
        </div>
      </div>
    </div>
  );
}
