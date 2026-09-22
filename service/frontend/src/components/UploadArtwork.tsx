import { useCallback, useEffect, useRef, type PointerEvent, type ReactNode } from 'react';

type UploadKind = 'image' | 'files' | 'document';

function UploadArtwork({ kind }: { kind: UploadKind }) {
  return (
    <span className={`upload-artwork upload-artwork-${kind}`}>
      <svg
        className="upload-artwork-svg"
        viewBox="0 0 112 96"
        fill="none"
        stroke="currentColor"
        strokeWidth="3"
        strokeLinecap="round"
        strokeLinejoin="round"
        aria-hidden="true"
        focusable="false"
      >
        {kind === 'document' ? (
          <>
            <path d="M80 64V31L63 14H28V80H72" />
            <path d="M63 14v18h17M40 44h25M40 55h19M40 66h12" />
          </>
        ) : (
          <>
            {kind === 'files' && <path d="M28 16V9h61v43" />}
            <path d="M78 74H17V23h70v32" />
            <path d="m18 62 18-19 19 18 11-11 12 12" />
            <circle cx="66" cy="38" r="6" />
          </>
        )}
        <path className="upload-artwork-plus" d="M88 65v24M76 77h24" strokeWidth="4" />
      </svg>
    </span>
  );
}

export function UploadPrompt({
  kind = 'image',
  title,
  description,
  meta,
  compact = false,
  disabled = false,
}: {
  kind?: UploadKind;
  title: ReactNode;
  description?: ReactNode;
  meta?: ReactNode;
  compact?: boolean;
  disabled?: boolean;
}) {
  const prompt = useRef<HTMLSpanElement>(null);
  const anchor = useRef<HTMLSpanElement>(null);
  const reduceMotion = useRef(true);
  const resetMotion = useCallback(() => {
    prompt.current?.style.setProperty('--upload-proximity', '0');
    prompt.current?.style.setProperty('--upload-rotate', '0deg');
    prompt.current?.style.setProperty('--upload-scale', '1');
  }, []);

  useEffect(() => {
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    const updatePreference = () => {
      reduceMotion.current = preference.matches;
      resetMotion();
    };
    updatePreference();
    preference.addEventListener('change', updatePreference);
    return () => preference.removeEventListener('change', updatePreference);
  }, [resetMotion]);

  useEffect(() => {
    if (disabled) resetMotion();
  }, [disabled, resetMotion]);

  const move = (event: PointerEvent<HTMLSpanElement>) => {
    if (disabled || reduceMotion.current || event.pointerType !== 'mouse') {
      resetMotion();
      return;
    }
    const bounds = anchor.current?.getBoundingClientRect();
    if (!bounds || !prompt.current) return;
    // Measure the stationary anchor; transforming the art must not move its attraction centre.
    const distance = Math.hypot(
      event.clientX - bounds.left - bounds.width / 2,
      event.clientY - bounds.top - bounds.height / 2,
    );
    const radius = Math.max(100, bounds.width * 1.75);
    const proximity = Math.max(0, 1 - distance / radius);
    const eased = proximity * proximity * (3 - 2 * proximity);
    prompt.current.style.setProperty('--upload-proximity', eased.toFixed(4));
    prompt.current.style.setProperty('--upload-rotate', `${(eased * 10).toFixed(3)}deg`);
    prompt.current.style.setProperty('--upload-scale', (1 + eased * 0.12).toFixed(4));
  };

  return (
    <span
      ref={prompt}
      className={`upload-prompt${compact ? ' upload-prompt-compact' : ''}`}
      onPointerEnter={move}
      onPointerMove={move}
      onPointerLeave={resetMotion}
      onPointerCancel={resetMotion}
    >
      <span ref={anchor} className="upload-artwork-anchor" aria-hidden="true">
        <UploadArtwork kind={kind} />
      </span>
      <span className="upload-copy">
        <strong>{title}</strong>
        {description && <span className="upload-description">{description}</span>}
        {meta && <small className="upload-meta">{meta}</small>}
      </span>
    </span>
  );
}
