import { createTypewriterClock } from './typewriterClock.ts';
import { createHoverActivationGuard } from './hoverActivation.ts';

const HIGHLIGHT = 'falcon-typewriter-hidden';
const TARGETS = [
  'button',
  'a[href]',
  'summary',
  'label',
  'input',
  'select',
  'textarea',
  '[role="button"]',
  '[role="link"]',
  '[role="tab"]',
  '[role="menuitem"]',
  '[role="checkbox"]',
  '[role="switch"]',
  '[role="spinbutton"]',
  '[contenteditable]:not([contenteditable="false"])',
].join(',');
const EXCLUDED_TEXT =
  'script, style, svg, canvas, input, select, textarea, [contenteditable]:not([contenteditable="false"]), [hidden], .visually-hidden';
const EDITABLE = 'input, select, textarea, [contenteditable]:not([contenteditable="false"])';

type TextPart = {
  node: Text;
  original: string;
  offsets: number[];
  first: number;
  range: Range;
};

/** A delegated visual effect; React keeps ownership of every original text node. */
export function installHoverTypewriter(root: HTMLElement): () => void {
  if (typeof CSS === 'undefined' || !CSS.highlights || typeof Highlight === 'undefined')
    return () => {};

  const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
  const segmenter = new Intl.Segmenter(undefined, { granularity: 'grapheme' });
  const clock = createTypewriterClock({
    schedule: (callback, delay) => window.setTimeout(callback, delay),
    cancel: (id) => window.clearTimeout(id),
  });
  let hovered: HTMLElement | null = null;
  let disposed = false;

  const ownerOf = (target: EventTarget | null): HTMLElement | null => {
    const element = target instanceof Element ? target : null;
    const found = element?.closest<HTMLElement>(TARGETS);
    if (!found || !root.contains(found)) return null;
    if (
      found instanceof HTMLLabelElement &&
      found.control &&
      !found.control.closest('[hidden], .visually-hidden') &&
      !found.control.matches('input[type="hidden"]')
    )
      return found.control;
    return found;
  };
  const enabled = (owner: HTMLElement) => {
    const control = owner instanceof HTMLLabelElement ? owner.control : owner;
    return (
      owner.isConnected &&
      root.contains(owner) &&
      !control?.matches(':disabled') &&
      !control?.closest('[inert], [aria-disabled="true"]') &&
      !owner.closest('[inert], [hidden], [aria-disabled="true"], .visually-hidden')
    );
  };

  const scopesFor = (owner: HTMLElement): HTMLElement[] => {
    if (!owner.matches(EDITABLE) && owner.getAttribute('role') !== 'spinbutton') return [owner];
    const labels =
      owner instanceof HTMLInputElement ||
      owner instanceof HTMLSelectElement ||
      owner instanceof HTMLTextAreaElement
        ? Array.from(owner.labels ?? [])
        : [];
    const labelledBy = (owner.getAttribute('aria-labelledby') ?? '')
      .split(/\s+/)
      .map((id) => document.getElementById(id))
      .filter((element): element is HTMLElement => element !== null);
    return [...new Set<HTMLElement>([...labels, ...labelledBy])].filter((element) =>
      root.contains(element),
    );
  };

  const visibleText = (scope: HTMLElement, tooltip = false): Text[] => {
    const nodes: Text[] = [];
    const walker = document.createTreeWalker(scope, NodeFilter.SHOW_TEXT);
    for (let node = walker.nextNode(); node; node = walker.nextNode()) {
      const text = node as Text;
      const parent = text.parentElement;
      if (
        !text.data.trim() ||
        !parent ||
        parent.closest(EXCLUDED_TEXT) ||
        (!tooltip && parent.closest('[aria-hidden="true"]'))
      )
        continue;
      let visible = true;
      for (let element: HTMLElement | null = parent; element; element = element.parentElement) {
        const style = getComputedStyle(element);
        if (style.display === 'none' || style.visibility !== 'visible' || style.opacity === '0') {
          visible = false;
          break;
        }
        if (element === scope) break;
      }
      if (visible) nodes.push(text);
    }
    return nodes;
  };

  const stop = () => clock.stop();
  const activation = createHoverActivationGuard(stop, {
    schedule: (callback, delay) => window.setTimeout(callback, delay),
    cancel: (id) => window.clearTimeout(id),
  });
  const begin = (owner: HTMLElement) => {
    stop();
    if (disposed || preference.matches || document.hidden || !enabled(owner)) return;
    const host = owner.closest<HTMLDialogElement>('dialog[open]') ?? document.body;
    const scopes = scopesFor(owner);
    let nodes = scopes.flatMap((scope) => visibleText(scope));
    let tooltip: HTMLSpanElement | null = null;

    if (!nodes.length) {
      const caption =
        owner.getAttribute('aria-label') ||
        owner.getAttribute('placeholder') ||
        owner.getAttribute('title');
      if (!caption?.trim()) return;
      tooltip = document.createElement('span');
      tooltip.className = 'hover-typewriter-tooltip';
      tooltip.setAttribute('aria-hidden', 'true');
      tooltip.textContent = caption;
      Object.assign(tooltip.style, {
        position: 'fixed',
        pointerEvents: 'none',
        left: '8px',
        top: '8px',
        maxWidth: `${Math.max(0, window.innerWidth - 16)}px`,
        whiteSpace: 'normal',
      });
      host.append(tooltip);
      const anchor = owner.getBoundingClientRect();
      const bounds = tooltip.getBoundingClientRect();
      tooltip.style.left = `${Math.max(8, Math.min(anchor.left, window.innerWidth - bounds.width - 8))}px`;
      tooltip.style.top = `${Math.max(
        8,
        Math.min(
          anchor.bottom + bounds.height + 14 <= window.innerHeight
            ? anchor.bottom + 6
            : anchor.top - bounds.height - 6,
          window.innerHeight - bounds.height - 8,
        ),
      )}px`;
      nodes = visibleText(tooltip, true);
    }
    if (!nodes.length) {
      tooltip?.remove();
      return;
    }

    const marked = new Map(
      [...new Set([...scopes, ...(tooltip ? [tooltip] : [])])].map(
        (source) => [source, source.getAttribute('data-hover-typewriter')] as const,
      ),
    );
    marked.forEach((_, source) => source.setAttribute('data-hover-typewriter', ''));
    let length = 0;
    const parts: TextPart[] = nodes.map((node) => {
      const offsets = Array.from(segmenter.segment(node.data), (part) => part.index);
      offsets.push(node.data.length);
      const first = length;
      length += offsets.length - 1;
      const range = document.createRange();
      range.selectNodeContents(node);
      return { node, original: node.data, offsets, first, range };
    });
    const hidden = new Highlight(...parts.map((part) => part.range));
    const caret = document.createElement('span');
    caret.className = 'hover-typewriter-caret';
    caret.setAttribute('aria-hidden', 'true');
    Object.assign(caret.style, { position: 'fixed', pointerEvents: 'none' });
    host.append(caret);
    CSS.highlights.set(HIGHLIGHT, hidden);

    const observer = new MutationObserver((records) => {
      if (
        !enabled(owner) ||
        parts.some((part) => !part.node.isConnected) ||
        (host instanceof HTMLDialogElement && !host.open) ||
        records.some(
          ({ target }) => owner.contains(target) || scopes.some((scope) => scope.contains(target)),
        )
      )
        stop();
    });
    // Observe removals without rescanning the document, including removal of the app root.
    observer.observe(document.documentElement, { childList: true, subtree: true });
    const control = owner instanceof HTMLLabelElement ? owner.control : null;
    for (const scope of new Set([owner, ...scopes, ...(control ? [control] : [])])) {
      observer.observe(scope, {
        childList: true,
        subtree: true,
        characterData: true,
        attributes: true,
        attributeFilter: [
          'disabled',
          'aria-disabled',
          'aria-valuenow',
          'data-current',
          'hidden',
          'inert',
          'class',
        ],
      });
    }
    if (host instanceof HTMLDialogElement)
      observer.observe(host, { attributes: true, attributeFilter: ['open'] });

    const placeCaret = (visible: number) => {
      const index = Math.max(0, Math.min(length - 1, visible - 1));
      const part = parts.find((entry) => index < entry.first + entry.offsets.length - 1);
      if (!part) return;
      const local = index - part.first;
      const range = document.createRange();
      range.setStart(part.node, part.offsets[local]);
      range.setEnd(part.node, part.offsets[local + 1]);
      const rects = Array.from(range.getClientRects()).filter((rect) => rect.height > 0);
      const rect = visible === 0 ? rects[0] : rects.at(-1);
      const parent = part.node.parentElement;
      if (!rect || !parent) {
        caret.hidden = true;
        return;
      }
      const style = getComputedStyle(parent);
      const trailing = visible > 0;
      const x = (style.direction === 'rtl') !== trailing ? rect.right : rect.left;
      let left = 0;
      let right = window.innerWidth;
      let top = 0;
      let bottom = window.innerHeight;
      for (let element: HTMLElement | null = parent; element; element = element.parentElement) {
        const clip = getComputedStyle(element);
        const masked = clip.maskImage !== 'none';
        if (clip.overflowX !== 'visible' || clip.overflowY !== 'visible' || masked) {
          const bounds = element.getBoundingClientRect();
          if (clip.overflowX !== 'visible' || masked) {
            left = Math.max(left, bounds.left);
            right = Math.min(right, bounds.right);
          }
          if (clip.overflowY !== 'visible' || masked) {
            top = Math.max(top, bounds.top);
            bottom = Math.min(bottom, bounds.bottom);
          }
        }
      }
      caret.hidden = x < left || x + 1 > right || rect.top < top || rect.bottom > bottom;
      Object.assign(caret.style, {
        left: `${x}px`,
        top: `${rect.top}px`,
        height: `${rect.height}px`,
        color: style.color,
      });
    };

    clock.start({
      length,
      paint(visible) {
        if (
          !enabled(owner) ||
          parts.some((part) => !part.node.isConnected || part.node.data !== part.original)
        ) {
          stop();
          return;
        }
        for (const part of parts) {
          const count = part.offsets.length - 1;
          const shown = Math.min(count, Math.max(0, visible - part.first));
          if (shown === count) hidden.delete(part.range);
          else part.range.setStart(part.node, part.offsets[shown]);
        }
        placeCaret(visible);
      },
      restore() {
        observer.disconnect();
        marked.forEach((previous, source) => {
          if (previous === null) source.removeAttribute('data-hover-typewriter');
          else source.setAttribute('data-hover-typewriter', previous);
        });
        CSS.highlights.delete(HIGHLIGHT);
        caret.remove();
        tooltip?.remove();
      },
    });
  };

  const over = (event: PointerEvent) => {
    if (event.pointerType !== 'mouse' && event.pointerType !== 'pen') return;
    if (event.buttons) return;
    const owner = ownerOf(event.target);
    if (owner === hovered) return;
    hovered = owner;
    if (owner) begin(owner);
    else stop();
  };
  const out = (event: PointerEvent) => {
    if (ownerOf(event.relatedTarget) === hovered) return;
    hovered = null;
    stop();
  };
  root.addEventListener('pointerover', over);
  root.addEventListener('pointerout', out);
  const pointerDown = (event: PointerEvent) => {
    if (event.button === 0) activation.pointerDown();
    else stop();
  };
  const pointerUp = () => activation.pointerUp();
  const pointerCancel = () => activation.pointerCancel();
  const focusIn = () => activation.focusIn();
  const activate = () => activation.activate();
  root.addEventListener('pointerdown', pointerDown, true);
  window.addEventListener('pointerup', pointerUp, true);
  window.addEventListener('pointercancel', pointerCancel, true);
  root.addEventListener('click', activate, true);
  root.addEventListener('focusin', focusIn, true);
  root.addEventListener('keydown', stop, true);
  root.addEventListener('wheel', stop, { capture: true, passive: true });
  document.addEventListener('scroll', stop, { capture: true, passive: true });
  document.addEventListener('visibilitychange', pointerCancel);
  window.addEventListener('resize', stop, { passive: true });
  window.addEventListener('blur', pointerCancel);
  preference.addEventListener('change', stop);

  return () => {
    disposed = true;
    activation.dispose();
    clock.dispose();
    root.removeEventListener('pointerover', over);
    root.removeEventListener('pointerout', out);
    root.removeEventListener('pointerdown', pointerDown, true);
    window.removeEventListener('pointerup', pointerUp, true);
    window.removeEventListener('pointercancel', pointerCancel, true);
    root.removeEventListener('click', activate, true);
    root.removeEventListener('focusin', focusIn, true);
    root.removeEventListener('keydown', stop, true);
    root.removeEventListener('wheel', stop, true);
    document.removeEventListener('scroll', stop, true);
    document.removeEventListener('visibilitychange', pointerCancel);
    window.removeEventListener('resize', stop);
    window.removeEventListener('blur', pointerCancel);
    preference.removeEventListener('change', stop);
  };
}
