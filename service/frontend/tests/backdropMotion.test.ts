import test, { afterEach, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { getEventListeners } from 'node:events';
import { startBackdropMotion } from '../src/visuals/backdropMotion.ts';

class MotionPreference extends EventTarget {
  matches = false;

  change(matches: boolean) {
    this.matches = matches;
    this.dispatchEvent(new Event('change'));
  }
}

class PageVisibility extends EventTarget {
  hidden = false;

  change(hidden: boolean) {
    this.hidden = hidden;
    this.dispatchEvent(new Event('visibilitychange'));
  }
}

class BrowserFrames {
  private nextId = 1;
  private callbacks = new Map<number, FrameRequestCallback>();

  requestAnimationFrame = (callback: FrameRequestCallback) => {
    const id = this.nextId++;
    this.callbacks.set(id, callback);
    return id;
  };

  cancelAnimationFrame = (id: number) => {
    this.callbacks.delete(id);
  };

  get pending() {
    return this.callbacks.size;
  }

  advance(time: number) {
    // Requests made by a callback belong to the next browser frame.
    for (const [id, callback] of [...this.callbacks]) {
      if (!this.callbacks.has(id)) continue;
      this.callbacks.delete(id);
      callback(time);
    }
  }
}

type Motion = ReturnType<typeof startBackdropMotion>;
let preference: MotionPreference;
let page: PageVisibility;
let frames: BrowserFrames;
let controls: Motion[];
let originalWindow: PropertyDescriptor | undefined;
let originalDocument: PropertyDescriptor | undefined;

beforeEach(() => {
  originalWindow = Object.getOwnPropertyDescriptor(globalThis, 'window');
  originalDocument = Object.getOwnPropertyDescriptor(globalThis, 'document');
  preference = new MotionPreference();
  page = new PageVisibility();
  frames = new BrowserFrames();
  controls = [];
  Object.defineProperty(globalThis, 'window', {
    configurable: true,
    value: {
      requestAnimationFrame: frames.requestAnimationFrame,
      cancelAnimationFrame: frames.cancelAnimationFrame,
      matchMedia: (query: string) => {
        assert.equal(query, '(prefers-reduced-motion: reduce)');
        return preference;
      },
    },
  });
  Object.defineProperty(globalThis, 'document', { configurable: true, value: page });
});

afterEach(() => {
  try {
    for (const control of controls) control.dispose();
  } finally {
    if (originalWindow) Object.defineProperty(globalThis, 'window', originalWindow);
    else Reflect.deleteProperty(globalThis, 'window');
    if (originalDocument) Object.defineProperty(globalThis, 'document', originalDocument);
    else Reflect.deleteProperty(globalThis, 'document');
  }
});

function begin(onPaint?: (angle: number) => void) {
  const angles: number[] = [];
  const availability: boolean[] = [];
  const control = startBackdropMotion(
    (angle) => {
      angles.push(angle);
      onPaint?.(angle);
    },
    (available) => availability.push(available),
  );
  controls.push(control);
  return { control, angles, availability };
}

function latest(angles: number[]) {
  const angle = angles.at(-1);
  assert.ok(angle !== undefined, 'the visible backdrop has been painted');
  return angle;
}

function assertSameIncrement(actual: number, expected: number) {
  assert.ok(Math.abs(actual - expected) < 1e-10, 'resume preserves the normal angular speed');
}

test('visible motion paints immediately and keeps one loop through repeated sync and redraw', () => {
  const { control, angles, availability } = begin();
  assert.equal(angles.length, 1);
  assert.ok(Number.isFinite(angles[0]));
  assert.deepEqual(availability, [true]);
  assert.equal(frames.pending, 1);

  for (let repetition = 0; repetition < 3; repetition++) {
    control.setPaused(false);
    control.redraw();
    page.change(false);
    preference.change(false);
    assert.equal(frames.pending, 1);
  }
  const beforeFrame = angles.length;
  frames.advance(0);
  frames.advance(50);
  assert.equal(angles.length, beforeFrame + 1);
  assert.ok(latest(angles) > angles[0]);
  assert.equal(frames.pending, 1);
});

test('manual pause cancels motion but permits static redraw and resumes without catch-up', () => {
  const { control, angles } = begin();
  frames.advance(0);
  frames.advance(50);
  const normalIncrement = latest(angles) - angles[0];
  const pausedAngle = latest(angles);
  control.setPaused(true);
  assert.equal(frames.pending, 0);
  assert.equal(latest(angles), pausedAngle);

  const beforeRedraw = angles.length;
  control.redraw();
  assert.equal(angles.length, beforeRedraw + 1);
  assert.equal(latest(angles), pausedAngle);
  frames.advance(300_000);
  assert.equal(angles.length, beforeRedraw + 1);

  control.setPaused(false);
  assert.equal(frames.pending, 1);
  assert.equal(latest(angles), pausedAngle);
  frames.advance(300_050);
  assert.equal(latest(angles), pausedAngle);
  frames.advance(300_100);
  assertSameIncrement(latest(angles) - pausedAngle, normalIncrement);
});

for (const suspension of ['hidden page', 'reduced motion'] as const) {
  test(`${suspension} stops the loop and resumes at the saved angle after a long absence`, () => {
    const { angles, availability } = begin();
    frames.advance(0);
    frames.advance(50);
    const normalIncrement = latest(angles) - angles[0];
    const savedAngle = latest(angles);
    if (suspension === 'hidden page') page.change(true);
    else preference.change(true);
    assert.equal(frames.pending, 0);
    assert.equal(latest(angles), savedAngle);
    if (suspension === 'reduced motion') assert.equal(availability.at(-1), false);

    const beforeSuspendedFrame = angles.length;
    frames.advance(300_000);
    assert.equal(angles.length, beforeSuspendedFrame);
    if (suspension === 'hidden page') page.change(false);
    else preference.change(false);
    assert.equal(frames.pending, 1);
    assert.equal(latest(angles), savedAngle);
    assert.equal(availability.at(-1), true);
    frames.advance(300_050);
    assert.equal(latest(angles), savedAngle);
    frames.advance(300_100);
    assertSameIncrement(latest(angles) - savedAngle, normalIncrement);
  });
}

test('starting in a hidden tab defers every paint, including redraw, until it becomes visible', () => {
  page.hidden = true;
  const { control, angles, availability } = begin();
  assert.deepEqual(angles, []);
  assert.deepEqual(availability, [true]);
  assert.equal(frames.pending, 0);
  control.redraw();
  assert.deepEqual(angles, []);
  page.change(false);
  assert.equal(angles.length, 1);
  assert.equal(frames.pending, 1);
});

test('initial reduced motion renders a static backdrop and only starts when motion is allowed', () => {
  preference.matches = true;
  const { control, angles, availability } = begin();
  assert.equal(angles.length, 1);
  assert.deepEqual(availability, [false]);
  assert.equal(frames.pending, 0);
  control.setPaused(false);
  control.redraw();
  assert.ok(angles.every((angle) => angle === angles[0]));
  assert.equal(frames.pending, 0);
  preference.change(false);
  assert.equal(availability.at(-1), true);
  assert.equal(frames.pending, 1);
});

test('visibility and preference changes preserve an explicit manual pause', () => {
  const { control, angles } = begin();
  control.setPaused(true);
  page.change(true);
  preference.change(true);
  page.change(false);
  preference.change(false);
  assert.equal(frames.pending, 0);
  assert.ok(angles.every((angle) => angle === angles[0]));
  control.setPaused(false);
  assert.equal(frames.pending, 1);
});

test('dispose cancels the outstanding frame, removes event handlers and makes controls inert', () => {
  const { control, angles, availability } = begin();
  assert.equal(getEventListeners(page, 'visibilitychange').length, 1);
  assert.equal(getEventListeners(preference, 'change').length, 1);
  control.dispose();
  assert.equal(frames.pending, 0);
  assert.equal(getEventListeners(page, 'visibilitychange').length, 0);
  assert.equal(getEventListeners(preference, 'change').length, 0);
  const painted = [...angles];
  const available = [...availability];
  page.change(true);
  page.change(false);
  preference.change(true);
  preference.change(false);
  control.redraw();
  control.setPaused(false);
  control.dispose();
  frames.advance(50);
  assert.deepEqual(angles, painted);
  assert.deepEqual(availability, available);
  assert.equal(frames.pending, 0);
});

test('disposal from an animation paint does not schedule another frame', () => {
  let disposeDuringPaint = false;
  const { control, angles } = begin(() => {
    if (disposeDuringPaint) control.dispose();
  });
  frames.advance(0);
  disposeDuringPaint = true;
  frames.advance(50);
  assert.equal(frames.pending, 0);
  assert.equal(getEventListeners(page, 'visibilitychange').length, 0);
  assert.equal(getEventListeners(preference, 'change').length, 0);
  const painted = angles.length;
  frames.advance(100);
  assert.equal(angles.length, painted);
});

test('disposal from a synchronous resume paint does not schedule another frame', () => {
  let disposeDuringPaint = false;
  const { control } = begin(() => {
    if (disposeDuringPaint) control.dispose();
  });
  control.setPaused(true);
  disposeDuringPaint = true;
  control.setPaused(false);
  assert.equal(frames.pending, 0);
  assert.equal(getEventListeners(page, 'visibilitychange').length, 0);
  assert.equal(getEventListeners(preference, 'change').length, 0);
});
