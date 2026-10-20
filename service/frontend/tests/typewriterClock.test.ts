import test from 'node:test';
import assert from 'node:assert/strict';
import { createTypewriterClock } from '../src/visuals/typewriterClock.ts';

class ManualTimer {
  private nextId = 1;
  private callbacks = new Map<number, () => void>();

  schedule = (callback: () => void, delay: number) => {
    assert.ok(Number.isFinite(delay) && delay > 0, 'characters are scheduled asynchronously');
    const id = this.nextId++;
    this.callbacks.set(id, callback);
    return id;
  };

  cancel = (id: number) => {
    this.callbacks.delete(id);
  };

  get pending() {
    return this.callbacks.size;
  }

  takeNext() {
    const next = this.callbacks.entries().next().value;
    assert.ok(next, 'a timer callback is pending');
    const [id, callback] = next;
    this.callbacks.delete(id);
    return callback;
  }

  tick() {
    this.takeNext()();
  }
}

function label(length: number, events: string[] = [], name = 'label') {
  const paints: number[] = [];
  let restorations = 0;
  const run = {
    length,
    paint(visible: number) {
      paints.push(visible);
      events.push(`${name}:${visible}`);
    },
    restore() {
      restorations += 1;
      events.push(`${name}:restore`);
    },
  };
  return {
    run,
    paints,
    get restorations() {
      return restorations;
    },
  };
}

test('a label paints zero immediately and reveals exactly one character per timer tick', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const heading = label(3);
  clock.start(heading.run);
  assert.deepEqual(heading.paints, [0]);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(heading.paints, [0, 1]);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(heading.paints, [0, 1, 2]);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(heading.paints, [0, 1, 2, 3]);
  assert.equal(timer.pending, 0);
  assert.equal(heading.restorations, 0);
  clock.dispose();
});

test('switching labels restores the old owner before blanking the new one and keeps one timer', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const events: string[] = [];
  const first = label(4, events, 'first');
  const second = label(2, events, 'second');
  clock.start(first.run);
  timer.tick();
  clock.start(second.run);
  assert.deepEqual(events, ['first:0', 'first:1', 'first:restore', 'second:0']);
  assert.equal(first.restorations, 1);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(first.paints, [0, 1]);
  assert.deepEqual(second.paints, [0, 1]);
  assert.equal(timer.pending, 1);
  clock.dispose();
});

test('a stale queued callback cannot paint or disturb the replacement owner', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const first = label(4);
  const second = label(4);
  clock.start(first.run);
  const alreadyQueued = timer.takeNext();
  clock.start(second.run);
  alreadyQueued();
  assert.deepEqual(first.paints, [0]);
  assert.deepEqual(second.paints, [0]);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(second.paints, [0, 1]);
  assert.equal(timer.pending, 1);
  clock.dispose();
});

test('a stale callback stays invalid when the same run object is restarted', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const heading = label(4);
  clock.start(heading.run);
  const alreadyQueued = timer.takeNext();
  clock.stop();
  clock.start(heading.run);
  alreadyQueued();
  assert.deepEqual(heading.paints, [0, 0]);
  assert.equal(heading.restorations, 1);
  assert.equal(timer.pending, 1);
  timer.tick();
  assert.deepEqual(heading.paints, [0, 0, 1]);
  clock.dispose();
});

for (const action of ['stop', 'dispose'] as const) {
  test(`${action} cancels a pending timer and restores its label exactly once`, () => {
    const timer = new ManualTimer();
    const clock = createTypewriterClock(timer);
    const heading = label(4);
    clock.start(heading.run);
    assert.equal(timer.pending, 1);
    clock[action]();
    clock[action]();
    assert.equal(heading.restorations, 1);
    assert.deepEqual(heading.paints, [0]);
    assert.equal(timer.pending, 0);
    clock.dispose();
    assert.equal(heading.restorations, 1);
  });

  test(`${action} also invalidates a callback already queued for delivery`, () => {
    const timer = new ManualTimer();
    const clock = createTypewriterClock(timer);
    const heading = label(4);
    clock.start(heading.run);
    const alreadyQueued = timer.takeNext();
    clock[action]();
    alreadyQueued();
    assert.deepEqual(heading.paints, [0]);
    assert.equal(heading.restorations, 1);
    assert.equal(timer.pending, 0);
    clock.dispose();
  });
}

test('a completed label has no idle timer and restores only when the pointer leaves', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const heading = label(1);
  clock.start(heading.run);
  timer.tick();
  assert.deepEqual(heading.paints, [0, 1]);
  assert.equal(timer.pending, 0);
  assert.equal(heading.restorations, 0);
  clock.stop();
  clock.stop();
  assert.equal(heading.restorations, 1);
  assert.equal(timer.pending, 0);
});

test('an empty label paints once without scheduling and is still restored on stop', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const heading = label(0);
  clock.start(heading.run);
  assert.deepEqual(heading.paints, [0]);
  assert.equal(timer.pending, 0);
  clock.stop();
  assert.equal(heading.restorations, 1);
});

test('dispose is permanent and a later start cannot blank another label', () => {
  const timer = new ManualTimer();
  const clock = createTypewriterClock(timer);
  const heading = label(2);
  clock.dispose();
  clock.start(heading.run);
  clock.stop();
  assert.deepEqual(heading.paints, []);
  assert.equal(heading.restorations, 0);
  assert.equal(timer.pending, 0);
});

for (const action of ['stop', 'dispose'] as const) {
  for (const during of ['initial paint', 'timer paint'] as const) {
    test(`${action} during ${during} restores once and cannot reschedule the old run`, () => {
      const timer = new ManualTimer();
      const clock = createTypewriterClock(timer);
      const heading = label(4);
      const originalPaint = heading.run.paint;
      heading.run.paint = (visible) => {
        originalPaint(visible);
        if (visible === (during === 'initial paint' ? 0 : 1)) clock[action]();
      };
      clock.start(heading.run);
      if (during === 'timer paint') timer.tick();
      assert.deepEqual(heading.paints, during === 'initial paint' ? [0] : [0, 1]);
      assert.equal(heading.restorations, 1);
      assert.equal(timer.pending, 0);
      clock.dispose();
      assert.equal(heading.restorations, 1);
    });
  }
}

for (const during of ['initial paint', 'timer paint'] as const) {
  test(`starting another label during ${during} leaves only the new owner and timer`, () => {
    const timer = new ManualTimer();
    const clock = createTypewriterClock(timer);
    const events: string[] = [];
    const first = label(4, events, 'first');
    const second = label(2, events, 'second');
    const originalPaint = first.run.paint;
    first.run.paint = (visible) => {
      originalPaint(visible);
      if (visible === (during === 'initial paint' ? 0 : 1)) clock.start(second.run);
    };
    clock.start(first.run);
    if (during === 'timer paint') timer.tick();
    assert.deepEqual(events.slice(-2), ['first:restore', 'second:0']);
    assert.equal(first.restorations, 1);
    assert.equal(timer.pending, 1);
    const previousPaints = [...first.paints];
    timer.tick();
    assert.deepEqual(first.paints, previousPaints);
    assert.deepEqual(second.paints, [0, 1]);
    assert.equal(timer.pending, 1);
    clock.dispose();
    assert.equal(first.restorations, 1);
    assert.equal(second.restorations, 1);
    assert.equal(timer.pending, 0);
  });
}
