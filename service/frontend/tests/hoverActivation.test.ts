import test from 'node:test';
import assert from 'node:assert/strict';
import { createHoverActivationGuard } from '../src/visuals/hoverActivation.ts';

class ManualTimer {
  private nextId = 1;
  private callbacks = new Map<number, () => void>();

  schedule = (callback: () => void) => {
    const id = this.nextId++;
    this.callbacks.set(id, callback);
    return id;
  };

  cancel = (id: number) => {
    this.callbacks.delete(id);
  };

  flush() {
    const pending = [...this.callbacks.values()];
    this.callbacks.clear();
    pending.forEach((callback) => callback());
  }
}

test('pointer focus keeps the hovered target stable until click dispatch begins', () => {
  const timer = new ManualTimer();
  const events: string[] = [];
  const guard = createHoverActivationGuard(() => events.push('stop'), timer);

  guard.pointerDown();
  guard.focusIn();
  assert.deepEqual(events, []);

  guard.activate();
  assert.deepEqual(events, ['stop']);
  timer.flush();
  assert.deepEqual(events, ['stop']);
});

test('keyboard focus still restores the hovered label immediately', () => {
  const timer = new ManualTimer();
  const events: string[] = [];
  const guard = createHoverActivationGuard(() => events.push('stop'), timer);

  guard.focusIn();
  assert.deepEqual(events, ['stop']);
});

test('a pointer release without activation restores on the next task', () => {
  const timer = new ManualTimer();
  const events: string[] = [];
  const guard = createHoverActivationGuard(() => events.push('stop'), timer);

  guard.pointerDown();
  guard.pointerUp();
  assert.deepEqual(events, []);
  timer.flush();
  assert.deepEqual(events, ['stop']);
});

test('an interrupted pointer activation resets before later keyboard focus', () => {
  const timer = new ManualTimer();
  const events: string[] = [];
  const guard = createHoverActivationGuard(() => events.push('stop'), timer);

  guard.pointerDown();
  guard.pointerCancel();
  guard.focusIn();
  assert.deepEqual(events, ['stop', 'stop']);
});
