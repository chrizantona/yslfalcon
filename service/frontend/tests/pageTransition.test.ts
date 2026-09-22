import test from 'node:test';
import assert from 'node:assert/strict';
import { createPageTransition } from '../src/visuals/pageTransition.ts';
import { pageRollDirection } from '../src/navigation.ts';

type Options = Parameters<typeof createPageTransition>[0];
type Direction = -1 | 0 | 1;

class ManualTransition {
  readonly finished: Promise<void>;
  resolve!: () => void;
  reject!: (error: Error) => void;
  skips = 0;
  readonly update: () => void;

  constructor(update: () => void) {
    this.update = update;
    this.finished = new Promise((resolve, reject) => {
      this.resolve = resolve;
      this.reject = reject;
    });
  }

  skipTransition() {
    this.skips += 1;
    // Native skipping does not cancel the update; tests may deliver it later.
  }
}

function setup(overrides: Partial<Options> = {}) {
  const commits: string[] = [];
  const directions: Direction[] = [];
  const routes: [string, string][] = [];
  const transitions: ManualTransition[] = [];
  const events: string[] = [];
  const controller = createPageTransition({
    initialPath: '/a',
    commit(path) {
      commits.push(path);
      events.push(`commit:${path}`);
    },
    direction(from, to) {
      routes.push([from, to]);
      return from === to ? 0 : from < to ? 1 : -1;
    },
    allowed: () => true,
    setDirection(direction) {
      directions.push(direction);
      events.push(`direction:${direction}`);
    },
    start(update) {
      events.push('start');
      const transition = new ManualTransition(update);
      transitions.push(transition);
      return transition;
    },
    ...overrides,
  });
  return { controller, commits, directions, routes, transitions, events };
}

for (const [reason, overrides] of [
  ['unsupported', { start: undefined }],
  ['motion disabled', { allowed: () => false }],
] as const) {
  test(`${reason} commits immediately without a transition`, () => {
    const run = setup(overrides);
    run.controller.navigate('/b');
    assert.deepEqual(run.commits, ['/b']);
    assert.deepEqual(run.directions, []);
    assert.equal(run.transitions.length, 0);
    run.controller.finish();
    assert.deepEqual(run.commits, ['/b']);
  });
}

test('zero direction, including the current page, commits immediately', () => {
  const run = setup();
  run.controller.navigate('/a');
  assert.deepEqual(run.commits, ['/a']);
  assert.equal(run.transitions.length, 0);
  assert.deepEqual(run.directions, []);
});

test('direction is set before native start and cleared after completion', async () => {
  const run = setup();
  run.controller.navigate('/b');
  assert.deepEqual(run.events, ['direction:1', 'start']);
  const transition = run.transitions[0];
  transition.update();
  transition.update();
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.directions, [1]);
  transition.resolve();
  await Promise.resolve();
  assert.deepEqual(run.directions, [1, 0]);
  run.controller.finish();
  run.controller.dispose();
  assert.equal(transition.skips, 0);
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.directions, [1, 0]);
});

test('rapid A to B to C ignores skipped updates and stale completion', async () => {
  const run = setup();
  run.controller.navigate('/b');
  const first = run.transitions[0];
  run.controller.navigate('/c');
  const latest = run.transitions[1];
  assert.equal(first.skips, 1);
  assert.deepEqual(run.routes, [
    ['/a', '/b'],
    ['/a', '/c'],
  ]);
  first.update();
  first.resolve();
  await Promise.resolve();
  assert.deepEqual(run.commits, []);
  assert.equal(run.directions.at(-1), 1);
  latest.update();
  first.update();
  assert.deepEqual(run.commits, ['/c']);
  latest.resolve();
  await Promise.resolve();
  assert.equal(run.directions.at(-1), 0);
});

test('returning to the committed page invalidates an uncommitted request', async () => {
  const run = setup();
  run.controller.navigate('/b');
  const first = run.transitions[0];
  run.controller.navigate('/a');
  first.update();
  first.resolve();
  await Promise.resolve();
  assert.deepEqual(run.commits, ['/a']);
  assert.equal(first.skips, 1);
  assert.equal(run.transitions.length, 1);
  assert.equal(run.directions.at(-1), 0);
});

test('requesting the same pending target still commits the latest request', async () => {
  const run = setup();
  run.controller.navigate('/b');
  const first = run.transitions[0];
  run.controller.navigate('/b');
  const latest = run.transitions[1];
  first.update();
  latest.update();
  latest.resolve();
  first.resolve();
  await Promise.resolve();
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.routes, [
    ['/a', '/b'],
    ['/a', '/b'],
  ]);
});

test('the inverse transition uses the page actually committed by the previous update', async () => {
  const run = setup();
  run.controller.navigate('/c');
  const first = run.transitions[0];
  first.update();
  run.controller.navigate('/b');
  const latest = run.transitions[1];
  first.resolve();
  await Promise.resolve();
  assert.deepEqual(run.routes, [
    ['/a', '/c'],
    ['/c', '/b'],
  ]);
  assert.equal(run.directions.at(-1), -1);
  latest.update();
  latest.resolve();
  await Promise.resolve();
  assert.deepEqual(run.commits, ['/c', '/b']);
  assert.equal(run.directions.at(-1), 0);
});

for (const alreadyUpdated of [false, true]) {
  test(`finish commits a pending target once (updated: ${alreadyUpdated})`, async () => {
    const run = setup();
    run.controller.navigate('/b');
    const transition = run.transitions[0];
    if (alreadyUpdated) transition.update();
    run.controller.finish();
    run.controller.finish();
    assert.equal(transition.skips, 1);
    assert.deepEqual(run.commits, ['/b']);
    assert.deepEqual(run.directions, [1, 0]);
    run.controller.navigate('/c');
    transition.update();
    transition.resolve();
    await Promise.resolve();
    assert.deepEqual(run.commits, ['/b']);
    assert.equal(run.directions.at(-1), 1);
    run.controller.finish();
    assert.deepEqual(run.commits, ['/b', '/c']);
  });
}

test('dispose invalidates queued callbacks permanently without committing', async () => {
  const run = setup();
  run.controller.navigate('/b');
  const transition = run.transitions[0];
  run.controller.dispose();
  run.controller.dispose();
  transition.update();
  transition.resolve();
  await Promise.resolve();
  run.controller.navigate('/c');
  run.controller.finish();
  assert.equal(transition.skips, 1);
  assert.deepEqual(run.commits, []);
  assert.deepEqual(run.directions, [1, 0]);
  assert.equal(run.transitions.length, 1);
});

test('start failure falls back once and invalidates a captured update', (context) => {
  const warning = context.mock.method(console, 'warn', () => {});
  let queuedUpdate: (() => void) | undefined;
  const failure = new Error('start failed');
  const run = setup({
    start(update) {
      queuedUpdate = update;
      throw failure;
    },
  });
  run.controller.navigate('/b');
  queuedUpdate?.();
  run.controller.finish();
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.directions, [1, 0]);
  assert.equal(warning.mock.callCount(), 1);
  assert.equal(warning.mock.calls[0].arguments[1], failure);
});

test('rejection before the update falls back once and releases direction', async (context) => {
  const warning = context.mock.method(console, 'warn', () => {});
  const run = setup();
  run.controller.navigate('/b');
  const transition = run.transitions[0];
  transition.reject(new Error('snapshot failed'));
  await Promise.resolve();
  transition.update();
  run.controller.finish();
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.directions, [1, 0]);
  assert.equal(warning.mock.callCount(), 1);
});

test('rejection after the update cleans up without committing again', async (context) => {
  const warning = context.mock.method(console, 'warn', () => {});
  const run = setup();
  run.controller.navigate('/b');
  const transition = run.transitions[0];
  transition.update();
  transition.reject(new Error('animation failed'));
  await Promise.resolve();
  assert.deepEqual(run.commits, ['/b']);
  assert.deepEqual(run.directions, [1, 0]);
  assert.equal(warning.mock.callCount(), 1);
});

test('a resolved skipped transition produces no warning', async (context) => {
  const warning = context.mock.method(console, 'warn', () => {});
  const run = setup();
  run.controller.navigate('/b');
  const first = run.transitions[0];
  run.controller.navigate('/c');
  first.update();
  first.resolve();
  await Promise.resolve();
  assert.equal(warning.mock.callCount(), 0);
  run.controller.finish();
});

test('synchronous navigation failures propagate without decoration fallback or retry', (context) => {
  const warning = context.mock.method(console, 'warn', () => {});
  const failure = new Error('navigation failed');
  let attempts = 0;
  const run = setup({
    commit() {
      attempts += 1;
      throw failure;
    },
    start(update) {
      update();
      return new ManualTransition(update);
    },
  });
  assert.throws(
    () => run.controller.navigate('/b'),
    (error) => error === failure,
  );
  run.controller.finish();
  assert.equal(attempts, 1);
  assert.deepEqual(run.directions, [1, 0]);
  assert.equal(warning.mock.callCount(), 0);
});

for (const [from, to] of [
  ['galleries', 'search'],
  ['search', 'history'],
  ['history', 'galleries'],
]) {
  test(`main-page cycle rolls forward from ${from} to ${to} and inversely back`, () => {
    assert.equal(pageRollDirection(from, to), 1);
    assert.equal(pageRollDirection(to, from), -1);
  });
}

for (const [from, to] of [
  ['search', 'search/request-id'],
  ['search/first', 'search/second'],
  ['quality', 'search'],
  ['search', 'integration'],
  ['system', 'quality'],
]) {
  test(`navigation from ${from} to ${to} has no page roll`, () => {
    assert.equal(pageRollDirection(from, to), 0);
  });
}
