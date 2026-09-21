import test from 'node:test';
import assert from 'node:assert/strict';
import {
  bboxFromPoints,
  heatOpacity,
  normalizeProgress,
  safeDownloadUrl,
  validBBox,
} from '../src/domain.ts';

test('bounding boxes use natural pixels and support reverse selection', () => {
  assert.deepEqual(bboxFromPoints([300, 200], [10, 20], 640, 480), [10, 20, 290, 180]);
  assert.deepEqual(bboxFromPoints([10.5, 20.5], [110.9, 220.9], 640, 480), [10, 20, 101, 201]);
});
test('bounding boxes stay inside the source image at edges', () => {
  assert.deepEqual(bboxFromPoints([630, 470], [900, 1000], 640, 480), [630, 470, 10, 10]);
  assert.deepEqual(bboxFromPoints([640, 480], [640, 480], 640, 480), [639, 479, 1, 1]);
});
test('manual bounding boxes reject invalid or fractional geometry', () => {
  assert.equal(validBBox([0, 0, 640, 480], 640, 480), true);
  for (const invalid of [
    [-1, 0, 2, 2],
    [0, 0, 0, 2],
    [600, 0, 50, 20],
    [0, 460, 20, 30],
    [0.5, 0, 5, 5],
    [NaN, 0, 5, 5],
  ]) {
    assert.equal(validBBox(invalid as [number, number, number, number], 640, 480), false);
  }
});
test('one percent progress remains one percent', () => {
  assert.equal(normalizeProgress(1), 1);
  assert.equal(normalizeProgress(0), 0);
  assert.equal(normalizeProgress(120), 100);
  assert.equal(normalizeProgress(-10), 0);
});
test('download links never leave the authenticated API origin', () => {
  assert.equal(
    safeDownloadUrl('/api/v1/exports/job-123/download'),
    '/api/v1/exports/job-123/download',
  );
  for (const unsafe of [
    'https://evil.example/file',
    '//evil.example/file',
    'javascript:alert(1)',
    '/api/\\evil.example',
    null,
    {},
    '/downloads/file',
  ])
    assert.equal(safeDownloadUrl(unsafe), null);
});
test('zero sensitivity produces a transparent map, not fabricated emphasis', () => {
  assert.equal(heatOpacity(0, 0), 0);
  assert.equal(heatOpacity(0.5, 1), 0.4);
  assert.equal(heatOpacity(-0.5, 1), 0.4);
  assert.equal(heatOpacity(5, 1), 0.8);
});
