import test from 'node:test';
import assert from 'node:assert/strict';
import {
  chunkImportFiles,
  matchImportFiles,
  parseGalleryCsv,
  transferImportFiles,
} from '../src/imports.ts';
import type { ImportSession } from '../src/types.ts';

const limits = {
  max_chunk_files: 100,
  max_chunk_bytes: 256 * 1024 * 1024,
  max_upload_bytes: 20 * 1024 * 1024,
};
function session(ids: string[], received: string[] = []): ImportSession {
  const missing = ids.filter((id) => !received.includes(id));
  return {
    id: 'import-1',
    gallery_id: 'gallery-1',
    status: 'uploading',
    expected_count: ids.length,
    uploaded_count: received.length,
    received_image_ids: received,
    missing_image_ids: missing,
    validation: { ready: missing.length === 0, issues: [] },
    job_id: null,
    version_id: null,
    created_at: '2026-09-21T12:00:00Z',
    limits,
  };
}

test('official CSV with 750 observations preserves IDs and pixel boxes without the old 500-file limit', () => {
  const rows = Array.from({ length: 750 }, (_, index) => `image-${index},12,24,360,220`);
  const parsed = parseGalleryCsv(`image_id,x,y,w,h\n${rows.join('\n')}`);
  assert.equal(parsed.issues.length, 0);
  assert.equal(parsed.entries.length, 750);
  assert.deepEqual(parsed.entries[749], { imageId: 'image-749', bbox: [12, 24, 360, 220] });
});

test('CSV supports UTF-8 BOM, reordered columns, CRLF and quoted IDs containing commas', () => {
  const parsed = parseGalleryCsv(
    '\uFEFFh,image_id,w,x,y,vehicle_id\r\n80,"front, view.jpg",120,3,4,ignored\r\n',
  );
  assert.deepEqual(parsed.issues, []);
  assert.deepEqual(parsed.entries, [{ imageId: 'front, view.jpg', bbox: [3, 4, 120, 80] }]);
  assert.deepEqual(parsed.ignoredColumns, ['vehicle_id']);
});

test('CSV preserves integer-valued decimals and partly clipped boxes for server intersection checks', () => {
  const parsed = parseGalleryCsv('image_id,x,y,w,h\nedge,-10.0,-20,100.0,80');
  assert.deepEqual(parsed.issues, []);
  assert.deepEqual(parsed.entries[0].bbox, [-10, -20, 100, 80]);
  assert.ok(parseGalleryCsv('image_id,x,y,w,h\nedge,10000001,0,100,80').issues.length);
});

test('explicit different filenames can share a stem, while implicit extension aliases must not overlap', () => {
  const parsed = parseGalleryCsv('image_id,x,y,w,h\na.jpg,0,0,10,10\na.png,0,0,10,10');
  assert.deepEqual(parsed.issues, []);
  assert.equal(parsed.entries.length, 2);
  assert.ok(parseGalleryCsv('image_id,x,y,w,h\na.tiff,0,0,10,10').issues.length);
  assert.ok(parseGalleryCsv('image_id,x,y,w,h\na,0x10,0,10,10').issues.length);
});

test('malformed, duplicate, ambiguous and invalid manifests fail before transfer', () => {
  for (const source of [
    'image_id,x,y,w\na,0,0,10',
    'image_id,x,y,w,h\n"a,0,0,10,20',
    'image_id,x,y,w,h\n"a"oops,0,0,10,20',
    'image_id,x,y,w,h\na,0,0,10,20\na,0,0,10,20',
    'image_id,x,y,w,h\na,0,0,10,20\na.JPG,0,0,10,20',
    'image_id,x,y,w,h\n../a,0,0,10,20',
    'image_id,x,y,w,h\na,,0,10,20',
    'image_id,x,y,w,h\na,0,0,-10,20',
    'image_id,x,y,w,h\na,0.5,0,10,20',
    'image_id,x,y,w,h\na,0,0,Infinity,20',
    'image_id,x,y,w,h\na,0,0,10',
    'image_id,x,y,w,h\n',
  ])
    assert.ok(parseGalleryCsv(source).issues.length > 0, source);
});
test('preflight rejects header whitespace and empty data records just as the server does', () => {
  const valid = 'image_id,x,y,w,h\na,0,0,10,10';
  assert.ok(parseGalleryCsv(' image_id ,x,y,w,h\na,0,0,10,10').issues.length);
  assert.ok(parseGalleryCsv(`${valid}\n,,,,\n`).issues.length);
  assert.ok(parseGalleryCsv(`${valid}\n\"\"\n`).issues.length);
  assert.deepEqual(parseGalleryCsv(`${valid}\n\n`).issues, []);
  assert.ok(parseGalleryCsv('image_id,x,y,w,h\n\"a\" ,0,0,10,10').issues.length);
});

test('preflight matches stems or filenames, excludes unrelated files and reports missing IDs', () => {
  const files = [
    new File(['a'], 'front.JPG'),
    new File(['b'], 'rear.png'),
    new File(['c'], 'query.jpg'),
    new File(['d'], '.DS_Store'),
  ];
  const result = matchImportFiles(['front', 'rear.PNG', 'absent'], files, 100);
  assert.deepEqual([...result.selected.keys()], ['front', 'rear.PNG']);
  assert.deepEqual(result.missing, ['absent']);
  assert.equal(result.unusedCount, 2);
  assert.deepEqual(result.issues, []);
  assert.deepEqual(matchImportFiles(['Front'], files, 100).missing, ['Front']);
});

test('preflight never chooses an arbitrary extension or duplicate directory basename', () => {
  const files = [
    new File(['a'], 'a.jpg'),
    new File(['b'], 'a.png'),
    new File(['c'], 'b.jpg'),
    new File(['d'], 'b.jpg'),
  ];
  const result = matchImportFiles(['a', 'b.jpg'], files, 100);
  assert.equal(result.issues.length, 2);
  assert.equal(result.selected.size, 0);
  assert.equal(matchImportFiles(['a.jpg'], files, 100).selected.size, 1);
});

test('preflight rejects empty and oversized files', () => {
  const result = matchImportFiles(
    ['a', 'b'],
    [new File([], 'a.jpg'), new File(['large'], 'b.jpg')],
    2,
  );
  assert.equal(result.issues.length, 2);
  assert.equal(result.selected.size, 0);
});

test('chunks respect both image count and request bytes including multipart headroom', () => {
  const files = Array.from({ length: 750 }, (_, index) => ({ name: `${index}.jpg`, size: 1000 }));
  const chunks = chunkImportFiles(files, limits);
  assert.deepEqual(
    chunks.map((chunk) => chunk.length),
    [100, 100, 100, 100, 100, 100, 100, 50],
  );
  assert.equal(chunks.flat().length, 750);
  const small = { ...limits, max_chunk_files: 10, max_chunk_bytes: 8192 };
  assert.deepEqual(
    chunkImportFiles(files.slice(0, 5), small).map((chunk) => chunk.length),
    [3, 2],
  );
  assert.throws(() => chunkImportFiles([{ name: 'big.jpg', size: small.max_chunk_bytes }], small));
});

test('resume reads server progress first and skips an already committed chunk after a lost response', async () => {
  const ids = ['a', 'b', 'c'];
  const files = ids.map((id) => new File([id], `${id}.jpg`));
  let received: string[] = [];
  let loseResponse = true;
  const transferred: string[][] = [];
  const state = () => ({ ...session(ids, received), limits: { ...limits, max_chunk_files: 2 } });
  const transport = {
    status: async () => state(),
    upload: async (chunk: File[]) => {
      const next = chunk.map((file) => file.name.replace('.jpg', ''));
      transferred.push(next);
      received = [...received, ...next];
      if (loseResponse) {
        loseResponse = false;
        throw new Error('Connection lost after commit');
      }
      return state();
    },
  };
  await assert.rejects(() => transferImportFiles(files, transport, () => {}));
  const result = await transferImportFiles(files, transport, () => {});
  assert.deepEqual(transferred, [['a', 'b'], ['c']]);
  assert.equal(result.uploaded_count, 3);
  assert.equal(result.validation.ready, true);
});

test('pause finishes the current atomic chunk and leaves the session resumable', async () => {
  const ids = ['a', 'b', 'c'];
  const files = ids.map((id) => new File([id], `${id}.jpg`));
  let received: string[] = [];
  let paused = false;
  const state = () => ({ ...session(ids, received), limits: { ...limits, max_chunk_files: 1 } });
  const result = await transferImportFiles(
    files,
    {
      status: async () => state(),
      upload: async (chunk) => {
        received.push(chunk[0].name.replace('.jpg', ''));
        return state();
      },
    },
    (next) => {
      if (next.uploaded_count) paused = true;
    },
    () => paused,
  );
  assert.equal(result.uploaded_count, 1);
  assert.deepEqual(result.missing_image_ids, ['b', 'c']);
  assert.equal(result.status, 'uploading');
});

test('partial file selection uploads only available observations and never finalizes implicitly', async () => {
  const ids = ['a', 'b'];
  let received: string[] = [];
  const result = await transferImportFiles(
    [new File(['a'], 'a.jpg')],
    {
      status: async () => session(ids, received),
      upload: async () => {
        received = ['a'];
        return session(ids, received);
      },
    },
    () => {},
  );
  assert.equal(result.uploaded_count, 1);
  assert.deepEqual(result.missing_image_ids, ['b']);
  assert.equal(result.status, 'uploading');
});

test('non-advancing server response stops transfer instead of resending forever', async () => {
  await assert.rejects(
    () =>
      transferImportFiles(
        [new File(['a'], 'a.jpg')],
        {
          status: async () => session(['a']),
          upload: async () => session(['a']),
        },
        () => {},
      ),
    /не подтвердил/,
  );
});
