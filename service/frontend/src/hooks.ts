import { useCallback, useEffect, useRef, useState } from 'react';
import { api } from './api';
import { errorMessage } from './domain';
import type { Job, PageResult } from './types';

export function useResource<T>(loader: () => Promise<T>, dependencies: unknown[] = []) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const reload = useCallback(() => setRevision((value) => value + 1), []);
  useEffect(() => {
    let alive = true;
    setLoading(true);
    setError(null);
    loaderRef
      .current()
      .then((value) => {
        if (alive) setData(value);
      })
      .catch((cause) => {
        if (alive) setError(errorMessage(cause));
      })
      .finally(() => {
        if (alive) setLoading(false);
      });
    return () => {
      alive = false;
    };
    // Callers provide stable primitives; loader itself intentionally does not trigger refetches.
  }, [...dependencies, revision]);
  return { data, error, loading, reload, setData };
}
export function useJob(
  initial: Job | null,
  onComplete?: (job: Job) => void,
  onUpdate?: (job: Job) => void,
) {
  const [job, setJob] = useState(initial);
  const [error, setError] = useState<string | null>(null);
  const completion = useRef(onComplete);
  completion.current = onComplete;
  const update = useRef(onUpdate);
  update.current = onUpdate;
  const reported = useRef<string | null>(null);
  useEffect(() => {
    setJob(initial);
    setError(null);
  }, [initial]);
  useEffect(() => {
    if (!job) return;
    if (job.status === 'completed') {
      if (reported.current !== job.id) {
        reported.current = job.id;
        completion.current?.(job);
      }
      return;
    }
    if (job.status === 'failed') return;
    let cancelled = false;
    const timer = window.setTimeout(() => {
      api
        .job(job.id)
        .then((next) => {
          if (!cancelled) {
            setJob(next);
            update.current?.(next);
            setError(null);
          }
        })
        .catch((cause) => {
          if (!cancelled) setError(errorMessage(cause));
        });
    }, 1300);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [job]);
  const retry = () => {
    if (job) {
      setError(null);
      setJob({ ...job });
    }
  };
  return { job, error, retry };
}

/** Owns page merging, duplicate suppression and retry; screens only decide where to render Load more. */
export function usePagedResource<T extends { id: string }>(
  loader: (offset?: number) => Promise<PageResult<T>>,
) {
  const [data, setData] = useState<PageResult<T> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [revision, setRevision] = useState(0);
  const generation = useRef(0);
  const loadedCount = useRef(0);
  const loaderRef = useRef(loader);
  loaderRef.current = loader;
  const reload = useCallback(() => {
    generation.current += 1;
    setLoadingMore(false);
    setRevision((value) => value + 1);
  }, []);
  useEffect(() => {
    const requestGeneration = ++generation.current;
    const targetCount = loadedCount.current;
    setLoading(true);
    setError(null);
    const loadWindow = async () => {
      let page = await loaderRef.current(0);
      // Keep already opened pages visible after imports, activation and manual refreshes.
      while (page.items.length < targetCount && page.next_offset != null) {
        if (requestGeneration !== generation.current) return;
        const next = await loaderRef.current(page.next_offset);
        const seen = new Set(page.items.map((item) => item.id));
        page = {
          ...next,
          items: [...page.items, ...next.items.filter((item) => !seen.has(item.id))],
        };
      }
      if (requestGeneration !== generation.current) return;
      loadedCount.current = page.items.length;
      setData(page);
    };
    loadWindow()
      .catch((cause) => {
        if (requestGeneration === generation.current) setError(errorMessage(cause));
      })
      .finally(() => {
        if (requestGeneration === generation.current) setLoading(false);
      });
    return () => {
      generation.current += 1;
    };
  }, [revision]);
  const loadMore = async () => {
    const offset = data?.next_offset;
    if (offset == null || loadingMore || loading) return;
    const requestGeneration = generation.current;
    setLoadingMore(true);
    setError(null);
    try {
      const page = await loader(offset);
      if (requestGeneration !== generation.current) return;
      setData((previous) => {
        if (!previous) return page;
        const seen = new Set(previous.items.map((item) => item.id));
        const merged = {
          ...page,
          items: [...previous.items, ...page.items.filter((item) => !seen.has(item.id))],
        };
        loadedCount.current = merged.items.length;
        return merged;
      });
    } catch (cause) {
      if (requestGeneration === generation.current) setError(errorMessage(cause));
    } finally {
      if (requestGeneration === generation.current) setLoadingMore(false);
    }
  };
  return { data, error, loading, reload, loadMore, loadingMore };
}
