/**
 * store.ts — where the images and the model live between visits.
 *
 * IndexedDB, not localStorage: the images are binary and there can be hundreds
 * of them, and localStorage would hold neither comfortably. A model is a few
 * hundred kilobytes of floats and structured-clones perfectly, so the same
 * store holds both.
 *
 * The point of this file is the sentence on the page: close the tab, come back,
 * and the model you taught is still there.
 */

import type { ModelFile, Sample } from './types.js';

const NAME = 'vision-demo';
const VERSION = 1;
const IMAGES = 'images';
const MODEL = 'model';
const SETTINGS = 'settings';

let handle: Promise<IDBDatabase> | null = null;

function open(): Promise<IDBDatabase> {
  if (handle) return handle;
  handle = new Promise((resolve, reject) => {
    const request = indexedDB.open(NAME, VERSION);
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(IMAGES)) db.createObjectStore(IMAGES, { keyPath: 'id' });
      if (!db.objectStoreNames.contains(MODEL)) db.createObjectStore(MODEL);
      if (!db.objectStoreNames.contains(SETTINGS)) db.createObjectStore(SETTINGS);
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error ?? new Error('IndexedDB refused to open.'));
  });
  return handle;
}

function run<T>(store: string, mode: IDBTransactionMode, work: (s: IDBObjectStore) => IDBRequest): Promise<T> {
  return open().then(
    (db) =>
      new Promise<T>((resolve, reject) => {
        const tx = db.transaction(store, mode);
        const request = work(tx.objectStore(store));
        request.onsuccess = () => resolve(request.result as T);
        request.onerror = () => reject(request.error ?? new Error('The database write failed.'));
      }),
  );
}

/* ---------------- images ---------------- */

export function allSamples(): Promise<Sample[]> {
  return run<Sample[]>(IMAGES, 'readonly', (s) => s.getAll()).then((rows) =>
    rows.sort((a, b) => (a.addedAt < b.addedAt ? -1 : a.addedAt > b.addedAt ? 1 : 0)),
  );
}

export function putSample(sample: Sample): Promise<IDBValidKey> {
  return run<IDBValidKey>(IMAGES, 'readwrite', (s) => s.put(sample));
}

export function putSamples(samples: Sample[]): Promise<void> {
  return open().then(
    (db) =>
      new Promise<void>((resolve, reject) => {
        const tx = db.transaction(IMAGES, 'readwrite');
        const store = tx.objectStore(IMAGES);
        for (const sample of samples) store.put(sample);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error ?? new Error('The database write failed.'));
      }),
  );
}

export function deleteSample(id: string): Promise<void> {
  return run<undefined>(IMAGES, 'readwrite', (s) => s.delete(id)).then(() => undefined);
}

export function clearSamples(): Promise<void> {
  return run<undefined>(IMAGES, 'readwrite', (s) => s.clear()).then(() => undefined);
}

/* ---------------- the model ---------------- */

export function saveModel(file: ModelFile): Promise<IDBValidKey> {
  return run<IDBValidKey>(MODEL, 'readwrite', (s) => s.put(file, 'current'));
}

export function loadModel(): Promise<ModelFile | null> {
  return run<ModelFile | undefined>(MODEL, 'readonly', (s) => s.get('current')).then((v) => v ?? null);
}

export function clearModel(): Promise<void> {
  return run<undefined>(MODEL, 'readwrite', (s) => s.delete('current')).then(() => undefined);
}

/* ---------------- settings ---------------- */

export function saveSetting(key: string, value: unknown): Promise<IDBValidKey> {
  return run<IDBValidKey>(SETTINGS, 'readwrite', (s) => s.put(value, key));
}

export function loadSetting<T>(key: string, fallback: T): Promise<T> {
  return run<T | undefined>(SETTINGS, 'readonly', (s) => s.get(key)).then((v) => (v === undefined ? fallback : (v as T)));
}

/** Is there anything stored at all? Used to decide whether to greet a returner. */
export async function hasStoredModel(): Promise<boolean> {
  try {
    return (await loadModel()) !== null;
  } catch {
    return false;
  }
}
