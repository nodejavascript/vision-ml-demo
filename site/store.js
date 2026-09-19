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
const NAME = 'vision-demo';
const VERSION = 1;
const IMAGES = 'images';
const MODEL = 'model';
const SETTINGS = 'settings';
let handle = null;
function open() {
    if (handle)
        return handle;
    handle = new Promise((resolve, reject) => {
        const request = indexedDB.open(NAME, VERSION);
        request.onupgradeneeded = () => {
            const db = request.result;
            if (!db.objectStoreNames.contains(IMAGES))
                db.createObjectStore(IMAGES, { keyPath: 'id' });
            if (!db.objectStoreNames.contains(MODEL))
                db.createObjectStore(MODEL);
            if (!db.objectStoreNames.contains(SETTINGS))
                db.createObjectStore(SETTINGS);
        };
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error('IndexedDB refused to open.'));
    });
    return handle;
}
function run(store, mode, work) {
    return open().then((db) => new Promise((resolve, reject) => {
        const tx = db.transaction(store, mode);
        const request = work(tx.objectStore(store));
        request.onsuccess = () => resolve(request.result);
        request.onerror = () => reject(request.error ?? new Error('The database write failed.'));
    }));
}
/* ---------------- images ---------------- */
export function allSamples() {
    return run(IMAGES, 'readonly', (s) => s.getAll()).then((rows) => rows.sort((a, b) => (a.addedAt < b.addedAt ? -1 : a.addedAt > b.addedAt ? 1 : 0)));
}
export function putSample(sample) {
    return run(IMAGES, 'readwrite', (s) => s.put(sample));
}
export function putSamples(samples) {
    return open().then((db) => new Promise((resolve, reject) => {
        const tx = db.transaction(IMAGES, 'readwrite');
        const store = tx.objectStore(IMAGES);
        for (const sample of samples)
            store.put(sample);
        tx.oncomplete = () => resolve();
        tx.onerror = () => reject(tx.error ?? new Error('The database write failed.'));
    }));
}
export function deleteSample(id) {
    return run(IMAGES, 'readwrite', (s) => s.delete(id)).then(() => undefined);
}
export function clearSamples() {
    return run(IMAGES, 'readwrite', (s) => s.clear()).then(() => undefined);
}
/* ---------------- the model ---------------- */
export function saveModel(file) {
    return run(MODEL, 'readwrite', (s) => s.put(file, 'current'));
}
export function loadModel() {
    return run(MODEL, 'readonly', (s) => s.get('current')).then((v) => v ?? null);
}
export function clearModel() {
    return run(MODEL, 'readwrite', (s) => s.delete('current')).then(() => undefined);
}
/* ---------------- settings ---------------- */
export function saveSetting(key, value) {
    return run(SETTINGS, 'readwrite', (s) => s.put(value, key));
}
export function loadSetting(key, fallback) {
    return run(SETTINGS, 'readonly', (s) => s.get(key)).then((v) => (v === undefined ? fallback : v));
}
/** Is there anything stored at all? Used to decide whether to greet a returner. */
export async function hasStoredModel() {
    try {
        return (await loadModel()) !== null;
    }
    catch {
        return false;
    }
}
