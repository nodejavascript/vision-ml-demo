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
const NAME = 'vision-ml-demo';
/** What this database was called before the September 2026 rename. */
const LEGACY_NAME = 'vision-demo';
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
/* ---------------- the name change ---------------- */
/** Every [key, value] pair in a store, without needing to know its shape. */
function readPairs(db, storeName) {
    return new Promise((resolve) => {
        if (!db.objectStoreNames.contains(storeName))
            return resolve([]);
        const out = [];
        const request = db.transaction(storeName, 'readonly').objectStore(storeName).openCursor();
        request.onsuccess = () => {
            const cursor = request.result;
            if (!cursor)
                return resolve(out);
            out.push([cursor.key, cursor.value]);
            cursor.continue();
        };
        request.onerror = () => resolve(out);
    });
}
/**
 * Carry over anything saved under the old database name.
 *
 * IndexedDB is keyed by name, so renaming the project on its own would leave every
 * picture and the trained model in a database nothing opens any more. The model is
 * half a minute of practice to rebuild; somebody's own photographs are not. So the
 * old store is read once and copied across.
 *
 * It runs only when the new database is completely empty, so it can never overwrite
 * newer work, and the old database is left where it is rather than deleted — there
 * is no reason to destroy the only copy of something.
 */
export async function migrateLegacy() {
    try {
        const already = await Promise.all([allSamples(), loadModel()]);
        if (already[0].length > 0 || already[1])
            return { carried: 0, failed: false };
        // Listing the databases first keeps this from creating an empty one each time.
        // Where the browser cannot list them the carry-over is skipped, which is the
        // safe way round: better to leave the old data alone than to open a database
        // just to find out whether it is there.
        const listed = await indexedDB.databases?.();
        if (!listed || !listed.some((entry) => entry.name === LEGACY_NAME))
            return { carried: 0, failed: false };
        const legacy = await new Promise((resolve) => {
            const request = indexedDB.open(LEGACY_NAME);
            request.onsuccess = () => resolve(request.result);
            request.onerror = () => resolve(null);
            request.onblocked = () => resolve(null);
        });
        if (!legacy)
            return { carried: 0, failed: true };
        const samples = await readPairs(legacy, IMAGES);
        const models = await readPairs(legacy, MODEL);
        const settings = await readPairs(legacy, SETTINGS);
        legacy.close();
        if (samples.length === 0 && models.length === 0)
            return { carried: 0, failed: false };
        // A store created with a key path takes its key from the value itself. Passing a
        // key as well throws, and the throw takes the entire transaction with it — which
        // is how the first version of this copied nothing and said nothing.
        const put = (target, key, value) => {
            if (target.keyPath === null)
                target.put(value, key);
            else
                target.put(value);
        };
        const db = await open();
        await new Promise((resolve, reject) => {
            const tx = db.transaction([IMAGES, MODEL, SETTINGS], 'readwrite');
            for (const [key, value] of samples)
                put(tx.objectStore(IMAGES), key, value);
            for (const [key, value] of models)
                put(tx.objectStore(MODEL), key, value);
            for (const [key, value] of settings)
                put(tx.objectStore(SETTINGS), key, value);
            tx.oncomplete = () => resolve();
            tx.onerror = () => reject(tx.error ?? new Error('The database write failed.'));
        });
        return { carried: samples.length, failed: false };
    }
    catch (error) {
        // A carry-over that fails must not stop the page from working — but it is
        // reported rather than swallowed, because the first version of this lost
        // everything and looked exactly like there being nothing to carry.
        console.warn('The carry-over from the old database name did not finish.', error);
        return { carried: 0, failed: true };
    }
}
