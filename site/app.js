/**
 * app.ts — the page.
 *
 * One rule runs through all of it: the images never leave. There is no fetch in
 * this file, no analytics, and no server to send anything to. Everything below
 * talks to IndexedDB on this machine and to a Web Worker in this tab.
 *
 * The loop the page exists to show is the one in section 3. The model looks at
 * every image it has no label for, guesses, and hands the visitor the ones it is
 * least sure about — worst first. The answer becomes training data, the model is
 * retrained, and the queue gets shorter and harder.
 */
import { Cnn } from './net.js';
import { decodeFile, paintSample } from './image.js';
import { makeSampleSet } from './samples.js';
import { VisionTrainer } from './trainer-host.js';
import * as store from './store.js';
import { drawBars, drawHeatmap, drawLine, drawTiles } from './charts.js';
/* ------------------------------------------------------------------ *
 * Small helpers
 * ------------------------------------------------------------------ */
function element(id) {
    const found = document.getElementById(id);
    if (!found)
        throw new Error(`The page is missing #${id}.`);
    return found;
}
function newId() {
    if (typeof crypto !== 'undefined' && 'randomUUID' in crypto)
        return crypto.randomUUID();
    return `s-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}
/** One colour per class, stable by index, used in every chart at once. */
function classColor(index) {
    return `hsl(${(265 + index * 47) % 360} 78% 68%)`;
}
/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */
const state = {
    samples: [],
    /** The vocabulary the UI offers — only ever grows, so indices never shift. */
    classes: [],
    model: null,
    file: null,
    history: [],
    predictions: new Map(),
    queue: [],
    askIndex: 0,
    training: false,
    selected: null,
    threshold: 0.6,
    verdict: null,
    /** The picture dropped into section 4, kept only long enough to draw it. */
    probe: null,
    /** The weights we handed the worker for the run in flight, for the charts. */
    pending: null,
    /** How many epochs that run was asked for, so the status line can count up. */
    planned: 0,
};
const el = {
    drop: element('drop'),
    fileInput: element('fileInput'),
    pickBtn: element('pickBtn'),
    sampleBtn: element('sampleBtn'),
    clearBtn: element('clearBtn'),
    imageCount: element('imageCount'),
    labelledCount: element('labelledCount'),
    classCount: element('classCount'),
    gridHint: element('gridHint'),
    grid: element('grid'),
    epochs: element('epochs'),
    lr: element('lr'),
    threshold: element('threshold'),
    augment: element('augment'),
    trainBtn: element('trainBtn'),
    stopBtn: element('stopBtn'),
    resetBtn: element('resetBtn'),
    status: element('statusMsg'),
    kpiParams: element('kpiParams'),
    kpiEpochs: element('kpiEpochs'),
    kpiLoss: element('kpiLoss'),
    kpiValAcc: element('kpiValAcc'),
    kpiConf: element('kpiConf'),
    lossChart: element('lossChart'),
    accChart: element('accChart'),
    askPanel: element('askPanel'),
    askImage: element('askImage'),
    askCaption: element('askCaption'),
    askProgress: element('askProgress'),
    askBars: element('askBars'),
    askButtons: element('askButtons'),
    askNewInput: element('askNewInput'),
    askNewBtn: element('askNewBtn'),
    askSkipBtn: element('askSkipBtn'),
    acceptBtn: element('acceptBtn'),
    predictDrop: element('predictDrop'),
    predictInput: element('predictInput'),
    predictLabel: element('predictLabel'),
    predictConfidence: element('predictConfidence'),
    predictBars: element('predictBars'),
    predictPreview: element('predictPreview'),
    predictMaps: element('predictMaps'),
    filterChart: element('filterChart'),
    saveBtn: element('saveBtn'),
    loadBtn: element('loadBtn'),
    loadInput: element('loadInput'),
    deleteBtn: element('deleteBtn'),
    modelCard: element('modelCard'),
    confusionChart: element('confusionChart'),
    balanceChart: element('balanceChart'),
    toast: element('toast'),
};
/* ------------------------------------------------------------------ *
 * The worker, with a main-thread fallback
 * ------------------------------------------------------------------ */
let worker = null;
try {
    worker = new Worker('./trainer.worker.js', { type: 'module' });
}
catch {
    worker = null;
}
const trainer = new VisionTrainer(handleMessage);
if (worker)
    worker.onmessage = (event) => handleMessage(event.data);
function send(request) {
    if (worker)
        worker.postMessage(request);
    else
        trainer.handle(request);
}
/* ------------------------------------------------------------------ *
 * Chatting to the visitor
 * ------------------------------------------------------------------ */
let toastTimer = 0;
function toast(message) {
    el.toast.textContent = message;
    el.toast.hidden = false;
    window.clearTimeout(toastTimer);
    toastTimer = window.setTimeout(() => {
        el.toast.hidden = true;
    }, 4200);
}
function setStatus(message, kind = '') {
    el.status.textContent = message;
    el.status.className = `status${kind ? ` ${kind}` : ''}`;
}
/* ------------------------------------------------------------------ *
 * Images
 * ------------------------------------------------------------------ */
async function addFiles(files) {
    const list = Array.from(files).filter((f) => f.type.startsWith('image/'));
    if (list.length === 0) {
        toast('Those were not image files.');
        return;
    }
    setStatus(`Reading ${list.length} image${list.length === 1 ? '' : 's'}…`);
    const added = [];
    let failed = 0;
    for (const file of list) {
        try {
            const decoded = await decodeFile(file);
            added.push({
                id: newId(),
                label: null,
                thumb: decoded.thumb,
                pixels: decoded.pixels,
                name: file.name,
                addedAt: new Date().toISOString(),
                origin: 'file',
            });
        }
        catch {
            failed += 1;
        }
    }
    if (added.length === 0) {
        setStatus('None of those could be read as images.', 'warn');
        return;
    }
    state.samples.push(...added);
    await store.putSamples(added);
    learnClasses();
    predictFor(added);
    refresh();
    setStatus(`Added ${added.length} image${added.length === 1 ? '' : 's'}${failed ? `, ${failed} unreadable` : ''}. ` +
        'Give the ones you recognise a name, then press Train.');
}
function addSampleSet() {
    const drawn = makeSampleSet(30, 7);
    // Most of the drawn set arrives labelled, because that is what a set you
    // gathered would look like — but a dozen are held back on purpose, so the
    // queue in section 3 has something real to ask about on the first run.
    const heldBack = new Set();
    const perClass = new Map();
    drawn.forEach((item, index) => {
        const list = perClass.get(item.label) ?? [];
        list.push(index);
        perClass.set(item.label, list);
    });
    for (const list of perClass.values()) {
        for (let i = 0; i < Math.min(4, list.length); i++)
            heldBack.add(list[i]);
    }
    const added = drawn.map((item, index) => ({
        id: newId(),
        label: heldBack.has(index) ? null : item.label,
        thumb: item.thumb,
        pixels: item.pixels,
        name: item.name,
        addedAt: new Date().toISOString(),
        origin: 'sample',
    }));
    state.samples.push(...added);
    void store.putSamples(added);
    learnClasses();
    predictFor(added);
    refresh();
    setStatus('Drew 90 pictures — three shapes, thirty each, in random colours and positions. ' +
        'Twelve were left unnamed deliberately, so the queue below has work to do.', 'good');
}
function removeSample(id) {
    state.samples = state.samples.filter((s) => s.id !== id);
    if (state.selected === id)
        state.selected = null;
    state.predictions.delete(id);
    void store.deleteSample(id);
    refresh();
}
async function removeAll() {
    state.samples = [];
    state.selected = null;
    state.probe = null;
    await store.clearSamples();
    refresh();
    setStatus('All images removed. The model is untouched — use “Forget everything” to clear that too.');
}
function setLabel(id, label) {
    const sample = state.samples.find((s) => s.id === id);
    if (!sample)
        return;
    sample.label = label;
    if (label && !state.classes.includes(label))
        state.classes.push(label);
    state.probe = null;
    void store.putSample(sample);
    refresh();
}
/**
 * Adopt every name the images already carry into the class vocabulary.
 *
 * This has to run after anything that adds named pictures. It was missing at
 * first, and the symptom was quiet rather than loud: the counters read zero
 * classes with seventy-eight images named, and the teach panel drew its buttons
 * from an empty list — so the one loop the page exists for had nothing to press.
 * The vocabulary only ever grows, so an index never has to move.
 */
function learnClasses() {
    for (const sample of state.samples) {
        if (sample.label && !state.classes.includes(sample.label))
            state.classes.push(sample.label);
    }
}
/* ------------------------------------------------------------------ *
 * Predicting
 * ------------------------------------------------------------------ */
function confidenceOf(id) {
    return state.predictions.get(id)?.confidence ?? 0;
}
/**
 * Run the model over a set of images. It is deliberately given only what
 * changed — the pictures just added, or everything after a training run —
 * because a forward pass each is not free, and re-running ninety of them
 * because one label changed is how a page starts to feel slow for no reason.
 */
function predictFor(samples) {
    const model = state.model;
    const classes = state.file?.classes ?? [];
    if (!model || classes.length === 0)
        return;
    for (const sample of samples) {
        const probs = model.predict(sample.pixels);
        let best = 0;
        for (let j = 1; j < classes.length; j++)
            if (probs[j] > probs[best])
                best = j;
        state.predictions.set(sample.id, {
            id: sample.id,
            probs,
            topIndex: best,
            topClass: classes[best] ?? '?',
            confidence: probs[best] ?? 0,
            unsure: (probs[best] ?? 0) < state.threshold,
        });
    }
}
/** Moving the threshold changes what counts as “not sure”, not what it thinks. */
function rethreshold() {
    for (const prediction of state.predictions.values()) {
        prediction.unsure = prediction.confidence < state.threshold;
    }
}
/**
 * The queue is the point of the page: everything with no name, least confident
 * first. An image the model has never been trained on scores zero and goes
 * straight to the front, which is exactly right.
 */
function rebuildQueue() {
    const unnamed = state.samples.filter((s) => s.label === null);
    unnamed.sort((a, b) => confidenceOf(a.id) - confidenceOf(b.id));
    state.queue = unnamed.map((s) => s.id);
    if (state.askIndex >= state.queue.length)
        state.askIndex = 0;
}
/* ------------------------------------------------------------------ *
 * Training
 * ------------------------------------------------------------------ */
function trainClasses() {
    const labelled = state.samples.filter((s) => s.label !== null);
    return state.classes.filter((c) => labelled.some((s) => s.label === c));
}
function startsWith(prefix, full) {
    if (prefix.length > full.length)
        return false;
    for (let i = 0; i < prefix.length; i++)
        if (prefix[i] !== full[i])
            return false;
    return true;
}
/**
 * What the page has, and what it still needs — in that order.
 *
 * The first version of this only ever spoke when Train was pressed, and only
 * about what was missing: four pictures loaded and the line read “it has 0
 * classes and 0 named images”, which reads as though the upload failed. It also
 * went stale the moment a name was given, because nothing recomputed it. So the
 * inventory comes first, the next step is specific, and `refresh` keeps it
 * current rather than waiting to be asked.
 */
function readiness() {
    const images = state.samples.length;
    const named = state.samples.filter((s) => s.label !== null).length;
    const classes = trainClasses();
    if (images === 0) {
        return 'No images yet. Add some above, or press “Draw a sample set”.';
    }
    if (named === 0) {
        return (`${images} image${images === 1 ? '' : 's'} loaded, none named yet — so there is nothing to learn from. ` +
            'Name them in the grid, or answer the questions in step 3.');
    }
    if (classes.length < 2) {
        return (`${named} of ${images} named, but all as “${classes[0] ?? ''}”. One class cannot be told apart from ` +
            'anything — it needs a second: a few pictures of something else (another person, or things that are ' +
            'not this one), named the same way.');
    }
    if (named < 4) {
        return (`${named} named across ${classes.length} classes (${classes.join(', ')}) — four named pictures is the ` +
            'least worth training on, and two of each class is better.');
    }
    const counts = new Map();
    for (const sample of state.samples) {
        if (sample.label)
            counts.set(sample.label, (counts.get(sample.label) ?? 0) + 1);
    }
    const thin = classes.filter((name) => (counts.get(name) ?? 0) < 2);
    if (thin.length > 0) {
        return (`${images} images across ${classes.length} classes, but ${thin.map((n) => `“${n}”`).join(' and ')} ` +
            `${thin.length === 1 ? 'has' : 'have'} fewer than two pictures. Two each is the minimum.`);
    }
    return `Ready — ${named} named images across ${classes.length} classes (${classes.join(', ')}).`;
}
function startTraining() {
    if (state.training)
        return;
    const labelled = state.samples.filter((s) => s.label !== null);
    const classes = trainClasses();
    if (labelled.length < 4 || classes.length < 2) {
        setStatus(readiness(), 'warn');
        return;
    }
    const samples = labelled.map((s) => ({
        id: s.id,
        classIndex: classes.indexOf(s.label),
        pixels: s.pixels,
    }));
    // Reuse the weights only when the class list has merely grown. If a class was
    // dropped from the middle, the output rows no longer mean what they meant, and
    // continuing would be training on a mislabelled model.
    const previous = state.file && startsWith(state.file.classes, classes) ? state.file : null;
    if (state.file && !previous) {
        setStatus('The classes changed order, so this run starts from fresh weights.', 'warn');
    }
    state.pending = previous;
    state.training = true;
    el.trainBtn.disabled = true;
    el.stopBtn.disabled = false;
    el.resetBtn.disabled = true;
    setStatus(`Training on ${samples.length} images across ${classes.length} classes…`);
    send({
        type: 'train',
        payload: {
            samples,
            classes,
            weights: previous,
            epochs: Math.max(1, Math.min(400, Number(el.epochs.value) || 25)),
            learningRate: Math.max(0.0001, Math.min(0.2, Number(el.lr.value) || 0.004)),
            validationSplit: 0.2,
            augment: el.augment.checked,
            seed: 20260919,
        },
    });
}
function stopTraining() {
    if (!state.training)
        return;
    send({ type: 'stop' });
    setStatus('Stopping — the weights are kept as they stand.');
}
function resetWeights() {
    state.model = null;
    state.file = null;
    state.history = [];
    state.predictions.clear();
    void store.clearModel();
    refresh();
    setStatus('The weights are gone. Your images and their names are still here.', 'warn');
}
function adopt(file) {
    state.file = file;
    state.history = file.meta.history.slice();
    for (const name of file.classes)
        if (!state.classes.includes(name))
            state.classes.push(name);
    try {
        state.model = Cnn.load(file);
    }
    catch (error) {
        state.model = null;
        setStatus(error instanceof Error ? error.message : 'That model could not be loaded.', 'warn');
    }
    void store.saveModel(file);
    predictFor(state.samples);
    refresh();
}
function handleMessage(message) {
    switch (message.type) {
        case 'started':
            state.history = state.pending ? state.pending.meta.history.slice() : [];
            state.planned = message.epochs;
            el.kpiParams.textContent = message.parameters.toLocaleString();
            renderCharts();
            return;
        case 'progress':
            state.history.push(message.metric);
            el.kpiEpochs.textContent = String(state.history.length);
            el.kpiLoss.textContent = message.metric.loss.toFixed(3);
            el.kpiValAcc.textContent = `${Math.round(message.metric.valAccuracy * 100)}%`;
            setStatus(`Epoch ${message.metric.epoch} of ${state.planned} — loss ${message.metric.loss.toFixed(3)}, ` +
                `right ${Math.round(message.metric.trainAccuracy * 100)}% of the time, ` +
                `${Math.round(message.metric.valAccuracy * 100)}% on the held-back images.`);
            renderCharts();
            return;
        case 'done':
        case 'stopped':
            state.training = false;
            el.trainBtn.disabled = false;
            el.stopBtn.disabled = true;
            el.resetBtn.disabled = false;
            adopt(message.weights);
            state.pending = null;
            setStatus(`${message.type === 'done' ? 'Trained' : 'Stopped'} at epoch ${message.weights.meta.epochsTrained} — ` +
                `loss ${message.metric.loss.toFixed(3)}, ${Math.round(message.metric.valAccuracy * 100)}% on the held-back images. ` +
                (state.queue.length > 0
                    ? `${state.queue.length} image${state.queue.length === 1 ? '' : 's'} still have no name — the queue below is ordered worst-first.`
                    : 'Every image has a name now.'), 'good');
            return;
        case 'error':
            state.training = false;
            el.trainBtn.disabled = false;
            el.stopBtn.disabled = true;
            el.resetBtn.disabled = false;
            setStatus(message.message, 'warn');
            return;
    }
}
/* ------------------------------------------------------------------ *
 * Rendering
 * ------------------------------------------------------------------ */
function refresh() {
    rebuildQueue();
    renderCounters();
    renderGrid();
    renderAsk();
    renderKpis();
    renderCharts();
    renderModelCard();
    renderVerdict();
    renderInspection();
    // Kept current rather than only spoken on demand. Silenced while training so it
    // cannot talk over the epoch count.
    if (!state.training)
        setStatus(readiness());
}
function renderCounters() {
    const labelled = state.samples.filter((s) => s.label !== null).length;
    el.imageCount.textContent = String(state.samples.length);
    el.labelledCount.textContent = String(labelled);
    el.classCount.textContent = String(trainClasses().length);
    el.gridHint.hidden = state.samples.length > 0;
}
function renderGrid() {
    el.grid.textContent = '';
    const fragment = document.createDocumentFragment();
    const classes = state.classes;
    for (const sample of state.samples) {
        const card = document.createElement('div');
        card.className = `card${state.selected === sample.id ? ' selected' : ''}`;
        const image = document.createElement('img');
        image.src = sample.thumb;
        image.alt = sample.name;
        image.title = 'Click to inspect what the model sees';
        image.addEventListener('click', () => {
            state.selected = sample.id;
            state.probe = null;
            renderGrid();
            renderInspection();
        });
        card.appendChild(image);
        const body = document.createElement('div');
        body.className = 'body';
        const badge = document.createElement('div');
        badge.className = 'badge';
        const said = document.createElement('span');
        const prediction = state.predictions.get(sample.id);
        if (!prediction) {
            said.className = 'said none';
            said.textContent = 'no guess yet';
        }
        else {
            said.className = `said${prediction.unsure ? ' unsure' : ''}`;
            said.textContent = prediction.unsure
                ? `not sure (${Math.round(prediction.confidence * 100)}%)`
                : `${prediction.topClass} ${Math.round(prediction.confidence * 100)}%`;
        }
        badge.appendChild(said);
        body.appendChild(badge);
        const select = document.createElement('select');
        const blank = document.createElement('option');
        blank.value = '';
        blank.textContent = '— name it —';
        select.appendChild(blank);
        for (const name of classes) {
            const option = document.createElement('option');
            option.value = name;
            option.textContent = name;
            select.appendChild(option);
        }
        if (sample.label && !classes.includes(sample.label)) {
            const option = document.createElement('option');
            option.value = sample.label;
            option.textContent = sample.label;
            select.appendChild(option);
        }
        select.value = sample.label ?? '';
        select.addEventListener('change', () => setLabel(sample.id, select.value === '' ? null : select.value));
        body.appendChild(select);
        card.appendChild(body);
        const kill = document.createElement('button');
        kill.type = 'button';
        kill.className = 'kill';
        kill.textContent = '×';
        kill.title = 'Remove this image';
        kill.addEventListener('click', (event) => {
            event.stopPropagation();
            removeSample(sample.id);
        });
        card.appendChild(kill);
        fragment.appendChild(card);
    }
    el.grid.appendChild(fragment);
}
function renderAsk() {
    const currentId = state.queue[state.askIndex];
    const sample = state.samples.find((s) => s.id === currentId);
    el.askButtons.textContent = '';
    if (!sample) {
        el.askImage.hidden = true;
        el.askCaption.textContent =
            state.samples.length === 0
                ? 'Add some images and the questions start here.'
                : 'Every image has a name. Add more, or train again and see if it changes its mind.';
        el.askProgress.textContent = '';
        drawBars(el.askBars, [], { emptyTitle: 'No question waiting.', emptyHint: 'Everything is named.' });
        return;
    }
    el.askImage.hidden = false;
    el.askImage.src = sample.thumb;
    const prediction = state.predictions.get(sample.id);
    if (prediction) {
        el.askCaption.textContent = `It thinks this is “${prediction.topClass}”, ${Math.round(prediction.confidence * 100)}% sure.`;
        el.askProgress.textContent =
            `${state.askIndex + 1} of ${state.queue.length} waiting — ordered least confident first.`;
        drawBars(el.askBars, state.file?.classes.map((name, index) => ({
            label: name,
            value: prediction.probs[index] ?? 0,
            color: classColor(index),
            caption: `${Math.round((prediction.probs[index] ?? 0) * 100)}%`,
        })) ?? [], { max: 1, format: (v) => `${Math.round(v * 100)}%` });
    }
    else {
        el.askCaption.textContent =
            state.classes.length === 0
                ? 'Nothing is named yet, so it has nothing to say. Type a name for what you see below and press “Add class”.'
                : 'It is not sure about this one yet. What is it?';
        el.askProgress.textContent = `${state.askIndex + 1} of ${state.queue.length} waiting.`;
        drawBars(el.askBars, [], { emptyTitle: 'Nothing learned yet.', emptyHint: 'Name a few, then press Train.' });
    }
    // One button per thing it could be, plus the escape hatch of a new one.
    state.classes.forEach((name, index) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.className = 'btn';
        button.textContent = name;
        button.style.borderColor = classColor(index);
        button.addEventListener('click', () => {
            setLabel(sample.id, name);
            toast(`Named “${name}”. That is one more picture it can learn from.`);
        });
        el.askButtons.appendChild(button);
    });
}
function renderKpis() {
    const last = state.history[state.history.length - 1];
    el.kpiParams.textContent = state.model ? state.model.parameterCount.toLocaleString() : '—';
    el.kpiEpochs.textContent = String(state.file?.meta.epochsTrained ?? state.history.length);
    el.kpiLoss.textContent = last ? last.loss.toFixed(3) : '—';
    el.kpiValAcc.textContent = last ? `${Math.round(last.valAccuracy * 100)}%` : '—';
    el.kpiConf.textContent = state.file ? `${Math.round(state.file.meta.meanConfidence * 100)}%` : '—';
}
function renderCharts() {
    const losses = state.history.map((m) => m.loss);
    const valLosses = state.history.map((m) => m.valLoss);
    const trainAcc = state.history.map((m) => m.trainAccuracy);
    const valAcc = state.history.map((m) => m.valAccuracy);
    drawLine(el.lossChart, {
        series: [
            { values: losses, color: getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#a78bfa', label: 'training' },
            { values: valLosses, color: getComputedStyle(document.body).getPropertyValue('--yellow').trim() || '#fbbf24', label: 'held back' },
        ],
        emptyTitle: 'The loss curve is drawn here as it trains.',
        emptyHint: 'Name some images and press Train.',
    });
    drawLine(el.accChart, {
        series: [
            { values: trainAcc, color: getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#a78bfa', label: 'training' },
            { values: valAcc, color: getComputedStyle(document.body).getPropertyValue('--yellow').trim() || '#fbbf24', label: 'held back' },
        ],
        floor: 0,
        ceil: 1,
        format: (v) => `${Math.round(v * 100)}%`,
        emptyTitle: 'Accuracy lands here.',
        emptyHint: 'One point per epoch, so it needs two.',
    });
    const balance = trainClasses().map((name, index) => ({
        label: name,
        value: state.samples.filter((s) => s.label === name).length,
        color: classColor(index),
    }));
    drawBars(el.balanceChart, balance, {
        emptyTitle: 'No classes yet.',
        emptyHint: 'Name a few images and the counts appear.',
    });
    const matrix = state.file?.meta.confusion ?? [];
    drawHeatmap(el.confusionChart, matrix, state.file?.classes ?? [], state.file?.classes ?? [], {
        emptyTitle: 'Nothing classified yet.',
        emptyHint: 'Train the model and this fills in — rows are the truth, columns are its answer.',
    });
    renderFilters();
}
function renderFilters() {
    if (!state.model) {
        drawTiles(el.filterChart, [], 3, {
            channels: 3,
            emptyTitle: 'No kernels yet.',
            emptyHint: 'These are what the first layer learned to look for.',
        });
        return;
    }
    const { tiles, max } = state.model.conv1Filters();
    drawTiles(el.filterChart, tiles, 3, { channels: 3, max, caption: 'red, green and blue are the three colour channels' });
}
function renderModelCard() {
    if (!state.file) {
        el.modelCard.textContent = 'No model yet. Trained weights are saved in this browser automatically.';
        el.modelCard.className = 'status';
        return;
    }
    const meta = state.file.meta;
    const when = new Date(meta.updatedAt);
    el.modelCard.className = 'status good';
    el.modelCard.textContent =
        `A model of ${meta.parameters.toLocaleString()} numbers, over ${state.file.classes.length} classes ` +
            `(${state.file.classes.join(', ')}), trained for ${meta.epochsTrained} epochs on ${meta.images} images. ` +
            `It is ${Math.round(meta.meanConfidence * 100)}% confident on average, and last changed ${when.toLocaleString()}. ` +
            'Saved in this browser, and downloadable below.';
}
function renderVerdict() {
    if (!state.verdict) {
        el.predictLabel.textContent = '—';
        el.predictConfidence.textContent = '';
        // An empty state, not three rows of zeros: the prompt is what tells a reader
        // what the box is for.
        drawBars(el.predictBars, [], {
            emptyTitle: 'Drop an image to classify it.',
            emptyHint: 'The bars fill in with its answer.',
        });
        return;
    }
    const classes = state.file?.classes ?? [];
    let best = 0;
    for (let j = 1; j < classes.length; j++)
        if (state.verdict.probs[j] > state.verdict.probs[best])
            best = j;
    const confidence = state.verdict.probs[best] ?? 0;
    el.predictLabel.textContent = classes[best] ?? '—';
    el.predictConfidence.textContent =
        confidence < state.threshold
            ? `${Math.round(confidence * 100)}% — under your threshold, so it is really saying “I do not know”.`
            : `${Math.round(confidence * 100)}% sure.`;
    drawBars(el.predictBars, classes.map((name, index) => ({
        label: name,
        value: state.verdict?.probs[index] ?? 0,
        color: classColor(index),
        caption: `${Math.round((state.verdict?.probs[index] ?? 0) * 100)}%`,
    })), { max: 1, format: (v) => `${Math.round(v * 100)}%` });
}
/** The two pictures that show what the network actually did with an image. */
function renderInspection() {
    const chosen = state.selected ?? state.queue[state.askIndex] ?? state.samples[0]?.id ?? null;
    const sample = state.probe ?? state.samples.find((s) => s.id === chosen) ?? null;
    if (!sample) {
        drawTiles(el.predictMaps, [], 48, {
            channels: 1,
            emptyTitle: 'Nothing to look at yet.',
            emptyHint: 'Add an image and this shows what the network noticed.',
        });
        return;
    }
    paintSample(el.predictPreview, sample.pixels);
    if (!state.model) {
        drawTiles(el.predictMaps, [], 48, {
            channels: 1,
            emptyTitle: 'No activations yet.',
            emptyHint: 'Train the model and these eight maps fill in.',
        });
        return;
    }
    const { maps, max } = state.model.featureMaps(sample.pixels);
    drawTiles(el.predictMaps, maps, 48, { channels: 1, max, caption: 'one map per first-layer kernel, white is a strong response' });
}
/* ------------------------------------------------------------------ *
 * Keeping it, and bringing it back
 * ------------------------------------------------------------------ */
function downloadModel() {
    if (!state.file) {
        toast('There is no model to download yet.');
        return;
    }
    const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const blob = new Blob([JSON.stringify(state.file)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = `vision-demo-model-${stamp}.json`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
    toast('Downloaded. Load it back with “Load a model file”.');
}
async function loadModelFile(file) {
    try {
        const parsed = JSON.parse(await file.text());
        if (parsed.format !== 'vision-demo-model')
            throw new Error('That is not a vision-demo model file.');
        adopt(parsed);
        setStatus(`Loaded the model from ${file.name}. It knows ${parsed.classes.length} classes and has trained for ${parsed.meta.epochsTrained} epochs.`, 'good');
    }
    catch (error) {
        setStatus(error instanceof Error ? error.message : 'That file could not be read.', 'warn');
    }
}
async function forgetEverything() {
    await store.clearSamples();
    await store.clearModel();
    state.samples = [];
    state.classes = [];
    state.model = null;
    state.file = null;
    state.history = [];
    state.predictions.clear();
    state.queue = [];
    state.selected = null;
    state.probe = null;
    state.verdict = null;
    refresh();
    setStatus('Everything is gone — images, names and weights.', 'warn');
}
/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */
function wireDropZone(zone, input, onFiles) {
    zone.addEventListener('click', () => input.click());
    zone.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            input.click();
        }
    });
    input.addEventListener('change', () => {
        if (input.files && input.files.length > 0)
            onFiles(input.files);
        input.value = '';
    });
    for (const type of ['dragenter', 'dragover']) {
        zone.addEventListener(type, (event) => {
            event.preventDefault();
            zone.classList.add('over');
        });
    }
    for (const type of ['dragleave', 'drop']) {
        zone.addEventListener(type, (event) => {
            event.preventDefault();
            zone.classList.remove('over');
        });
    }
    zone.addEventListener('drop', (event) => {
        const files = event.dataTransfer?.files;
        if (files && files.length > 0)
            onFiles(files);
    });
}
async function boot() {
    wireDropZone(el.drop, el.fileInput, (files) => void addFiles(files));
    wireDropZone(el.predictDrop, el.predictInput, (files) => void classifyOne(files));
    el.pickBtn.addEventListener('click', () => el.fileInput.click());
    el.sampleBtn.addEventListener('click', () => addSampleSet());
    el.clearBtn.addEventListener('click', () => void removeAll());
    el.trainBtn.addEventListener('click', () => startTraining());
    el.stopBtn.addEventListener('click', () => stopTraining());
    el.resetBtn.addEventListener('click', () => resetWeights());
    el.saveBtn.addEventListener('click', () => downloadModel());
    el.loadBtn.addEventListener('click', () => el.loadInput.click());
    el.loadInput.addEventListener('change', () => {
        const file = el.loadInput.files?.[0];
        if (file)
            void loadModelFile(file);
        el.loadInput.value = '';
    });
    el.deleteBtn.addEventListener('click', () => void forgetEverything());
    el.askSkipBtn.addEventListener('click', () => {
        state.askIndex = (state.askIndex + 1) % Math.max(1, state.queue.length);
        renderAsk();
        renderInspection();
    });
    el.askNewBtn.addEventListener('click', () => {
        const name = el.askNewInput.value.trim().toLowerCase();
        const id = state.queue[state.askIndex];
        if (!name) {
            toast('Type a name for the new thing first.');
            return;
        }
        if (!state.classes.includes(name))
            state.classes.push(name);
        el.askNewInput.value = '';
        if (id) {
            setLabel(id, name);
            toast(`Added “${name}” and gave it this picture.`);
        }
        else {
            refresh();
            toast(`Added “${name}”. Give it a picture or two.`);
        }
    });
    el.acceptBtn.addEventListener('click', () => acceptConfident());
    el.threshold.addEventListener('change', () => {
        state.threshold = Math.max(0.3, Math.min(0.95, Number(el.threshold.value) || 0.6));
        el.threshold.value = state.threshold.toFixed(2);
        void store.saveSetting('threshold', state.threshold);
        rethreshold();
        refresh();
    });
    for (const field of [el.epochs, el.lr, el.augment]) {
        field.addEventListener('change', () => {
            void store.saveSetting('training', {
                epochs: Number(el.epochs.value),
                lr: Number(el.lr.value),
                augment: el.augment.checked,
            });
        });
    }
    window.addEventListener('resize', () => {
        renderCharts();
        renderVerdict();
        renderInspection();
    });
    // Come back where you left off: the images, then the model.
    const [samples, model, threshold, training] = await Promise.all([
        store.allSamples().catch(() => []),
        store.loadModel().catch(() => null),
        store.loadSetting('threshold', 0.6),
        store.loadSetting('training', { epochs: 25, lr: 0.004, augment: true }),
    ]);
    state.samples = samples;
    state.threshold = threshold;
    el.threshold.value = threshold.toFixed(2);
    el.epochs.value = String(training.epochs ?? 25);
    el.lr.value = String(training.lr ?? 0.004);
    el.augment.checked = training.augment !== false;
    learnClasses();
    if (model) {
        adopt(model);
        setStatus(`Picked up where you left off — ${samples.length} images and a model trained for ${model.meta.epochsTrained} epochs, ` +
            `remembered from this browser.`, 'good');
    }
    else if (samples.length > 0) {
        refresh();
        setStatus(`${samples.length} images remembered from last time, with no model yet. Press Train.`);
    }
    else {
        // No status call here on purpose: `refresh` already states the inventory, and
        // two sources for one line is how it drifts.
        refresh();
    }
}
async function classifyOne(files) {
    const file = Array.from(files).find((f) => f.type.startsWith('image/'));
    if (!file) {
        toast('That was not an image.');
        return;
    }
    if (!state.model || !state.file) {
        toast('Train a model first — then it has something to say.');
        return;
    }
    try {
        const decoded = await decodeFile(file);
        state.verdict = { name: file.name, probs: state.model.predict(decoded.pixels) };
        // The dropped picture is shown in the inspection charts without ever joining
        // the grid: it has no name, and the model must not learn from a picture the
        // visitor has not decided about.
        state.probe = {
            id: 'probe',
            label: null,
            thumb: decoded.thumb,
            pixels: decoded.pixels,
            name: file.name,
            addedAt: new Date().toISOString(),
            origin: 'file',
        };
        state.selected = null;
        renderVerdict();
        renderInspection();
    }
    catch {
        toast('That image could not be read.');
    }
}
/** Pseudo-labelling, and it is honest about it: only the confident ones. */
function acceptConfident() {
    const accepted = [];
    for (const sample of state.samples) {
        if (sample.label !== null)
            continue;
        const prediction = state.predictions.get(sample.id);
        if (!prediction || prediction.confidence < state.threshold)
            continue;
        sample.label = prediction.topClass;
        accepted.push(sample.id);
    }
    if (accepted.length === 0) {
        toast('Nothing is confident enough yet — name a few yourself and train again.');
        return;
    }
    for (const id of accepted) {
        const sample = state.samples.find((s) => s.id === id);
        if (sample)
            void store.putSample(sample);
    }
    refresh();
    toast(`Accepted ${accepted.length} of its own guesses, at ${Math.round(state.threshold * 100)}% or better.`);
}
void boot();
