/**
 * app.ts — the simple version.
 *
 * One picture at a time, and one question at a time. The whole loop is:
 *
 *   show a picture  →  it guesses  →  you say what it is  →  it studies  →  again
 *
 * There is no Train button and no numbers to set on the front of the page. It
 * studies by itself after every answer, because the person this is for should
 * not have to know what a learning rate is to teach it something. The numbers
 * still exist, behind the "More detail" door, for whoever wants them.
 *
 * The engine underneath (net.ts, image.ts, store.ts) is unchanged from the
 * detailed version — only this shell is different.
 */

import { Cnn } from './net.js';
import { decodeFile } from './image.js';
import { makeSampleSet } from './samples.js';
import { VisionTrainer } from './trainer-host.js';
import * as store from './store.js';
import { drawHeatmap, drawLine, drawTiles } from './charts.js';
import type { EpochMetric, HostMessage, HostRequest, ModelFile, Sample, TrainSample } from './types.js';

/* ------------------------------------------------------------------ *
 * Helpers
 * ------------------------------------------------------------------ */

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`The page is missing #${id}.`);
  return found as T;
}

function newId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) return crypto.randomUUID();
  return `s-${Math.random().toString(36).slice(2)}-${Date.now().toString(36)}`;
}

/** "a cat" · "a cat and Sarah" · "a cat, Sarah and Bob" */
function listWords(names: string[]): string {
  if (names.length === 0) return 'nothing yet';
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

function plural(count: number, one: string, many: string): string {
  return count === 1 ? one : many;
}

/** Only reuse weights when the list of names has merely grown. */
function startsWith(prefix: string[], full: string[]): boolean {
  if (prefix.length > full.length) return false;
  for (let i = 0; i < prefix.length; i++) if (prefix[i] !== full[i]) return false;
  return true;
}

/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */

type Stage = 'start' | 'asking' | 'ready';

const state = {
  samples: [] as Sample[],
  /** The names it knows. Only ever grows, so a name never changes meaning. */
  names: [] as string[],
  model: null as Cnn | null,
  file: null as ModelFile | null,
  history: [] as EpochMetric[],
  /** The picture on the stage right now — not saved until it is named. */
  current: null as Sample | null,
  guess: null as { name: string; sure: number } | null,
  stage: 'start' as Stage,
  /** The weights handed to the worker for the run in flight. */
  approved: null as ModelFile | null,
  sure: 0.55,
  passes: 60,
  /** A safety net only. The run is background work, so it is not there to make
      anybody wait — it stops a runaway on a very large set of pictures. */
  budgetMs: 45000,
  /** The pass count for the run in flight, and how far through it is. */
  plannedPasses: 60,
  currentPass: 0,
  studying: false,
  /** An answer arrived while a run was going; take it up when that run ends. */
  needsStudy: false,
  augment: true,
  seed: 1,
};

const el = {
  picture: element<HTMLImageElement>('picture'),
  picEmpty: element<HTMLDivElement>('picEmpty'),
  says: element<HTMLElement>('says'),
  sub: element<HTMLElement>('sub'),
  answers: element<HTMLDivElement>('answers'),
  newThing: element<HTMLDivElement>('newThing'),
  newName: element<HTMLInputElement>('newName'),
  newBtn: element<HTMLButtonElement>('newBtn'),
  controls: element<HTMLDivElement>('controls'),
  progress: element<HTMLElement>('progress'),
  studying: element<HTMLElement>('studying'),
  confusedChart: element<HTMLCanvasElement>('confusedChart'),
  rightChart: element<HTMLCanvasElement>('rightChart'),
  mixChart: element<HTMLCanvasElement>('mixChart'),
  filtersChart: element<HTMLCanvasElement>('filtersChart'),
  mapsChart: element<HTMLCanvasElement>('mapsChart'),
  passes: element<HTMLInputElement>('passes'),
  sureInput: element<HTMLInputElement>('sure'),
  augment: element<HTMLInputElement>('augment'),
  card: element<HTMLElement>('card'),
  saveBtn: element<HTMLButtonElement>('saveBtn'),
  loadBtn: element<HTMLButtonElement>('loadBtn'),
  loadInput: element<HTMLInputElement>('loadInput'),
  forgetBtn: element<HTMLButtonElement>('forgetBtn'),
  toast: element<HTMLDivElement>('toast'),
};

/* ------------------------------------------------------------------ *
 * The worker, with a main-thread fallback
 * ------------------------------------------------------------------ */

let worker: Worker | null = null;
try {
  worker = new Worker('./trainer.worker.js', { type: 'module' });
} catch {
  worker = null;
}
const trainer = new VisionTrainer(handleMessage);
if (worker) worker.onmessage = (event: MessageEvent<HostMessage>): void => handleMessage(event.data);

function send(request: HostRequest): void {
  if (worker) worker.postMessage(request);
  else trainer.handle(request);
}

/* ------------------------------------------------------------------ *
 * Chatting to the person
 * ------------------------------------------------------------------ */

let toastTimer = 0;

function toast(message: string): void {
  el.toast.textContent = message;
  el.toast.hidden = false;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    el.toast.hidden = true;
  }, 4200);
}

/**
 * The big line. Built from text nodes rather than markup, because the names come
 * from the person typing and a name is not a place to put HTML.
 */
function say(lead: string, strong?: string, tail?: string): void {
  el.says.textContent = '';
  el.says.append(lead);
  if (strong !== undefined) {
    const bold = document.createElement('b');
    bold.textContent = strong;
    el.says.append(bold);
    if (tail) el.says.append(tail);
  }
}

function but(text: string, kind: '' | 'primary' | 'ghost' | 'danger' | 'guess', onClick: () => void): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `btn${kind ? ` ${kind}` : ''}`;
  button.textContent = text;
  button.addEventListener('click', onClick);
  return button;
}

/* ------------------------------------------------------------------ *
 * What it knows
 * ------------------------------------------------------------------ */

function namedSamples(): Sample[] {
  return state.samples.filter((s) => s.label !== null);
}

/** The names that actually have a picture behind them — the ones worth guessing. */
function knownNames(): string[] {
  const named = namedSamples();
  return state.names.filter((name) => named.some((s) => s.label === name));
}

function learnNames(): void {
  for (const sample of state.samples) {
    if (sample.label && !state.names.includes(sample.label)) state.names.push(sample.label);
  }
}

function guessFor(sample: Sample): { name: string; sure: number } | null {
  const model = state.model;
  const classes = state.file?.classes ?? [];
  if (!model || classes.length < 2) return null;
  const probs = model.predict(sample.pixels);
  let best = 0;
  for (let j = 1; j < classes.length; j++) if (probs[j] > probs[best]) best = j;
  return { name: classes[best] ?? '', sure: probs[best] ?? 0 };
}

/* ------------------------------------------------------------------ *
 * The stage
 * ------------------------------------------------------------------ */

function render(): void {
  const named = namedSamples();
  const names = knownNames();
  const confident = state.guess !== null && state.guess.sure >= state.sure;

  // Drawn first and unconditionally: it read as a stale line whenever one of the
  // branches below returned early without refreshing it.
  renderProgress();

  // The picture.
  el.picture.hidden = state.current === null;
  el.picEmpty.hidden = state.current !== null;
  if (state.current) el.picture.src = state.current.thumb;

  // Everything else is rebuilt, because there is only ever one moment on screen.
  el.answers.textContent = '';
  el.controls.textContent = '';
  el.newThing.hidden = true;

  if (state.stage === 'start') {
    say('Show me a picture.');
    el.sub.textContent =
      names.length < 2
        ? `I ${named.length === 0 ? "don't know anything yet" : `only know ${listWords(names)} so far`}. ` +
          'Show me one and tell me what it is, and I will start learning. I need at least two different things before I can tell them apart.'
        : `I know ${names.length} things — ${listWords(names)} — from ${named.length} ${plural(named.length, 'picture', 'pictures')}. ` +
          'Show me a picture and I will try to guess what it is.';
    el.answers.append(but('Choose a picture', 'primary', () => choosePicture()));
    if (named.length === 0) {
      el.controls.append(but('No pictures handy? Let it practise on 90 drawn shapes', 'ghost', () => void practise()));
    }
    return;
  }

  if (state.stage === 'ready') {
    say('Thanks — I will remember that.');
    el.sub.textContent =
      names.length < 2
        ? `That was ${names[0] ?? 'a new thing'}. Now show me something different — two things is the least I can tell apart.`
        : `I have now seen ${named.length} ${plural(named.length, 'picture', 'pictures')} of ${names.length} things. ` +
          'Show me another and I will try to guess it.';
    el.answers.append(but('Show me another picture', 'primary', () => another()));
    return;
  }

  // stage === 'asking'
  if (state.guess && confident) {
    say('I think this is ', state.guess.name, ' — am I right?');
    el.sub.textContent = "If I'm wrong, just tap the right name.";
  } else if (state.guess) {
    say('I am not sure, but this might be ', state.guess.name, '.');
    el.sub.textContent = 'What is it really?';
  } else {
    say('I do not know what this is yet.');
    el.sub.textContent =
      names.length < 2
        ? 'Tell me what it is, and show me a second kind of thing too — two is the least I can tell apart.'
        : 'Tell me what it is, and I will remember it.';
  }

  // Every name it knows, with its own guess marked as the likely one.
  for (const name of names) {
    const isGuess = state.guess !== null && state.guess.name === name && confident;
    el.answers.append(but(isGuess ? `${name} ✓` : name, isGuess ? 'guess' : '', () => void answer(name)));
  }
  el.answers.append(but('Something new…', 'ghost', () => showNewName()));
  el.controls.append(but('Show me a different picture', 'ghost', () => another()));
}

function showNewName(): void {
  el.newThing.hidden = false;
  el.newName.value = '';
  el.newName.focus();
}

function renderProgress(): void {
  const named = namedSamples();
  const names = knownNames();
  if (named.length === 0) {
    el.progress.textContent = "I haven't seen any pictures yet.";
    return;
  }
  el.progress.textContent = '';
  el.progress.append(`I know ${names.length} ${plural(names.length, 'thing', 'things')} — `);
  const bold = document.createElement('b');
  bold.textContent = listWords(names);
  el.progress.append(bold, ` — from ${named.length} ${plural(named.length, 'picture', 'pictures')}.`);
}

/* ------------------------------------------------------------------ *
 * The loop
 * ------------------------------------------------------------------ */

const filePicker = document.createElement('input');
filePicker.type = 'file';
filePicker.accept = 'image/*';
filePicker.addEventListener('change', () => {
  const file = filePicker.files?.[0];
  filePicker.value = '';
  if (file) void showPicture(file);
});

function choosePicture(): void {
  filePicker.click();
}

function another(): void {
  state.current = null;
  state.guess = null;
  state.stage = 'start';
  render();
  choosePicture();
}

async function showPicture(file: File): Promise<void> {
  try {
    const decoded = await decodeFile(file);
    state.current = {
      id: newId(),
      label: null,
      thumb: decoded.thumb,
      pixels: decoded.pixels,
      name: file.name,
      addedAt: new Date().toISOString(),
      origin: 'file',
    };
    state.guess = guessFor(state.current);
    state.stage = 'asking';
    render();
  } catch {
    toast('That file could not be read as a picture.');
  }
}

async function answer(name: string): Promise<void> {
  const sample = state.current;
  const clean = name.trim().toLowerCase();
  if (!sample || clean === '') return;

  if (!state.names.includes(clean)) state.names.push(clean);
  sample.label = clean;
  if (!state.samples.includes(sample)) state.samples.push(sample);
  await store.putSample(sample);

  state.guess = null;
  state.stage = 'ready';
  render();
  study();
}

async function practise(): Promise<void> {
  const drawn = makeSampleSet(30, 7);
  const added: Sample[] = drawn.map((item) => ({
    id: newId(),
    label: item.label,
    thumb: item.thumb,
    pixels: item.pixels,
    name: item.name,
    addedAt: new Date().toISOString(),
    origin: 'sample',
  }));
  state.samples.push(...added);
  learnNames();
  await store.putSamples(added);
  toast('Drew 90 pictures — circles, squares and triangles — and told it what they are. Watch it learn.');
  render();
  study();
}

/**
 * Study, in the background.
 *
 * This used to block the page on "Studying…". The measurement is blunt about why
 * that was wrong: 12 passes took 6 seconds and got 6 of 9 on pictures it had
 * never seen, and 60 passes took 27 seconds and got 8 of 9. Capping the wait is
 * what left it guessing at chance — and asking somebody to sit through half a
 * minute after every picture is not a page anyone would use. So it studies on its
 * own thread, the page never waits, and because each run continues from the last
 * set of weights the improvement simply accumulates.
 */
function study(): void {
  const named = namedSamples();
  const names = knownNames();

  // One thing is not a classification problem. The stage explains that instead.
  if (named.length < 2 || names.length < 2) return;

  // Already studying: remember there is newer work and take it up when this run
  // ends. Two runs at once would fight over the same weights.
  if (state.studying) {
    state.needsStudy = true;
    return;
  }

  const samples: TrainSample[] = named.map((s) => ({
    id: s.id,
    classIndex: names.indexOf(s.label as string),
    pixels: s.pixels,
  }));

  const previous = state.file && startsWith(state.file.classes, names) ? state.file : null;
  state.approved = previous;
  state.seed = (state.seed + 1) >>> 0;
  state.studying = true;
  state.currentPass = 0;
  state.plannedPasses = Math.max(2, Math.min(80, state.passes));
  renderStudying();

  send({
    type: 'train',
    payload: {
      samples,
      classes: names,
      weights: previous,
      epochs: state.plannedPasses,
      learningRate: 0.004,
      validationSplit: 0.2,
      augment: state.augment,
      seed: state.seed,
      budgetMs: state.budgetMs,
    },
  });
}

/** The small line that says it is still working, so the wait is never silent. */
function renderStudying(): void {
  if (!state.studying) {
    el.studying.hidden = true;
    return;
  }
  el.studying.hidden = false;
  el.studying.textContent = state.model
    ? `Still studying — pass ${state.currentPass} of ${state.plannedPasses}. It keeps getting better while you carry on.`
    : `Studying the pictures. I will start guessing in a moment — pass ${state.currentPass} of ${state.plannedPasses}.`;
}

function handleMessage(message: HostMessage): void {
  switch (message.type) {
    case 'started':
      state.history = state.approved ? state.approved.meta.history.slice() : [];
      renderCurves();
      return;

    case 'progress':
      state.currentPass = message.metric.epoch;
      state.history.push(message.metric);
      renderStudying();
      // Only the curves, on purpose. The filter and activation pictures each need a
      // forward pass through the network, and doing that on every pass would slow
      // the studying down for a picture nobody is looking at yet.
      renderCurves();
      return;

    case 'done':
    case 'stopped': {
      adopt(message.weights);
      state.studying = false;
      renderStudying();
      if (state.needsStudy) {
        state.needsStudy = false;
        study();
      } else if (state.stage === 'asking' && state.current && state.current.label === null) {
        // It got better while the picture sat on the stage — so say what it thinks now.
        state.guess = guessFor(state.current);
        render();
      }
      return;
    }

    case 'error':
      state.studying = false;
      renderStudying();
      toast(message.message);
      return;
  }
}

/* ------------------------------------------------------------------ *
 * The model
 * ------------------------------------------------------------------ */

function adopt(file: ModelFile): void {
  state.file = file;
  state.history = file.meta.history.slice();
  for (const name of file.classes) if (!state.names.includes(name)) state.names.push(name);
  try {
    state.model = Cnn.load(file);
  } catch {
    state.model = null;
  }
  void store.saveModel(file);
  renderCard();
  renderCharts();
}

async function forgetEverything(): Promise<void> {
  await store.clearSamples();
  await store.clearModel();
  state.samples = [];
  state.names = [];
  state.model = null;
  state.file = null;
  state.history = [];
  state.current = null;
  state.guess = null;
  state.stage = 'start';
  render();
  renderCard();
  renderCharts();
  toast('It has forgotten everything — the pictures and everything it learned.');
}

function saveToFile(): void {
  if (!state.file) {
    toast('There is nothing to save yet.');
    return;
  }
  const stamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const blob = new Blob([JSON.stringify(state.file)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = `vision-demo-memory-${stamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  toast('Saved. Load it back with "Load a saved memory".');
}

async function loadFromFile(file: File): Promise<void> {
  try {
    const parsed = JSON.parse(await file.text()) as ModelFile;
    if (parsed.format !== 'vision-demo-model') throw new Error('That is not one of these files.');
    adopt(parsed);
    render();
    toast(`Loaded — it remembers ${parsed.classes.length} things.`);
  } catch (error) {
    toast(error instanceof Error ? error.message : 'That file could not be read.');
  }
}

function renderCard(): void {
  if (!state.file) {
    el.card.textContent = 'Nothing learned yet. It saves itself in this browser as you teach it.';
    return;
  }
  const named = namedSamples();
  const when = new Date(state.file.meta.updatedAt);
  el.card.textContent =
    `It knows ${state.file.classes.length} things (${state.file.classes.join(', ')}) from ${named.length} pictures, ` +
    `and has studied ${state.file.meta.epochsTrained} times in total. Saved in this browser, and last changed ${when.toLocaleString()}.`;
}

/* ------------------------------------------------------------------ *
 * Charts
 * ------------------------------------------------------------------ */

function renderCharts(): void {
  renderCurves();
  renderModelPictures();
}

/** The two line charts. Cheap, so these can be redrawn on every epoch. */
function renderCurves(): void {
  const accent = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#a78bfa';
  const yellow = getComputedStyle(document.body).getPropertyValue('--yellow').trim() || '#fbbf24';

  drawLine(el.confusedChart, {
    series: [
      { values: state.history.map((m) => m.loss), color: accent, label: 'while studying' },
      { values: state.history.map((m) => m.valLoss), color: yellow, label: 'on pictures it had not seen' },
    ],
    emptyTitle: 'This line is drawn as it studies.',
    emptyHint: 'Show it a picture and tell it what the picture is.',
  });

  drawLine(el.rightChart, {
    series: [
      { values: state.history.map((m) => m.trainAccuracy), color: accent, label: 'pictures it studied' },
      { values: state.history.map((m) => m.valAccuracy), color: yellow, label: 'pictures it had not seen' },
    ],
    floor: 0,
    ceil: 1,
    format: (v) => `${Math.round(v * 100)}%`,
    emptyTitle: 'This fills in as it studies.',
    emptyHint: 'It needs at least two kinds of thing first.',
  });
}

/** The pictures of what it learned. Each one needs a forward pass, so these are
    drawn only when the model or the picture on the stage actually changes. */
function renderModelPictures(): void {
  drawHeatmap(el.mixChart, state.file?.meta.confusion ?? [], state.file?.classes ?? [], state.file?.classes ?? [], {
    emptyTitle: 'Nothing to mix up yet.',
    emptyHint: 'Once it has guessed a few times, this shows where it went wrong.',
  });

  // Computed once: it is a pass over the first layer's weights, not a free read.
  const filters = state.model ? state.model.conv1Filters() : null;
  drawTiles(el.filtersChart, filters ? filters.tiles : [], 3, {
    channels: 3,
    max: filters ? filters.max : 1,
    caption: 'red, green and blue are the three colour channels',
    emptyTitle: 'Nothing learned yet.',
    emptyHint: 'These are the little patterns it teaches itself to look for.',
  });

  renderMaps();
}

/** The feature maps for whatever is on the stage — or the last picture it saw. */
function renderMaps(): void {
  const picture = state.current ?? state.samples[state.samples.length - 1] ?? null;
  if (!picture) {
    drawTiles(el.mapsChart, [], 48, {
      channels: 1,
      emptyTitle: 'Nothing to look at yet.',
      emptyHint: 'Show it a picture and this lights up.',
    });
    return;
  }
  if (!state.model) {
    drawTiles(el.mapsChart, [], 48, {
      channels: 1,
      emptyTitle: 'It has not studied yet.',
      emptyHint: 'Teach it two kinds of thing and these appear.',
    });
    return;
  }
  const { maps, max } = state.model.featureMaps(picture.pixels);
  drawTiles(el.mapsChart, maps, 48, { channels: 1, max, caption: 'white is a strong reaction' });
}

/* ------------------------------------------------------------------ *
 * Wiring
 * ------------------------------------------------------------------ */

async function boot(): Promise<void> {
  el.newBtn.addEventListener('click', () => {
    const name = el.newName.value.trim().toLowerCase();
    if (name === '') {
      toast('Type a name first.');
      return;
    }
    void answer(name);
  });
  el.newName.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') el.newBtn.click();
  });

  el.saveBtn.addEventListener('click', () => saveToFile());
  el.loadBtn.addEventListener('click', () => el.loadInput.click());
  el.loadInput.addEventListener('change', () => {
    const file = el.loadInput.files?.[0];
    el.loadInput.value = '';
    if (file) void loadFromFile(file);
  });
  el.forgetBtn.addEventListener('click', () => void forgetEverything());

  el.passes.addEventListener('change', () => {
    state.passes = Math.max(2, Math.min(80, Number(el.passes.value) || 60));
    el.passes.value = String(state.passes);
    void store.saveSetting('passes', state.passes);
  });
  el.sureInput.addEventListener('change', () => {
    state.sure = Math.max(0.3, Math.min(0.95, (Number(el.sureInput.value) || 55) / 100));
    el.sureInput.value = String(Math.round(state.sure * 100));
    void store.saveSetting('sure', state.sure);
    render();
  });
  el.augment.addEventListener('change', () => {
    state.augment = el.augment.checked;
    void store.saveSetting('augment', state.augment);
  });

  // A picture dropped anywhere on the window is the same as choosing one.
  for (const type of ['dragover', 'drop'] as const) {
    window.addEventListener(type, (event) => event.preventDefault());
  }
  window.addEventListener('drop', (event) => {
    const file = (event as DragEvent).dataTransfer?.files?.[0];
    if (file && file.type.startsWith('image/')) void showPicture(file);
  });

  window.addEventListener('resize', () => {
    renderCharts();
  });

  // Pick up where they left off.
  const [samples, model, passes, sure, augment] = await Promise.all([
    store.allSamples().catch(() => [] as Sample[]),
    store.loadModel().catch(() => null),
    store.loadSetting<number>('passes', 60),
    store.loadSetting<number>('sure', 0.55),
    store.loadSetting<boolean>('augment', true),
  ]);

  state.samples = samples;
  learnNames();
  state.passes = passes;
  state.sure = sure;
  state.augment = augment !== false;
  el.passes.value = String(passes);
  el.sureInput.value = String(Math.round(sure * 100));
  el.augment.checked = state.augment;

  if (model) {
    adopt(model);
    state.stage = 'start';
    render();
    renderStudying();
    toast(`Picked up where you left off — it remembers ${model.classes.length} things from ${samples.length} pictures.`);
    return;
  }

  render();
  renderCharts();
}

void boot();
