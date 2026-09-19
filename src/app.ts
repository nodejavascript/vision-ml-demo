/**
 * app.ts — the simple version, multi-label.
 *
 * One picture at a time, and one question at a time. The whole loop is:
 *
 *   show a picture  →  it says what it can see  →  you list what you see
 *                   →  it studies  →  again
 *
 * The answer is a LIST, separated by commas, and that is not a detail of the
 * wording — it is what makes the learning better. A photograph of a dog on a
 * beach teaches it "dog" and "beach" at once instead of forcing a choice between
 * them, and the same picture gets reused by everything in it. Underneath, that
 * means the network answers an independent yes/no for each thing (sigmoid and
 * binary cross-entropy) rather than picking one winner from all of them (softmax).
 *
 * There is no Train button, because the person this is for should not have to know
 * what a learning rate is to teach it something.
 */

import { Cnn } from './net.js';
import { decodeFile } from './image.js';
import { makeSampleSet } from './samples.js';
import { VisionTrainer } from './trainer-host.js';
import * as store from './store.js';
import { drawBars, drawLine, drawTiles } from './charts.js';
import type { EpochMetric, HostMessage, HostRequest, ModelFile, Sample, TrainSample } from './types.js';

/** Bumped whenever the weights stop meaning what they meant. See types.ts. */
const MODEL_VERSION = 2;

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

/** "a bear" · "a bear and a tree" · "a bear, a tree and the sky" */
function listWords(names: string[]): string {
  if (names.length === 0) return 'nothing yet';
  if (names.length === 1) return names[0];
  return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}

function plural(count: number, one: string, many: string): string {
  return count === 1 ? one : many;
}

function startsWith(prefix: string[], full: string[]): boolean {
  if (prefix.length > full.length) return false;
  for (let i = 0; i < prefix.length; i++) if (prefix[i] !== full[i]) return false;
  return true;
}

/**
 * Words that cannot begin a name.
 *
 * A picture is named by the things in it — dog, sea, beach, rail house — so "the
 * dog" and "my dog" are both just dog. If a label starts with one of these it is
 * not a name at all, it is the start of a sentence, and that is the signal used
 * below to decide whether to clean it up.
 */
const OPENS_A_SENTENCE = new Set([
  'a', 'an', 'the', 'this', 'that', 'these', 'those', 'some',
  'my', 'our', 'your', 'his', 'her', 'their', 'its',
  'i', 'we', 'you', 'it', 'there', 'they', 'he', 'she',
]);

/** Once a label is known to be a sentence, these come off the front of it too. */
const FRAMING = new Set([
  ...OPENS_A_SENTENCE,
  'is', 'are', 'was', 'were', 'am', 'be', 'been', 'being', 'will', 'would', 'can', 'could',
  'see', 'sees', 'saw', 'seen', 'look', 'looks', 'looking', 'like',
  'photo', 'photos', 'picture', 'pictures', 'image', 'images', 'shot',
  'of', 'in', 'on', 'at', 'with', 'and', 'or', 'to', 'very', 'just', 'really', 'only',
]);

/** How to answer, said the same way everywhere so it only has to be learned once. */
const HOW_TO_LIST = 'Name the things you can see — nouns only, separated by commas.';

/**
 * What the person typed, as a list of things.
 *
 * They are asked for nouns, because a name is what the model can use: a class called
 * "this is a photo of my cat" is not a thing, it is a sentence, and it would sit in
 * the vocabulary forever getting in the way of everything else. Three things are
 * done about that, and only these three:
 *
 *   1. Split on the separators people actually type — commas, semicolons, new
 *      lines — and on the words "and" and "or", because "a dog and a beach" is two
 *      things rather than one long one.
 *   2. Strip framing words **off the front** of a piece, turning the sentence above
 *      into "cat".
 *   3. Leave everything else exactly as typed.
 *
 * The front is the only safe place to cut. A word removed from the middle destroys a
 * name — "cup of tea", "rail house", "fish and chips" — so nothing is ever removed
 * from the middle or the end, and cleaning only starts at all when the first word is
 * one that could not begin a name. "photo frame" and "can opener" therefore come
 * through untouched, because they do not start with one.
 *
 * There is no dictionary here and this does not pretend to be one: it cannot know
 * that "sitting" is not a thing. What it can do is stop at the first real word, and
 * never invent a meaning by cutting into one.
 */
function parseLabels(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const piece of text.toLowerCase().split(/[,;\n]+|\band\b|\bor\b/)) {
    const words = piece
      .replace(/[.!?]+/g, ' ')
      .split(/\s+/)
      .filter((word) => word !== '');

    let start = 0;
    if (words.length > 0 && OPENS_A_SENTENCE.has(words[0])) {
      while (start < words.length && FRAMING.has(words[start])) start += 1;
    }

    const name = words.slice(start).join(' ');
    if (name === '' || seen.has(name)) continue;
    seen.add(name);
    out.push(name);
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * State
 * ------------------------------------------------------------------ */

type Stage = 'start' | 'asking' | 'ready';

const state = {
  samples: [] as Sample[],
  /** The things it knows. Only ever grows, so a name never changes meaning. */
  names: [] as string[],
  model: null as Cnn | null,
  file: null as ModelFile | null,
  history: [] as EpochMetric[],
  /** The picture on the stage right now — not saved until it is named. */
  current: null as Sample | null,
  /** What it can see in the current picture, most sure first. */
  sees: [] as Array<{ name: string; sure: number }>,
  /**
   * Its single strongest answer, whatever its confidence.
   *
   * Kept because "I do not know" is a dead end for the person standing in front of
   * the page — the confidence line is theirs to move, and knowing what it leans
   * towards is what makes that line mean something. It is only ever shown as a
   * guess, never as an answer.
   */
  best: null as { name: string; sure: number } | null,
  /**
   * Pictures handed over together and still to be named, in order.
   *
   * Held as files rather than decoded pictures on purpose: fifty photographs
   * decoded up front would sit in memory as fifty bitmaps for no reason.
   */
  pending: [] as File[],
  /** How many pictures of this run have been finished — named, or skipped. */
  runDone: 0,
  /**
   * What the person listed for the picture just finished.
   *
   * Carried onto the next picture's line so a run reads as one continuous
   * conversation. Without it, naming a picture and being shown the next one looks
   * like the answer was thrown away.
   */
  lastNoted: [] as string[],
  stage: 'start' as Stage,
  approved: null as ModelFile | null,
  sure: 0.5,
  passes: 60,
  /** A runaway guard only: studying is background work, not a wait. */
  budgetMs: 45000,
  plannedPasses: 60,
  currentPass: 0,
  studying: false,
  needsStudy: false,
  augment: true,
  seed: 1,
};

const el = {
  stage: element<HTMLElement>('stage'),
  pic: element<HTMLDivElement>('pic'),
  picture: element<HTMLImageElement>('picture'),
  picEmpty: element<HTMLDivElement>('picEmpty'),
  says: element<HTMLElement>('says'),
  sub: element<HTMLElement>('sub'),
  queue: element<HTMLElement>('queue'),
  tell: element<HTMLDivElement>('tell'),
  list: element<HTMLInputElement>('list'),
  tellBtn: element<HTMLButtonElement>('tellBtn'),
  answers: element<HTMLDivElement>('answers'),
  controls: element<HTMLDivElement>('controls'),
  aside: element<HTMLElement>('aside'),
  progress: element<HTMLElement>('progress'),
  studying: element<HTMLElement>('studying'),
  confusedChart: element<HTMLCanvasElement>('confusedChart'),
  rightChart: element<HTMLCanvasElement>('rightChart'),
  knowsChart: element<HTMLCanvasElement>('knowsChart'),
  filtersChart: element<HTMLCanvasElement>('filtersChart'),
  mapsChart: element<HTMLCanvasElement>('mapsChart'),
  passes: element<HTMLInputElement>('passes'),
  sureInput: element<HTMLInputElement>('sure'),
  augment: element<HTMLInputElement>('augment'),
  more: element<HTMLDetailsElement>('more'),
  card: element<HTMLElement>('card'),
  saveBtn: element<HTMLButtonElement>('saveBtn'),
  loadBtn: element<HTMLButtonElement>('loadBtn'),
  loadInput: element<HTMLInputElement>('loadInput'),
  forgetBtn: element<HTMLButtonElement>('forgetBtn'),
  toast: element<HTMLDivElement>('toast'),
};

/* ------------------------------------------------------------------ *
 * The run — pictures handed over together
 * ------------------------------------------------------------------ */

/**
 * How many pictures this run holds, counted from what is actually here rather than
 * kept in a counter of its own.
 *
 * A stored total can drift from reality — a picture skipped, a second drop arriving
 * mid-run, a queue emptied — and a page that says "4 of 12" about a queue of nine is
 * worse than one that says nothing. Deriving it from the three things that exist
 * makes that impossible.
 */
function runTotal(): number {
  const onStage = state.stage === 'asking' && state.current ? 1 : 0;
  return state.runDone + onStage + state.pending.length;
}

/** Which picture of the run is on the stage, counting from one. */
function runPosition(): number {
  const onStage = state.stage === 'asking' && state.current ? 1 : 0;
  return state.runDone + onStage;
}

/** More than one picture: the only time the position is worth saying. */
function hasRun(): boolean {
  return runTotal() > 1;
}

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

/** Built from text nodes rather than markup: the names are typed by a person. */
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

function but(text: string, kind: '' | 'primary' | 'ghost' | 'danger', onClick: () => void): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `btn${kind ? ` ${kind}` : ''}`;
  button.textContent = text;
  button.addEventListener('click', onClick);
  return button;
}

/**
 * A control that reads as a link.
 *
 * A button rather than an anchor, because it does something rather than going
 * somewhere — but it is styled down to the weight of a link, so the quiet way out
 * of a panel does not compete with the thing the panel is asking for.
 */
function link(text: string, onClick: () => void): HTMLButtonElement {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = 'link';
  button.textContent = text;
  button.addEventListener('click', onClick);
  return button;
}

/* ------------------------------------------------------------------ *
 * What it knows
 * ------------------------------------------------------------------ */

function namedSamples(): Sample[] {
  return state.samples.filter((s) => s.labels.length > 0);
}

/** The names that have at least one picture behind them. */
function knownNames(): string[] {
  const named = namedSamples();
  return state.names.filter((name) => named.some((s) => s.labels.includes(name)));
}

function learnNames(): void {
  for (const sample of state.samples) {
    for (const name of sample.labels) if (!state.names.includes(name)) state.names.push(name);
  }
}

/** How many pictures each name appears in — its support, worth showing plainly. */
function labelCounts(): Map<string, number> {
  const counts = new Map<string, number>();
  for (const sample of namedSamples()) {
    for (const name of sample.labels) counts.set(name, (counts.get(name) ?? 0) + 1);
  }
  return counts;
}

/**
 * Look at a picture once and report everything it can see, plus its strongest
 * answer whether or not that answer clears the bar.
 *
 * One call, one forward pass: the network is the expensive part here and the page
 * would otherwise ask twice for the same picture on every redraw.
 */
function look(sample: Sample | null): { sees: Array<{ name: string; sure: number }>; best: { name: string; sure: number } | null } {
  const model = state.model;
  const classes = state.file?.classes ?? [];
  if (!model || !sample || classes.length === 0) return { sees: [], best: null };

  const probs = model.predict(sample.pixels);
  const found: Array<{ name: string; sure: number }> = [];
  let bestIndex = -1;
  let bestSure = -1;
  for (let j = 0; j < classes.length; j++) {
    if (probs[j] >= state.sure) found.push({ name: classes[j], sure: probs[j] });
    if (probs[j] > bestSure) {
      bestSure = probs[j];
      bestIndex = j;
    }
  }
  found.sort((a, b) => b.sure - a.sure);
  const best = bestIndex >= 0 && bestSure > 0.05 ? { name: classes[bestIndex], sure: bestSure } : null;
  return { sees: found, best };
}

/* ------------------------------------------------------------------ *
 * The stage
 * ------------------------------------------------------------------ */

function render(): void {
  const named = namedSamples();
  const names = knownNames();

  // Drawn first and unconditionally: it read as a stale line whenever one of the
  // branches below returned early without refreshing it.
  renderProgress();
  renderQueue();

  el.picture.hidden = state.current === null;
  el.picEmpty.hidden = state.current !== null;
  if (state.current) el.picture.src = state.current.thumb;

  el.answers.textContent = '';
  el.controls.textContent = '';
  el.aside.textContent = '';
  el.aside.hidden = true;
  el.tell.hidden = true;

  if (state.stage === 'start') {
    say('Show me a picture.');
    el.sub.textContent =
      names.length < 2
        ? `I ${named.length === 0 ? "don't know anything yet" : `only know ${listWords(names)} so far`}. ` +
          `Show me one and ${HOW_TO_LIST} I need at least two different things before I can tell them apart.`
        : `I know ${names.length} things — ${listWords(names)} — from ${named.length} ${plural(named.length, 'picture', 'pictures')}. ` +
          'Show me a picture and I will say what I can see in it.';
    el.answers.append(but('Choose a picture', 'primary', () => choosePicture()));
    // Only while there is nothing to work with: the offer is a way out of an empty
    // page, not something to reach for once you have pictures of your own.
    if (named.length === 0) {
      el.aside.append('No pictures handy? ');
      el.aside.append(link('practise', () => void practise()));
      el.aside.append(' on 90 drawn shapes.');
      el.aside.hidden = false;
    }
    return;
  }

  if (state.stage === 'ready') {
    const said = state.current?.labels ?? [];
    say('Thanks — I will remember that.');
    if (hasRun()) {
      // The whole run is finished, so say so: twelve pictures is a sitting, not a
      // single answer, and the person should see the end of it. The counts differ
      // when something was skipped, which is why both are given.
      el.sub.textContent =
        `That was the last of the ${runTotal()}. You now have ${named.length} ` +
        `${plural(named.length, 'picture', 'pictures')} and I know ${names.length} ` +
        `${plural(names.length, 'thing', 'things')}. Drop in another batch whenever you like.`;
    } else {
      el.sub.textContent =
        said.length > 0
          ? `I have noted ${listWords(said)}. I now know ${names.length} ${plural(names.length, 'thing', 'things')} ` +
            `from ${named.length} ${plural(named.length, 'picture', 'pictures')}. Show me another — drag one onto the ` +
            'box, or press the button.'
          : 'Show me another picture.';
    }
    el.answers.append(but('Show me a picture', 'primary', () => another()));
    return;
  }

  // stage === 'asking'
  // What was just named rides on this line, and so does where you are in the run,
  // so naming picture after picture reads as one continuous thing.
  const noted = state.lastNoted.length > 0 ? `Noted ${listWords(state.lastNoted)}. ` : '';
  const where = hasRun() ? `Picture ${runPosition()} of ${runTotal()}. ` : '';

  if (state.sees.length > 0) {
    say('I can see ', listWords(state.sees.map((s) => s.name)), '.');
    el.sub.textContent = `${noted}${where}${HOW_TO_LIST} I will remember all of it.`;
    el.answers.append(but('Yes — that is what I see', '', () => void answer(state.sees.map((s) => s.name))));
  } else if (state.best) {
    // Naming the strongest answer even when it is unsure is not a hedge: the
    // confidence line is the person's to move, and "I do not know" tells them
    // nothing about which way it leans.
    const percent = Math.round(state.best.sure * 100);
    say('I am not sure yet — my best guess is ', state.best.name, `, and I am only ${percent}% on that.`);
    el.sub.textContent = `${noted}${where}${HOW_TO_LIST}`;
  } else {
    say('I do not know what is in this picture yet.');
    el.sub.textContent =
      names.length < 2
        ? `${noted}${where}${HOW_TO_LIST} Show me a second kind of picture too — two things is the least I can tell apart.`
        : `${noted}${where}${HOW_TO_LIST}`;
  }

  el.tell.hidden = false;
  el.list.value = '';
  // The button says how many are behind this one, because that is the thing you want
  // to know before deciding whether to bother with a hard picture.
  el.controls.append(
    but(
      state.pending.length > 0
        ? `Skip this one (${state.pending.length} ${plural(state.pending.length, 'picture', 'pictures')} to go)`
        : 'Show me a different picture',
      'ghost',
      () => skip(),
    ),
  );
}

/** Where the run has got to, said plainly, and only while a picture is on the stage. */
function renderQueue(): void {
  // Hidden outside the asking stage as well as outside a run: once a batch is
  // finished the count is the previous run's, and leaving "5 of 5" on a page that
  // has moved on reads as a stuck number.
  if (state.stage !== 'asking' || !hasRun()) {
    el.queue.hidden = true;
    return;
  }
  el.queue.hidden = false;
  el.queue.textContent =
    `Picture ${runPosition()} of ${runTotal()}` +
    (state.pending.length > 0
      ? ` · ${state.pending.length} still to name`
      : ' · last one');
}

function renderProgress(): void {
  const named = namedSamples();
  const names = knownNames();
  if (named.length === 0) {
    el.progress.textContent = "I haven't seen any pictures yet.";
    return;
  }
  const counts = labelCounts();
  el.progress.textContent = '';
  el.progress.append(`I know ${names.length} ${plural(names.length, 'thing', 'things')} — `);
  const bold = document.createElement('b');
  // The count sits beside each name, because "I know 2 things from 2 pictures"
  // hides the interesting part: which one has a single picture behind it.
  bold.textContent = names.map((n) => `${n} (${counts.get(n) ?? 0})`).join(', ');
  el.progress.append(bold, ` — from ${named.length} ${plural(named.length, 'picture', 'pictures')}.`);
}

/* ------------------------------------------------------------------ *
 * The loop
 * ------------------------------------------------------------------ */

const filePicker = document.createElement('input');
filePicker.type = 'file';
filePicker.accept = 'image/*';
// Several at once, because gathering twenty pictures and handing them over one at a
// time is the tedious part of teaching something like this.
filePicker.multiple = true;
// In the document rather than detached: an input that is not attached behaves
// inconsistently about opening, and being hidden is what keeps it off the page.
filePicker.hidden = true;
document.body.appendChild(filePicker);
filePicker.addEventListener('change', () => {
  const files = Array.from(filePicker.files ?? []);
  filePicker.value = '';
  if (files.length > 0) enqueue(files);
});

function choosePicture(): void {
  filePicker.click();
}

/**
 * Hand over pictures: the first one onto the stage, the rest queued behind it.
 *
 * Files are not decoded here. Each one is decoded when it reaches the stage, so a
 * batch of fifty costs one bitmap at a time.
 */
function enqueue(files: File[]): void {
  const pictures = files.filter((file) => file.type.startsWith('image/'));
  if (pictures.length === 0) {
    toast('None of those were pictures.');
    return;
  }
  if (pictures.length < files.length) {
    const missed = files.length - pictures.length;
    toast(
      missed === 1
        ? 'One file was not a picture, so it was left out.'
        : `${missed} files were not pictures, so they were left out.`,
    );
  }

  if (state.stage === 'ready') {
    // Whatever is on the stage has already been named, so it is not part of what
    // comes next and must not be counted as one of them.
    state.current = null;
    state.runDone = 0;
    state.lastNoted = [];
  }

  state.pending.push(...pictures);
  if (state.current === null) next();
  else render();
}

/**
 * The next picture in the queue, or the end of the run.
 *
 * Every way out of a picture arrives here — named it, or skipped it — so the count
 * on the page and the picture on the stage cannot disagree.
 */
function next(): void {
  const file = state.pending.shift();
  if (file) {
    void showPicture(file);
    return;
  }
  // The queue is empty. That is either the end of a run of several, or the end of a
  // single picture handed over on its own — and the two want different pages.
  const total = runTotal();
  state.pending = [];
  state.runDone = 0;
  state.stage = total > 1 ? 'ready' : 'start';
  render();
  if (total <= 1) choosePicture();
}

/**
 * Set the picture aside without naming it. It teaches nothing — which is the honest
 * thing to do with a picture you cannot describe, rather than guessing at it.
 */
function skip(): void {
  if (state.current === null) return;
  if (hasRun()) state.runDone += 1;
  state.current = null;
  state.sees = [];
  state.best = null;
  state.lastNoted = [];

  if (state.pending.length > 0) {
    next();
    return;
  }
  state.runDone = 0;
  state.stage = 'start';
  render();
  choosePicture();
}

/**
 * Start again from the picker.
 *
 * Used by the "Show me a picture" button once a run is finished, so it clears the
 * run with it — otherwise the next single picture would be counted as picture two
 * of a batch that had already ended.
 */
function another(): void {
  state.current = null;
  state.sees = [];
  state.best = null;
  state.pending = [];
  state.runDone = 0;
  state.lastNoted = [];
  state.stage = 'start';
  render();
  choosePicture();
}

async function showPicture(file: File): Promise<void> {
  try {
    const decoded = await decodeFile(file);
    state.current = {
      id: newId(),
      labels: [],
      thumb: decoded.thumb,
      pixels: decoded.pixels,
      name: file.name,
      addedAt: new Date().toISOString(),
      origin: 'file',
    };
    const now = look(state.current);
    state.sees = now.sees;
    state.best = now.best;
    state.stage = 'asking';
    render();
    el.list.focus();
  } catch {
    toast('That file could not be read as a picture.');
  }
}

async function answer(labels: string[]): Promise<void> {
  const sample = state.current;
  if (!sample) return;
  if (labels.length === 0) {
    toast('Name at least one thing you can see in it — just the thing itself.');
    el.list.focus();
    return;
  }

  for (const name of labels) if (!state.names.includes(name)) state.names.push(name);
  sample.labels = labels;
  if (!state.samples.includes(sample)) state.samples.push(sample);
  await store.putSample(sample);

  if (hasRun()) state.runDone += 1;
  state.lastNoted = labels;
  state.sees = [];
  state.best = null;

  if (state.pending.length > 0) {
    // Straight on to the next one. The acknowledgement is carried onto its line, so
    // a run of twenty reads as twenty answers rather than twenty confirmations.
    next();
  } else {
    state.stage = 'ready';
    render();
  }
  study();
}

async function practise(): Promise<void> {
  // The drawn pictures are honest multi-label examples: every one is a shape in a
  // colour, so a blue circle is both "circle" and "blue", and every picture gets
  // reused by both answers.
  const drawn = makeSampleSet(30, 7);
  const added: Sample[] = drawn.map((item) => ({
    id: newId(),
    labels: [item.label, item.colour],
    thumb: item.thumb,
    pixels: item.pixels,
    name: item.name,
    addedAt: new Date().toISOString(),
    origin: 'sample',
  }));
  state.samples.push(...added);
  learnNames();
  await store.putSamples(added);
  toast('Drew 90 pictures — each is a shape on a coloured background, so every one of them teaches two things at once.');
  render();
  study();
}

/**
 * Study, in the background.
 *
 * The measurement is blunt: 12 passes took 6 seconds and got 6 of 9 on pictures it
 * had never seen, while 60 passes took 26 seconds and got 8-9 of 9. Capping the
 * wait is what left it guessing at chance, and asking somebody to sit through half
 * a minute after every picture is not a page anyone would use. So it studies on its
 * own thread, the page never waits, and because each run continues from the last
 * set of weights the improvement accumulates.
 */
function study(): void {
  const named = namedSamples();
  const names = knownNames();

  // One thing is not something you can tell apart from anything.
  if (named.length < 2 || names.length < 2) return;

  // Already studying: remember there is newer work and take it up when this run
  // ends. Two runs at once would fight over the same weights.
  if (state.studying) {
    state.needsStudy = true;
    return;
  }

  const samples: TrainSample[] = named.map((s) => ({
    id: s.id,
    targets: s.labels.map((name) => names.indexOf(name)).filter((index) => index >= 0),
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

function renderStudying(): void {
  if (!state.studying) {
    el.studying.hidden = true;
    return;
  }
  el.studying.hidden = false;
  el.studying.textContent = state.model
    ? `Still studying — pass ${state.currentPass} of ${state.plannedPasses}. It keeps getting better while you carry on.`
    : `Studying the pictures. I will start saying what I can see in a moment — pass ${state.currentPass} of ${state.plannedPasses}.`;
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
      // forward pass, and doing that on every pass slows the studying down for a
      // picture nobody is looking at yet.
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
      } else if (state.stage === 'asking' && state.current && state.current.labels.length === 0) {
        // It got better while the picture sat on the stage — so say what it sees now.
        const now = look(state.current);
        state.sees = now.sees;
        state.best = now.best;
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
  state.sees = [];
  state.best = null;
  state.stage = 'start';
  render();
  renderCard();
  renderCharts();
  // Closing the panel puts the page back to how it looked on the first visit, and
  // the button that was just pressed is inside it — so leaving it open would answer
  // "forget everything" with a still-open drawer of settings about nothing.
  el.more.open = false;
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
  link.download = `vision-ml-demo-memory-${stamp}.json`;
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
  toast('Saved. Load it back with "Load a saved memory".');
}

async function loadFromFile(file: File): Promise<void> {
  try {
    const parsed = JSON.parse(await file.text()) as ModelFile;
    // Both ids are accepted: the project was renamed after this file format was
    // already in use, and a memory somebody saved to disk is theirs, not ours to
    // invalidate over a name.
    if (parsed.format !== 'vision-ml-demo-model' && parsed.format !== 'vision-demo-model') {
      throw new Error('That is not one of these files.');
    }
    if (parsed.version !== MODEL_VERSION) throw new Error('That memory was saved by an older version and cannot be read.');
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
  const counts = state.file.meta.perClassCount ?? [];
  const thinnest = counts.length > 0 ? Math.min(...counts) : 0;
  el.card.textContent =
    `It knows ${state.file.classes.length} things (${state.file.classes.join(', ')}) from ${named.length} pictures, ` +
    `and has studied ${state.file.meta.epochsTrained} times in total. ` +
    `The thinnest thing has ${thinnest} ${plural(thinnest, 'picture', 'pictures')} behind it. Saved in this browser.`;
}

/* ------------------------------------------------------------------ *
 * Charts
 * ------------------------------------------------------------------ */

function renderCharts(): void {
  renderCurves();
  renderKnows();
  renderModelPictures();
}

function renderCurves(): void {
  const accent = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#a78bfa';
  const yellow = getComputedStyle(document.body).getPropertyValue('--yellow').trim() || '#fbbf24';

  drawLine(el.confusedChart, {
    series: [
      { values: state.history.map((m) => m.loss), color: accent, label: 'while studying' },
      { values: state.history.map((m) => m.valLoss), color: yellow, label: 'on pictures held back' },
    ],
    emptyTitle: 'This line is drawn as it studies.',
    emptyHint: 'Show it a picture and list what you can see in it.',
  });

  drawLine(el.rightChart, {
    series: [
      { values: state.history.map((m) => m.trainAccuracy), color: accent, label: 'pictures it studied' },
      { values: state.history.map((m) => m.valAccuracy), color: yellow, label: 'pictures held back' },
    ],
    floor: 0,
    ceil: 1,
    format: (v) => `${Math.round(v * 100)}%`,
    emptyTitle: 'This fills in as it studies.',
    emptyHint: 'It needs at least two kinds of thing first.',
  });
}

/** How well it knows each thing — and how many pictures are behind that number. */
function renderKnows(): void {
  const classes = state.file?.classes ?? [];
  const accuracy = state.file?.meta.perClassAccuracy ?? [];
  const counts = state.file?.meta.perClassCount ?? [];
  drawBars(
    el.knowsChart,
    classes.map((name, index) => ({
      label: name,
      value: accuracy[index] ?? 0,
      caption: `${Math.round((accuracy[index] ?? 0) * 100)}% · ${counts[index] ?? 0} ${plural(counts[index] ?? 0, 'picture', 'pictures')}`,
    })),
    {
      max: 1,
      emptyTitle: 'Nothing to score yet.',
      emptyHint: 'Once it has studied, this shows how well it knows each thing.',
    },
  );
}

function renderModelPictures(): void {
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
  el.tellBtn.addEventListener('click', () => void answer(parseLabels(el.list.value)));
  el.list.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      void answer(parseLabels(el.list.value));
    }
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
    state.sure = Math.max(0.3, Math.min(0.95, (Number(el.sureInput.value) || 50) / 100));
    el.sureInput.value = String(Math.round(state.sure * 100));
    void store.saveSetting('sure', state.sure);
    if (state.current) {
      const now = look(state.current);
      state.sees = now.sees;
      state.best = now.best;
    }
    render();
  });
  el.augment.addEventListener('change', () => {
    state.augment = el.augment.checked;
    void store.saveSetting('augment', state.augment);
  });

  // Drag a picture anywhere onto the page. The box it will land in lights up, so
  // the gesture is visible rather than something you have to already know about.
  // The whole window is the target on purpose — aiming at one small rectangle is a
  // fussy thing to ask of somebody.
  el.pic.addEventListener('click', () => choosePicture());
  el.pic.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      choosePicture();
    }
  });

  const carriesFiles = (event: DragEvent): boolean => Array.from(event.dataTransfer?.types ?? []).includes('Files');
  // dragenter fires again for every element entered, so a depth count is what keeps
  // the highlight on until the pointer actually leaves the page.
  let dragDepth = 0;

  window.addEventListener('dragenter', (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    dragDepth += 1;
    el.pic.classList.add('dragging');
  });
  window.addEventListener('dragover', (event: DragEvent) => {
    if (!carriesFiles(event)) return;
    // Without this the browser opens the dropped file instead of letting the page
    // have it, and the drop event never arrives.
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = 'copy';
  });
  window.addEventListener('dragleave', () => {
    dragDepth = Math.max(0, dragDepth - 1);
    if (dragDepth === 0) el.pic.classList.remove('dragging');
  });
  window.addEventListener('drop', (event: DragEvent) => {
    dragDepth = 0;
    el.pic.classList.remove('dragging');
    event.preventDefault();

    const files = Array.from(event.dataTransfer?.files ?? []);
    if (files.length === 0) {
      toast('That was not a picture.');
      return;
    }
    // However many arrive, they all go in. Dropping twelve photographs and being
    // shown one is the behaviour this replaces.
    el.stage.scrollIntoView({ block: 'nearest' });
    enqueue(files);
  });

  window.addEventListener('resize', () => renderCharts());

  // A rename must not strand what is already saved. See store.migrateLegacy.
  const carried = await store.migrateLegacy().catch(() => ({ carried: 0, failed: true }));

  const [samples, model, passes, sure, augment] = await Promise.all([
    store.allSamples().catch(() => [] as Sample[]),
    store.loadModel().catch(() => null as ModelFile | null),
    store.loadSetting<number>('passes', 60),
    store.loadSetting<number>('sure', 0.5),
    store.loadSetting<boolean>('augment', true),
  ]);

  state.samples = samples.map((s) => ({ ...s, labels: s.labels ?? [] }));
  learnNames();
  state.passes = passes;
  state.sure = sure;
  state.augment = augment !== false;
  el.passes.value = String(passes);
  el.sureInput.value = String(Math.round(sure * 100));
  el.augment.checked = state.augment;

  // A memory saved by the older single-answer version would load its weights happily
  // and then answer nonsense, because those numbers mean something else now. Better
  // to leave it behind and say so than to show confident rubbish.
  if (model && model.version !== MODEL_VERSION) {
    await store.clearModel();
    toast('A memory from the older version was set aside. The pictures are still here.');
  } else if (model) {
    adopt(model);
    render();
    renderStudying();
    toast(`Picked up where you left off — it knows ${model.classes.length} things from ${samples.length} pictures.`);
    return;
  }

  render();
  renderCharts();
  if (carried.failed) {
    toast('Something saved under the old name could not be brought across.');
  } else if (carried.carried > 0) {
    toast(`Carried over ${carried.carried} ${plural(carried.carried, 'picture', 'pictures')} saved under the old name.`);
  }
}

void boot();
