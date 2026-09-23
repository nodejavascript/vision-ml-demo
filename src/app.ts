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
import { loadPicture } from './image.js';
import type { Loaded } from './image.js';
import { detectFaces } from './faces.js';
import type { Box } from './faces.js';
import { makeSampleFiles, PRACTISE_COUNT, SHAPE_NAMES } from './samples.js';
import { VisionTrainer } from './trainer-host.js';
import * as store from './store.js';
import { drawBars, drawLine, drawProgress, drawTiles } from './charts.js';
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
const HOW_TO_LIST = 'name the thing in the box — one name.';

/**
 * The one name in the box.
 *
 * Each rectangle holds a single person, place or thing, so the answer is a single
 * name and the field no longer takes a list. That is what makes the separators go: a
 * name with "and" in it — fish and chips, salt and pepper — is now perfectly safe,
 * because nothing splits on it any more. Only a real separator can, and that can only
 * mean somebody is still typing a list, which is worth saying out loud rather than
 * quietly turning their second name into the first.
 *
 * The framing words come off the front exactly as before, and only from the front:
 * "this is my gran" is gran. Nothing is ever taken from the middle or the end, because
 * "cup of tea" and "rail house" are single things.
 */
function parseName(text: string): { name: string; extra: number } {
  const pieces = text
    .toLowerCase()
    .split(/[,;\n]+/)
    .map((piece) =>
      piece
        .replace(/[.!?]+/g, ' ')
        .split(/\s+/)
        .filter((word) => word !== ''),
    )
    .filter((words) => words.length > 0);

  const cleaned = pieces.map((words) => {
    let start = 0;
    if (OPENS_A_SENTENCE.has(words[0])) {
      while (start < words.length && FRAMING.has(words[start])) start += 1;
    }
    return words.slice(start).join(' ');
  });
  const usable = cleaned.filter((name) => name !== '');

  return { name: usable[0] ?? '', extra: Math.max(0, usable.length - 1) };
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
  /**
   * What it can see in the current picture, most sure first.
   *
   * Only ever what cleared the confidence line. Its strongest answer *under* the line
   * used to be kept here too and shown as "I am not sure yet — my best guess is X, and I
   * am only 34% on that". George, 2026-09-19: "is the confidense is lot, say im not sure
   * yet. thats it. dont make dumb guesses." So there is no such value any more, and
   * nothing in this file can produce one.
   */
  sees: [] as Array<{ name: string; sure: number }>,
  /**
   * Pictures handed over together and not opened yet, in order.
   *
   * Held as files rather than opened pictures on purpose: fifty photographs opened
   * up front would sit in memory as fifty full-size canvases for no reason.
   */
  files: [] as File[],
  /** Pictures that have been opened and searched for faces, waiting their turn. */
  turns: [] as Turn[],
  /** The picture on the stage, and which of its faces is being asked about. */
  turn: null as Turn | null,
  turnIndex: 0,
  /** Faces of this picture already named, so their boxes can show as done. */
  namedBoxes: [] as Box[],
  /**
   * Faces the person declined to name.
   *
   * Tracked separately from the named ones because they are a different answer — "I
   * do not know who this is" rather than "this is Sarah" — and because a skipped box
   * that looks exactly like an untouched one makes the skip look as though it did not
   * register at all.
   */
  skippedBoxes: [] as Box[],
  /** How many pictures of this run have been finished — every face named, or skipped. */
  photosDone: 0,
  /**
   * How many pictures the run that just ended held, or 0 if it was a single picture.
   *
   * Kept because the closing line is drawn after the counters have been cleared, and
   * recomputing it there is how "that was the last of the 6" got said about five.
   */
  finishedTotal: 0,
  /** True while the next picture is being opened, so the page can say so. */
  opening: false,
  /** A box being dragged out by hand, before it is let go of. */
  draft: null as Box | null,
  /**
   * The name just given, and the run it was given in, so a spelling mistake can be
   * taken back. Only ever the most recent one — see `LastAnswer` for why.
   */
  lastAnswer: null as LastAnswer | null,
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
  boxes: element<HTMLDivElement>('boxes'),
  picEmpty: element<HTMLDivElement>('picEmpty'),
  says: element<HTMLElement>('says'),
  guesses: element<HTMLDivElement>('guesses'),
  sub: element<HTMLElement>('sub'),
  queue: element<HTMLElement>('queue'),
  tell: element<HTMLDivElement>('tell'),
  tellLabel: element<HTMLLabelElement>('tellLabel'),
  who: element<HTMLDivElement>('who'),
  face: element<HTMLImageElement>('face'),
  skipBox: element<HTMLButtonElement>('skipBox'),
  skipImage: element<HTMLButtonElement>('skipImage'),
  progressChart: element<HTMLCanvasElement>('progressChart'),
  list: element<HTMLInputElement>('list'),
  tellBtn: element<HTMLButtonElement>('tellBtn'),
  answers: element<HTMLDivElement>('answers'),
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
  card: element<HTMLElement>('card'),
  saveBtn: element<HTMLButtonElement>('saveBtn'),
  loadBtn: element<HTMLButtonElement>('loadBtn'),
  loadInput: element<HTMLInputElement>('loadInput'),
  forgetBtn: element<HTMLButtonElement>('forgetBtn'),
  note: element<HTMLElement>('note'),
};

/* ------------------------------------------------------------------ *
 * The run — pictures handed over together, face by face
 * ------------------------------------------------------------------ */

/**
 * The file names the practise set handed over.
 *
 * Kept so a picture can say where it came from — a practise run and a visitor's own
 * photographs lead to different answers about the same page — without the file name being
 * read back at the moment it is named. The set's names are its own (`practise-01.png`), and
 * nothing in a name says which shape is in the picture.
 */
const practiseFiles = new Set<string>();

/** A picture waiting its turn, with the boxes found in it. */
interface Turn {
  picture: Loaded;
  /** Empty means nothing was found, so the whole picture gets named instead. */
  boxes: Box[];
}

/**
 * The run exactly as it stood the moment before an answer was taken.
 *
 * A copy of the counters and a *reference* to the picture that was on the stage — the
 * same `Turn`, not the file reopened — which is what makes going back free instead of a
 * second face search. The counters and the queues are copied because the loop shifts
 * them in place, and a snapshot the thing it describes can change underneath is not a
 * snapshot.
 */
interface RunSnapshot {
  turn: Turn | null;
  turnIndex: number;
  namedBoxes: Box[];
  skippedBoxes: Box[];
  photosDone: number;
  finishedTotal: number;
  turns: Turn[];
  files: File[];
  sees: Array<{ name: string; sure: number }>;
  current: Sample | null;
  stage: Stage;
}

/**
 * The name just given, kept so that it can be taken back.
 *
 * One deep, not a history. Undoing anything older than the last name would have to
 * unwind whatever came after it, and a page that silently reverses two answers when one
 * is asked for is worse than a page with no undo at all. It is dropped the moment
 * anything else touches the run — a skip, a box added or removed, another drop — so the
 * way back can never lead to a picture that is no longer what it was.
 */
interface LastAnswer {
  sample: Sample;
  name: string;
  nameWasNew: boolean;
  before: RunSnapshot;
}

/**
 * How many pictures this run holds, counted from what is actually here rather than
 * kept in a counter of its own.
 *
 * A stored total can drift from reality — a face skipped, a second drop arriving
 * mid-run, a queue emptied — and a page that says "4 of 12" about a queue of nine is
 * worse than one that says nothing. Deriving it from the things that exist makes
 * that impossible.
 */
function runTotal(): number {
  return state.photosDone + (state.turn ? 1 : 0) + state.turns.length + state.files.length;
}

/** Which picture of the run is on the stage, counting from one. */
function runPosition(): number {
  return state.photosDone + (state.turn ? 1 : 0);
}

/** More than one picture: the only time the position across the run is worth saying. */
function hasRun(): boolean {
  return runTotal() > 1;
}

/** The face being asked about, or null when the whole picture is the subject. */
function currentBox(): Box | null {
  const turn = state.turn;
  if (!turn || turn.boxes.length === 0) return null;
  return turn.boxes[state.turnIndex] ?? null;
}

/** How many pictures are still waiting behind this one. */
function waitingPhotos(): number {
  return state.turns.length + state.files.length;
}

/** The run, copied, so that it can be put back. See `RunSnapshot`. */
function snapshotRun(): RunSnapshot {
  return {
    turn: state.turn,
    turnIndex: state.turnIndex,
    namedBoxes: state.namedBoxes.slice(),
    skippedBoxes: state.skippedBoxes.slice(),
    photosDone: state.photosDone,
    finishedTotal: state.finishedTotal,
    turns: state.turns.slice(),
    files: state.files.slice(),
    sees: state.sees.slice(),
    current: state.current,
    stage: state.stage,
  };
}

/** Put the run back exactly as the snapshot found it. */
function restoreRun(before: RunSnapshot): void {
  state.turn = before.turn;
  state.turnIndex = before.turnIndex;
  state.namedBoxes = before.namedBoxes.slice();
  state.skippedBoxes = before.skippedBoxes.slice();
  state.photosDone = before.photosDone;
  state.finishedTotal = before.finishedTotal;
  state.turns = before.turns.slice();
  state.files = before.files.slice();
  state.sees = before.sees.slice();
  state.current = before.current;
  state.stage = before.stage;
}

/**
 * Drop the way back.
 *
 * Called by everything that changes the run after an answer. The snapshot describes one
 * moment, and once the queue, the boxes or the pictures have moved, restoring it would
 * take back work the person did not ask to take back.
 */
function forgetUndo(): void {
  state.lastAnswer = null;
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

/**
 * Say something that is not a question.
 *
 * A line on the page, under the box it is usually about — not a toast floating over the
 * page and vanishing on a timer. George, 2026-09-19: "i dont want toasts." It stays until
 * the next thing happens, because a message that fades while it is being read has not been
 * delivered, and `note('')` takes it away.
 */
function note(message: string): void {
  el.note.textContent = message;
  el.note.hidden = message === '';
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
 * Look at a picture once and report everything it can see.
 *
 * One call, one forward pass: the network is the expensive part here and the page
 * would otherwise ask twice for the same picture on every redraw.
 */
function look(sample: Sample | null): { sees: Array<{ name: string; sure: number }> } {
  const model = state.model;
  const classes = state.file?.classes ?? [];
  if (!model || !sample || classes.length === 0) return { sees: [] };

  const probs = model.predict(sample.pixels);
  const found: Array<{ name: string; sure: number }> = [];
  for (let j = 0; j < classes.length; j++) {
    if (probs[j] >= state.sure) found.push({ name: classes[j], sure: probs[j] });
  }
  found.sort((a, b) => b.sure - a.sure);
  return { sees: found };
}

/* ------------------------------------------------------------------ *
 * The stage
 * ------------------------------------------------------------------ */

/**
 * The question the name box was last emptied for.
 *
 * The box is emptied when the question changes, and not on every redraw: a study run
 * finishing redraws this column on its own, and emptying the box then deletes half a name
 * somebody is in the middle of typing.
 */
let clearedFor: Sample | null = null;

/**
 * Whether the run now starting began by wiping the store.
 *
 * Set by `practise()` and read once, by the first picture's opening line. The wipe itself
 * is instant and the line `practise()` writes lasts about a tenth of a second before the
 * face search finishes and the first picture says what it found — so the news has to
 * travel on that line, or the person never sees it. A state they cannot see is a state
 * they cannot trust.
 */
let freshStart = false;

/**
 * Whether the first model of this sitting has already been counted.
 *
 * `model_trained` is the demo's own conversion — "it learned something" is what this page
 * is for — and it is sent once, when a model first exists, rather than on every study run.
 * Sixty passes per picture is sixty events, and a number that moves for no reason is a
 * number nobody reads.
 */
let trainedOnce = false;

function render(): void {
  const named = namedSamples();
  const names = knownNames();

  // Drawn first and unconditionally: it read as a stale line whenever one of the
  // branches below returned early without refreshing it.
  renderProgress();
  renderQueue();

  // The stage shows the whole picture — a face has to be seen in place to know who it
  // is — while the thing actually being named is the crop of one box.
  const showing = state.turn !== null;
  el.picture.hidden = !showing;
  el.picEmpty.hidden = showing;
  if (state.turn && el.picture.src !== state.turn.picture.thumb) el.picture.src = state.turn.picture.thumb;

  el.answers.textContent = '';
  el.guesses.textContent = '';
  el.guesses.hidden = true;
  el.says.hidden = false;
  el.says.classList.remove('as-label');
  el.sub.hidden = false;
  el.aside.textContent = '';
  el.aside.hidden = true;
  el.tell.hidden = true;
  el.who.hidden = true;

  renderBoxes();
  renderImageProgress();

  if (state.stage === 'start') {
    say('Show me a picture.');
    el.sub.textContent =
      names.length < 2
        ? `I ${named.length === 0 ? "don't know anything yet" : `only know ${listWords(names)} so far`}. ` +
          `Show me one and ${HOW_TO_LIST} ` +
          'I need at least two different things before I can tell them apart.'
        : `I know ${names.length} things — ${listWords(names)} — from ${named.length} ${plural(named.length, 'picture', 'pictures')}. ` +
          'Show me a picture and I will draw a box round what it finds.';
    el.answers.append(but('Choose a picture', 'primary', () => choosePicture()));
    showPractiseOffer(named.length === 0);
    return;
  }

  if (state.stage === 'ready') {
    say('Thanks — I will remember that.');
    if (state.finishedTotal > 0) {
      // The whole run is finished, so say so: a batch of photographs is a sitting,
      // not a single answer, and the person should see the end of it. The total was
      // counted while the run still existed — recomputing it here, from counters
      // that have already been cleared, is how "last of the 6" got said about five.
      el.sub.textContent =
        `That was the last of the ${state.finishedTotal}. You now have ${named.length} ` +
        `${plural(named.length, 'picture', 'pictures')} and I know ${names.length} ` +
        `${plural(names.length, 'thing', 'things')}. Drop in another batch whenever you like.`;
    } else {
      el.sub.textContent =
        `I now know ${names.length} ${plural(names.length, 'thing', 'things')} from ${named.length} ` +
        `${plural(named.length, 'picture', 'pictures')}. Show me another — drag one onto the box, or press the button.`;
    }
    el.answers.append(but('Show me a picture', 'primary', () => another()));
    // The last box of the last picture is the likeliest place to want the way back: the run
    // is over, the page says so, and only then does the spelling in the name show up.
    if (state.lastAnswer) el.answers.append(undoControl());
    showPractiseOffer(false);
    return;
  }

  // stage === 'asking'
  const askingAboutFace = currentBox() !== null;
  // What to do, and nothing else. This line used to open with "Noted X." and close with
  // "I will remember every one of them." — the first repeats what the box on the picture
  // and the chart under it already show, and the second is not an instruction. George,
  // 2026-09-19: "dont put Noted ngle. I will remember every one of them. langaurge."
  //
  // "Name a second one" only means anything once there has been a first — and how many
  // pictures are actually named is the honest way to know, where the counter that used to
  // decide it could be left behind by a skip or a second drop.
  const nudge =
    names.length < 2
      ? named.length > 0
        ? 'Name a second one and I can start telling them apart.'
        : 'Name this one and I can start learning.'
      : '';
  el.sub.textContent = nudge;
  el.sub.hidden = nudge === '';

  if (state.sees.length > 0) {
    // A label over its buttons rather than the model speaking in its own voice: it is the
    // same size as the question it answers, because that is what it is — the head of the
    // answer, not a sentence. See `.says.as-label` in the stylesheet.
    el.says.classList.add('as-label');
    // One button per thing it can see, each labelled with the thing itself, so agreeing
    // with it is a single press and the box to disagree with is right above. Buttons
    // rather than a sentence naming them, because the network answers a separate yes/no
    // per name and more than one can clear the line at once — "I think this is sarah and
    // dog" is not a sentence, and a row of six would be nonsense.
    say(state.sees.length === 1 ? 'My guess:' : 'My guesses:');
    el.guesses.hidden = false;
    for (const seen of state.sees) {
      el.guesses.append(but(seen.name, '', () => void answer(seen.name)));
    }
  } else {
    // Nothing cleared the confidence line, so there is no guess — and no guess is exactly
    // what should be on screen. George, 2026-09-19: "if it doesnt know, i dont want to see
    // my guess." What used to be here was the reason it had nothing to say, and the line
    // under it then said the same thing again as an instruction — "Name this one and I can
    // start learning." is the whole of it, so the page asks once and explains once.
    //
    // Emptied as well as hidden: a line nobody can see still reads back out of the DOM, and
    // a stale "My guess:" behind a hidden element is a lie waiting for the next reader.
    el.says.textContent = '';
    el.says.hidden = true;
  }

  // The question changes with the subject: a rectangle is a person to name, and a
  // picture with no face found in it is the picture itself.
  el.tell.hidden = false;
  // Emptied when the question changes, not on every redraw. A study run finishing redraws
  // this column on its own, and emptying the box then deletes half a name somebody is in
  // the middle of typing.
  if (state.current !== clearedFor) {
    clearedFor = state.current;
    el.list.value = '';
  }

  // What can be done to the boxes used to be spelled out here — three sentences of
  // instructions that sat on the page the entire time a picture was up, for a gesture
  // most people never need. The × is visible on the box itself.
  el.who.hidden = false;
  el.face.hidden = !askingAboutFace;
  if (askingAboutFace && state.current) el.face.src = state.current.thumb;

  if (askingAboutFace) {
    el.tellLabel.textContent = 'Who or what is in this box? One name:';
    el.list.placeholder = 'sarah, or the dog, or the beach';
  } else {
    el.tellLabel.textContent = 'What is in this picture? One name:';
    el.list.placeholder = 'the kitchen, or the garden';
  }

  // Two ways to move on, each sitting under the thing it acts on rather than in a
  // button row beside them: skip the box, or skip the whole picture.
  const leftInPicture = state.turn ? state.turn.boxes.length - state.turnIndex - 1 : 0;
  el.skipBox.hidden = !askingAboutFace;
  el.skipImage.hidden = false;
  el.skipImage.title =
    leftInPicture > 0
      ? `Leave this picture — ${leftInPicture} ${plural(leftInPicture, 'box', 'boxes')} will not be named`
      : 'Leave this picture';

  if (state.lastAnswer) el.answers.append(undoControl());
}

/**
 * Where the run has got to, said plainly, and only while a picture is on the stage.
 *
 * Two numbers rather than one, because there are two things to know: which face of
 * this photograph, and which photograph of the batch. A single "3 of 12" could not
 * say both.
 */
function renderQueue(): void {
  // Hidden outside the asking stage as well as outside a run: once a batch is
  // finished the count is the previous run's, and leaving "5 of 5" on a page that
  // has moved on reads as a stuck number.
  if (state.stage !== 'asking' || state.turn === null) {
    el.queue.hidden = true;
    return;
  }
  const parts: string[] = [];
  const boxes = state.turn.boxes.length;
  // "box" throughout, because a box is what the page draws and what the question asks
  // about. It used to say "face" here and "box" everywhere else, which is two words for
  // one thing — and with the practise set, which holds shapes and no faces at all, the old
  // word was simply wrong.
  if (boxes > 0) parts.push(`box ${state.turnIndex + 1} of ${boxes}`);
  // "picture", not "photo": the page says picture everywhere else.
  if (hasRun()) parts.push(`picture ${runPosition()} of ${runTotal()}`);
  if (state.opening) parts.push('opening the next one…');
  if (parts.length === 0) {
    el.queue.hidden = true;
    return;
  }
  el.queue.hidden = false;
  el.queue.textContent = parts.join(' · ');
}

/* ------------------------------------------------------------------ *
 * The boxes on the picture
 * ------------------------------------------------------------------ */

/**
 * Where the picture actually is inside its square box.
 *
 * The image is fitted whole rather than cropped — `object-fit: contain` — so on a
 * tall or wide photograph there is empty space above or beside it. A box measured in
 * the picture's own pixels has to be mapped through that letterboxing or every
 * rectangle would sit in the wrong place, which is worse than drawing none.
 *
 * Measured against the picture's OWN size, never the image element's `naturalWidth`:
 * what is on screen is a thumbnail at a different size again, so scaling source
 * pixels by the thumbnail's ratio would put every box out by that factor.
 */
function pictureInElement(): { scale: number; left: number; top: number } | null {
  const source = state.turn?.picture;
  if (!source) return null;
  const shown = { width: el.picture.clientWidth, height: el.picture.clientHeight };
  if (shown.width === 0 || shown.height === 0) return null;
  const scale = Math.min(shown.width / source.width, shown.height / source.height);
  return {
    scale,
    left: (shown.width - source.width * scale) / 2,
    top: (shown.height - source.height * scale) / 2,
  };
}

/** The box, in the picture's own pixels, under a point in element coordinates. */
function toPictureSpace(clientX: number, clientY: number): { x: number; y: number } | null {
  const fit = pictureInElement();
  if (!fit) return null;
  const area = el.picture.getBoundingClientRect();
  return {
    x: Math.round((clientX - area.left - fit.left) / fit.scale),
    y: Math.round((clientY - area.top - fit.top) / fit.scale),
  };
}

function renderBoxes(): void {
  el.boxes.textContent = '';
  const turn = state.turn;
  const fit = turn ? pictureInElement() : null;
  // The overlay stays on the picture whenever there IS a picture, even with no boxes
  // in it — otherwise there is nothing to drag on, and a face the detector missed
  // could never be added. Only the boxes inside it come and go.
  const active = turn !== null && fit !== null && state.stage === 'asking';
  el.boxes.hidden = !active;
  if (!active || !turn || !fit) return;

  turn.boxes.forEach((box, index) => {
    const element = document.createElement('div');
    element.className = 'box';
    if (index === state.turnIndex) element.classList.add('current');
    if (state.namedBoxes.includes(box)) element.classList.add('done');
    if (state.skippedBoxes.includes(box)) element.classList.add('skipped');
    element.style.left = `${fit.left + box.x * fit.scale}px`;
    element.style.top = `${fit.top + box.y * fit.scale}px`;
    element.style.width = `${box.w * fit.scale}px`;
    element.style.height = `${box.h * fit.scale}px`;
    element.title = `Box ${index + 1} of ${turn.boxes.length} — click to name this one`;

    const number = document.createElement('span');
    number.className = 'box-num';
    number.textContent = String(index + 1);
    element.append(number);

    // Only the box being asked about gets the ×, so a photo of twenty people is not
    // a thicket of delete buttons.
    if (index === state.turnIndex) {
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'box-x';
      remove.textContent = '×';
      remove.title = 'This box is wrong — remove it';
      remove.addEventListener('click', (event) => {
        event.stopPropagation();
        void removeBox(index);
      });
      element.append(remove);
    }

    element.addEventListener('click', (event) => {
      event.stopPropagation();
      if (index !== state.turnIndex) void goToFace(index);
    });

    el.boxes.append(element);
  });

  // The box being dragged out by hand, before it is let go of.
  if (state.draft) {
    const draft = document.createElement('div');
    draft.className = 'draft';
    draft.style.left = `${fit.left + state.draft.x * fit.scale}px`;
    draft.style.top = `${fit.top + state.draft.y * fit.scale}px`;
    draft.style.width = `${state.draft.w * fit.scale}px`;
    draft.style.height = `${state.draft.h * fit.scale}px`;
    el.boxes.append(draft);
  }
}

/** Drop a box the person says is wrong, and carry on. */
async function removeBox(index: number): Promise<void> {
  const turn = state.turn;
  if (!turn) return;
  forgetUndo();
  turn.boxes.splice(index, 1);
  if (state.turnIndex >= turn.boxes.length) state.turnIndex = Math.max(0, turn.boxes.length - 1);
  // A picture whose boxes have all been removed becomes the whole picture, which is
  // the honest fallback: something gets named either way.
  await showFace();
}

/** Move to a face the person picked out of the picture themselves. */
async function goToFace(index: number): Promise<void> {
  const turn = state.turn;
  if (!turn || index < 0 || index >= turn.boxes.length) return;
  forgetUndo();
  state.turnIndex = index;
  await showFace();
}

/**
 * How far through this picture you are, drawn under it.
 *
 * One segment per rectangle, so it is a picture of the picture rather than a number
 * about it: a bar showing three filled segments and five hollow ones says "three of
 * eight" in a way a numeral cannot. Nothing is drawn when there is no picture, or when
 * the picture has no boxes — a chart of zero things is not information, it is noise.
 */
function renderImageProgress(): void {
  const turn = state.turn;
  const show = turn !== null && turn.boxes.length > 0 && state.stage === 'asking';
  el.progressChart.hidden = !show;
  if (!show || !turn) return;
  drawProgress(
    el.progressChart,
    turn.boxes.map((box, index) => ({
      state: state.namedBoxes.includes(box)
        ? ('named' as const)
        : state.skippedBoxes.includes(box)
          ? ('skipped' as const)
          : index === state.turnIndex
            ? ('current' as const)
            : ('left' as const),
    })),
  );
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
 * Hand over pictures: one goes on the stage, the rest queue behind it.
 *
 * Files are not opened here. Each one is opened when its turn comes, so a batch of
 * fifty costs one full-size picture in memory at a time rather than fifty.
 */
function enqueue(files: File[]): void {
  note('');
  const pictures = files.filter((file) => file.type.startsWith('image/'));
  if (pictures.length === 0) {
    note('None of those were pictures.');
    return;
  }
  if (pictures.length < files.length) {
    const missed = files.length - pictures.length;
    note(
      missed === 1
        ? 'One file was not a picture, so it was left out.'
        : `${missed} files were not pictures, so they were left out.`,
    );
  }

  if (state.stage === 'ready') {
    // Whatever is on the stage has already been named, so it is not part of what
    // comes next and must not be counted as one of them.
    state.turn = null;
    state.current = null;
    state.photosDone = 0;
    state.finishedTotal = 0;
  }

  // Anything that adds pictures takes the way back with it: the snapshot describes one
  // moment, and restoring it afterwards would drop the pictures just handed over.
  forgetUndo();
  state.files.push(...pictures);
  if (state.turn === null) void nextPhoto();
  else render();
}

/**
 * Open the next picture waiting, skipping any that will not open.
 *
 * Returns nothing when the run is over — which the caller tells apart from "there
 * was nothing to open" only by the run counters, and does not need to.
 */
async function openNextPhoto(): Promise<Turn | null> {
  while (state.turns.length === 0 && state.files.length > 0) {
    const file = state.files.shift();
    if (!file) break;
    state.opening = true;
    try {
      const picture = await loadPicture(file);
      // 🔴 THE FINDER RUNS ON EVERY PICTURE A VISITOR BRINGS, AND NOT ON THE DRAWINGS THE
      // PAGE MADE ITSELF — and it is a measured decision, not a convenience.
      //
      // The finder is a face detector: a Haar cascade looking for the light-dark pattern of
      // eyes, cheeks and nose. A drawn shape is a solid blob, and on the practise set it
      // INVENTS boxes — measured 2026-09-23, through this very pipeline: 9 of the 20
      // pictures got a box, ten boxes in all, each covering between 4% and 18% of the frame.
      // A 4% crop of a circle is not a circle, so leaving those in place would hand the
      // network a fragment of a shape and call it the shape's name — the practise would be
      // teaching something other than what it says it teaches.
      //
      // So the drawing is handed over with nothing found, and the question is about the
      // whole picture, which is the honest description of a page that drew one shape on it.
      // ⚠️ This is NOT the forbidden thing: nothing here tunes the detector, and every
      // photograph a visitor drops in still goes through it exactly as before. The page is
      // only declining to run a face finder over its own drawing of a hexagon.
      const drawn = practiseFiles.has(file.name);
      state.turns.push({ picture, boxes: drawn ? [] : detectFaces(picture.frame) });
    } catch {
      note(`${file.name} could not be read as a picture.`);
    } finally {
      state.opening = false;
    }
  }
  return state.turns.shift() ?? null;
}

/** Move on: the next face of this picture, or the next picture, or the end. */
async function nextPhoto(): Promise<void> {
  const turn = state.turn;
  if (turn && state.turnIndex + 1 < turn.boxes.length) {
    state.turnIndex += 1;
    await showFace();
    return;
  }

  if (turn) {
    state.photosDone += 1;
    state.turn = null;
    state.turnIndex = 0;
    state.namedBoxes = [];
    state.skippedBoxes = [];
  }

  if (waitingPhotos() > 0) render(); // say that it is opening the next one
  const nextTurn = await openNextPhoto();
  if (!nextTurn) {
    endRun();
    return;
  }
  state.turn = nextTurn;
  state.turnIndex = 0;
  state.namedBoxes = [];
  state.skippedBoxes = [];
  await showFace();

  const boxes = nextTurn.boxes.length;
  const cleared = freshStart;
  freshStart = false;
  // The wipe is instant, and the line `practise()` writes lasts about a tenth of a second
  // before the first picture says what it found — so the news travels on this line or the
  // person never sees it. For the practise set it carries what the set is as well: thirty
  // shapes, four kinds, several of each — and the one instruction that makes the set teach
  // anything, the same word for the same shape.
  //
  // ⚠️ "SEVERAL OF EACH" RATHER THAN A NUMBER, and that is deliberate: seven or eight of the
  // four is a consequence of the count, and a phrase that says "seven or eight" has to be kept
  // in step with two constants the moment either moves. The set is thirty of four; the number
  // each gets is arithmetic, and arithmetic belongs in `samples.ts` rather than in a sentence.
  const opening = cleared
    ? `Memory cleared. I drew ${PRACTISE_COUNT} shapes of ${SHAPE_NAMES.length} kinds, several of each. ` +
      'Name each picture, using the same word for the same shape. '
    : '';
  if (boxes > 0) {
    note(
      opening +
        (boxes === 1
          ? 'Found one box. Name it, or drag a box of your own if it has the wrong one.'
          : `Found ${boxes} boxes. They are named one at a time.`),
    );
  } else if (cleared) {
    // Nothing else would be said here, so the wipe speaks on its own.
    note(opening.trim());
  }
}

/** Put the face being asked about on the stage, and ask. */
async function showFace(): Promise<void> {
  const turn = state.turn;
  if (!turn) return;
  const box = currentBox();
  const picture = turn.picture;

  state.current = {
    id: newId(),
    labels: [],
    // With a box, the thing being named is the crop of it; without one, the picture
    // is the subject and the crop is the picture.
    thumb: box ? picture.cropThumb(box) : picture.thumb,
    pixels: box ? picture.crop(box) : picture.whole(),
    name: box ? `${picture.filename} — box ${state.turnIndex + 1}` : picture.filename,
    addedAt: new Date().toISOString(),
    // Where it came from: the visitor's disk, or the practise set the page drew. Recorded
    // here rather than read back from the file name later.
    origin: practiseFiles.has(picture.filename) ? 'sample' : 'file',
  };

  const now = look(state.current);
  state.sees = now.sees;
  state.stage = 'asking';
  render();
  el.list.focus();
}

/**
 * Set one face aside without naming it.
 *
 * It teaches nothing, which is the honest thing to do with somebody you cannot name
 * rather than guessing at it. Skipping is not removing: the box stays, because it is
 * still a face the detector found and pretending otherwise would be a lie about what
 * the picture holds — but it is marked, so it is visibly dealt with and the person can
 * see that the skip registered.
 */
async function skip(): Promise<void> {
  const box = currentBox();
  if (state.current === null) return;
  forgetUndo();
  note('');
  if (box && !state.skippedBoxes.includes(box)) state.skippedBoxes.push(box);
  state.current = null;
  state.sees = [];

  if (state.turn && state.turnIndex + 1 < state.turn.boxes.length) {
    await nextPhoto();
    return;
  }
  if (waitingPhotos() > 0) {
    await nextPhoto();
    return;
  }
  // Nothing behind it: the run is over, with the boxes marked rather than the page
  // jumping straight to a file dialog nobody asked for.
  endRun();
}

/**
 * Leave this picture alone and go on to the next.
 *
 * The action for both kinds of item: a picture whose faces you do not want, and a
 * picture with no face in it at all. Marking every remaining box is what makes the
 * difference visible rather than the page simply jumping away — a group photograph is
 * the case this exists for, eight faces found and none of them wanted, and declining
 * them one at a time is the kind of small tediousness that stops a page being used.
 */
async function skipImage(): Promise<void> {
  const turn = state.turn;
  if (!turn) return;
  forgetUndo();
  note('');
  for (const box of turn.boxes) {
    if (!state.skippedBoxes.includes(box) && !state.namedBoxes.includes(box)) state.skippedBoxes.push(box);
  }
  state.current = null;
  state.sees = [];
  // Put the pointer on the last face so the ordinary "this picture is finished" path
  // runs, rather than a second way of finishing a picture existing beside it.
  state.turnIndex = Math.max(0, turn.boxes.length - 1);

  if (waitingPhotos() > 0) {
    await nextPhoto();
    return;
  }
  endRun();
}

/**
 * The run is over: back to an empty page, or the closing line if there was a batch.
 *
 * It deliberately does NOT open the file dialog. Landing on the picker's own page with
 * a button on it is a stable place to stop; a dialog appearing by itself the instant a
 * skip is pressed is not, and it is the last thing somebody who just declined six
 * faces wants thrown at them.
 */
function endRun(): void {
  state.finishedTotal = runTotal() > 1 ? runTotal() : 0;
  state.turn = null;
  state.current = null;
  state.sees = [];
  state.turns = [];
  state.files = [];
  state.photosDone = 0;
  state.turnIndex = 0;
  state.namedBoxes = [];
  state.skippedBoxes = [];
  // Deliberately NOT cleared here: the closing line of a run is reached through answer(),
  // not through this, and the last name of the last picture is the likeliest of all to
  // want taking back. Every caller that does reach here has already dropped it.
  state.stage = state.finishedTotal > 0 ? 'ready' : 'start';
  render();
}

/** Start again from the picker. */
function another(): void {
  state.turn = null;
  state.current = null;
  state.sees = [];
  state.turns = [];
  state.files = [];
  state.photosDone = 0;
  state.finishedTotal = 0;
  state.turnIndex = 0;
  state.namedBoxes = [];
  state.skippedBoxes = [];
  forgetUndo();
  state.stage = 'start';
  render();
  choosePicture();
}

async function answer(name: string, extra = 0): Promise<void> {
  const sample = state.current;
  if (!sample) return;
  note('');
  if (name === '') {
    note('Name the thing in the box — one name is enough.');
    el.list.focus();
    return;
  }
  // Somebody still typing a list. Said out loud rather than quietly taking the first,
  // because their second name would otherwise vanish with no sign it had.
  if (extra > 0) {
    note(`One name per box — I used "${name}".`);
  }

  // An event about the page, never about the picture: it carries where the picture came from
  // and how many things are known, and never a name that was typed or a pixel that was seen.
  // It only exists once a visitor has allowed analytics — `siteTrack` is defined by the gate.
  window.siteTrack?.('picture_named', { source: sample.origin, names: knownNames().length });

  const before = snapshotRun();
  const nameWasNew = !state.names.includes(name);

  if (nameWasNew) state.names.push(name);
  // One label, because one rectangle is one person, place or thing. The network still
  // answers per name underneath; a sample that happens to carry a single one is
  // simply the ordinary case now.
  sample.labels = [name];
  if (!state.samples.includes(sample)) state.samples.push(sample);
  await store.putSample(sample);
  // The way back, held from here until something else touches the run. The snapshot is
  // taken above, before a single thing was written, because that is the state the box was
  // asked in — the state to put back.
  state.lastAnswer = { sample, name, nameWasNew, before };

  // The box is remembered as done so the overlay can show it as settled rather than
  // leaving every box looking equally unanswered.
  const box = currentBox();
  if (box && !state.namedBoxes.includes(box)) state.namedBoxes.push(box);

  state.sees = [];

  // Straight on to the next face, and then the next picture. The acknowledgement is
  // carried onto the next question's line, so a run of forty faces reads as forty
  // answers rather than forty confirmations.
  study();
  if (state.turn && state.turnIndex + 1 < state.turn.boxes.length) {
    await nextPhoto();
  } else if (waitingPhotos() > 0) {
    await nextPhoto();
  } else {
    // The last one. The total is worked out here, while the counters still exist,
    // and the closing line reads it rather than recomputing it from nothing.
    state.finishedTotal = runTotal() > 1 ? runTotal() : 0;
    state.photosDone = 0;
    state.turn = null;
    state.turnIndex = 0;
    state.namedBoxes = [];
    state.skippedBoxes = [];
    state.stage = 'ready';
    render();
  }
}

/**
 * Take the last name back.
 *
 * For the commonest mistake on this page: a name typed with the spelling wrong, or the
 * wrong name for the box, pressed through before it was read back. The run is put back
 * exactly as it stood while that box was being asked about, and the name goes back into
 * the box with it selected — so correcting a typo is a retype, and not a hunt back
 * through the queue for the picture it was on.
 */
async function undoLastAnswer(): Promise<void> {
  const last = state.lastAnswer;
  if (!last) return;
  forgetUndo();
  note('');

  // The picture, the box, the counters: exactly as they were.
  restoreRun(last.before);

  // The name was never really given, so it comes off the picture, out of the list and out
  // of the store. The sample was made a moment ago by showFace and carries a fresh id, so
  // it cannot be one of the pictures saved from an earlier sitting — nothing here can
  // touch older work.
  const at = state.samples.indexOf(last.sample);
  if (at >= 0) state.samples.splice(at, 1);
  last.sample.labels = [];
  void store.deleteSample(last.sample.id).catch(() => undefined);

  // A name that only ever existed in this one answer goes with it. A name that was
  // already here — typed earlier, or learned from a saved memory — is left alone: it is
  // not this answer's to remove.
  if (last.nameWasNew && !namedSamples().some((s) => s.labels.includes(last.name))) {
    state.names = state.names.filter((name) => name !== last.name);
  }

  // Ask the same box again, with the name in it ready to correct.
  clearedFor = null;
  render();
  el.list.value = last.name;
  el.list.focus();
  el.list.select();

  // The wrong name may already be studying. Stop that run and study again from the
  // pictures that are actually named now, or the model keeps the very mistake the undo
  // was meant to take back.
  if (state.studying) {
    state.needsStudy = true;
    send({ type: 'stop' });
  } else {
    study();
  }

  note(`Taken back — "${last.name}" is in the box. Correct it and press Tell it.`);
}

/**
 * The way back from a name typed wrong.
 *
 * A control in the row rather than words in the line above it: the moment a mistake is
 * noticed is the moment the page has already moved on, and a link inside a sentence reads
 * as part of the sentence. It names what it does, because "undo" alone beside a fresh
 * question could as easily mean the question as the answer.
 */
function undoControl(): HTMLButtonElement {
  const undo = link('Undo the last name', () => void undoLastAnswer());
  undo.title = 'Take the last name back, so you can correct it';
  return undo;
}

async function practise(): Promise<void> {
  // Practising hands the practise set over the way an upload does — George, 2026-09-19:
  // "load them all in the drag pictures here, like 10, so i can train the model as if i
  // uploaded them". So nothing is taught on the person's behalf: the pictures queue up, each
  // is opened in turn and searched, and every box is named by hand. What is practised is
  // therefore the real loop, not a shortcut through it.
  //
  // The set is twenty shapes the page draws itself — two of each of ten — because a set of
  // things it drew is a set it can be measured against, and because the colour is random on
  // every one of them, so the only thing there is to learn is the shape.
  //
  // And it starts from NOTHING — George, 2026-09-19: *"when i click practise, you should
  // delete memory"*. It used to add to whatever was already in the store, so a second
  // practise run queued its pictures behind the first run's and carried a model trained on
  // names from that run: a name taught five minutes ago answered for a box in this set, and
  // the chart's heading counted boxes from both. A practise set is a fresh start, pictures
  // and model both.
  //
  // The pictures are drawn FIRST, so a failure leaves what was already learned alone rather
  // than clearing it for nothing.
  const files = await makeSampleFiles(PRACTISE_COUNT);
  if (files.length === 0) {
    note('The practise shapes could not be drawn.');
    return;
  }
  practiseFiles.clear();
  for (const file of files) practiseFiles.add(file.name);
  window.siteTrack?.('practise_started', { pictures: files.length, names: SHAPE_NAMES.length });
  await forgetEverything();
  enqueue(files);
  // Read by the first picture's opening line — see `freshStart`. Set after `enqueue` on
  // purpose: `enqueue` starts the run without awaiting it, so this is still set before the
  // line is written.
  freshStart = true;
}

/**
 * The way to the practise set, said the same way wherever the page is at rest.
 *
 * It used to show only while nothing had been named, and it went as soon as a picture
 * arrived — so somebody who had practised once could never get back to it. George,
 * 2026-09-19: *"where did the practise images go?"* It is now offered at the start and
 * again whenever a run finishes; only the first line changes.
 */
function showPractiseOffer(nothingNamed: boolean): void {
  el.aside.append(nothingNamed ? 'No pictures handy? ' : 'Want the practise set again? ');
  el.aside.append(link('practise', () => void practise()));
  el.aside.append(
    ` on ${PRACTISE_COUNT} shapes it draws itself — ${SHAPE_NAMES.length} kinds, several of each.`,
  );
  el.aside.hidden = false;
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
      // The demo's own conversion, sent once when a model first exists. See `trainedOnce`.
      if (!trainedOnce) {
        trainedOnce = true;
        window.siteTrack?.('model_trained', {
          pictures: namedSamples().length,
          names: knownNames().length,
        });
      }
      if (state.needsStudy) {
        state.needsStudy = false;
        study();
      } else if (state.stage === 'asking' && state.current && state.current.labels.length === 0) {
        // It got better while the picture sat on the stage — so say what it sees now.
        const now = look(state.current);
        state.sees = now.sees;
        render();
      }
      return;
    }

    case 'error':
      state.studying = false;
      renderStudying();
      note(message.message);
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
  state.turn = null;
  state.turns = [];
  state.files = [];
  state.photosDone = 0;
  state.finishedTotal = 0;
  state.turnIndex = 0;
  state.namedBoxes = [];
  state.skippedBoxes = [];
  state.draft = null;
  forgetUndo();
  state.stage = 'start';
  render();
  renderCard();
  renderCharts();
  note('It has forgotten everything — the pictures and everything it learned.');
}

function saveToFile(): void {
  if (!state.file) {
    note('There is nothing to save yet.');
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
  window.siteTrack?.('memory_saved', { names: state.file.classes.length });
  note('Saved. Load it back with "Load a saved memory".');
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
    // A different model is in play now, so a way back that would put the old model's last
    // line of guesses on screen again is dropped.
    forgetUndo();
    render();
    note(`Loaded — it remembers ${parsed.classes.length} things.`);
  } catch (error) {
    note(error instanceof Error ? error.message : 'That file could not be read.');
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
  const accent = getComputedStyle(document.body).getPropertyValue('--accent').trim() || '#f97316';
  const second = getComputedStyle(document.body).getPropertyValue('--alt').trim() || '#60a5fa';

  drawLine(el.confusedChart, {
    series: [
      { values: state.history.map((m) => m.loss), color: accent, label: 'while studying' },
      { values: state.history.map((m) => m.valLoss), color: second, label: 'on pictures held back' },
    ],
    emptyTitle: 'This line is drawn as it studies.',
    emptyHint: 'Show it a picture and list what you can see in it.',
  });

  drawLine(el.rightChart, {
    series: [
      { values: state.history.map((m) => m.trainAccuracy), color: accent, label: 'pictures it studied' },
      { values: state.history.map((m) => m.valAccuracy), color: second, label: 'pictures held back' },
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
  el.tellBtn.addEventListener('click', () => {
    const typed = parseName(el.list.value);
    void answer(typed.name, typed.extra);
  });
  el.list.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      event.preventDefault();
      const typed = parseName(el.list.value);
      void answer(typed.name, typed.extra);
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

  // The two ways to move on. Both are links under the thing they act on rather than
  // buttons beside them, because that is what they are about: this face, or this
  // picture.
  el.skipBox.addEventListener('click', () => void skip());
  el.skipImage.addEventListener('click', () => void skipImage());

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
  el.pic.addEventListener('click', () => {
    // Only opens the picker when there is nothing on the stage. With a picture
    // showing, a click is either picking a face or starting to drag a new box, and
    // re-opening the file dialog on top of that would be maddening.
    if (state.turn === null) choosePicture();
  });
  el.pic.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' || event.key === ' ') {
      event.preventDefault();
      if (state.turn === null) choosePicture();
    }
  });

  /* ---------- drawing a box by hand ---------- */

  // The detector will miss faces — a black-and-white photograph, a face in shadow —
  // and it will occasionally draw a box around something that is not a face at all.
  // Being able to add one and remove one is what keeps the page usable when the
  // guess is wrong, which it will be.
  let dragging: { x: number; y: number } | null = null;

  el.boxes.addEventListener('pointerdown', (event) => {
    if (state.turn === null || state.stage !== 'asking') return;
    // A press that landed on a box is that box's business, not a new box.
    if (event.target !== el.boxes) return;
    const point = toPictureSpace(event.clientX, event.clientY);
    if (!point) return;
    event.preventDefault();
    dragging = point;
    state.draft = { x: point.x, y: point.y, w: 0, h: 0 };
    el.boxes.setPointerCapture(event.pointerId);
  });

  el.boxes.addEventListener('pointermove', (event) => {
    if (!dragging) return;
    const point = toPictureSpace(event.clientX, event.clientY);
    if (!point) return;
    const turn = state.turn;
    // Clamped to the picture: the overlay covers the letterboxing either side of a
    // tall photo too, and a box dragged out there would be off the image entirely.
    const x = turn ? Math.max(0, Math.min(turn.picture.width, point.x)) : point.x;
    const y = turn ? Math.max(0, Math.min(turn.picture.height, point.y)) : point.y;
    state.draft = {
      x: Math.min(dragging.x, x),
      y: Math.min(dragging.y, y),
      w: Math.abs(x - dragging.x),
      h: Math.abs(y - dragging.y),
    };
    renderBoxes();
  });

  const finishDrag = (event: PointerEvent): void => {
    if (!dragging) return;
    const draft = state.draft;
    dragging = null;
    state.draft = null;
    if (el.boxes.hasPointerCapture(event.pointerId)) el.boxes.releasePointerCapture(event.pointerId);

    const turn = state.turn;
    if (!turn || !draft) {
      renderBoxes();
      return;
    }
    // Anything smaller than a tenth of the picture across is a mis-click rather than
    // a box, and a box that size would be a face nobody could recognise anyway.
    const tooSmall = draft.w < turn.picture.width * 0.06 || draft.h < turn.picture.height * 0.06;
    if (tooSmall) {
      renderBoxes();
      note('That box was too small — drag across the thing you want to name.');
      return;
    }

    forgetUndo();
    turn.boxes.push(draft);
    // Left to right again, so the numbering stays in the order a person reads them.
    turn.boxes.sort((a, b) => a.x - b.x);
    state.turnIndex = turn.boxes.indexOf(draft);
    state.namedBoxes = state.namedBoxes.filter((named) => named !== draft);
    state.skippedBoxes = state.skippedBoxes.filter((skipped) => skipped !== draft);
    void showFace();
  };

  el.boxes.addEventListener('pointerup', finishDrag);
  el.boxes.addEventListener('pointercancel', finishDrag);

  // The boxes are measured from the rendered picture, and it has no size until its
  // thumbnail has arrived — so they are drawn again the moment it does.
  el.picture.addEventListener('load', () => renderBoxes());

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
      note('That was not a picture.');
      return;
    }
    // However many arrive, they all go in. Dropping twelve photographs and being
    // shown one is the behaviour this replaces.
    el.stage.scrollIntoView({ block: 'nearest' });
    enqueue(files);
  });

  // The boxes are placed from the picture's rendered size, so a resize moves the
  // picture and every box has to move with it.
  window.addEventListener('resize', () => {
    renderCharts();
    renderBoxes();
    renderImageProgress();
  });

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
    note('A memory from the older version was set aside. The pictures are still here.');
  } else if (model) {
    adopt(model);
    render();
    renderStudying();
    note(`Picked up where you left off — it knows ${model.classes.length} things from ${samples.length} pictures.`);
    return;
  }

  render();
  renderCharts();
  if (carried.failed) {
    note('Something saved under the old name could not be brought across.');
  } else if (carried.carried > 0) {
    note(`Carried over ${carried.carried} ${plural(carried.carried, 'picture', 'pictures')} saved under the old name.`);
  }
}

void boot();
