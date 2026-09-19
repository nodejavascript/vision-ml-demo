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
import { detectFaces } from './faces.js';
import { makeSampleFiles } from './samples.js';
import { VisionTrainer } from './trainer-host.js';
import * as store from './store.js';
import { drawBars, drawLine, drawProgress, drawTiles } from './charts.js';
/** Bumped whenever the weights stop meaning what they meant. See types.ts. */
const MODEL_VERSION = 2;
/* ------------------------------------------------------------------ *
 * Helpers
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
/** "a bear" · "a bear and a tree" · "a bear, a tree and the sky" */
function listWords(names) {
    if (names.length === 0)
        return 'nothing yet';
    if (names.length === 1)
        return names[0];
    return `${names.slice(0, -1).join(', ')} and ${names[names.length - 1]}`;
}
function plural(count, one, many) {
    return count === 1 ? one : many;
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
function parseName(text) {
    const pieces = text
        .toLowerCase()
        .split(/[,;\n]+/)
        .map((piece) => piece
        .replace(/[.!?]+/g, ' ')
        .split(/\s+/)
        .filter((word) => word !== ''))
        .filter((words) => words.length > 0);
    const cleaned = pieces.map((words) => {
        let start = 0;
        if (OPENS_A_SENTENCE.has(words[0])) {
            while (start < words.length && FRAMING.has(words[start]))
                start += 1;
        }
        return words.slice(start).join(' ');
    });
    const usable = cleaned.filter((name) => name !== '');
    return { name: usable[0] ?? '', extra: Math.max(0, usable.length - 1) };
}
const state = {
    samples: [],
    /** The things it knows. Only ever grows, so a name never changes meaning. */
    names: [],
    model: null,
    file: null,
    history: [],
    /** The picture on the stage right now — not saved until it is named. */
    current: null,
    /** What it can see in the current picture, most sure first. */
    sees: [],
    /**
     * Its single strongest answer, whatever its confidence.
     *
     * Kept because "I do not know" is a dead end for the person standing in front of
     * the page — the confidence line is theirs to move, and knowing what it leans
     * towards is what makes that line mean something. It is only ever shown as a
     * guess, never as an answer.
     */
    best: null,
    /**
     * Pictures handed over together and not opened yet, in order.
     *
     * Held as files rather than opened pictures on purpose: fifty photographs opened
     * up front would sit in memory as fifty full-size canvases for no reason.
     */
    files: [],
    /** Pictures that have been opened and searched for faces, waiting their turn. */
    turns: [],
    /** The picture on the stage, and which of its faces is being asked about. */
    turn: null,
    turnIndex: 0,
    /** Faces of this picture already named, so their boxes can show as done. */
    namedBoxes: [],
    /**
     * Faces the person declined to name.
     *
     * Tracked separately from the named ones because they are a different answer — "I
     * do not know who this is" rather than "this is Sarah" — and because a skipped box
     * that looks exactly like an untouched one makes the skip look as though it did not
     * register at all.
     */
    skippedBoxes: [],
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
    draft: null,
    /**
     * What the person listed for the picture just finished.
     *
     * Carried onto the next picture's line so a run reads as one continuous
     * conversation. Without it, naming a picture and being shown the next one looks
     * like the answer was thrown away.
     */
    lastNoted: [],
    stage: 'start',
    approved: null,
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
    stage: element('stage'),
    pic: element('pic'),
    picture: element('picture'),
    boxes: element('boxes'),
    picEmpty: element('picEmpty'),
    says: element('says'),
    sub: element('sub'),
    queue: element('queue'),
    tell: element('tell'),
    tellLabel: element('tellLabel'),
    who: element('who'),
    face: element('face'),
    skipBox: element('skipBox'),
    skipImage: element('skipImage'),
    progressChart: element('progressChart'),
    list: element('list'),
    tellBtn: element('tellBtn'),
    answers: element('answers'),
    aside: element('aside'),
    progress: element('progress'),
    studying: element('studying'),
    confusedChart: element('confusedChart'),
    rightChart: element('rightChart'),
    knowsChart: element('knowsChart'),
    filtersChart: element('filtersChart'),
    mapsChart: element('mapsChart'),
    passes: element('passes'),
    sureInput: element('sure'),
    augment: element('augment'),
    more: element('more'),
    card: element('card'),
    saveBtn: element('saveBtn'),
    loadBtn: element('loadBtn'),
    loadInput: element('loadInput'),
    forgetBtn: element('forgetBtn'),
    toast: element('toast'),
};
/* ------------------------------------------------------------------ *
 * The run — pictures handed over together, face by face
 * ------------------------------------------------------------------ */
/** How many drawn pictures the practise link hands over. */
const PRACTISE_COUNT = 10;
/**
 * How many pictures this run holds, counted from what is actually here rather than
 * kept in a counter of its own.
 *
 * A stored total can drift from reality — a face skipped, a second drop arriving
 * mid-run, a queue emptied — and a page that says "4 of 12" about a queue of nine is
 * worse than one that says nothing. Deriving it from the things that exist makes
 * that impossible.
 */
function runTotal() {
    return state.photosDone + (state.turn ? 1 : 0) + state.turns.length + state.files.length;
}
/** Which picture of the run is on the stage, counting from one. */
function runPosition() {
    return state.photosDone + (state.turn ? 1 : 0);
}
/** More than one picture: the only time the position across the run is worth saying. */
function hasRun() {
    return runTotal() > 1;
}
/** The face being asked about, or null when the whole picture is the subject. */
function currentBox() {
    const turn = state.turn;
    if (!turn || turn.boxes.length === 0)
        return null;
    return turn.boxes[state.turnIndex] ?? null;
}
/** How many pictures are still waiting behind this one. */
function waitingPhotos() {
    return state.turns.length + state.files.length;
}
/**
 * Why it has nothing to say about this picture.
 *
 * The reason is always its own state, never the picture — and saying which state is
 * the whole point. A flat "I do not know what is in this picture yet" was true, but
 * it read as though every picture had been considered and turned down, which is
 * exactly what a stuck page looks like. Three separate sentences, and the third one
 * resolves on its own because finishing a study run re-reads whatever is on the
 * stage.
 */
function cannotReadYet(names, named) {
    if (named === 0) {
        return 'I have not been taught anything yet.';
    }
    if (names.length < 2) {
        return `I know only ${listWords(names)} so far — one thing is not enough to tell two pictures apart.`;
    }
    return 'I am still learning — I will have something to say in a moment.';
}
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
 * Chatting to the person
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
/** Built from text nodes rather than markup: the names are typed by a person. */
function say(lead, strong, tail) {
    el.says.textContent = '';
    el.says.append(lead);
    if (strong !== undefined) {
        const bold = document.createElement('b');
        bold.textContent = strong;
        el.says.append(bold);
        if (tail)
            el.says.append(tail);
    }
}
function but(text, kind, onClick) {
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
function link(text, onClick) {
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
function namedSamples() {
    return state.samples.filter((s) => s.labels.length > 0);
}
/** The names that have at least one picture behind them. */
function knownNames() {
    const named = namedSamples();
    return state.names.filter((name) => named.some((s) => s.labels.includes(name)));
}
function learnNames() {
    for (const sample of state.samples) {
        for (const name of sample.labels)
            if (!state.names.includes(name))
                state.names.push(name);
    }
}
/** How many pictures each name appears in — its support, worth showing plainly. */
function labelCounts() {
    const counts = new Map();
    for (const sample of namedSamples()) {
        for (const name of sample.labels)
            counts.set(name, (counts.get(name) ?? 0) + 1);
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
function look(sample) {
    const model = state.model;
    const classes = state.file?.classes ?? [];
    if (!model || !sample || classes.length === 0)
        return { sees: [], best: null };
    const probs = model.predict(sample.pixels);
    const found = [];
    let bestIndex = -1;
    let bestSure = -1;
    for (let j = 0; j < classes.length; j++) {
        if (probs[j] >= state.sure)
            found.push({ name: classes[j], sure: probs[j] });
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
function render() {
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
    if (state.turn && el.picture.src !== state.turn.picture.thumb)
        el.picture.src = state.turn.picture.thumb;
    el.answers.textContent = '';
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
                    'Show me a picture and I will find the faces in it.';
        el.answers.append(but('Choose a picture', 'primary', () => choosePicture()));
        // Only while there is nothing to work with: the offer is a way out of an empty
        // page, not something to reach for once you have pictures of your own.
        if (named.length === 0) {
            el.aside.append('No pictures handy? ');
            el.aside.append(link('practise', () => void practise()));
            el.aside.append(` on ${PRACTISE_COUNT} drawn shapes.`);
            el.aside.hidden = false;
        }
        return;
    }
    if (state.stage === 'ready') {
        const said = state.current?.labels ?? [];
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
        }
        else {
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
    // The instruction is in the label on the answer box and the position is in the chip
    // above, so this line only adds what neither of those says: what was just noted,
    // and — while it still cannot tell two things apart — why that matters.
    const noted = state.lastNoted.length > 0 ? `Noted ${listWords(state.lastNoted)}. ` : '';
    const askingAboutFace = currentBox() !== null;
    // What to do, not why it cannot — the why is on the line above, in the model's own
    // voice, and the same reason said twice on one screen is how a page starts to read as
    // noise. And "name a second one" only means anything once there has been a first: on
    // the opening picture it was asking for a second before any first existed.
    const nudge = names.length < 2
        ? state.lastNoted.length > 0
            ? 'Name a second one and I can start telling them apart.'
            : 'Name this one and I can start learning.'
        : 'I will remember every one of them.';
    if (state.sees.length > 0) {
        // One name per box now, so the answer is the single strongest thing it can see
        // rather than everything above the line. "I think this is sarah" is a claim
        // somebody can agree with; a list of six is not.
        const guess = state.sees[0].name;
        say('I think this is ', guess, '.');
        el.sub.textContent = `${noted}${nudge}`;
        el.answers.append(but('Yes — that is it', '', () => void answer(guess)));
    }
    else if (state.best) {
        // Naming the strongest answer even when it is unsure is not a hedge: the
        // confidence line is the person's to move, and "I do not know" tells them
        // nothing about which way it leans.
        const percent = Math.round(state.best.sure * 100);
        say('I am not sure yet — my best guess is ', state.best.name, `, and I am only ${percent}% on that.`);
        el.sub.textContent = `${noted}${nudge}`;
    }
    else {
        say(cannotReadYet(names, named.length));
        el.sub.textContent = `${noted}${nudge}`;
    }
    // The question changes with the subject: a rectangle is a person to name, and a
    // picture with no face found in it is the picture itself.
    el.tell.hidden = false;
    el.list.value = '';
    // What can be done to the boxes used to be spelled out here — three sentences of
    // instructions that sat on the page the entire time a picture was up, for a gesture
    // most people never need. The × is visible on the box itself.
    el.who.hidden = false;
    el.face.hidden = !askingAboutFace;
    if (askingAboutFace && state.current)
        el.face.src = state.current.thumb;
    if (askingAboutFace) {
        el.tellLabel.textContent = 'Who or what is in this box? One name:';
        el.list.placeholder = 'sarah, or the dog, or the beach';
    }
    else {
        el.tellLabel.textContent = 'What is in this picture? One name:';
        el.list.placeholder = 'the kitchen, or the garden';
    }
    // Two ways to move on, each sitting under the thing it acts on rather than in a
    // button row beside them: skip the box, or skip the whole picture.
    const leftInPhoto = state.turn ? state.turn.boxes.length - state.turnIndex - 1 : 0;
    el.skipBox.hidden = !askingAboutFace;
    el.skipImage.hidden = false;
    el.skipImage.title =
        leftInPhoto > 0
            ? `Leave this picture — ${leftInPhoto} ${plural(leftInPhoto, 'face', 'faces')} will not be named`
            : 'Leave this picture';
}
/**
 * Where the run has got to, said plainly, and only while a picture is on the stage.
 *
 * Two numbers rather than one, because there are two things to know: which face of
 * this photograph, and which photograph of the batch. A single "3 of 12" could not
 * say both.
 */
function renderQueue() {
    // Hidden outside the asking stage as well as outside a run: once a batch is
    // finished the count is the previous run's, and leaving "5 of 5" on a page that
    // has moved on reads as a stuck number.
    if (state.stage !== 'asking' || state.turn === null) {
        el.queue.hidden = true;
        return;
    }
    const parts = [];
    const faces = state.turn.boxes.length;
    if (faces > 0)
        parts.push(`face ${state.turnIndex + 1} of ${faces}`);
    if (hasRun())
        parts.push(`photo ${runPosition()} of ${runTotal()}`);
    if (state.opening)
        parts.push('opening the next one…');
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
function pictureInElement() {
    const source = state.turn?.picture;
    if (!source)
        return null;
    const shown = { width: el.picture.clientWidth, height: el.picture.clientHeight };
    if (shown.width === 0 || shown.height === 0)
        return null;
    const scale = Math.min(shown.width / source.width, shown.height / source.height);
    return {
        scale,
        left: (shown.width - source.width * scale) / 2,
        top: (shown.height - source.height * scale) / 2,
    };
}
/** The box, in the picture's own pixels, under a point in element coordinates. */
function toPictureSpace(clientX, clientY) {
    const fit = pictureInElement();
    if (!fit)
        return null;
    const area = el.picture.getBoundingClientRect();
    return {
        x: Math.round((clientX - area.left - fit.left) / fit.scale),
        y: Math.round((clientY - area.top - fit.top) / fit.scale),
    };
}
function renderBoxes() {
    el.boxes.textContent = '';
    const turn = state.turn;
    const fit = turn ? pictureInElement() : null;
    // The overlay stays on the picture whenever there IS a picture, even with no boxes
    // in it — otherwise there is nothing to drag on, and a face the detector missed
    // could never be added. Only the boxes inside it come and go.
    const active = turn !== null && fit !== null && state.stage === 'asking';
    el.boxes.hidden = !active;
    if (!active || !turn || !fit)
        return;
    turn.boxes.forEach((box, index) => {
        const element = document.createElement('div');
        element.className = 'box';
        if (index === state.turnIndex)
            element.classList.add('current');
        if (state.namedBoxes.includes(box))
            element.classList.add('done');
        if (state.skippedBoxes.includes(box))
            element.classList.add('skipped');
        element.style.left = `${fit.left + box.x * fit.scale}px`;
        element.style.top = `${fit.top + box.y * fit.scale}px`;
        element.style.width = `${box.w * fit.scale}px`;
        element.style.height = `${box.h * fit.scale}px`;
        element.title = `Face ${index + 1} of ${turn.boxes.length} — click to name this one`;
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
            if (index !== state.turnIndex)
                void goToFace(index);
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
async function removeBox(index) {
    const turn = state.turn;
    if (!turn)
        return;
    turn.boxes.splice(index, 1);
    if (state.turnIndex >= turn.boxes.length)
        state.turnIndex = Math.max(0, turn.boxes.length - 1);
    // A picture whose boxes have all been removed becomes the whole picture, which is
    // the honest fallback: something gets named either way.
    await showFace();
}
/** Move to a face the person picked out of the picture themselves. */
async function goToFace(index) {
    const turn = state.turn;
    if (!turn || index < 0 || index >= turn.boxes.length)
        return;
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
function renderImageProgress() {
    const turn = state.turn;
    const show = turn !== null && turn.boxes.length > 0 && state.stage === 'asking';
    el.progressChart.hidden = !show;
    if (!show || !turn)
        return;
    drawProgress(el.progressChart, turn.boxes.map((box, index) => ({
        state: state.namedBoxes.includes(box)
            ? 'named'
            : state.skippedBoxes.includes(box)
                ? 'skipped'
                : index === state.turnIndex
                    ? 'current'
                    : 'left',
    })));
}
function renderProgress() {
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
    if (files.length > 0)
        enqueue(files);
});
function choosePicture() {
    filePicker.click();
}
/**
 * Hand over pictures: one goes on the stage, the rest queue behind it.
 *
 * Files are not opened here. Each one is opened when its turn comes, so a batch of
 * fifty costs one full-size picture in memory at a time rather than fifty.
 */
function enqueue(files) {
    const pictures = files.filter((file) => file.type.startsWith('image/'));
    if (pictures.length === 0) {
        toast('None of those were pictures.');
        return;
    }
    if (pictures.length < files.length) {
        const missed = files.length - pictures.length;
        toast(missed === 1
            ? 'One file was not a picture, so it was left out.'
            : `${missed} files were not pictures, so they were left out.`);
    }
    if (state.stage === 'ready') {
        // Whatever is on the stage has already been named, so it is not part of what
        // comes next and must not be counted as one of them.
        state.turn = null;
        state.current = null;
        state.photosDone = 0;
        state.finishedTotal = 0;
        state.lastNoted = [];
    }
    state.files.push(...pictures);
    if (state.turn === null)
        void nextPhoto();
    else
        render();
}
/**
 * Open the next picture waiting, skipping any that will not open.
 *
 * Returns nothing when the run is over — which the caller tells apart from "there
 * was nothing to open" only by the run counters, and does not need to.
 */
async function openNextPhoto() {
    while (state.turns.length === 0 && state.files.length > 0) {
        const file = state.files.shift();
        if (!file)
            break;
        state.opening = true;
        try {
            const picture = await loadPicture(file);
            // The search runs on the picture's own pixels, then the boxes are handed back
            // in that same space, which is the space the overlay draws them in.
            state.turns.push({ picture, boxes: detectFaces(picture.frame) });
        }
        catch {
            toast(`${file.name} could not be read as a picture.`);
        }
        finally {
            state.opening = false;
        }
    }
    return state.turns.shift() ?? null;
}
/** Move on: the next face of this picture, or the next picture, or the end. */
async function nextPhoto() {
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
    if (waitingPhotos() > 0)
        render(); // say that it is opening the next one
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
    const faces = nextTurn.boxes.length;
    if (faces > 0) {
        toast(faces === 1
            ? 'Found one face. Name it, or drag a box if it has the wrong one.'
            : `Found ${faces} faces. They are named one at a time.`);
    }
}
/** Put the face being asked about on the stage, and ask. */
async function showFace() {
    const turn = state.turn;
    if (!turn)
        return;
    const box = currentBox();
    const picture = turn.picture;
    state.current = {
        id: newId(),
        labels: [],
        // With a box, the thing being named is the crop of it; without one, the picture
        // is the subject and the crop is the picture.
        thumb: box ? picture.cropThumb(box) : picture.thumb,
        pixels: box ? picture.crop(box) : picture.whole(),
        name: box ? `${picture.filename} — face ${state.turnIndex + 1}` : picture.filename,
        addedAt: new Date().toISOString(),
        origin: 'file',
    };
    const now = look(state.current);
    state.sees = now.sees;
    state.best = now.best;
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
async function skip() {
    const box = currentBox();
    if (state.current === null)
        return;
    if (box && !state.skippedBoxes.includes(box))
        state.skippedBoxes.push(box);
    state.current = null;
    state.sees = [];
    state.best = null;
    state.lastNoted = [];
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
async function skipImage() {
    const turn = state.turn;
    if (!turn)
        return;
    for (const box of turn.boxes) {
        if (!state.skippedBoxes.includes(box) && !state.namedBoxes.includes(box))
            state.skippedBoxes.push(box);
    }
    state.current = null;
    state.sees = [];
    state.best = null;
    state.lastNoted = [];
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
function endRun() {
    state.finishedTotal = runTotal() > 1 ? runTotal() : 0;
    state.turn = null;
    state.current = null;
    state.sees = [];
    state.best = null;
    state.turns = [];
    state.files = [];
    state.photosDone = 0;
    state.turnIndex = 0;
    state.namedBoxes = [];
    state.skippedBoxes = [];
    state.lastNoted = [];
    state.stage = state.finishedTotal > 0 ? 'ready' : 'start';
    render();
}
/** Start again from the picker. */
function another() {
    state.turn = null;
    state.current = null;
    state.sees = [];
    state.best = null;
    state.turns = [];
    state.files = [];
    state.photosDone = 0;
    state.finishedTotal = 0;
    state.turnIndex = 0;
    state.namedBoxes = [];
    state.skippedBoxes = [];
    state.lastNoted = [];
    state.stage = 'start';
    render();
    choosePicture();
}
async function answer(name, extra = 0) {
    const sample = state.current;
    if (!sample)
        return;
    if (name === '') {
        toast('Name the thing in the box — one name is enough.');
        el.list.focus();
        return;
    }
    // Somebody still typing a list. Said out loud rather than quietly taking the first,
    // because their second name would otherwise vanish with no sign it had.
    if (extra > 0) {
        toast(`One name per box — I used "${name}".`);
    }
    if (!state.names.includes(name))
        state.names.push(name);
    // One label, because one rectangle is one person, place or thing. The network still
    // answers per name underneath; a sample that happens to carry a single one is
    // simply the ordinary case now.
    sample.labels = [name];
    if (!state.samples.includes(sample))
        state.samples.push(sample);
    await store.putSample(sample);
    // The box is remembered as done so the overlay can show it as settled rather than
    // leaving every box looking equally unanswered.
    const box = currentBox();
    if (box && !state.namedBoxes.includes(box))
        state.namedBoxes.push(box);
    state.lastNoted = [name];
    state.sees = [];
    state.best = null;
    // Straight on to the next face, and then the next picture. The acknowledgement is
    // carried onto the next question's line, so a run of forty faces reads as forty
    // answers rather than forty confirmations.
    study();
    if (state.turn && state.turnIndex + 1 < state.turn.boxes.length) {
        await nextPhoto();
    }
    else if (waitingPhotos() > 0) {
        await nextPhoto();
    }
    else {
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
async function practise() {
    // Practising hands the drawn pictures over the way an upload does — George, 2026-09-19:
    // "load them all in the drag pictures here, like 10, so i can train the model as if i
    // uploaded them". So nothing is taught on the person's behalf: the shapes queue up,
    // each is opened in turn and searched for faces, and every box is named by hand. What
    // is practised is therefore the real loop, not a shortcut through it.
    const files = makeSampleFiles(PRACTISE_COUNT);
    if (files.length === 0) {
        toast('The drawn shapes could not be prepared.');
        return;
    }
    toast(`${files.length} drawn shapes queued — a shape on a coloured background, so naming one teaches two things. ` +
        'These hold no faces, so a box on one is the detector guessing: the × removes it.');
    enqueue(files);
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
function study() {
    const named = namedSamples();
    const names = knownNames();
    // One thing is not something you can tell apart from anything.
    if (named.length < 2 || names.length < 2)
        return;
    // Already studying: remember there is newer work and take it up when this run
    // ends. Two runs at once would fight over the same weights.
    if (state.studying) {
        state.needsStudy = true;
        return;
    }
    const samples = named.map((s) => ({
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
function renderStudying() {
    if (!state.studying) {
        el.studying.hidden = true;
        return;
    }
    el.studying.hidden = false;
    el.studying.textContent = state.model
        ? `Still studying — pass ${state.currentPass} of ${state.plannedPasses}. It keeps getting better while you carry on.`
        : `Studying the pictures. I will start saying what I can see in a moment — pass ${state.currentPass} of ${state.plannedPasses}.`;
}
function handleMessage(message) {
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
            }
            else if (state.stage === 'asking' && state.current && state.current.labels.length === 0) {
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
function adopt(file) {
    state.file = file;
    state.history = file.meta.history.slice();
    for (const name of file.classes)
        if (!state.names.includes(name))
            state.names.push(name);
    try {
        state.model = Cnn.load(file);
    }
    catch {
        state.model = null;
    }
    void store.saveModel(file);
    renderCard();
    renderCharts();
}
async function forgetEverything() {
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
    state.turn = null;
    state.turns = [];
    state.files = [];
    state.photosDone = 0;
    state.finishedTotal = 0;
    state.turnIndex = 0;
    state.namedBoxes = [];
    state.skippedBoxes = [];
    state.draft = null;
    state.lastNoted = [];
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
function saveToFile() {
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
async function loadFromFile(file) {
    try {
        const parsed = JSON.parse(await file.text());
        // Both ids are accepted: the project was renamed after this file format was
        // already in use, and a memory somebody saved to disk is theirs, not ours to
        // invalidate over a name.
        if (parsed.format !== 'vision-ml-demo-model' && parsed.format !== 'vision-demo-model') {
            throw new Error('That is not one of these files.');
        }
        if (parsed.version !== MODEL_VERSION)
            throw new Error('That memory was saved by an older version and cannot be read.');
        adopt(parsed);
        render();
        toast(`Loaded — it remembers ${parsed.classes.length} things.`);
    }
    catch (error) {
        toast(error instanceof Error ? error.message : 'That file could not be read.');
    }
}
function renderCard() {
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
function renderCharts() {
    renderCurves();
    renderKnows();
    renderModelPictures();
}
function renderCurves() {
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
function renderKnows() {
    const classes = state.file?.classes ?? [];
    const accuracy = state.file?.meta.perClassAccuracy ?? [];
    const counts = state.file?.meta.perClassCount ?? [];
    drawBars(el.knowsChart, classes.map((name, index) => ({
        label: name,
        value: accuracy[index] ?? 0,
        caption: `${Math.round((accuracy[index] ?? 0) * 100)}% · ${counts[index] ?? 0} ${plural(counts[index] ?? 0, 'picture', 'pictures')}`,
    })), {
        max: 1,
        emptyTitle: 'Nothing to score yet.',
        emptyHint: 'Once it has studied, this shows how well it knows each thing.',
    });
}
function renderModelPictures() {
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
function renderMaps() {
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
async function boot() {
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
        if (file)
            void loadFromFile(file);
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
    el.pic.addEventListener('click', () => {
        // Only opens the picker when there is nothing on the stage. With a picture
        // showing, a click is either picking a face or starting to drag a new box, and
        // re-opening the file dialog on top of that would be maddening.
        if (state.turn === null)
            choosePicture();
    });
    el.pic.addEventListener('keydown', (event) => {
        if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            if (state.turn === null)
                choosePicture();
        }
    });
    /* ---------- drawing a box by hand ---------- */
    // The detector will miss faces — a black-and-white photograph, a face in shadow —
    // and it will occasionally draw a box around something that is not a face at all.
    // Being able to add one and remove one is what keeps the page usable when the
    // guess is wrong, which it will be.
    let dragging = null;
    el.boxes.addEventListener('pointerdown', (event) => {
        if (state.turn === null || state.stage !== 'asking')
            return;
        // A press that landed on a box is that box's business, not a new box.
        if (event.target !== el.boxes)
            return;
        const point = toPictureSpace(event.clientX, event.clientY);
        if (!point)
            return;
        event.preventDefault();
        dragging = point;
        state.draft = { x: point.x, y: point.y, w: 0, h: 0 };
        el.boxes.setPointerCapture(event.pointerId);
    });
    el.boxes.addEventListener('pointermove', (event) => {
        if (!dragging)
            return;
        const point = toPictureSpace(event.clientX, event.clientY);
        if (!point)
            return;
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
    const finishDrag = (event) => {
        if (!dragging)
            return;
        const draft = state.draft;
        dragging = null;
        state.draft = null;
        if (el.boxes.hasPointerCapture(event.pointerId))
            el.boxes.releasePointerCapture(event.pointerId);
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
            toast('That box was too small — drag across the face you want to name.');
            return;
        }
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
    const carriesFiles = (event) => Array.from(event.dataTransfer?.types ?? []).includes('Files');
    // dragenter fires again for every element entered, so a depth count is what keeps
    // the highlight on until the pointer actually leaves the page.
    let dragDepth = 0;
    window.addEventListener('dragenter', (event) => {
        if (!carriesFiles(event))
            return;
        dragDepth += 1;
        el.pic.classList.add('dragging');
    });
    window.addEventListener('dragover', (event) => {
        if (!carriesFiles(event))
            return;
        // Without this the browser opens the dropped file instead of letting the page
        // have it, and the drop event never arrives.
        event.preventDefault();
        if (event.dataTransfer)
            event.dataTransfer.dropEffect = 'copy';
    });
    window.addEventListener('dragleave', () => {
        dragDepth = Math.max(0, dragDepth - 1);
        if (dragDepth === 0)
            el.pic.classList.remove('dragging');
    });
    window.addEventListener('drop', (event) => {
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
        store.allSamples().catch(() => []),
        store.loadModel().catch(() => null),
        store.loadSetting('passes', 60),
        store.loadSetting('sure', 0.5),
        store.loadSetting('augment', true),
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
    }
    else if (model) {
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
    }
    else if (carried.carried > 0) {
        toast(`Carried over ${carried.carried} ${plural(carried.carried, 'picture', 'pictures')} saved under the old name.`);
    }
}
void boot();
