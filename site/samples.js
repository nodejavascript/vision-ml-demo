/**
 * samples.ts — the practise set: thirty pictures of four hard shapes, drawn here rather than fetched.
 *
 * The demo asks you to bring your own pictures, and it should. But an empty page with a
 * file dialog is a poor first minute: nothing at all happens until you have gathered
 * twenty photographs. So the page also draws a set of its own and hands it over the way
 * an upload arrives — queued, opened one at a time, each one named by hand.
 *
 * George, 2026-09-23: *"the samples for a practise is 20 of basic shapes to train"* — and then,
 * having looked at it: ***"the problem with the 20, there is not enough of similar shapes to learn
 * from, make it 30 and only 4 different but difficult shapes"***.
 *
 * **THE COUNT AND THE CHOICE OF SHAPES ARE THE SAME DECISION, and that is the whole point.**
 * Twenty pictures across ten shapes is **two examples each**, and two is not a set to learn from:
 * a network can hold one picture and echo its answer, and the second example is the only test of
 * whether anything was learned — and it has already been seen. Thirty across four shapes is
 * **seven or eight of each**, so the same shape comes back again and again in different colours,
 * rotations and sizes, and what has to be learned is the shape.
 *
 * **The four are the hard ones** — star, moon, heart and arrow — because a set only teaches
 * something if the answer is not obvious from a glance at the extent of the drawing. The six that
 * went with them (circle, square, triangle, diamond, cross, hexagon) are convex and symmetric: a
 * circle and a hexagon are told apart by a couple of corners, and a square is a diamond rotated
 * 45 degrees. The four left have concave or curved detail that survives rotation and does not
 * survive being memorised — a crescent and a star are the two hardest things here to tell apart,
 * and telling them apart on the seventh example is the test this set exists to set.
 *
 * Three things about the drawing are deliberate, and each is there for the learning:
 *
 *   - **The colour is random, every picture.** If a star were always orange the network would
 *     learn "orange" and call it a star. With a random hue on a random dark ground, colour
 *     carries no information at all and the only thing left to learn is the shape — which is
 *     what a practise set is for.
 *   - **Nothing is in the same place twice.** Each shape is rotated, scaled and nudged by its own
 *     seed, so a network that memorises one picture's pixels rather than the shape gets the next
 *     one wrong, and the chart of held-back pictures says so.
 *   - **It is a PNG, at the width the face search reads.** The picture is drawn at `ANALYSIS`
 *     wide, so the search scales it by one and never resamples, and a hard edge stays one
 *     pixel wide. A drawn shape handed over as a JPEG arrives fringed with colour that is not
 *     in the picture — a measured fault from the retired set, not a preference.
 *
 * These are pictures of **things, not of people**, and the face finder is neither told that
 * nor asked to skip them: it runs on every picture the page opens, and on a shape it
 * correctly finds nothing. The page then asks about the picture itself, which is the same
 * path a photograph of a kitchen takes. Practise is the real loop, or it is not practise.
 */
import { ANALYSIS } from './image.js';
/** How many pictures the practise link hands over — seven or eight of each of the four shapes. */
export const PRACTISE_COUNT = 30;
/**
 * The four shapes, in the order the pool is built.
 *
 * Exported so the page can say what the set holds without repeating the list, and so the
 * suite can check that what the copy promises is what is drawn.
 */
export const SHAPE_NAMES = [
    'star',
    'moon',
    'heart',
    'arrow',
];
/**
 * The canvas the picture is drawn on.
 *
 * `ANALYSIS` imported rather than a number of its own: the picture has to be drawn at the
 * size the face search reads, or the search resamples it and the edges soften. Writing 320
 * here would be a second copy of a number that has already changed once.
 */
const CANVAS = ANALYSIS;
/** A tiny deterministic generator, so picture three is the same picture for everybody. */
function seededRandom(seed) {
    let state = seed >>> 0;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}
/** A number between `low` and `high`. */
function between(random, low, high) {
    return low + random() * (high - low);
}
/**
 * One shape's outline, drawn about the origin inside a box two units across.
 *
 * Everything is built around (0, 0) and fitted by the transform, so the jitter is applied
 * once, in one place, rather than being arithmetic inside every path.
 */
function path(ctx, shape) {
    switch (shape) {
        case 'star': {
            const points = 5;
            const inner = 0.44;
            for (let i = 0; i < points * 2; i++) {
                const radius = i % 2 === 0 ? 1 : inner;
                // Started at the top rather than at three o'clock, which is where a star looks wrong.
                const angle = (i * Math.PI) / points - Math.PI / 2;
                const x = Math.cos(angle) * radius;
                const y = Math.sin(angle) * radius;
                if (i === 0)
                    ctx.moveTo(x, y);
                else
                    ctx.lineTo(x, y);
            }
            ctx.closePath();
            break;
        }
        case 'heart':
            // Two lobes and a point, as curves rather than as a character: a glyph would arrive
            // drawn by whatever machine the reader has, which is not the same picture twice.
            ctx.moveTo(0, 0.95);
            ctx.bezierCurveTo(-1.45, -0.2, -0.62, -1.15, 0, -0.42);
            ctx.bezierCurveTo(0.62, -1.15, 1.45, -0.2, 0, 0.95);
            ctx.closePath();
            break;
        case 'arrow':
            // Up and to the right, with a tail long enough to be an arrow and not a wedge.
            ctx.moveTo(0.15, -1);
            ctx.lineTo(1, -0.15);
            ctx.lineTo(0.5, -0.15);
            ctx.lineTo(0.5, 0.95);
            ctx.lineTo(-0.2, 0.95);
            ctx.lineTo(-0.2, -0.15);
            ctx.lineTo(-0.7, -0.15);
            ctx.closePath();
            break;
        case 'moon':
            // A crescent, and it has to be built from two ARCS rather than from a disc with a
            // second disc taken out of it.
            //
            // Two things were measured wrong on the way here (2026-09-23), and both are recorded
            // so neither is tried again. First, two overlapping circles filled even-odd leave a
            // LENS where the bite sticks out past the disc — the part of the bite outside the disc
            // is enclosed once, so it is filled. Second, and subtler: coming back along the bite's
            // RIGHT side fills the disc with a dent in it, because the enclosed area is what lies
            // between the two arcs, and the bite's right side is outside the disc.
            //
            // The crescent is the band between the disc's left edge and the bite's left edge, so
            // the path runs the long way round the LEFT of the disc and back along the LEFT of the
            // bite (anticlockwise). With the bite at centre (0.28, 0) and radius 0.95:
            //   they meet at x ≈ 0.314, y ≈ ±0.949 — 1.25 rad on the disc, 1.535 rad on the bite.
            // That leaves a crescent about a sixth of the disc across, which is a moon rather than
            // a gibbous blob and still five pixels thick at the size the network reads.
            ctx.arc(0, 0, 1, 1.25, -1.25, false);
            ctx.arc(0.28, 0, 0.95, -1.535, 1.535, true);
            ctx.closePath();
            break;
        default:
            // Unreachable while the list above is the list — and a CIRCLE, which is deliberately not
            // one of the four shapes this set draws: a name added without its drawing then fails
            // visibly, as a shape nobody asked for, instead of silently painting a blank canvas.
            ctx.arc(0, 0, 1, 0, Math.PI * 2);
            break;
    }
}
/** One picture: a dark ground, one bright shape, jittered by its own seed. */
function drawShape(canvas, shape, seed) {
    const random = seededRandom(seed);
    const ctx = canvas.getContext('2d');
    if (!ctx)
        throw new Error('This browser would not give a 2D canvas context.');
    canvas.width = CANVAS;
    canvas.height = CANVAS;
    // The ground: a dark hue of its own, so no two pictures of the same shape sit on the
    // same coloured page.
    ctx.fillStyle = `hsl(${Math.round(between(random, 0, 360))} 42% 10%)`;
    ctx.fillRect(0, 0, CANVAS, CANVAS);
    // The shape: a bright hue of its own, so colour says nothing about which shape it is.
    const hue = Math.round(between(random, 0, 360));
    ctx.save();
    ctx.translate(CANVAS / 2 + between(random, -0.06, 0.06) * CANVAS, CANVAS / 2 + between(random, -0.06, 0.06) * CANVAS);
    ctx.rotate(between(random, -Math.PI / 7, Math.PI / 7));
    const size = between(random, 0.3, 0.4) * CANVAS;
    ctx.scale(size, size);
    ctx.beginPath();
    path(ctx, shape);
    ctx.fillStyle = `hsl(${hue} 78% 64%)`;
    // `evenodd` because the crescent's two arcs enclose the shape between them, and a hole in
    // a path (if one is ever added) should be a hole rather than a second shape on top.
    ctx.fill('evenodd');
    ctx.restore();
}
/**
 * Draw the set and hand it over as files, exactly as `enqueue` receives an upload.
 *
 * Deterministic on purpose: picture *n* is the same picture for every visitor and on every
 * run, so the set can be measured and the suite can assert what is in it. `random` is a
 * parameter so that a test can shuffle the same way twice.
 */
export async function makeSampleFiles(count = PRACTISE_COUNT, random = Math.random) {
    // 🔴 SEVEN OR EIGHT OF EACH — the change George asked for on 23 September 2026: *"there is not
    // enough of similar shapes to learn from, make it 30 and only 4 different but difficult
    // shapes"*. The pool is built by CYCLING the four, so thirty across four comes out EVEN —
    // 8, 8, 7, 7 — rather than as a run of one shape and a couple of another.
    const pool = [];
    while (pool.length < count)
        pool.push(SHAPE_NAMES[pool.length % SHAPE_NAMES.length]);
    // Then shuffled, and the shuffle is not decoration: **a queue that always reached the star
    // fourth would hand the person the answer.** "The fifth one is the moon" is a rule about the
    // set, not about the picture, and a set you can read the answer off teaches nothing. The ORDER
    // is different on every visit; the drawing at a position is not, because its seed is the
    // position and the shape only decides which outline goes on it.
    for (let i = pool.length - 1; i > 0; i--) {
        const j = Math.floor(random() * (i + 1));
        const swap = pool[i];
        pool[i] = pool[j];
        pool[j] = swap;
    }
    const canvas = document.createElement('canvas');
    const out = [];
    const take = Math.max(0, Math.min(count, pool.length));
    for (let i = 0; i < take; i++) {
        try {
            // The seed is the picture's position in the set and never its name: the file name is
            // the one field that travels with the picture into the model's own record, and a file
            // called `circle-2.png` would have the answer written on it.
            drawShape(canvas, pool[i], 4000 + i * 97);
            const blob = await new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));
            if (!blob)
                continue;
            out.push(new File([blob], `practise-${String(i + 1).padStart(2, '0')}.png`, { type: 'image/png' }));
        }
        catch {
            // One picture that will not draw is not worth losing the other nineteen over.
        }
    }
    return out;
}
