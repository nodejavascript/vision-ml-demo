/**
 * samples.ts — a small set of pictures the page can draw for you.
 *
 * The demo asks you to bring your own images, and it should. But an empty page
 * with a file dialog is a poor first minute: you cannot see the charts do
 * anything until you have gathered twenty photographs. So the page can also
 * draw its own — three shapes, many colours, varied size and position — which
 * is enough to watch a network learn something real, and enough to try the
 * whole loop without leaving your chair.
 *
 * The colour, position and size are randomised per image and the shape is not,
 * so colour is a distractor rather than a shortcut: the only thing that
 * predicts the label is the outline, which is exactly the trap a real image set
 * would set.
 */
import { pixelsFromCanvas, thumbFromCanvas } from './image.js';
import { mulberry32 } from './net.js';
export const SAMPLE_CLASS_LABELS = ['circle', 'square', 'triangle'];
/** The eight colours a shape is allowed to be, by hue bucket. */
const HUE_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta'];
function hueName(hue) {
    const wrapped = ((hue % 360) + 360) % 360;
    return HUE_NAMES[Math.round(wrapped / 45) % 8];
}
const CANVAS = 128;
function hsl(h, s, l) {
    return `hsl(${Math.round(h)} ${Math.round(s)}% ${Math.round(l)}%)`;
}
/**
 * Draw `perClass` pictures of each shape. Returns them decoded and ready, so
 * the caller treats a drawn image exactly like a chosen one.
 */
export function makeSampleSet(perClass = 30, seed = 7) {
    const rng = mulberry32(seed);
    const out = [];
    const canvas = document.createElement('canvas');
    canvas.width = CANVAS;
    canvas.height = CANVAS;
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx)
        return out;
    for (const label of SAMPLE_CLASS_LABELS) {
        for (let n = 0; n < perClass; n++) {
            const hue = rng() * 360;
            // A dark background with a bright shape keeps the contrast high whatever
            // hue comes up; a light-on-light pair would be an unfair picture.
            ctx.fillStyle = hsl(hue, 30, 14 + rng() * 8);
            ctx.fillRect(0, 0, CANVAS, CANVAS);
            const cx = CANVAS / 2 + (rng() - 0.5) * 26;
            const cy = CANVAS / 2 + (rng() - 0.5) * 26;
            const reach = 38 + rng() * 20;
            const shapeHue = hue + 120 + rng() * 120;
            ctx.fillStyle = hsl(shapeHue, 72, 58 + rng() * 10);
            ctx.beginPath();
            if (label === 'circle') {
                ctx.arc(cx, cy, reach * 0.52, 0, Math.PI * 2);
            }
            else if (label === 'square') {
                const side = reach * 0.92;
                ctx.rect(cx - side / 2, cy - side / 2, side, side);
            }
            else {
                const angle = rng() * Math.PI * 2;
                const radius = reach * 0.6;
                for (let i = 0; i < 3; i++) {
                    const a = angle + (i * 2 * Math.PI) / 3;
                    const x = cx + Math.cos(a) * radius;
                    const y = cy + Math.sin(a) * radius;
                    if (i === 0)
                        ctx.moveTo(x, y);
                    else
                        ctx.lineTo(x, y);
                }
                ctx.closePath();
            }
            ctx.fill();
            out.push({
                label,
                colour: hueName(shapeHue),
                pixels: pixelsFromCanvas(canvas),
                thumb: thumbFromCanvas(canvas),
                name: `${label}-${n + 1}.png`,
            });
        }
    }
    return out;
}
