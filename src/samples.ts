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

import { ANALYSIS, pixelsFromCanvas } from './image.js';
import { mulberry32 } from './net.js';

export const SAMPLE_CLASS_LABELS = ['circle', 'square', 'triangle'];

/** The eight colours a shape is allowed to be, by hue bucket. */
const HUE_NAMES = ['red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'purple', 'magenta'];

function hueName(hue: number): string {
  const wrapped = ((hue % 360) + 360) % 360;
  return HUE_NAMES[Math.round(wrapped / 45) % 8];
}

export interface GeneratedSample {
  label: string;
  /**
   * The colour of the shape, as a plain word.
   *
   * This is the second thing every drawn picture teaches, and it is what makes the
   * practice set honestly multi-label: a circle drawn in blue is both "circle" and
   * "blue", which is exactly the sort of thing a photograph of a dog on a beach
   * teaches. It is taken from the shape's colour rather than the background's on
   * purpose — the background is deliberately dark and desaturated, so naming its
   * hue would be asking the model to learn something a person could not see either.
   */
  colour: string;
  pixels: Uint8Array;
  /**
   * The drawing itself, losslessly, as a PNG data URL.
   *
   * Lossless rather than the small JPEG the rest of the page uses for display, and that is
   * the whole point. A drawn shape is nothing but hard edges, and JPEG spends its budget
   * exactly there: the ringing it leaves along an outline is a band of in-between colours,
   * which is one of the two things the face search reads as skin. Handed over as a JPEG,
   * the practise pictures arrive at the detector blurred.
   */
  png: string;
  name: string;
}

/**
 * The size every drawn picture is painted at — the size the face search looks at.
 *
 * Not a round number of its own choosing: it is image.ts's ANALYSIS, so the search reads
 * the drawing's own pixels and never enlarges it. Drawn at 128 and scaled up to 224, every
 * hard edge reached the detector as a soft gradient instead. George, 2026-09-19: "the test
 * pictures should be sharper. getting too many boxes because of blurred lines."
 */
const CANVAS = ANALYSIS;

function hsl(h: number, s: number, l: number): string {
  return `hsl(${Math.round(h)} ${Math.round(s)}% ${Math.round(l)}%)`;
}

/**
 * Draw `perClass` pictures of each shape. Returns them decoded and ready, so
 * the caller treats a drawn image exactly like a chosen one.
 */
export function makeSampleSet(perClass = 30, seed = 7): GeneratedSample[] {
  const rng = mulberry32(seed);
  const out: GeneratedSample[] = [];
  const canvas = document.createElement('canvas');
  canvas.width = CANVAS;
  canvas.height = CANVAS;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  if (!ctx) return out;

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
      } else if (label === 'square') {
        const side = reach * 0.92;
        ctx.rect(cx - side / 2, cy - side / 2, side, side);
      } else {
        const angle = rng() * Math.PI * 2;
        const radius = reach * 0.6;
        for (let i = 0; i < 3; i++) {
          const a = angle + (i * 2 * Math.PI) / 3;
          const x = cx + Math.cos(a) * radius;
          const y = cy + Math.sin(a) * radius;
          if (i === 0) ctx.moveTo(x, y);
          else ctx.lineTo(x, y);
        }
        ctx.closePath();
      }
      ctx.fill();

      out.push({
        label,
        colour: hueName(shapeHue),
        pixels: pixelsFromCanvas(canvas),
        png: canvas.toDataURL('image/png'),
        name: `${label}-${n + 1}.png`,
      });
    }
  }
  return out;
}

/**
 * Some of the drawn pictures, handed over as files.
 *
 * Practising must not become a second way through the page. If the drawn pictures went
 * straight into the model they would skip the whole loop — the queue, the picture being
 * searched for faces, one box named at a time — and what got practised would not be what
 * the page actually does when pictures arrive. So they leave here as files and are
 * handed to exactly the code an upload goes through.
 *
 * Taken at even intervals across the whole set rather than from the front: the set is
 * generated grouped by shape, so the first ten would be ten circles and there would be
 * nothing in them to tell apart.
 */
export function makeSampleFiles(count = 10, seed = 7): File[] {
  const all = makeSampleSet(30, seed);
  if (all.length === 0 || count < 1) return [];

  const step = all.length / count;
  const out: File[] = [];
  for (let i = 0; i < count; i++) {
    const item = all[Math.floor(i * step)];
    if (!item) continue;
    // The drawing, at its own full size, decoded back out of the data URL rather than
    // painted a second time — so it is the same picture that was measured, not another
    // like it. A PNG, and the name says so: a file called .png holding JPEG bytes is the
    // small lie that costs an hour later, and JPEG is what put the blur on these outlines
    // in the first place.
    const base64 = item.png.slice(item.png.indexOf(',') + 1);
    const bytes = Uint8Array.from(atob(base64), (ch) => ch.charCodeAt(0));
    out.push(new File([bytes], item.name, { type: 'image/png' }));
  }
  return out;
}
