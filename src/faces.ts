/**
 * faces.ts — where the faces are.
 *
 * This page has no server, no libraries and no downloaded weights, so the face
 * finder is built out of the picture itself: skin has a fairly narrow range of
 * colour regardless of who it belongs to, so a mask of "skin-coloured pixels" plus
 * a tidy-up and a search for blobs lands on faces often enough to be useful.
 *
 * It is a PROPOSER, not an authority. It will miss faces in black-and-white
 * photographs, in heavy shadow, and behind a mask or a hand; it will sometimes fire
 * on a wooden floor or a skin-coloured wall. That is why every box it draws can be
 * removed by hand and a missing one can be drawn by hand — the page must stay usable
 * when the guess is wrong, because it will be wrong.
 *
 * Two honest notes, stated because pretending otherwise would be worse than the
 * limitation:
 *
 *   - The colour rule is the standard one from the literature, and like the rest of
 *     that literature it was tuned mostly on lighter skin. It is measurably less
 *     reliable on darker skin in dim light.
 *   - It knows nothing about what a face IS. It cannot tell a face from a knee, and
 *     "one blob of skin" is all the structure it understands, so two people standing
 *     close together can come out as one box.
 */

/** A rectangle in the picture's own pixels. */
export interface Box {
  x: number;
  y: number;
  w: number;
  h: number;
}

/**
 * Raw RGB, three bytes per pixel, plus the size of the picture it was taken from.
 *
 * The source size is carried because the search runs on a smaller copy while every
 * box has to come back in the picture's own pixels — the space the boxes are drawn
 * and cropped in. Leaving that conversion to the caller is how they silently end up
 * measured against the wrong picture.
 */
export interface Frame {
  rgb: Uint8Array;
  width: number;
  height: number;
  sourceWidth: number;
  sourceHeight: number;
}

/** Nobody needs more boxes than this, and it bounds the work. */
const MOST_FACES = 6;

/** A blob smaller than this fraction of the picture is speckle, not a face. */
const LEAST_AREA = 0.004;

/** A blob bigger than this is a wall, not a face. */
const MOST_AREA = 0.62;

/** Faces are taller than they are wide, but a tilted head is wide too. */
const NARROWEST = 0.5;
const WIDEST = 1.9;

/**
 * How much of the box a blob has to fill.
 *
 * A face is roughly oval, so it covers about 0.6–0.8 of its own box. Loose bounds
 * because hair, a beard and a neck all change the shape — but a long diagonal arm
 * covers much less, and a solid rectangle of wood covers nearly all of it.
 */
const LEAST_FILL = 0.42;
const MOST_FILL = 0.94;

/**
 * Is this pixel skin-coloured?
 *
 * Judged in YCbCr rather than RGB, because that is where skin clusters regardless
 * of how bright the light was: the blue-difference and red-difference channels of
 * skin stay in a narrow band while the RGB values move all over the place.
 * A white wall fails it (its Cb is neutral, 128) and so does a shadow.
 */
function isSkin(r: number, g: number, b: number): boolean {
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  const y = 0.299 * r + 0.587 * g + 0.114 * b;
  return cb >= 77 && cb <= 127 && cr >= 133 && cr <= 173 && y > 45 && y < 250;
}

/** One 3x3 pass that grows the set pixels outward. */
function grow(src: Uint8Array, width: number, height: number, dst: Uint8Array): void {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let on = 0;
      for (let dy = -1; dy <= 1 && on === 0; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          if (src[ny * width + nx] !== 0) {
            on = 1;
            break;
          }
        }
      }
      dst[y * width + x] = on;
    }
  }
}

/** One 3x3 pass that pulls the set pixels inward. */
function shrink(src: Uint8Array, width: number, height: number, dst: Uint8Array): void {
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      let on = 1;
      for (let dy = -1; dy <= 1 && on === 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) {
          on = 0;
          break;
        }
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) {
            on = 0;
            break;
          }
          if (src[ny * width + nx] === 0) {
            on = 0;
            break;
          }
        }
      }
      dst[y * width + x] = on;
    }
  }
}

interface Blob {
  left: number;
  top: number;
  right: number;
  bottom: number;
  count: number;
}

/**
 * One pass of "grow then shrink".
 *
 * Growing first closes the gaps skin has inside it — an open mouth, eyes behind
 * glasses, a nostril shadow — so a face becomes one solid blob instead of a ring of
 * fragments. Shrinking afterwards puts the outline back where it was rather than
 * leaving every blob fattened. Deliberately not the other way round: growing after
 * shrinking is "open", which erases small faces entirely.
 */
function closeHoles(mask: Uint8Array, width: number, height: number): void {
  const scratch = new Uint8Array(mask.length);
  grow(mask, width, height, scratch);
  grow(scratch, width, height, mask);
  shrink(mask, width, height, scratch);
  shrink(scratch, width, height, mask);
}

/**
 * Find the solid blobs of a mask.
 *
 * A stack rather than recursion: a blob can be tens of thousands of pixels and the
 * call stack would not survive it.
 */
function blobsOf(mask: Uint8Array, width: number, height: number): Blob[] {
  const seen = new Uint8Array(mask.length);
  const stack = new Int32Array(mask.length);
  const blobs: Blob[] = [];

  for (let start = 0; start < mask.length; start++) {
    if (mask[start] === 0 || seen[start] !== 0) continue;

    let top = 0;
    stack[top++] = start;
    seen[start] = 1;
    const blob: Blob = { left: width, top: height, right: -1, bottom: -1, count: 0 };

    while (top > 0) {
      const index = stack[--top];
      const x = index % width;
      const y = (index - x) / width;
      blob.count += 1;
      if (x < blob.left) blob.left = x;
      if (x > blob.right) blob.right = x;
      if (y < blob.top) blob.top = y;
      if (y > blob.bottom) blob.bottom = y;

      // Eight neighbours, not four: an eight-connected blob is what a diagonal
      // edge of a cheek looks like, and four-connectivity would saw it in two.
      for (let dy = -1; dy <= 1; dy++) {
        const ny = y + dy;
        if (ny < 0 || ny >= height) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx;
          if (nx < 0 || nx >= width) continue;
          const next = ny * width + nx;
          if (mask[next] !== 0 && seen[next] === 0) {
            seen[next] = 1;
            stack[top++] = next;
          }
        }
      }
    }
    blobs.push(blob);
  }
  return blobs;
}

/** How much two boxes cover each other, as a share of the smaller one. */
function overlap(a: Box, b: Box): number {
  const left = Math.max(a.x, b.x);
  const top = Math.max(a.y, b.y);
  const right = Math.min(a.x + a.w, b.x + b.w);
  const bottom = Math.min(a.y + a.h, b.y + b.h);
  if (right <= left || bottom <= top) return 0;
  const shared = (right - left) * (bottom - top);
  return shared / Math.min(a.w * a.h, b.w * b.h);
}

/**
 * The faces in a picture, left to right.
 *
 * Empty when there are none — which is a real answer, not a failure, and the page
 * treats it as "name the whole picture instead".
 */
export function detectFaces(frame: Frame): Box[] {
  const { rgb, width, height } = frame;
  if (width < 16 || height < 16) return [];

  const mask = new Uint8Array(width * height);
  for (let i = 0, p = 0; i < mask.length; i++, p += 3) {
    mask[i] = isSkin(rgb[p], rgb[p + 1], rgb[p + 2]) ? 1 : 0;
  }

  closeHoles(mask, width, height);

  const pictureArea = width * height;
  const found: Box[] = [];

  for (const blob of blobsOf(mask, width, height)) {
    const w = blob.right - blob.left + 1;
    const h = blob.bottom - blob.top + 1;
    const boxArea = w * h;
    if (boxArea <= 0) continue;

    const share = boxArea / pictureArea;
    if (share < LEAST_AREA || share > MOST_AREA) continue;

    const shape = w / h;
    if (shape < NARROWEST || shape > WIDEST) continue;

    const fill = blob.count / boxArea;
    if (fill < LEAST_FILL || fill > MOST_FILL) continue;

    // Flesh the box out a little. The mask covers skin, and a face is hair and jaw
    // as well — but not much: the mask is usually most of the face already, so a
    // wide margin would put a box around somebody's shoulders.
    const padX = Math.round(w * 0.12);
    const padY = Math.round(h * 0.12);
    found.push({
      x: Math.max(0, blob.left - padX),
      y: Math.max(0, blob.top - padY),
      w: Math.min(width, blob.right + padX) - Math.max(0, blob.left - padX) + 1,
      h: Math.min(height, blob.bottom + padY) - Math.max(0, blob.top - padY) + 1,
    });
  }

  // Two blobs on one face — split by a pair of sunglasses, say — come back together
  // as the larger of the two, since they are the same person.
  found.sort((a, b) => b.w * b.h - a.w * a.h);
  const kept: Box[] = [];
  for (const box of found) {
    if (kept.some((other) => overlap(other, box) > 0.35)) continue;
    kept.push(box);
    if (kept.length === MOST_FACES) break;
  }

  // Left to right, so naming them one at a time goes across the picture the way a
  // person would point at them.
  kept.sort((a, b) => a.x - b.x);

  // Back out of the search copy and into the picture's own pixels.
  const backX = width === 0 ? 1 : frame.sourceWidth / width;
  const backY = height === 0 ? 1 : frame.sourceHeight / height;
  return kept.map((box) => ({
    x: Math.round(box.x * backX),
    y: Math.round(box.y * backY),
    w: Math.round(box.w * backX),
    h: Math.round(box.h * backY),
  }));
}
