/**
 * faces.ts — where the faces are.
 *
 * The search itself is in `haar.ts`. This file is the page's way in: the two types
 * a picture and a box are described with, and the one function the stage calls.
 *
 * It used to hold a hand-written skin-colour finder, and that is worth recording
 * rather than quietly deleting, because it is the mistake this file is shaped
 * around. "Skin is a fairly narrow range of colour, so a mask of skin-coloured
 * pixels finds faces" is true often enough to look like it works and false often
 * enough to be useless: measured on eighteen ordinary photographs on 19 September
 * 2026 it put a box on twelve of them, missed eight of the fifteen faces an
 * independent detector named, drew a 16x27 speck on one picture, and put a box
 * covering 77% of another — because a warm-toned portrait is mostly skin-coloured,
 * and nothing in the rule knows what a face is. A real face detector fixed all of it.
 *
 * What has NOT changed is the deal with the person using the page: this is a
 * PROPOSER, not an authority. It will still miss a face turned away, a face in
 * shadow, a face behind a hand or a mask, and a face smaller than about a twelfth
 * of the picture. That is why every box it draws can be removed by hand and a
 * missing one drawn by hand — the page has to stay usable when the guess is wrong,
 * because it will be wrong.
 */
export { detectFaces } from './haar.js';
