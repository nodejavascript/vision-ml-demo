/**
 * build-cascade.mjs — turn OpenCV's Haar cascade for frontal faces into a
 * TypeScript module this page can import, with no library and no download.
 *
 * WHAT IS BEING COPIED, AND WHY IT IS ALLOWED
 *
 * `haarcascade_frontalface_default.xml` is the stump-based 24x24 AdaBoost face
 * detector created by Rainer Lienhart and shipped with OpenCV. It carries the
 * Intel licence (reproduced beside the generated file and in `src/cascade-data.ts`),
 * which permits redistribution with the notice kept. The *detector* is not being
 * copied — only the trained numbers. `src/haar.ts` is written here, from the
 * published algorithm, and its arithmetic was checked against OpenCV 4.10 on the
 * same pictures before it was wired into the page.
 *
 * WHY THE NUMBERS ARE PACKED AND BASE64'd
 *
 * The XML is 930 KB of pretty-printed decimal, and it would have to be fetched,
 * served with the right content type and kept in step with the build. Emitted as
 * typed arrays it is about 160 KB, it loads with the module, and — the useful
 * part — every threshold and leaf value goes back through `Float32Array`, which
 * is the type OpenCV itself stores them in. So the detector in the page does the
 * same arithmetic on the same numbers, rather than on a decimal reading of them.
 *
 *   node tools/build-cascade.mjs [path-to-xml]
 */
import { readFile, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const DEFAULT_XML = '/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml';
const FALLBACK_URL =
  'https://raw.githubusercontent.com/opencv/opencv/4.x/data/haarcascades/haarcascade_frontalface_default.xml';

const here = dirname(fileURLToPath(import.meta.url));
const OUT = join(here, '..', 'src', 'cascade-data.ts');

/** Read the cascade XML from disk, or from the OpenCV repository if it is not installed. */
async function readCascade(path) {
  if (existsSync(path)) return readFile(path, 'utf8');
  process.stderr.write(`no cascade at ${path} — fetching OpenCV's copy\n`);
  const res = await fetch(FALLBACK_URL);
  if (!res.ok) throw new Error(`could not fetch the cascade: HTTP ${res.status}`);
  return res.text();
}

/** Everything between the open tag of `name` and its close tag. */
function section(body, name) {
  const open = body.indexOf(`<${name}>`);
  const close = body.indexOf(`</${name}>`);
  if (open < 0 || close < 0) throw new Error(`no <${name}> section in the cascade`);
  return body.slice(open + name.length + 2, close);
}

/** The licence block at the top of the file, kept so it can be written beside the data. */
function licence(text) {
  const open = text.indexOf('<!--');
  const close = text.indexOf('-->');
  return text.slice(open + 4, close).trim();
}

function parse(text) {
  const head = text.slice(text.indexOf('<cascade'));
  const body = head.slice(head.indexOf('>') + 1);

  const height = Number(/<height>(\d+)<\/height>/.exec(body)?.[1]);
  const width = Number(/<width>(\d+)<\/width>/.exec(body)?.[1]);
  const stageNum = Number(/<stageNum>(\d+)<\/stageNum>/.exec(body)?.[1]);

  // Features: each is a list of rectangles (x, y, w, h, weight), plus a `tilted`
  // flag that the default cascade leaves out entirely and which means 0 when absent.
  const rectXYWH = [];
  const rectWeight = [];
  const featureRectFirst = [];
  const featureRectCount = [];
  const featureRe = /<rects>([\s\S]*?)<\/rects>(\s*<tilted>([\s\S]*?)<\/tilted>)?/g;
  for (const chunk of section(body, 'features').matchAll(featureRe)) {
    if (chunk[3] !== undefined && chunk[3].trim() !== '0') {
      throw new Error('this cascade uses tilted features, which this port does not read');
    }
    featureRectFirst.push(rectXYWH.length / 4);
    let count = 0;
    for (const nums of chunk[1].matchAll(/<_\s*>([\s\S]*?)<\/_>/g)) {
      const v = nums[1].trim().split(/\s+/).map(Number);
      if (v.length !== 5) throw new Error(`a rectangle with ${v.length} numbers`);
      rectXYWH.push(v[0], v[1], v[2], v[3]);
      rectWeight.push(v[4]);
      count += 1;
    }
    featureRectCount.push(count);
  }

  // Stages: each holds a threshold and a list of weak classifiers, every one a stump.
  const stageThreshold = [];
  const stageStumpFirst = [];
  const stageStumpCount = [];
  const stumpFeature = [];
  const stumpThreshold = [];
  const stumpLeft = [];
  const stumpRight = [];

  const stagesText = section(body, 'stages');
  const stageChunks = stagesText.split('<stageThreshold>').slice(1);
  for (const chunk of stageChunks) {
    stageThreshold.push(Number(chunk.slice(0, chunk.indexOf('<'))));
    stageStumpFirst.push(stumpFeature.length);
    let count = 0;
    const weak = /<weakClassifiers>([\s\S]*?)<\/weakClassifiers>/.exec(chunk);
    if (!weak) throw new Error('a stage with no weak classifiers');
    for (const wc of weak[1].matchAll(/<internalNodes>([\s\S]*?)<\/internalNodes>\s*<leafValues>([\s\S]*?)<\/leafValues>/g)) {
      const nodes = wc[1].trim().split(/\s+/);
      const leaves = wc[2].trim().split(/\s+/).map(Number);
      // A stump is four numbers: a left and a right branch that the reader ignores,
      // the feature it tests, and its threshold. The two leaves are the answers.
      if (nodes.length !== 4) throw new Error(`a weak classifier with ${nodes.length / 4} nodes`);
      if (leaves.length !== 2) throw new Error(`a weak classifier with ${leaves.length} leaves`);
      stumpFeature.push(Number(nodes[2]));
      stumpThreshold.push(Number(nodes[3]));
      stumpLeft.push(leaves[0]);
      stumpRight.push(leaves[1]);
      count += 1;
    }
    stageStumpCount.push(count);
  }

  if (stageThreshold.length !== stageNum) {
    throw new Error(`stageNum says ${stageNum}, the file holds ${stageThreshold.length}`);
  }
  return {
    width, height,
    stageThreshold, stageStumpFirst, stageStumpCount,
    stumpFeature, stumpThreshold, stumpLeft, stumpRight,
    featureRectFirst, featureRectCount, rectXYWH, rectWeight,
  };
}

const b64 = {
  f32: (a) => Buffer.from(new Float32Array(a).buffer).toString('base64'),
  i32: (a) => Buffer.from(new Int32Array(a).buffer).toString('base64'),
  u8: (a) => Buffer.from(new Uint8Array(a)).toString('base64'),
};

const path = process.argv[2] || DEFAULT_XML;
const text = await readCascade(path);
const c = parse(text);

const file = `/**
 * cascade-data.ts — GENERATED. Do not edit by hand.
 *
 *   node tools/build-cascade.mjs [path-to-xml]
 *
 * OpenCV's stump-based 24x24 frontal-face cascade, created by Rainer Lienhart,
 * packed as typed arrays and base64'd so the page can import it directly: no
 * fetch, no asset, no content type to get wrong. Every threshold and leaf value
 * is stored as a \`Float32\`, which is the type OpenCV keeps them in, so the
 * arithmetic here matches OpenCV's rather than a decimal reading of it.
 *
 * ${c.stageThreshold.length} stages · ${c.stumpFeature.length} stumps · ${c.featureRectFirst.length} features

 * · ${c.rectXYWH.length / 4} rectangles. The detector that reads this is
 * \`src/haar.ts\`, written from the published algorithm and checked against
 * OpenCV 4.10 before it was wired into the page.
 *
 * LICENCE — the cascade is OpenCV's, and the notice is kept as its licence requires:
 *
${licence(text).split('\n').map((line) => ` *   ${line}`.trimEnd()).join('\n')}
 */

export const CASCADE_WIDTH = ${c.width};
export const CASCADE_HEIGHT = ${c.height};
export const CASCADE_NORM = { x: 1, y: 1, w: ${c.width} - 2, h: ${c.height} - 2 };

/** Per stage: the score a window must reach to be carried to the next stage. */
export const STAGE_THRESHOLD = ${JSON.stringify(b64.f32(c.stageThreshold))};
/** Per stage: where its stumps start, and how many there are. */
export const STAGE_STUMP_FIRST = ${JSON.stringify(b64.i32(c.stageStumpFirst))};
export const STAGE_STUMP_COUNT = ${JSON.stringify(b64.i32(c.stageStumpCount))};
/** Per stump: the feature it tests, the threshold, and the two leaf values. */
export const STUMP_FEATURE = ${JSON.stringify(b64.i32(c.stumpFeature))};
export const STUMP_THRESHOLD = ${JSON.stringify(b64.f32(c.stumpThreshold))};
export const STUMP_LEFT = ${JSON.stringify(b64.f32(c.stumpLeft))};
export const STUMP_RIGHT = ${JSON.stringify(b64.f32(c.stumpRight))};
/** Per feature: where its rectangles start, and how many there are. */
export const FEATURE_RECT_FIRST = ${JSON.stringify(b64.i32(c.featureRectFirst))};
export const FEATURE_RECT_COUNT = ${JSON.stringify(b64.i32(c.featureRectCount))};
/** Per rectangle: x, y, w, h as bytes (a 24-pixel window needs no more), then the weight. */
export const RECT_XYWH = ${JSON.stringify(b64.u8(c.rectXYWH))};
export const RECT_WEIGHT = ${JSON.stringify(b64.f32(c.rectWeight))};
`;

await writeFile(OUT, file);

// 🔴 THE NOTICE IS WRITTEN FROM THE CASCADE'S OWN TEXT, NOT TYPED — 23 September 2026.
//
// The licence permits redistributing this cascade **with the notice kept**, and a notice kept only
// inside a GENERATED file is a notice a reader has to open the source to find. So the same block
// that goes into `src/cascade-data.ts` is written here as well, and it comes from the cascade's own
// comment rather than from anything typed into this repository — which is what stops the two from
// drifting apart.
const NOTICES = join(here, '..', 'THIRD-PARTY-NOTICES.md');
const asText = licence(text).replace(/`/g, '\\`');

const notices = `# Third-party notices

This project redistributes one thing it did not create. **The notice is kept as its licence
requires**, and it is kept twice on purpose: inside the generated file a reader may never open, and
here, where a reader looks.

## Rainer Lienhart's frontal-face cascade, as shipped with OpenCV

| | |
|---|---|
| what it is | a stump-based 24×24 AdaBoost frontal-face detector |
| where it came from | \`haarcascade_frontalface_default.xml\`, from OpenCV's \`data/haarcascades\` |
| what is copied | the **trained numbers only** — ${c.stageThreshold.length} stages, ${c.stumpFeature.length} features, ${c.rectXYWH.length / 4} rectangles, packed as typed arrays |
| what is not copied | the detector. \`src/haar.ts\` is written here from the published Viola-Jones algorithm, and its arithmetic was checked against OpenCV 4.10 before it was used |
| how it is regenerated | \`npm run cascade\` — \`node tools/build-cascade.mjs [path-to-xml]\`, which writes both \`src/cascade-data.ts\` and this file |
| why the numbers are packed | the XML is ~930 KB of pretty-printed decimal; emitted as typed arrays it is about 160 KB and every value goes back through the same \`Float32\` OpenCV stores it in |

### The licence, exactly as the cascade file carries it

\`\`\`
${asText}
\`\`\`
`;

await writeFile(NOTICES, notices);

const bytes = Buffer.byteLength(file);
process.stdout.write(
  `wrote ${OUT}\n` +
    `  ${c.stageThreshold.length} stages · ${c.stumpFeature.length} stumps · ` +
    `${c.featureRectFirst.length} features · ${c.rectXYWH.length / 4} rectangles\n` +
    `  ${(bytes / 1024).toFixed(0)} KB of TypeScript from ${(text.length / 1024).toFixed(0)} KB of XML\n` +
    `wrote ${NOTICES}\n`,
);
