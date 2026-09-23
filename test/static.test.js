import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync, existsSync, readdirSync, statSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

/**
 * Unit tests for the vision-ml-demo page.
 *
 * These earn their place on the things only reading can settle: that no Google tag is in
 * the page, that the title IS the host, that the footer door is delegated, that the policy
 * is a section rather than a page of its own. Behaviour is the end-to-end suite's job and
 * delivery is the live check's; a unit test that tries to prove behaviour is a test that
 * passes for the wrong reason.
 */

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const SITE = join(ROOT, 'site');
const read = (name) => readFileSync(join(SITE, name), 'utf8');

const HOST = 'vision-ml-demo.nodejavascript.com';
const html = read('index.html');
const css = read('styles.css');
const gate = read('consent.js');
const app = read('app.js');
const charts = read('charts.js');
const samples = read('samples.js');
const manifest = JSON.parse(read('manifest.webmanifest'));

/**
 * The practise set's two numbers, read out of the emitted module — the one place they are
 * declared — so every check that talks about the set agrees with the set rather than with a
 * sentence somebody wrote about it.
 */
const practiseCount = Number(samples.match(/PRACTISE_COUNT = (\d+)/)[1]);
const practiseKinds = samples.match(/SHAPE_NAMES = \[([\s\S]*?)\];/)[1].match(/'[a-z]+'/g).length;

const NUMBER_WORDS = {
  one: 1, two: 2, three: 3, four: 4, five: 5, six: 6, seven: 7, eight: 8, nine: 9, ten: 10,
  eleven: 11, twelve: 12, fifteen: 15, eighteen: 18, twenty: 20, twentyfive: 25, thirty: 30,
  thirtyfive: 35, forty: 40, fifty: 50, sixty: 60, seventy: 70, eighty: 80, ninety: 90,
};

/** A written number as a number — `null` for a word that is not a number at all. */
const numberIn = (word) => {
  const w = String(word).toLowerCase().replace(/[^a-z0-9]/g, '');
  if (/^\d+$/.test(w)) return Number(w);
  return NUMBER_WORDS[w] ?? null;
};

/** Strip comments before reading code, so a doc comment can never be read as a fault. */
const withoutComments = (text) =>
  text.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');

/** The page without its HTML comments — the notes in this file discuss the very tags the
    checks look for, and a check that reads its own documentation fails for the wrong
    reason. A false failure is the expensive kind: it teaches the reader to distrust it. */
const pageOnly = (text) => text.replace(/<!--[\s\S]*?-->/g, '');

const htmlNoComments = pageOnly(html);

/* ------------------------------------------------------------------ *
 * Privacy: nothing the visitor brings can leave the page
 * ------------------------------------------------------------------ */

test('no client-side file can send anything anywhere', () => {
  // The privacy section makes this claim in as many words — "no request in the model code
  // could carry a picture anywhere" — so it is asserted rather than believed. Everything
  // the page does happens in this tab: the pictures are read with `createImageBitmap`, the
  // model is trained in a worker, and the memory is kept in IndexedDB. There is no server
  // behind this page to send anything to.
  const clientFiles = readdirSync(SITE).filter((name) => name.endsWith('.js'));
  assert.ok(clientFiles.length >= 8, `expected the built modules, found ${clientFiles.length}`);
  const forbidden = [/\bfetch\s*\(/, /XMLHttpRequest/, /sendBeacon/, /new\s+WebSocket/, /EventSource/, /importScripts\s*\(/];
  for (const file of clientFiles) {
    const source = withoutComments(read(file));
    for (const pattern of forbidden) {
      assert.ok(!pattern.test(source), `${file} must not contain ${pattern}`);
    }
  }
});

test('the page loads NOTHING from another origin, and no Google tag sits in it', () => {
  const loads = [...htmlNoComments.matchAll(/<(script|link)\b[^>]*>/g)]
    .map((m) => m[0])
    .filter((tag) => !/rel="(canonical|alternate|preconnect|dns-prefetch)"/.test(tag));
  const remote = loads
    .map((tag) => (tag.match(/(?:src|href)="(https?:\/\/[^"]+)"/) || [])[1])
    .filter(Boolean);
  assert.deepEqual(remote, [], `the page loads something remote: ${remote.join(', ')}`);
  assert.ok(
    !/googletagmanager|google-analytics\.com|googleapis|fonts\.google/.test(htmlNoComments),
    'nothing from Google may sit in the page — the gate is the only thing that may add it',
  );
});

test('the measurement id rides on the gate, which is the only thing allowed to load Google', () => {
  const tag = html.match(/<script[^>]*data-ga-id="([^"]+)"[^>]*>/);
  assert.ok(tag, 'the consent script must carry the measurement id as an attribute');
  assert.match(tag[1], /^G-[A-Z0-9]{6,}$/, `that does not look like a measurement id: ${tag[1]}`);
  assert.match(tag[0], /consent\.js/, 'the id must ride on the gate itself');
  assert.match(gate, /googletagmanager\.com\/gtag\/js/, 'the gate is what appends the tag');
});

/* ------------------------------------------------------------------ *
 * The page and the script have to agree
 * ------------------------------------------------------------------ */

test('every id the cookie gate looks up exists in the page', () => {
  const wanted = new Set([
    ...[...gate.matchAll(/getElementById\('([A-Za-z0-9_-]+)'\)/g)].map((m) => m[1]),
    ...[...gate.matchAll(/closest\s*\??\.?\s*\(\s*'#([A-Za-z0-9_-]+)'/g)].map((m) => m[1]),
  ]);
  for (const id of ['consentAccept', 'consentDecline', 'consentAnalytics', 'consentBtn', 'consentBar']) {
    assert.ok(wanted.has(id), `the gate no longer wires #${id}`);
  }
  const missing = [...wanted].filter((id) => !html.includes(`id="${id}"`));
  assert.deepEqual(missing, [], `index.html is missing: ${missing.join(', ')}`);
});

test('the footer door is DELEGATED, not bound when the script loads', () => {
  // 🔴 This shipped as a bug once, on a React site in the family: the gate bound
  // `#consentBtn` with addEventListener as the script ran, the footer did not exist at that
  // moment, and the button rendered correctly in every screenshot and did nothing at all.
  // On this site both forms work, so no browser can tell them apart — the only place to
  // hold the rule is the source, which is what this reads.
  assert.match(gate, /document\.addEventListener\(\s*'click'/, 'the door must be listened for on the document');
  assert.match(gate, /closest\s*\??\.?\s*\(\s*'#consentBtn'/, 'the door must be matched as the click rises');
  assert.doesNotMatch(
    gate,
    /getElementById\('consentBtn'\)/,
    'a delegated gate must not look the footer door up by id at all',
  );
});

test('every element id the page looks up exists in the markup', () => {
  const pageIds = new Set([...html.matchAll(/id="([A-Za-z0-9_-]+)"/g)].map((m) => m[1]));
  const wanted = new Set([...app.matchAll(/element\('([A-Za-z0-9_-]+)'\)/g)].map((m) => m[1]));
  assert.ok(wanted.size >= 25, `expected the page's own element list, found ${wanted.size}`);
  const missing = [...wanted].filter((id) => !pageIds.has(id));
  assert.deepEqual(missing, [], `app.js looks up an id that is not in the page: ${missing.join(', ')}`);
});

/* ------------------------------------------------------------------ *
 * Standard part 1 — the title IS the host, and there is no .html anywhere
 * ------------------------------------------------------------------ */

test('the html title IS the full domain name', () => {
  const title = html.match(/<title>([^<]*)<\/title>/)[1];
  assert.equal(title, HOST, 'the title must BE the host, not contain it');
  assert.equal(manifest.short_name, 'vision-ml-demo');
});

test('the canonical, the og:url and the sitemap all agree, and none of them ends in .html', () => {
  assert.equal(html.match(/<link rel="canonical" href="([^"]+)"/)[1], `https://${HOST}/`);
  assert.equal(html.match(/<meta property="og:url" content="([^"]+)"/)[1], `https://${HOST}/`);
  const sitemap = read('sitemap.xml');
  const locs = [...sitemap.matchAll(/<loc>([^<]+)<\/loc>/g)].map((m) => m[1]);
  assert.deepEqual(locs, [`https://${HOST}/`], 'one document, one URL');
  for (const href of [...html.matchAll(/href="([^"]+)"/g)].map((m) => m[1])) {
    assert.ok(!/\.html?($|[?#])/.test(href), `an href points at an .html URL: ${href}`);
  }
});

test('there is no privacy page anywhere — the policy is a section of the one page', () => {
  const files = readdirSync(SITE);
  assert.ok(!files.some((name) => /^privacy/i.test(name)), 'no privacy.* file may exist');
  assert.match(html, /<section class="card" id="privacy">/, 'the policy must be a section of the page');
  assert.match(html, /<a href="#privacy">Privacy<\/a>/, 'the footer must link to that section');
  const repoFiles = [];
  const walk = (dir) => {
    for (const entry of readdirSync(dir, { withFileTypes: true })) {
      if (entry.name === 'node_modules' || entry.name === '.git') continue;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) walk(full);
      else repoFiles.push(full);
    }
  };
  walk(ROOT);
  const stray = repoFiles.filter((file) => /(^|\/)privacy\.[a-z]+$/i.test(file));
  assert.deepEqual(stray, [], `a privacy page exists: ${stray.join(', ')}`);
});

/* ------------------------------------------------------------------ *
 * Standard parts 3 and 4 — the bar, the footer, and no nav
 * ------------------------------------------------------------------ */

test('the header carries the brand and nothing else, and the brand goes to THIS site', () => {
  const header = html.match(/<header class="top">[\s\S]*?<\/header>/)[0];
  assert.ok(!/<nav[\s>]/.test(header), 'the header carries no nav (standard part 3)');
  assert.equal(header.match(/<a class="brand" href="([^"]+)"/)[1], '/', 'the brand goes to self, not to the parent');
  assert.match(header, /<b>vision-ml-demo<\/b>/);
  assert.match(header, /<small>nodejavascript\.com<\/small>/);
  // The bar carries the FAMILY's mark (five circles); the favicon is the site's own.
  assert.equal((header.match(/<circle /g) ?? []).length, 5, 'the bar shows the family mark');
});

test('the footer is brand · links · copyright, with the mother-site link exactly once', () => {
  const footer = pageOnly(html).match(/<footer class="site-footer">[\s\S]*?<\/footer>/)[0];
  assert.equal((footer.match(/href="https:\/\/nodejavascript\.com\/"/g) ?? []).length, 1, 'the link home appears once, on the brand line');
  assert.match(footer, /© 2026 vision-ml-demo\.nodejavascript\.com\. All rights reserved\./);
  assert.match(footer, /id="consentBtn"/, 'the footer carries the door to the cookie answer');
  assert.ok(!/back to top/i.test(footer), 'there is no Back to top in a footer');
  // ⚠️ THIS SAID "the repository is private, so nothing links it" — the repository went PUBLIC on
  // 23 September 2026, so the reason changed while the rule did not: **this page still links no
  // repository.** The project's card on `nodejavascript.com` carries the source link (part 11c); a
  // visitor who came to teach it a picture is not sent to a git host to use it.
  assert.ok(!/gitlab\.com|github\.com/.test(footer), 'the demo page links no repository — the card does');
});

test('the two cookie answers carry the same class, so neither can be made easier than the other', () => {
  const accept = html.match(/<button id="consentAccept" class="([^"]*)"/)[1];
  const decline = html.match(/<button id="consentDecline" class="([^"]*)"/)[1];
  assert.equal(accept, decline, 'the answers must be styled identically');
  assert.match(html, /id="consentAnalytics"[^>]*aria-checked="false"/, 'the switch starts off');
  assert.ok(!/count my visits|this device/i.test(html), "the owner's counting switch is not in the panel");
});

/* ------------------------------------------------------------------ *
 * Standard part 5 — this site's identity, and the abstract that is drawn
 * ------------------------------------------------------------------ */

test('the theme colour and the page accent are the same colour', () => {
  // A relationship rather than a pinned value: a check that holds the literal goes stale the
  // moment the identity is re-claimed, and a check on the relationship cannot.
  const accent = css.match(/--accent:\s*([^;]+);/)[1].trim();
  assert.equal(html.match(/<meta name="theme-color" content="([^"]+)"/)[1], accent);
  assert.equal(manifest.theme_color, accent, 'the manifest must agree with the tab');
});

test('the bar is the page’s OWN near-black at 0.82 alpha', () => {
  const bg = css.match(/--bg:\s*#([0-9a-f]{6});/i)[1];
  const rgb = [0, 2, 4].map((offset) => parseInt(bg.slice(offset, offset + 2), 16));
  const bar = css.match(/\.top \{[\s\S]*?background: rgba\(([^)]+)\)/)[1].replace(/\s/g, '');
  assert.equal(bar, `${rgb.join(',')},0.82`, 'the bar must be the page’s own colour, not another site’s');
  assert.equal(manifest.background_color, `#${bg}`, 'the manifest must agree with the page');
});

test('the background abstract is a drawing in the page, masked so it has no edge', () => {
  // Standard part 5d-ii, and the half the family missed: nine of ten sites painted a
  // gradient wash and no geometry, and every one of them passed a check that counted
  // gradients. So this asserts BOTH halves — that a texture is drawn, and that it is
  // painted by an element the page actually carries.
  assert.match(html, /<div class="dvs-pattern" aria-hidden="true"><\/div>/, 'the layer must be in the page');
  const pattern = css.match(/\.dvs-pattern \{[\s\S]*?\n\}/)[0];
  assert.match(pattern, /conic-gradient\(/, 'the drawing must be geometry, not a second glow');
  assert.match(pattern, /background-size:\s*\d+px \d+px/, 'a checkerboard is a tiled drawing');
  assert.match(pattern, /mask-image:/, 'it must fade, or it has an edge and an edge is a band');
  assert.match(pattern, /-webkit-mask-image:/, 'both masks, for the engines that need the prefix');
  assert.match(pattern, /position: fixed/, 'its own fixed layer, never the base gradient stack');
  assert.match(pattern, /pointer-events: none/);
});

/* ------------------------------------------------------------------ *
 * Standard parts 2 and 7 — the mechanics a screenshot cannot check
 * ------------------------------------------------------------------ */

test('`[hidden]` beats any display rule', () => {
  // The gate swaps the question for the panel by setting `hidden`, and two of the views are
  // laid out with flex — so without this the ask stays on screen underneath its own panel.
  assert.match(css, /\[hidden\] \{ display: none !important; \}/);
});

test('the page reserves the cookie bar’s height, so the footer stays clickable', () => {
  assert.match(css, /padding-bottom: var\(--consent-height, 0px\)/);
  assert.match(gate, /--consent-height/, 'the gate measures the bar and writes the reservation');
});

/* ------------------------------------------------------------------ *
 * SEO and the icon set (parts 14 and 15b)
 * ------------------------------------------------------------------ */

test('titles and descriptions are inside the house limits', () => {
  const description = html.match(/<meta\s+name="description"\s+content="([^"]*)"/)[1];
  assert.ok(
    description.length >= 120 && description.length <= 160,
    `the description is ${description.length} characters`,
  );
  const og = html.match(/<meta\s+property="og:description"\s+content="([^"]*)"/)[1];
  assert.ok(og.length <= 200, `the social description is ${og.length} characters`);
});

test('every social and structured-data tag is present and parses', () => {
  for (const tag of [
    'og:type',
    'og:site_name',
    'og:title',
    'og:description',
    'og:url',
    'og:image',
    'twitter:card',
    'twitter:title',
    'twitter:description',
    'twitter:image',
  ]) {
    assert.ok(new RegExp(`(property|name)="${tag}"`).test(html), `${tag} is missing`);
  }
  assert.equal(html.match(/<meta property="og:site_name" content="([^"]+)"/)[1], HOST);
  const json = JSON.parse(html.match(/<script type="application\/ld\+json">([\s\S]*?)<\/script>/)[1]);  const kinds = json['@graph'].map((node) => node['@type']);
  assert.deepEqual(kinds, ['WebApplication', 'FAQPage']);
  for (const question of json['@graph'][1].mainEntity) {
    assert.ok(question.acceptedAnswer.text.length > 40, 'every FAQ answer must say something');
  }
  assert.match(html, /<img id="picture" alt="[^"]+"/, 'the picture needs alt text');
});

/** The width and height inside a PNG's IHDR, and whether it carries an alpha channel. */
function pngInfo(file) {
  const bytes = readFileSync(join(SITE, file));
  assert.equal(bytes.subarray(0, 8).toString('hex'), '89504e470d0a1a0a', `${file} is not a PNG`);
  const colourType = bytes[25];
  return {
    width: bytes.readUInt32BE(16),
    height: bytes.readUInt32BE(20),
    hasAlpha: colourType === 4 || colourType === 6,
  };
}

test('the icon set is complete, the right size, and the apple icon is opaque', () => {
  assert.ok(existsSync(join(SITE, 'favicon.svg')), 'favicon.svg is missing');
  assert.equal(pngInfo('favicon-32.png').width, 32);
  assert.equal(pngInfo('favicon-32.png').height, 32);
  const apple = pngInfo('apple-touch-icon.png');
  assert.equal(apple.width, 180);
  assert.equal(apple.height, 180);
  // iOS paints transparency black, so a transparent icon arrives on a home screen with
  // black wedges. Measured, not assumed: the colour type is read out of the file.
  assert.equal(apple.hasAlpha, false, 'the apple icon must have no alpha channel');

  const ico = readFileSync(join(SITE, 'favicon.ico'));
  assert.equal(ico.readUInt16LE(0), 0, 'the .ico signature');
  assert.equal(ico.readUInt16LE(2), 1, 'the .ico type');
  const count = ico.readUInt16LE(4);
  const sizes = Array.from({ length: count }, (_, i) => ico[6 + i * 16]);
  assert.ok(sizes.includes(16) && sizes.includes(32) && sizes.includes(48), `the .ico holds ${sizes.join('/')}`);

  for (const icon of manifest.icons) {
    const file = icon.src.replace(/^\//, '');
    assert.ok(existsSync(join(SITE, file)), `the manifest names a file that is not there: ${file}`);
  }
  const pngIcons = manifest.icons.filter((icon) => icon.src.endsWith('.png'));
  for (const icon of pngIcons) {
    const [w, h] = icon.sizes.split('x').map(Number);
    const info = pngInfo(icon.src.replace(/^\//, ''));
    assert.ok(info.width === w && info.height === h, `${icon.src} claims ${icon.sizes} and measures ${info.width}x${info.height}`);
  }
});

test('robots.txt points at the sitemap, and the sitemap is on disk', () => {
  const robots = read('robots.txt');
  assert.match(robots, new RegExp(`^Sitemap: https://${HOST}/sitemap\\.xml$`, 'm'));
  assert.ok(statSync(join(SITE, 'sitemap.xml')).size > 100);
});

/* ------------------------------------------------------------------ *
 * The practise set, and the words the page uses
 * ------------------------------------------------------------------ */

test('the practise set is thirty pictures of four hard shapes, several of each', () => {
  // Read out of the emitted module rather than the TypeScript, so a set that fails to
  // compile or an export that is dropped is caught here rather than in the browser.
  const shapes = samples.match(/SHAPE_NAMES = \[([\s\S]*?)\];/)[1].match(/'[a-z]+'/g).map((s) => s.slice(1, -1));
  // 🔴 FOUR SHAPES AND THIRTY PICTURES ARE ONE DECISION, NOT TWO (George, 23 September 2026:
  // *"there is not enough of similar shapes to learn from, make it 30 and only 4 different but
  // difficult shapes"*). Twenty across ten gave two examples each, which is not a set to learn
  // from; thirty across four gives seven or eight, which is. Both numbers are asserted together
  // here on purpose: a later change that moved one without the other would quietly undo the fix.
  assert.equal(shapes.length, 4, `the set draws four kinds of shape, found ${shapes.length}`);
  assert.deepEqual(shapes, ['star', 'moon', 'heart', 'arrow'],
    'the four are the hard ones — concave, curved or directional, so the answer is not readable off the drawing');
  assert.deepEqual([...new Set(shapes)], shapes, 'no shape may appear twice in the list');
  assert.equal(samples.match(/PRACTISE_COUNT = (\d+)/)[1], '30');
  // ⚠️ `\s+` RATHER THAN A SPACE: `tsc` re-prints a long statement onto two lines, so an
  // assertion that pinned the formatting of the TypeScript failed on the JavaScript that ships.
  // The check is about the pool being built by cycling, and the line break is not the rule.
  assert.match(samples, /while \(pool\.length < count\)\s+pool\.push\(SHAPE_NAMES\[pool\.length % SHAPE_NAMES\.length\]\)/,
    'the pool is built by cycling the shapes, so thirty across four comes out even');
  assert.match(samples, /'image\/png'/, 'a drawn shape handed over as a JPEG arrives fringed with colour');
  assert.match(samples, /ANALYSIS/, 'drawn at the width the search reads, so it is never resampled');
  assert.match(samples, /practise-\$\{/, 'the file names are the set’s own');
  assert.ok(/practise-\$\{/.test(samples), 'the file name must be built from the position alone');
  assert.ok(
    !/practise-\$\{[^}]*\}[^`]*shape/.test(samples),
    'a file name holding the shape would put the answer on screen'
  );
  // The six shapes that went are not merely unused — they are GONE from the drawing code, so a
  // name left in the list with no outline behind it cannot come back unnoticed.
  for (const gone of ['circle', 'square', 'triangle', 'diamond', 'cross', 'hexagon']) {
    assert.ok(!new RegExp(`case '${gone}'`).test(samples), `the retired shape is still drawn: ${gone}`);
  }
});

test('every count the page states about its own practise set agrees with the set', () => {
  // 🔴 THE DEFECT THIS EXISTS FOR, measured 23 September 2026. The set went from twenty pictures
  // of ten shapes to thirty of four; the module was changed, the visible prose was changed, and
  // the three `<head>` descriptions were NOT — so the page went on telling Google and every
  // sharing card "Twenty practise shapes are built in" while drawing thirty. The prose that
  // travels FURTHEST, into a search result nobody stands next to, was the one nothing read. The
  // practise-set test above pins the module; this one pins the words.
  //
  // The pairing is deliberately narrow so it cannot cry wolf: a NUMBER WORD (or digits)
  // immediately before "shapes" in a passage that is talking about the practise set. In "thirty
  // shapes of four kinds" the four is paired with "kinds", so the shape count is checked by its
  // own assertion below rather than guessed at here.
  // Every passage that LEAVES the page: the description Google shows, and the two sharing cards.
  const headCopy = [...html.matchAll(
    /<meta[^>]*(?:name|property)="(description|og:description|twitter:description)"[^>]*content="([^"]*)"/g,
  )].map((m) => ({ where: `the ${m[1]}`, text: m[2] }));
  assert.equal(headCopy.length, 3, 'the page carries three descriptions: search, Open Graph, Twitter');

  // And the passages a visitor reads.
  const prose = [...html.matchAll(/>([^<>{}]*(?:practise|Practise)[^<>{}]*)</g)]
    .map((m) => ({ where: 'the visible prose', text: m[1] }));

  const passages = [...headCopy, ...prose];
  assert.ok(prose.length > 0, 'the page must say somewhere what the practise set is');

  for (const { where, text } of passages) {
    for (const m of text.matchAll(/\b([A-Za-z]+|\d+)\s+(?:practise\s+)?shapes\b/gi)) {
      const said = numberIn(m[1]);
      if (said === null) continue; // "practise shapes", "drawn shapes" — not a count
      assert.equal(said, practiseCount,
        `${where} says "${m[0]}" and the set is ${practiseCount} — the number that travels furthest is the one to change`);
    }
  }

  // The count of KINDS is the other half of the same decision, and it is stated in words too.
  assert.ok(passages.some((p) => /\bshapes\b/.test(p.text)),
    'at least one passage must state the set outright, or this test passes by saying nothing');
  for (const m of html.matchAll(/\b([A-Za-z]+|\d+)\s+kinds\b/gi)) {
    const said = numberIn(m[1]);
    if (said === null) continue;
    assert.equal(said, practiseKinds, `the page says "${m[0]}" and the set draws ${practiseKinds}`);
  }
});

test('a count of the practise set that is HISTORY is marked as history', () => {
  // The same rule one layer down: a comment saying the set is twenty regenerates the stale copy
  // above the next time somebody reads it as current. A former value is worth keeping — the
  // reason the set changed is the whole argument for four shapes — so the rule is not "never
  // write twenty". It is "do not write it as though it were true now".
  //
  // 🔴 THE PAIRING IS NARROW ON PURPOSE, AND THE NARROWNESS IS THE POINT. The first draft of this
  // check paired any number with `shapes|pictures|photographs` and failed on `src/app.ts:158` —
  // "fifty photographs opened up front would sit in memory as fifty full-size canvases", which is
  // a cap on the visitor's OWN uploads — and would then have failed on `src/haar.ts:252`, "more
  // than eight pictures to choose a practise set from", which is the thirty-eight-photograph
  // survey. `photographs` in `haar.ts` means that survey throughout, and `samples.ts` says "until
  // you have gathered twenty photographs" about the visitor's own disk. **A check that flagged
  // those is a check that would be switched off, which is worse than no check at all** — so this
  // one reads only a count that is QUALIFIED as the set: "N practise shapes|pictures|photographs",
  // "N shapes the page draws", or a number right after "the practise set".
  const FORMER = /\b(?:was|were|used to|formerly|retired|replaced|before|no longer|of the day|at the time)\b/i;
  const QUALIFIED = new RegExp(
    '\\b([A-Za-z]+|\\d+)\\s+practise\\s+(?:shapes|pictures|photographs)\\b'
    + '|\\b([A-Za-z]+|\\d+)\\s+shapes\\s+the\\s+page\\s+draws\\b'
    + '|\\bpractise set\\b[^.]{0,24}?\\b([A-Za-z]+|\\d+)\\s+(?:pictures|shapes|photographs)\\b',
    'gi',
  );
  const files = [
    ['src/app.ts', readFileSync(join(ROOT, 'src', 'app.ts'), 'utf8')],
    ['src/samples.ts', readFileSync(join(ROOT, 'src', 'samples.ts'), 'utf8')],
    ['src/haar.ts', readFileSync(join(ROOT, 'src', 'haar.ts'), 'utf8')],
  ];

  let seen = 0;
  for (const [name, text] of files) {
    for (const [i, line] of text.split('\n').entries()) {
      for (const m of line.matchAll(QUALIFIED)) {
        const said = numberIn(m[1] ?? m[2] ?? m[3]);
        if (said === null) continue;
        seen++;
        if (said === practiseCount) continue;
        assert.ok(FORMER.test(line),
          `${name}:${i + 1} reads "${m[0]}" as though it were true now, and the set is ${practiseCount} — say it was, or it will be copied forward`);
      }
    }
  }

  // And it may not pass by finding nothing to read.
  assert.ok(seen >= 2, `these files do state the size of the practise set and this check must read it (read ${seen})`);
});

test('the page says ONE word for a box and ONE word for a picture', () => {
  // Standard part 6b: two names for one thing leaves the reader deciding whether they are
  // the same thing. The page's word for what it draws is "box", and for an image it is
  // "picture" — and the practise set holds shapes, so "face" is not merely inconsistent
  // here, it is wrong.
  for (const stale of [
    'Found one face',
    'find the faces in it',
    'drag across the face',
    "plural(leftInPhoto, 'face', 'faces')",
    'box ${state.turnIndex + 1} of ${faces}',
    'photo ${runPosition()}',
  ]) {
    assert.ok(!app.includes(stale), `the page still says: ${stale}`);
  }
  assert.match(app, /box \$\{state\.turnIndex \+ 1\} of \$\{boxes\}/, 'the queue counts boxes');
  assert.match(app, /picture \$\{runPosition\(\)\} of \$\{runTotal\(\)\}/, 'the queue counts pictures');
});

test('the page sends only the events the house standard asks for, and never what was typed', () => {
  const tracked = [...app.matchAll(/siteTrack\?\.\('([a-z_]+)'/g)].map((m) => m[1]);
  assert.deepEqual(tracked.sort(), ['memory_saved', 'model_trained', 'picture_named', 'practise_started']);
  for (const call of [...app.matchAll(/siteTrack\?\.\([^)]*\)/g)].map((m) => m[0])) {
    assert.ok(!/el\.list\.value|sample\.name|labels/.test(call), `an event carries what the visitor typed: ${call}`);
  }
  assert.match(gate, /'page_view'/, 'the page sends its own page_view');
  assert.match(gate, /'element_click'/);
  assert.match(gate, /'scroll_depth'/);
  assert.match(gate, /send_page_view: false/, 'or the landing is counted twice');
});

/* ------------------------------------------------------------------ *
 * The numbers panel — the gutters the charts need, and one row per field
 * ------------------------------------------------------------------ */

test('the number and its unit are one field, not three rows of a grid', () => {
  // 🔴 MEASURED 23 September 2026, George at an 801px window: *"the charts are crowded and have
  // overlaps. the Only say it can see something at this confidence is wrapping"*. One shape caused
  // that half: `.controls label` was a **grid**, and a grid gives EVERY child its own row — text
  // nodes included. So the setting was the words, the box, and a bare `%` on a line of its own
  // underneath (measured: y=1715, 1738, 1787), which also left the two settings different heights.
  assert.match(
    htmlNoComments,
    /<span class="field"><input type="number" id="sure"[^>]*><span class="unit">%<\/span><\/span>/,
    'the box and its unit belong to one field',
  );
  assert.match(css, /\.controls label \{[^}]*flex-direction: column/s, 'the label is a column: the words, then the field');
  assert.match(css, /\.controls \.field \{[^}]*display: flex/s, 'the field keeps the box and the unit on one line');
  assert.match(
    css,
    /\.controls label\.check \{[^}]*flex-direction: row/s,
    'the checkbox is still a row — a column label centres it and drops its words onto the next line',
  );
});

test('the charts measure their gutters instead of drawing over the plot', () => {
  // 🔴 THE SAME REPORT, THE OTHER HALF. The plot used to span the whole canvas and the axis labels
  // were drawn INTO it: `format(max)` sat on the top gridline with the series under it, `format(min)`
  // shared the bottom corner with the legend, and the bars chart left a flat 52px for a caption that
  // reads `100% · 2 pictures` and measures about 120px — so the number printed over the end of its
  // own bar and the tail of it was clipped by the canvas edge. A gutter is a measurement, not a
  // guess, and these assertions are about the measurement rather than about a pinned number.
  assert.match(charts, /measureText\(format\(max\)\)/, 'the left gutter is sized from the widest axis label');
  assert.match(charts, /const padLeft = axisWidth \+ \d+/, 'and the plot is inset by it');
  assert.match(charts, /moveTo\(plot\.x, y\)/, 'the gridlines run across the plot, not across the labels');
  assert.match(
    charts,
    /measureText\(i\.caption \?\? format\(i\.value\)\)\.width/,
    'the caption gutter is measured from the captions themselves',
  );
  assert.doesNotMatch(charts, /const valueWidth = 52;/, 'a flat 52px for a 120px caption is the bug');
});

test('the finder runs on what a visitor brings, and not on the page’s own drawings', () => {
  // Measured 2026-09-23 through the page's own pipeline: on the drawn practise shapes the cascade
  // invents a box on some of them, each covering a small share of the frame — and a 4% crop of a
  // crescent is not a crescent. So the drawing is handed over with nothing found, and every
  // photograph a visitor drops in still goes through the finder unchanged. This asserts the
  // distinction rather than the outcome, because it is the distinction that has to survive.
  assert.match(app, /practiseFiles\.has\(file\.name\)/, 'the page must know which files it drew');
  assert.match(
    app,
    /boxes: drawn \? \[\] : detectFaces\(picture\.frame\)/,
    'the finder must run on everything except the drawings the page made itself',
  );
  // The owner's switch is the live check's business (`?ga=off` against a remembered yes);
  // asserting the string here as well would be the same rule held twice, in two places.
});
