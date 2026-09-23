import { test, before, after } from 'node:test';
import assert from 'node:assert/strict';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { chromium } from 'playwright';

/**
 * End-to-end tests: a real Chrome, the real page, the real worker, the real practise set —
 * against `tools/serve.js` on a test port.
 *
 * The unit suite proves the page says what the standard requires; this suite proves the
 * page DOES it, and it is the only place two of the claims can be checked at all: that a
 * refusal makes no request to Google, and that the practise set really hands over thirty
 * pictures through the same loop an upload goes through.
 */

const here = dirname(fileURLToPath(import.meta.url));
const ROOT = join(here, '..');
const PORT = 4397;
const BASE = `http://127.0.0.1:${PORT}`;

let server;
let browser;

async function waitForServer(tries = 60) {
  for (let i = 0; i < tries; i++) {
    try {
      const response = await fetch(`${BASE}/`, { signal: AbortSignal.timeout(1000) });
      if (response.ok) return;
    } catch {
      /* not listening yet */
    }
    await new Promise((r) => setTimeout(r, 100));
  }
  throw new Error(`the local server never came up on ${BASE}`);
}

before(async () => {
  server = spawn(process.execPath, [join(ROOT, 'tools', 'serve.js'), String(PORT)], { stdio: 'ignore' });
  await waitForServer();
  browser = await chromium.launch({ channel: 'chrome' });
});

after(async () => {
  await browser?.close();
  server?.kill();
});

/**
 * A fresh page that records every request it makes and every analytics event.
 *
 * `consent` seeds the stored answer before any page script runs, which is how a RETURNING
 * visitor is simulated — the choice made last week rather than one made by clicking. The
 * analytics beacons are answered with an empty 204 instead of being aborted, because
 * aborting logs a console error and this suite asserts there are none; the request is still
 * recorded either way, and nothing leaves the machine.
 */
async function openPage(options = {}) {
  const { consent, path = '/' } = options;
  const context = await browser.newContext();
  if (consent) {
    await context.addInitScript((value) => {
      try {
        localStorage.setItem('analytics_consent', value);
      } catch {
        /* storage blocked */
      }
    }, consent);
  }
  const page = await context.newPage();
  const requests = [];
  const pageErrors = [];
  const consoleErrors = [];

  page.on('request', (request) => requests.push(request.url()));
  page.on('pageerror', (error) => pageErrors.push(String(error.message)));
  page.on('console', (message) => {
    if (message.type() === 'error') consoleErrors.push(message.text());
  });
  await page.route(/google-analytics\.com|googletagmanager\.com/, (route) => route.fulfill({ status: 204, body: '' }));

  await page.goto(`${BASE}${path}`, { waitUntil: 'load' });
  await page.waitForSelector('#says');
  return { context, page, requests, pageErrors, consoleErrors };
}

const toGoogle = (requests) => requests.filter((url) => /google/.test(url));
const dataLayer = (page) => page.evaluate(() => (window.dataLayer ?? []).map((entry) => Array.from(entry)));
const eventNames = async (page) => (await dataLayer(page)).filter((e) => e[0] === 'event').map((e) => e[1]);

/* ------------------------------------------------------------------ *
 * Delivery
 * ------------------------------------------------------------------ */

test('every asset is served no-store, which is what keeps a deploy visible', async () => {
  const context = await browser.newContext();
  for (const path of ['/', '/app.js', '/consent.js', '/styles.css', '/samples.js', '/haar.js']) {
    const response = await context.request.get(BASE + path);
    assert.equal(response.status(), 200, `${path} did not answer 200`);
    const cacheControl = response.headers()['cache-control'] ?? '';
    assert.match(
      cacheControl,
      /no-store/,
      `${path} is cacheable (${cacheControl || 'no cache-control'}) — a deploy would be invisible to a returning visitor`,
    );
  }
  await context.close();
});

test('the page loads clean, with the title that IS the host', async () => {
  const { context, page, pageErrors, consoleErrors } = await openPage();
  // Derived from the page's own canonical rather than hard-coded: a pinned title breaks the
  // suite the day the copy changes, which turns the test into a record of what it was.
  const canonicalHost = await page.$eval('link[rel="canonical"]', (el) => new URL(el.href).host);
  assert.equal(await page.title(), canonicalHost);
  assert.equal(await page.locator('h1').count(), 1);
  assert.equal(await page.locator('header.top nav').count(), 0, 'the header carries no nav');
  assert.deepEqual(pageErrors, []);
  assert.deepEqual(consoleErrors, []);
  await context.close();
});

/* ------------------------------------------------------------------ *
 * Standard part 2 — the cookie gate
 * ------------------------------------------------------------------ */

test('before an answer: no request to Google at all, and no way to send one', async () => {
  const { context, page, requests } = await openPage();
  assert.ok(await page.isVisible('#consentBar'), 'the question must be asked');
  assert.equal(await page.evaluate(() => typeof window.gtag), 'undefined', 'gtag must not exist without consent');
  assert.deepEqual(toGoogle(requests), [], 'nothing may be requested from Google before a visitor chooses');
  await context.close();
});

test('the two answers are the same size and the same weight', async () => {
  const { context, page } = await openPage();
  const accept = await page.locator('#consentAccept').boundingBox();
  const decline = await page.locator('#consentDecline').boundingBox();
  assert.ok(Math.abs(accept.width - decline.width) <= 1, 'the answers must be the same width');
  assert.ok(Math.abs(accept.height - decline.height) <= 1, 'the answers must be the same height');
  const styles = await page.evaluate(() => {
    const read = (id) => {
      const computed = getComputedStyle(document.getElementById(id));
      return `${computed.fontSize}/${computed.fontWeight}/${
        document.getElementById(id).className
      }`;
    };
    return [read('consentAccept'), read('consentDecline')];
  });
  assert.equal(styles[0], styles[1], 'neither answer may be made less inviting than the other');
  // Neither may be drawn with the focus ring when the bar opens: a ring is a recommendation.
  assert.equal(await page.evaluate(() => document.activeElement?.id ?? ''), 'consentBar');
  await context.close();
});

test('accepting loads the tag, counts the landing once, and releases the page', async () => {
  const { context, page, requests } = await openPage();
  await page.click('#consentAccept');
  await page.waitForFunction(() => typeof window.gtag === 'function');
  await page.waitForTimeout(400);

  assert.ok(await page.isHidden('#consentBar'), 'the bar goes when the question is answered');
  const events = await eventNames(page);
  assert.equal(events.filter((name) => name === 'page_view').length, 1, 'the landing is counted exactly once');
  assert.ok(toGoogle(requests).some((url) => url.includes('googletagmanager')), 'the tag loads only after a yes');
  assert.equal(
    await page.evaluate(() => getComputedStyle(document.body).paddingBottom),
    '0px',
    'the space reserved for the bar must be given back',
  );
  assert.equal(await page.evaluate(() => localStorage.getItem('analytics_consent')), 'granted');
  await context.close();
});

test('refusing sends NOTHING to Google, and is not asked again', async () => {
  const { context, page, requests } = await openPage();
  await page.click('#consentDecline');
  await page.waitForTimeout(400);

  assert.deepEqual(toGoogle(requests), [], 'a refusal makes no request to Google — none at all');
  assert.equal(await page.evaluate(() => typeof window.gtag), 'undefined');
  assert.equal(await page.evaluate(() => localStorage.getItem('analytics_consent')), 'denied');
  assert.ok(await page.isHidden('#consentBar'));

  await page.reload({ waitUntil: 'load' });
  await page.waitForSelector('#says');
  assert.ok(await page.isHidden('#consentBar'), 'rejecting is a real answer, not a nag');
  assert.deepEqual(toGoogle(requests), [], 'still nothing, on the second visit');
  await context.close();
});

test('the footer door reopens the panel and leaves the answer alone', async () => {
  const { context, page, requests } = await openPage({ consent: 'denied' });
  assert.ok(await page.isHidden('#consentBar'), 'an answered visitor is not asked again');
  await page.click('#consentBtn');
  assert.ok(await page.isVisible('#consentPrefs'), 'the door opens the panel, not the question');
  assert.equal(
    await page.getAttribute('#consentAnalytics', 'aria-checked'),
    'false',
    'the switch shows the answer that is actually in force',
  );
  // Looking at the settings must not start anything.
  assert.deepEqual(toGoogle(requests), []);
  await page.keyboard.press('Escape');
  assert.ok(await page.isHidden('#consentBar'));
  await context.close();
});

/* ------------------------------------------------------------------ *
 * The practise set, through the real loop
 * ------------------------------------------------------------------ */

test('the practise set hands over thirty shapes, and they go through the real loop', async () => {
  const { context, page, pageErrors, consoleErrors } = await openPage({ consent: 'denied' });
  const offer = page.locator('#aside .link');
  assert.ok(await offer.isVisible(), 'the way to the practise set is offered before anything is named');
  await offer.click();

  // Thirty pictures, queued the way an upload is queued — four shapes, seven or eight of each,
  // which is the whole point of the change: enough of a similar shape to learn from.
  await page.waitForFunction(() => document.getElementById('queue')?.textContent?.includes('of 30'), null, {
    timeout: 20000,
  });
  assert.match(await page.textContent('#queue'), /^picture 1 of 30$/);

  // A shape is not a face, so the finder finds nothing and the question is about the whole
  // picture — which is the same path a photograph of a kitchen takes.
  assert.equal(await page.textContent('#tellLabel'), 'What is in this picture? One name:');
  assert.equal(await page.locator('#boxes .box').count(), 0, 'a shape holds no faces, and none is invented');

  // Naming it moves the run on, and the page says what it now knows. The name typed is the
  // person's own word — nothing checks it against the set, and `star` is only a label here.
  await page.fill('#list', 'star');
  await page.click('#tellBtn');
  await page.waitForFunction(() => document.getElementById('queue')?.textContent?.includes('picture 2 of 30'), null, {
    timeout: 20000,
  });
  assert.match(await page.textContent('#progress'), /I know 1 thing/);

  assert.deepEqual(pageErrors, []);
  assert.deepEqual(consoleErrors, []);
  await context.close();
});

test('the background abstract is painted, not merely written down', async () => {
  // The unit suite reads the stylesheet; this reads the page a visitor actually gets, which
  // is the half that went wrong for the family — a rule satisfied in CSS nobody paints.
  const { context, page } = await openPage();
  const pattern = await page.evaluate(() => {
    const layer = document.querySelector('.dvs-pattern');
    if (!layer) return null;
    const computed = getComputedStyle(layer);
    return {
      backgroundImage: computed.backgroundImage,
      mask: computed.maskImage === 'none' ? computed.webkitMaskImage : computed.maskImage,
      position: computed.position,
      box: layer.getBoundingClientRect().width,
    };
  });
  assert.ok(pattern, 'the layer must be in the page');
  assert.match(pattern.backgroundImage, /conic-gradient/, 'a gradient wash is not a drawing');
  assert.notEqual(pattern.mask, 'none', 'a pattern with an edge is a band, and bands were rejected');
  assert.equal(pattern.position, 'fixed');
  assert.ok(pattern.box > 200, 'it must actually cover the viewport');
  await context.close();
});
