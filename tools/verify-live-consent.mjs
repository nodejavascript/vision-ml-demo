/**
 * The live check: what a visitor actually receives right now, from the deployed host.
 *
 *   node tools/verify-live-consent.mjs [url]
 *
 * Not part of `test:e2e` — that suite has to stay runnable offline against its own build,
 * and a local build cannot prove what the deploy is serving. This is where the DNS, the TLS,
 * the Caddy headers and the cache purge all have to be right at the same time, and it is the
 * only check that can see a stale artefact.
 *
 * It reads the shell cache header, the icon set, the sitemap and the robots file over HTTP,
 * then drives a real Chrome for the half that only a browser can see: that a refusal makes
 * no request to Google, that the bar is the page's own colour, and that the background
 * abstract is painted rather than merely declared.
 */
import { chromium } from 'playwright';

const BASE = process.argv[2] || 'https://vision-ml-demo.nodejavascript.com/';
const HOST = new URL(BASE).host;
const GOOGLE = /google-analytics\.com|googletagmanager\.com/;

const checks = [];
const note = (name, ok, detail = '') => checks.push([name, ok, detail]);

/* ------------------------------------------------------------------ *
 * 1. What the server sends
 * ------------------------------------------------------------------ */

const shell = await fetch(BASE);
const html = await shell.text();
// The page's comments discuss the very tags these checks look for, so they are stripped
// first: a check that reads its own documentation reports a fault that is not there, and a
// false failure is the expensive kind.
const served = html.replace(/<!--[\s\S]*?-->/g, '');
note('the shell answers 200', shell.status === 200, `status ${shell.status}`);

const cacheHeaders = shell.headers.getSetCookie ? shell.headers.getSetCookie() : [];
void cacheHeaders;
const cacheControl = shell.headers.get('cache-control') ?? '';
note('the shell is no-store, so a deploy is visible immediately', /no-store/.test(cacheControl), cacheControl);
note('the served HTML carries no Google tag', !/googletagmanager|google-analytics\.com/.test(served));
note('the measurement id rides on the gate', /data-ga-id="G-[A-Z0-9]+"/.test(html));
const title = (html.match(/<title>([^<]*)<\/title>/) ?? [])[1] ?? '';
note('the html title IS the host', title === HOST, title);
note('the policy is a section of the page, not a page of its own', /id="privacy"/.test(html));
note('the footer carries the door', /id="consentBtn"/.test(html));

for (const path of ['/consent.js', '/app.js', '/styles.css', '/favicon.svg', '/favicon.ico', '/favicon-32.png', '/apple-touch-icon.png', '/manifest.webmanifest', '/og.png', '/sitemap.xml', '/robots.txt']) {
  const response = await fetch(new URL(path, BASE));
  note(`${path} answers 200`, response.status === 200, `status ${response.status}`);
  if (path === '/sitemap.xml' || path === '/robots.txt') {
    const body = await response.text();
    note(`${path} names this host`, body.includes(HOST));
  }
}

/* ------------------------------------------------------------------ *
 * 2. What a browser does
 * ------------------------------------------------------------------ */

const browser = await chromium.launch({ channel: 'chrome' });
const pageErrors = [];

async function visit(label, { click, seed, query = '' } = {}) {
  // 🔴 THE OWNER'S NETWORK IS SERVED A STUB `/consent.js` (the `no-ga-for-me` rule on dvs-sites),
  // and this live check runs from inside that range — so the bar never appears, the click times
  // out, and the gate fails on a site that is correct. Measured 24 September 2026 on the fourth
  // site in one day. The header only un-suppresses a file every other visitor already gets.
  const context = await browser.newContext({ extraHTTPHeaders: { 'X-Nodejs-Audit': '1' } });
  if (seed) {
    await context.addInitScript((value) => {
      try {
        localStorage.setItem('analytics_consent', value);
      } catch {
        /* storage blocked */
      }
    }, seed);
  }
  const page = await context.newPage();
  const google = [];
  page.on('request', (request) => {
    if (GOOGLE.test(request.url())) google.push(request.url());
  });
  page.on('pageerror', (error) => pageErrors.push(`${label}: ${error.message}`));
  await page.route(GOOGLE, (route) => route.fulfill({ status: 204, body: '' }));

  await page.goto(BASE + query, { waitUntil: 'load' });
  await page.waitForSelector('#says');

  const before = google.length;
  if (click) {
    await page.click(click);
    await page.waitForTimeout(1200);
  }

  const state = await page.evaluate(() => {
    const bar = document.getElementById('consentBar');
    return {
      gtag: typeof window.gtag,
      track: typeof window.siteTrack,
      barOpen: bar ? !bar.hidden : false,
      reserved: parseFloat(getComputedStyle(document.body).paddingBottom) || 0,
      stored: localStorage.getItem('analytics_consent'),
      practiseOffered: !!document.querySelector('#aside .link'),
      pattern: (() => {
        const layer = document.querySelector('.dvs-pattern');
        if (!layer) return null;
        const computed = getComputedStyle(layer);
        return {
          drawing: /conic-gradient/.test(computed.backgroundImage),
          masked: (computed.maskImage === 'none' ? computed.webkitMaskImage : computed.maskImage) !== 'none',
        };
      })(),
      theme: (() => {
        const accent = getComputedStyle(document.body).getPropertyValue('--accent').trim();
        const meta = document.querySelector('meta[name="theme-color"]')?.content ?? '';
        const bar = getComputedStyle(document.querySelector('header.top')).backgroundColor;
        return { accent, meta, bar };
      })(),
    };
  });

  console.log(
    `${label}\n` +
      `  requests to Google before any click: ${before}\n` +
      `  requests to Google after:            ${google.length}\n` +
      `  window.gtag / window.siteTrack:      ${state.gtag} / ${state.track}\n` +
      `  bar open / height reserved:          ${state.barOpen} / ${state.reserved}px\n` +
      `  stored answer:                       ${state.stored}`,
  );
  await context.close();
  return { before, after: google.length, state };
}

console.log(`\n=== live check, ${BASE} ===\n`);

const untouched = await visit('1. A visit that answers nothing');
const rejected = await visit('2. Reject all, clicked', { click: '#consentDecline' });
const accepted = await visit('3. Accept all, clicked', { click: '#consentAccept' });
const remembered = await visit('4. A remembered yes', { seed: 'granted' });
const owner = await visit('5. ?ga=off, the owner switch, against a remembered yes', {
  seed: 'granted',
  query: '?ga=off',
});

note('nothing is asked of Google before a choice', untouched.before === 0);
note('there is no gtag to call before a choice', untouched.state.gtag === 'undefined');
note('the page has no way to send an event before a choice', untouched.state.track === 'undefined');
note('the question is on screen', untouched.state.barOpen === true);
note('refusing makes no request to Google at all', rejected.after === 0);
note('refusing is remembered', rejected.state.stored === 'denied');
note('accepting fetches the tag', accepted.after > 0);
note('accepting opens the page-events route', accepted.state.track === 'function');
note('a remembered yes is not re-asked', remembered.state.barOpen === false);
note('the owner switch beats a remembered yes', owner.after === 0 && owner.state.gtag === 'undefined');
note('the bar reserves its own height', untouched.state.reserved > 0);
note('the practise set is offered on the deployed page', untouched.state.practiseOffered);
note('the background abstract is DRAWN, not merely declared', !!untouched.state.pattern?.drawing);
note('the abstract is masked, so it has no edge', !!untouched.state.pattern?.masked);
note('the theme colour is the page’s own accent', untouched.state.theme.meta === untouched.state.theme.accent, JSON.stringify(untouched.state.theme));
note('no page error on any visit', pageErrors.length === 0, pageErrors.join(' | '));

console.log('\n=== checks ===');
let failed = 0;
for (const [name, ok, detail] of checks) {
  if (!ok) failed += 1;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${name}${!ok && detail ? `  (${detail})` : ''}`);
}
console.log(`\n${checks.length - failed}/${checks.length} passed`);

await browser.close();
process.exit(failed ? 1 : 0);
