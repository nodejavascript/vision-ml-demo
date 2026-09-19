/**
 * serve.js — a static file server for local checking.
 *
 * Module workers need a real origin, so the page cannot be opened from file://.
 * This serves site/ on http://127.0.0.1:4330 with the content types a static
 * host would send.
 *
 *   node tools/serve.js [port]
 */

import { createServer } from 'node:http';
import { readFile, stat } from 'node:fs/promises';
import { extname, join, normalize, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const SITE = join(here, '..', 'site');
const port = Number(process.argv[2]) || 4330;

const TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'text/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.ico': 'image/x-icon',
  '.txt': 'text/plain; charset=utf-8',
};

const server = createServer(async (req, res) => {
  const url = new URL(req.url, `http://localhost:${port}`);
  let path = normalize(decodeURIComponent(url.pathname));
  if (path === '/' || path === '') path = '/index.html';
  const file = join(SITE, path);
  if (!file.startsWith(SITE)) {
    res.writeHead(403).end('forbidden');
    return;
  }
  try {
    const info = await stat(file);
    if (info.isDirectory()) throw new Error('directory');
    const body = await readFile(file);
    res.writeHead(200, {
      'Content-Type': TYPES[extname(file)] || 'application/octet-stream',
      'Content-Length': body.length,
      // No caching, ever: a browser holding an old module is the single most
      // expensive way to lose an afternoon.
      'Cache-Control': 'no-store',
    });
    res.end(body);
  } catch {
    res.writeHead(404, { 'Content-Type': 'text/plain' }).end('not found');
  }
});

server.listen(port, '127.0.0.1', () => {
  console.log(`vision-demo on http://127.0.0.1:${port}/  (serving ${SITE})`);
});
