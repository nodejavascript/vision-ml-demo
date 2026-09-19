# vision-ml-demo

Teach a small computer program to recognise your pictures, one picture at a time.

A prototype for `vision-ml-demo.nodejavascript.com`. **Not deployed** — it is a local
build with no analytics, no tracking, and no server behind it.

## Run it

```bash
npm install
npm run dev        # tsc, then serve on http://127.0.0.1:4330/
```

## The two versions

| Branch | What it is |
|---|---|
| **`master`** | the simple one — one picture at a time, plain words, no numbers on the front of the page |
| **`v1-full`** | the earlier detailed one — five sections, eight charts, every setting exposed |

Both share the same engine (`src/net.ts` and friends). Only the page differs.

## What it does

1. **Show it a picture.** It guesses what the picture is.
2. **Tell it what the picture is.** Tap a name it already knows, or type a new one.
3. **It studies, in the background, and remembers.**

That is the whole page. There is no Train button, because the person this is for
should not have to know what a learning rate is to teach it something. The numbers
still exist, behind the *More detail* door.

No pictures handy? **It will practise on 90 drawn shapes** — circles, squares and
triangles in random colours and positions — so the loop can be tried without
hunting for photographs first.

## The model

`src/net.ts` is the whole thing: convolutions, pooling, dense layers, softmax,
backpropagation and Adam, as plain loops over typed arrays. No framework, no matrix
library, no runtime dependencies.

```
3x48x48 → conv 3x3 x8 → relu → pool → conv 3x3 x16 → relu → pool → 2304 → 32 → names
```

**75,251 numbers** for three names — about the same size as the sibling `llm-demo`.

## Three traps that cost time

- **It has to study on its own thread, and it has to study enough.** Measured on
  pictures it had never seen: **12 passes → 6 of 9** in 6 seconds, **60 passes →
  8–9 of 9** in 26 seconds. Blocking the page for 26 seconds after every picture is
  not a page anyone would use, and capping the wait is what left it guessing at
  chance (3 of 9). It now studies in a Worker, the page never waits, and because
  each run continues from the last weights the learning accumulates.
- **Yield with a `MessageChannel`, never a `setTimeout`.** Chromium clamps timers
  in a hidden tab — one second, then a minute under intensive throttling. Yielding
  per pass with `setTimeout(…, 0)` made 12 passes take **80 s in a background tab
  against 11 s in a focused one**, same work, nothing on screen to say why.
- **Canvas heights belong in CSS.** Each canvas is `width: 100%`; if its height
  comes from the `height` attribute then the box's aspect ratio decides how tall it
  renders, which feeds back into the drawing code, because every chart reads
  `clientHeight` to size its backing store. The measured result was a 110-pixel
  canvas rendering **247 pixels tall**.

## Two bugs the page promised and the engine refused

- **"Show me two things and I'll tell them apart" — and it would not.** The trainer
  demanded at least **four** samples. Naming two pictures, which the page tells you
  is enough, returned an error nobody was looking for, no model was ever built, and
  from then on every guess came back *"I do not know what this is yet."* One
  picture of each of two things is the honest minimum.
- **The class list never learned its own names.** Adopting the names the pictures
  already carried was missing, so the teach buttons were built from an empty list.

## Files

| | |
|---|---|
| `src/net.ts` | the network, its gradients, and the Adam update |
| `src/image.ts` | a file → 48×48 bytes, and the thumbnails |
| `src/store.ts` | IndexedDB — the pictures and the model |
| `src/trainer-host.ts` | the study loop, one pass at a time, in 40 ms slices |
| `src/trainer.worker.ts` | runs it off the main thread |
| `src/charts.ts` | every chart, drawn by hand on a canvas |
| `src/samples.ts` | the drawn practice shapes |
| `src/app.ts` | the page |
| `site/` | the hand-written page, plus **generated** JavaScript |

**`site/*.js` is compiled from `src/*.ts` — never edit it there.**

## Not here

No tests, no analytics, no sitemap, no deployment. Deliberately.
