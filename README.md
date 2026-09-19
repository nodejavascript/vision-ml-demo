# vision-demo

Teach a small convolutional network to recognise your images, in your browser.

A prototype for `vision-demo.nodejavascript.com`. **Not deployed** — it is a local
build with no analytics, no tracking and no server behind it.

## Run it

```bash
npm install
npm run dev        # tsc, then serve on http://127.0.0.1:4330/
```

`npm run watch` recompiles in the background if you are editing.

## What it does

1. **Give it images.** Drop files in, or press *Draw a sample set* and it paints
   90 pictures of its own — three shapes, thirty each, in random colours and
   positions. Twelve are left unnamed deliberately.
2. **Train it.** Loss and accuracy curves are drawn as it goes.
3. **Teach it what it could not name.** Every image with no label gets a guess,
   and the ones it is least sure about are queued worst-first. Each answer becomes
   training data. *Accept its confident guesses* pseudo-labels the rest.
4. **Use it.** Drop in a picture and it answers, with the eight first-layer
   activations and the eight learned kernels underneath.
5. **Keep it.** Saved in IndexedDB as you go, and downloadable as one JSON file.

Everything stays on the machine. There is no fetch in `src/`.

## The model

`src/net.ts` is the whole thing: convolutions, pooling, dense layers, softmax,
backpropagation and Adam, written as plain loops over typed arrays. No framework,
no matrix library, no runtime dependencies.

```
3x48x48 → conv 3x3 x8 → relu → pool → conv 3x3 x16 → relu → pool → 2304 → 32 → classes
```

**75,251 parameters** for three classes — about the same size as the sibling
`llm-demo`. Roughly 90% on the labelled set and 55–65% on images held back, which
is the honest number for two convolutions and a few dozen pictures.

## Files

| | |
|---|---|
| `src/net.ts` | the network, its gradients, and the Adam update |
| `src/image.ts` | file → 48×48 bytes, and the thumbnails |
| `src/samples.ts` | the drawn sample set |
| `src/charts.ts` | every chart, drawn by hand on a canvas |
| `src/store.ts` | IndexedDB — images and model |
| `src/trainer-host.ts` | the training loop, one epoch at a time |
| `src/trainer.worker.ts` | runs it off the main thread |
| `src/app.ts` | the page |
| `site/` | the hand-written page, plus **generated** JavaScript |

**`site/*.js` is compiled from `src/*.ts` — never edit it there.**

## Three things that cost time

- **Learning-rate warmup, 10 steps (`src/net.ts`).** Adam's bias-corrected first
  steps are the full learning rate taken on random weights, so the loss curve
  jumps *up* before it falls. Measured first-epoch mean loss at lr 0.01: **8.9
  without warmup, 2.6 with it**, and every rate still reaches 100% on the little
  synthetic task. A curve that spikes before it drops reads as a broken chart.
- **Yield with a `MessageChannel`, not a `setTimeout` (`src/trainer-host.ts`).**
  Chromium clamps timers in a hidden tab — one second, then a minute under
  intensive throttling. Yielding per epoch with a zero-delay timeout made 12
  epochs take **80 s in a background tab against 11 s in a focused one**, same
  work, nothing on screen to say why. A `MessageChannel` task is not a timer.
- **Canvas heights belong in CSS (`site/styles.css`).** Each canvas is
  `width: 100%`; if its height comes from the `height` attribute then the box's
  aspect ratio decides how tall it renders, which feeds back into the drawing
  code, because every chart reads `clientHeight` to size its backing store. The
  measured result was a 110-pixel canvas rendering **247 pixels tall.** Fixing
  the height in CSS breaks the loop.

## Not here

No tests, no analytics, no sitemap, no manifest, no deployment. Deliberately.
