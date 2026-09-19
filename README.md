# vision-ml-demo

Teach a small computer program to recognise your pictures, one picture at a time.

A prototype for `vision-ml-demo.nodejavascript.com`. **Not deployed** — it is a local
build with no analytics, no tracking, and no server behind it.

## Run it

```bash
npm install
npm run dev        # tsc, then serve on http://127.0.0.1:4330/
```

Running now as the `vision-ml-demo` systemd user service (same port, restart on
failure): `systemctl --user status vision-ml-demo`.

## The two versions

| Branch | What it is |
|---|---|
| **`master`** | the simple one — one picture at a time, plain words, no numbers on the front of the page |
| **`v1-full`** | the earlier detailed one — five sections, eight charts, every setting exposed |

Both share the same engine (`src/net.ts` and friends). Only the page differs.

## What it does

1. **Hand it pictures — as many as you like.** Drop a batch, or choose several. It
   puts them in a queue and goes through them **one at a time**, saying where you
   are: *photo 3 of 12*.
2. **It looks for faces in each one and draws a box around what it finds.** Two
   people in a photograph are named **one at a time** — *face 1 of 2*, then *face 2
   of 2* — and each face is saved as its own thing to recognise, not the whole
   photograph.
3. **For each box, name the person.** Names only, separated by commas. A photograph
   with no face found in it falls back to naming the picture itself.
4. **It studies, in the background, and remembers.**

The run keeps its rhythm: naming a face goes straight on to the next one, and what
you wrote rides along on the next question's line. A second batch dropped mid-run
queues up behind what is already waiting.

**A box around something you do not want to name can be skipped**, and the box shows
that it was — dimmed, with its number struck through. Skipping is deliberately *not*
the same as removing: the face is still there and still counted, because it is a face
the detector found and pretending otherwise would be a lie about what the picture
holds. Removing is the **×**, for a box that should never have been drawn. `Skip the
other N` appears when more than one face is left, because eight faces found and one
wanted is the ordinary case and declining the other seven one at a time is the kind of
small tediousness that stops a page being used.

That is the whole page. There is no Train button, because the person this is for
should not have to know what a learning rate is to teach it something. The numbers
still exist, behind the *More detail* door.

## Where the faces come from

There is **no face detector in this browser** — `window.FaceDetector` exists on
Chrome OS and Android, not on desktop Linux — and the page has no server, no
libraries and no downloaded weights. So `src/faces.ts` finds them out of the picture
itself: skin has a fairly narrow range of colour, so it builds a mask of
skin-coloured pixels in YCbCr, closes the gaps inside a face, searches for solid
blobs, and keeps the ones shaped and sized like a head.

**It is a proposer, not an authority, and the page is built around that:**

- Every box has an **×** to remove it.
- **Dragging on the picture draws a new box**, for a face it missed.

That is not a nicety. It will miss faces in black-and-white photographs, in heavy
shadow, and behind a mask or a hand; it will sometimes fire on a wooden floor. Two
honest notes, said because pretending otherwise would be worse than the limitation:
the colour rule is the standard one from the literature and, like the rest of that
literature, was tuned mostly on lighter skin, so it is measurably less reliable on
darker skin in dim light; and it knows nothing about what a face *is* — "one blob of
skin" is all the structure it understands, so two people standing close together can
come out as one box.

## Why a list

Asking for a list rather than a single name is what makes one picture worth several.
A photograph of a dog on a beach teaches **dog** *and* **beach**, and the same
picture is reused by both answers — so twenty photographs go much further than they
would if each one had to pick a winner. It is also how real image collections are
labelled.

Underneath, that is a different network: a separate yes/no for each thing it knows
(`sigmoid` + binary cross-entropy) instead of one choice out of all of them
(`softmax`). It is why the page can answer with more than one thing at a time.

Measured on the 90 drawn shapes / 11 names: loss **0.85 → 0.13**, **85%** on
pictures held back from training, **72–100%** per name.

## Nouns only

A name is what the model can use. A class called *"this is a photo of my cat"* is not
a thing, it is a sentence, and it would sit in the vocabulary forever getting in the
way of everything else. So the page asks for **nouns, separated by commas** —
`dog, cat, sky, beach, rail house` — and `parseLabels()` does three things about it:

1. **Split** on commas, semicolons and new lines, and on the words *and* / *or* —
   "a dog and a beach" is two things, not one long one.
2. **Strip framing words off the front**, but only when the piece starts with a word
   that could not begin a name — *a, the, my, this, i, there, some…* So
   "this is a photo of my cat" becomes **cat**, and "the sky" becomes **sky**.
3. **Leave everything else exactly as typed.**

**The front is the only safe place to cut.** A word removed from the middle destroys
a name — *cup of tea*, *rail house*, *fish and chips* — so nothing is ever taken from
the middle or the end, and cleaning only starts at all if the first word could not
begin a name. That is why **photo frame** and **can opener** come through untouched:
they do not start with one.

There is **no dictionary** here and this does not pretend to be one. It cannot know
that *sitting* is not a thing, so "a cat sitting on a wall" comes out as
**cat sitting on a wall**. It stops at the first real word rather than guessing where
the noun ends — and the page echoes back every name it stored, so a bad one is visible
immediately rather than buried in the vocabulary.

No pictures handy? **It will practise on that same set** — 90 drawn shapes, each a
shape in a colour, so every one of them is a two-label picture. The offer is a
small link in the corner of the stage, and it only appears while you have nothing
named of your own.

## The model

`src/net.ts` is the whole thing: convolutions, pooling, dense layers, sigmoid,
binary cross-entropy, backpropagation and Adam, as plain loops over typed arrays. No
framework, no matrix library, no runtime dependencies.

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
