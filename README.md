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
3. **For each box, give it one name — one person, place or thing.** A photograph
   with no face found in it falls back to naming the picture itself.
4. **Watch the chart under the picture fill in.** One segment per box: named,
   skipped, the one being asked, and the ones still to come.
5. **It studies, in the background, and remembers.**

The run keeps its rhythm: naming a face goes straight on to the next one. A second batch
dropped mid-run queues up behind what is already waiting.

**A box around something you do not want to name can be skipped**, and the box shows
that it was — dimmed, with its number struck through. Skipping is deliberately *not*
the same as removing: the face is still there and still counted, because it is a face
the detector found and pretending otherwise would be a lie about what the picture
holds. Removing is the **×**, for a box that should never have been drawn.

There are two ways out, and both are links sitting under the thing they act on:
**Skip this box** beneath the thumbnail of the box being asked about, and **Skip this
image** beneath the picture, for a photograph that turns out to hold nothing you want to
name. The second is the answer to eight boxes found and none of them wanted.

**A name typed wrong is taken back, not lived with.** *Undo the last name* sits in the row
under the line above, and puts the run back exactly as it stood while that box was being
asked about: the same picture, the same box, and the name in the box with it selected — so
correcting a misspelling is a retype rather than a hunt back through the queue for the
photograph it was on. If the wrong name had already started a study run, that run is stopped
and the studying starts again from the names that are actually there. It holds **one** name
and not a history, because undoing anything older would have to unwind whatever came after
it — and it is dropped the moment anything else touches the run: a skip, a box added or
removed, another batch dropped in. A page that silently reverses two answers when one is
asked for is worse than a page with no undo at all.

**There is no instruction note.** Three sentences used to sit beside the thumbnail the
whole time a picture was up — *press a box to pick it, × to remove it, or drag across a
face to add one it missed* — for a gesture most people never need, on a page that is
otherwise one question and one answer. The **×** is visible on the box itself, which is
where somebody looking for it looks. The cost is that drawing a missed box by dragging
is no longer announced anywhere; that is the trade, made deliberately.

That is the whole page. There is no Train button, because the person this is for
should not have to know what a learning rate is to teach it something. The numbers
still exist, behind the *More detail* door.

**One line, and only while it has something to say.** That line used to open with *Noted
sarah.* and close with *I will remember every one of them.* — the first repeats what the box
on the picture and the chart under it already show, and the second is not an instruction.
George, 2026-09-19: *"dont put Noted ngle. I will remember every one of them. langaurge."*
What is left says one thing and stops: *Name this one and I can start learning*, then *Name
a second one and I can start telling them apart*, and after that **no line at all**. How
many pictures are actually named decides which — derived from what is there rather than from
a counter, so a skip or a second drop cannot leave it saying something untrue.

**The question is at the top of the column, and the guess is the first thing under it.**
The order used to be the model's line, then what to do, then the box to type in — so the one
thing on the page you have to act on sat underneath two lines of narration about it. It now
reads downwards: where you are, the question, *My guess:* with a button per thing it can
see, then what to do next. The buttons are labelled with the names themselves, so agreeing
with the model is one press and disagreeing is the box directly above it.

**A guess, or nothing — and nothing means the line is not there at all.** Everything it
offers has cleared the confidence line, which is the number behind the *More detail* door
and is yours to move. Under that line there is nothing: **the line is not drawn** and no
buttons with it. It used to name its strongest answer anyway — *I am not sure yet — my best
guess is sarah, and I am only 34% on that* — on the reasoning that "I do not know" tells you
nothing about which way it leans; then it said *I'm not sure yet* on its own. George,
2026-09-19: *"if it doesnt know, i dont want to see my guess."* So there is no guess line
when there is no guess, and the value that produced the old sentence is gone from the code
rather than hidden in it — there is no strongest-answer reading anywhere in the page any
more. A name under the line is one the model has no reason to give, and a button on it
invites somebody to accept it without looking.

**The guess is a label, not a sentence.** *My guess:* over its buttons is the head of the
answer, so it is set at the same size as the question it answers — not in the model's own
voice, which is the larger type on the opening and closing lines — and it carries a little
more air above it, so the box you type in and the guesses under it do not read as one block.

## Where the faces come from

There is **no face detector in this browser** — `window.FaceDetector` exists on
Chrome OS and Android, not on desktop Linux — and the page has no server, no
libraries and no runtime. So the detector is written here, in `src/haar.ts`: the
**Viola-Jones algorithm**, the one behind OpenCV's frontal-face cascade, as plain
loops over typed arrays.

A window 24 pixels square walks the picture at every position and every size. At each
stop it asks a few hundred black-and-white questions — *is the eye band darker than
the cheeks, is the bridge of the nose lighter than its sides* — in order of how
decisive they are, and stops at the first one that is answered the wrong way. Almost
every window is thrown out after two or three questions. A window that answers all
twenty-five rounds is a face.

The questions and their thresholds are the **one thing here that is not written from
scratch**, because they cannot be: they are trained, and training needs thousands of
labelled faces. They are OpenCV's own cascade — Rainer Lienhart's stump-based 24×24
frontal-face detector — which the Intel licence permits redistributing with the notice
kept. `tools/build-cascade.mjs` packs them into a TypeScript module so there is nothing
to fetch and no asset to get wrong, and `src/haar.ts` is the detector that reads them.

It was checked against OpenCV 4.10 before it was wired in: the same cascade run over
the same pictures keeps the same windows — 10 of 10 on a test photograph — and 409
fixed 24×24 patches put to both are answered identically, 63 yes and 346 no, with no
disagreement in either direction.

**It is a proposer, not an authority, and the page is built around that:**

- Every box has an **×** to remove it.
- **Dragging on the picture draws a new box**, for a face it missed.

That is not a nicety. It will still miss a face turned away from the camera, a face in
heavy shadow, a face behind a mask or a hand, and a face smaller than about a twelfth
of the picture.

### What it replaced, and why the numbers matter

Until 19 September 2026 the search was a **skin-colour finder**: mask the
skin-coloured pixels in YCbCr, close the gaps, look for solid blobs. It is worth
writing down what that actually did, because "skin is a narrow range of colour" is
true often enough to look like it works.

Measured against eighteen ordinary photographs — twelve of which hold a face,
according to dlib's `face_recognition` — and, separately, against the twenty practise
photographs the page ships:

| | the colour rule, at 224 | the cascade, at 320 |
|---|---|---|
| photographs with a face that got a box | 7 of 12 | **12 of 12** |
| faces named by the reference that were missed | 8 of 15 | **0 of 12 photographs** |
| face-free photographs that got an invented box | 5 of 6 | **1 of 6** |
| practise photographs whose box was the whole picture | 6 of 20 (boxes 67–82% of the frame) | **0 of 20** |

Two changes did that, and it is worth keeping them apart. **A real detector** is one.
**Looking at a bigger copy of the picture** is the other, and it is the cheaper half:
the cascade at the old 224 found a face in 10 of the 12 photographs; at 320 it found
12 of 12. A 24-pixel window at 224 wide means a face has to be a tenth of the picture
to be visible to the search at all, and a photograph is usually not a headshot.

A colour rule also cannot be checked the way this can. The cascade's arithmetic was
compared against OpenCV's, which a heuristic has no equivalent of — and a heuristic has
no answer at all to *"why did you say that is a face?"*

## One box, one name

The page used to take a comma-separated list — `dog, cat, sky, beach`. It does not
any more, and the reason is the comma. Eight faces in a photograph means eight
questions, and a box holding one person is not a place a list belongs: the separator
that helped when a whole picture was being described is what silently split
**fish and chips** into two strangers.

So each box gets **one** name, and the separators are gone. Type *fish and chips* and
that is what is stored; type *dog, cat* and the page says so — *One name per box — I
used "dog"* — rather than quietly keeping half of it. One picture is still worth
several names, because a photograph with three faces is asked about three times.

Underneath, that is a multi-label network: a separate yes/no for each thing it knows
(`sigmoid` + binary cross-entropy) instead of one choice out of all of them
(`softmax`). It is why the page can answer with more than one thing at a time.

Measured on the 90 drawn shapes / 11 names: loss **0.85 → 0.13**, **85%** on
pictures held back from training, **72–100%** per name.

## How a name is read

A name is what the model can use. A class called *"this is a photo of my cat"* is not
a thing, it is a sentence, and it would sit in the vocabulary forever getting in the
way of everything else. So the page asks for **one name per box** — `dog`, `cat`,
`sky`, `beach`, `rail house` — and `parseName()` does two things about it:

1. **Strip framing words off the front**, but only when the first word is one that
   could not begin a name — *a, the, my, this, i, there, some…* So
   "this is a photo of my cat" becomes **cat**, and "the sky" becomes **sky**.
2. **Leave everything else exactly as typed.** There is nothing to split, because one
   box holds one thing.

**The front is the only place anything is cut, and the only words cut are ones that
could not start a name.** Nothing is ever taken from the middle or the end, and
nothing is cut on at all — which is what keeps *cup of tea*, *rail house* and
**fish and chips** whole. The earlier version split those, because it treated *and* /
*or* as separators: right for a sentence, wrong for a name. **photo frame** and
**can opener** come through untouched because they do not start with a framing word.

Type more than one name and it is not silently dropped — the first is stored and the
page says what it did, *One name per box — I used "dog"*. A name kept without being
mentioned is how a vocabulary fills up with things nobody meant to teach it.

There is **no dictionary** here and this does not pretend to be one. It cannot know
that *sitting* is not a thing, so "a cat sitting on a wall" comes out as
**cat sitting on a wall**. It stops at the first real word rather than guessing where
the noun ends — and the page echoes back every name it stored, so a bad one is visible
immediately rather than buried in the vocabulary.

No pictures handy? **The practise link queues the page's own drawn shapes** — ten of
them, taken at even intervals across the set so all three shapes are represented, since
the set is generated grouped by shape and the first ten would be ten circles. They are
handed over as **files**, not pushed into the model: they queue, each is opened in turn,
each is searched for faces, and every box is named by hand. Nothing is taught on your
behalf — what you practise is the loop a real upload goes through, not a shortcut beside
it. The offer is a small link just above the stage card, and it only appears while you
have nothing named of your own.

**The drawn pictures are painted sharp, and that is a fix rather than a preference.**
George, 2026-09-19: *"the test pictures should be sharper. getting too many boxes because of
blurred lines."* He was right about the cause. The shapes were painted at **128 px**, saved as
a **JPEG at quality 0.72**, and then enlarged to the 224 px the face search looks at — so
every hard edge reached the detector as a four-pixel gradient, and the in-between colours
along that gradient are exactly what the detector reads as skin. Painted at **224 px** (the
search's own width, so it is never resampled at all) and handed over as a **PNG**, the edge
stays one pixel wide.

Measured over the whole 90-picture set, four seeds, before and after:

| | before — 128 px, JPEG 0.72 | after — 224 px, PNG |
|---|---|---|
| pictures that got a box | **19 of 90** | **3 of 90** |
| boxes drawn in total | 22 | 3 |
| smallest box | 8×14 — a speck | 61×61 — the whole shape |
| display: pixels that are neither the background nor the shape | 2.30% | **0.48%** |

Every speck is gone. What is left is the other kind — the whole shape, when its hue happens
to land in the skin range — and that is a colour coincidence rather than blur, so it is left
alone. Tuning the shapes' colours, or the threshold, to hide it would start rejecting small
faces in real photographs, which is the thing the detector exists to find. The practise toast
says a box on a drawn shape is the detector guessing, and the **×** is the answer.

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
| `src/faces.ts` | the page's way in to the face search: the `Box` and `Frame` types |
| `src/haar.ts` | the face detector — Viola-Jones, over typed arrays |
| `src/cascade-data.ts` | **generated**: OpenCV's cascade, packed as base64 typed arrays |
| `tools/build-cascade.mjs` | regenerates `src/cascade-data.ts` from the cascade XML |
| `src/store.ts` | IndexedDB — the pictures and the model |
| `src/trainer-host.ts` | the study loop, one pass at a time, in 40 ms slices |
| `src/trainer.worker.ts` | runs it off the main thread |
| `src/charts.ts` | every chart, drawn by hand on a canvas |
| `src/samples.ts` | the practise photographs, and where each came from |
| `src/app.ts` | the page |
| `site/` | the hand-written page, plus **generated** JavaScript |

**`site/*.js` is compiled from `src/*.ts` — never edit it there.**

## Not here

No tests, no analytics, no sitemap, no deployment. Deliberately.
