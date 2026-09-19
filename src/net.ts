/**
 * net.ts — a small convolutional network, written from scratch.
 *
 * There is no framework here and no matrix library: the convolutions, the
 * pooling, the dense layers, the softmax and the Adam update are all plain
 * loops over typed arrays, and every gradient is derived in the comment beside
 * the loop that computes it. The whole model is about 75,000 numbers, which is
 * small enough to read and small enough to train in front of you.
 *
 * Shape, for a 48-pixel input and three classes:
 *
 *   image 3x48x48
 *     conv 3x3, 8 filters, same padding      -> 8x48x48
 *     relu, max-pool 2x2                     -> 8x24x24
 *     conv 3x3, 16 filters, same padding     -> 16x24x24
 *     relu, max-pool 2x2                     -> 16x12x12
 *     flatten                                -> 2304
 *     dense -> 32, relu                      -> 32
 *     dense -> classes                       -> classes, softmax
 *
 * Two things are deliberately true:
 *
 *  1. `forward` and `backward` are separate, and `backward` is given the
 *     gradient of the loss with respect to the logits. Nothing is inferred.
 *  2. The flatten is free: `a2` is stored channel-major with the spatial axes
 *     last, so its memory order IS the flat vector the dense layer wants. That
 *     removes a copy and, more usefully, removes a place for an index bug.
 */

import {
  BATCH,
  CONV1,
  CONV2,
  HIDDEN,
  SAMPLE,
  emptyMeta,
} from './types.js';
import type { EpochMetric, ModelFile, TrainSample } from './types.js';

const K = 3; // convolution width
const PAD = 1; // 'same' padding for a 3x3 kernel

/**
 * One learnable tensor with everything the optimiser needs beside it. Keeping
 * the moment estimates next to the weights rather than in a side table means
 * there is no way to add a parameter and forget to give it an optimiser.
 */
class Param {
  size: number;
  data: Float32Array;
  grad: Float32Array;
  m: Float32Array;
  v: Float32Array;

  constructor(size: number) {
    this.size = size;
    this.data = new Float32Array(size);
    this.grad = new Float32Array(size);
    this.m = new Float32Array(size);
    this.v = new Float32Array(size);
  }

  zeroGrad(): void {
    this.grad.fill(0);
  }

  /** Append room for new rows, keeping what is already learned in place. */
  grow(newSize: number): void {
    if (newSize <= this.size) return;
    const copy = (src: Float32Array): Float32Array => {
      const next = new Float32Array(newSize);
      next.set(src.subarray(0, this.size));
      return next;
    };
    this.data = copy(this.data);
    this.grad = copy(this.grad);
    this.m = copy(this.m);
    this.v = copy(this.v);
    this.size = newSize;
  }
}

export function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** A normal sample, from two uniforms, so initialisation needs no library. */
function gaussian(rng: () => number): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

/** He initialisation: the variance ReLU needs to keep its activations alive. */
function fillHe(target: Float32Array, fanIn: number, rng: () => number): void {
  const scale = Math.sqrt(2 / fanIn);
  for (let i = 0; i < target.length; i++) target[i] = gaussian(rng) * scale;
}

/* ------------------------------------------------------------------ *
 * Pixels
 * ------------------------------------------------------------------ */

/**
 * The model never sees a byte. It sees 0-centred floats, which is what a
 * convolution with a bias wants: a mid-grey pixel should contribute nothing.
 */
export function normalise(pixels: Uint8Array, out: Float32Array): Float32Array {
  const n = Math.min(pixels.length, out.length);
  for (let i = 0; i < n; i++) out[i] = pixels[i] / 127.5 - 1;
  return out;
}

/**
 * A deliberately blunt augmentation: flip, nudge up to two pixels, and change
 * the exposure. It is not clever, but on a set this small it is the difference
 * between a model that recognises the object and one that recognises the photo.
 */
export function augment(pixels: Uint8Array, size: number, rng: () => number): Uint8Array {
  const out = new Uint8Array(size * size * 3);
  const flip = rng() < 0.5;
  const dx = Math.round((rng() - 0.5) * 4);
  const dy = Math.round((rng() - 0.5) * 4);
  const gain = 0.82 + rng() * 0.36;
  for (let y = 0; y < size; y++) {
    const sy = y - dy;
    if (sy < 0 || sy >= size) continue;
    for (let x = 0; x < size; x++) {
      const sx = flip ? size - 1 - x : x;
      const tx = sx - dx;
      if (tx < 0 || tx >= size) continue;
      const si = (sy * size + tx) * 3;
      const di = (y * size + x) * 3;
      for (let c = 0; c < 3; c++) {
        const v = pixels[si + c] * gain + (1 - gain) * 128;
        out[di + c] = v < 0 ? 0 : v > 255 ? 255 : v;
      }
    }
  }
  return out;
}

/* ------------------------------------------------------------------ *
 * The network
 * ------------------------------------------------------------------ */

export class Cnn {
  readonly size: number;
  readonly q1: number;
  readonly q2: number;
  readonly flat: number;
  readonly hidden: number;
  classes: number;

  // Weights, in the order they are serialised.
  readonly w1: Param;
  readonly b1: Param;
  readonly w2: Param;
  readonly b2: Param;
  readonly w3: Param;
  readonly b3: Param;
  readonly w4: Param;
  readonly b4: Param;

  // Activations. Held between forward and backward on purpose: recomputing them
  // would cost more than the memory does.
  private readonly x: Float32Array;
  private readonly a1: Float32Array;
  private readonly p1: Float32Array;
  private readonly a2: Float32Array;
  private readonly p2: Float32Array;
  private readonly h: Float32Array;
  private logits: Float32Array;
  private probs: Float32Array;
  private readonly arg1: Int32Array;
  private readonly arg2: Int32Array;

  // Gradients. Separate buffers from the activations, so nothing is written over
  // something that is still to be read.
  private gLogits: Float32Array;
  private readonly gH: Float32Array;
  private readonly gP2: Float32Array;
  private readonly gA2: Float32Array;
  private readonly gP1: Float32Array;
  private readonly gA1: Float32Array;

  private step = 0;

  /**
   * Adam's first steps are the full learning rate, taken on random weights, and
   * that shows up as a loss which jumps UP before it comes down. Measured on
   * this network: the first epoch's mean loss was 8.9 at a learning rate of 0.01
   * and 2.4 at 0.001 — the spike scales with the rate, so the weights are being
   * hurt before they are helped. Ramping in over the first few batches removes
   * it, and it is what a real training script does anyway.
   */
  warmupSteps = 10;

  constructor(classes: number, size = SAMPLE, hidden = HIDDEN, seed = 1234) {
    this.size = size;
    this.q1 = size / 2;
    this.q2 = size / 4;
    this.flat = CONV2 * this.q2 * this.q2;
    this.hidden = hidden;
    this.classes = Math.max(1, classes);

    const rng = mulberry32(seed);
    this.w1 = new Param(CONV1 * 3 * K * K);
    this.b1 = new Param(CONV1);
    this.w2 = new Param(CONV2 * CONV1 * K * K);
    this.b2 = new Param(CONV2);
    this.w3 = new Param(this.hidden * this.flat);
    this.b3 = new Param(this.hidden);
    this.w4 = new Param(this.classes * this.hidden);
    this.b4 = new Param(this.classes);

    fillHe(this.w1.data, 3 * K * K, rng);
    fillHe(this.w2.data, CONV1 * K * K, rng);
    fillHe(this.w3.data, this.flat, rng);
    fillHe(this.w4.data, this.hidden, rng);

    this.x = new Float32Array(3 * size * size);
    this.a1 = new Float32Array(CONV1 * size * size);
    this.p1 = new Float32Array(CONV1 * this.q1 * this.q1);
    this.a2 = new Float32Array(CONV2 * this.q1 * this.q1);
    this.p2 = new Float32Array(CONV2 * this.q2 * this.q2);
    this.h = new Float32Array(this.hidden);
    this.logits = new Float32Array(this.classes);
    this.probs = new Float32Array(this.classes);
    this.arg1 = new Int32Array(CONV1 * this.q1 * this.q1);
    this.arg2 = new Int32Array(CONV2 * this.q2 * this.q2);

    this.gLogits = new Float32Array(this.classes);
    this.gH = new Float32Array(this.hidden);
    this.gP2 = new Float32Array(this.flat);
    this.gA2 = new Float32Array(CONV2 * this.q1 * this.q1);
    this.gP1 = new Float32Array(CONV1 * this.q1 * this.q1);
    this.gA1 = new Float32Array(CONV1 * size * size);
  }

  private get params(): Param[] {
    return [this.w1, this.b1, this.w2, this.b2, this.w3, this.b3, this.w4, this.b4];
  }

  get parameterCount(): number {
    let total = 0;
    for (const p of this.params) total += p.size;
    return total;
  }

  /**
   * Make room for more classes, keeping the ones already learned. The output
   * layer is laid out class-major, so a new class appends cleanly and the rows
   * above it keep their meaning — which is exactly what "add a new thing to
   * look for" has to do to be worth offering.
   */
  setClasses(count: number): void {
    const next = Math.max(1, count);
    if (next === this.classes) return;
    this.w4.grow(next * this.hidden);
    this.b4.grow(next);
    this.classes = next;
    // The scratch buffers are sized by the class count, so they are rebuilt.
    this.logits = new Float32Array(next);
    this.probs = new Float32Array(next);
    this.gLogits = new Float32Array(next);
  }

  /* ---------------- forward ---------------- */

  private conv(
    input: Float32Array,
    channels: number,
    height: number,
    width: number,
    weight: Param,
    bias: Param,
    filters: number,
    out: Float32Array,
  ): void {
    const w = weight.data;
    for (let f = 0; f < filters; f++) {
      const b = bias.data[f];
      for (let y = 0; y < height; y++) {
        for (let x = 0; x < width; x++) {
          let sum = b;
          for (let c = 0; c < channels; c++) {
            for (let ky = 0; ky < K; ky++) {
              const iy = y + ky - PAD;
              if (iy < 0 || iy >= height) continue;
              for (let kx = 0; kx < K; kx++) {
                const ix = x + kx - PAD;
                if (ix < 0 || ix >= width) continue;
                sum += input[(c * height + iy) * width + ix] * w[((f * channels + c) * K + ky) * K + kx];
              }
            }
          }
          out[(f * height + y) * width + x] = sum;
        }
      }
    }
  }

  private pool(input: Float32Array, channels: number, height: number, width: number, out: Float32Array, arg: Int32Array): void {
    const oh = height / 2;
    const ow = width / 2;
    for (let c = 0; c < channels; c++) {
      for (let oy = 0; oy < oh; oy++) {
        for (let ox = 0; ox < ow; ox++) {
          let best = -Infinity;
          let at = 0;
          for (let dy = 0; dy < 2; dy++) {
            for (let dx = 0; dx < 2; dx++) {
              const iy = oy * 2 + dy;
              const ix = ox * 2 + dx;
              const v = input[(c * height + iy) * width + ix];
              if (v > best) {
                best = v;
                at = iy * width + ix;
              }
            }
          }
          out[(c * oh + oy) * ow + ox] = best;
          arg[(c * oh + oy) * ow + ox] = at;
        }
      }
    }
  }

  /** Run one image and keep every activation, for the backward pass. */
  forward(pixels: Uint8Array): Float32Array {
    normalise(pixels, this.x);
    const s = this.size;
    this.conv(this.x, 3, s, s, this.w1, this.b1, CONV1, this.a1);
    relu(this.a1);
    this.pool(this.a1, CONV1, s, s, this.p1, this.arg1);
    this.conv(this.p1, CONV1, this.q1, this.q1, this.w2, this.b2, CONV2, this.a2);
    relu(this.a2);
    this.pool(this.a2, CONV2, this.q1, this.q1, this.p2, this.arg2);

    // p2 is already channel-major with spatial last, so it is the flat vector.
    dense(this.p2, this.flat, this.h, this.w3, this.b3, this.hidden);
    relu(this.h);
    dense(this.h, this.hidden, this.logits, this.w4, this.b4, this.classes);
    softmax(this.logits, this.probs, this.classes);
    return this.probs;
  }

  /** Probabilities only — no caches, so it is safe to call anywhere. */
  predict(pixels: Uint8Array): Float32Array {
    return this.forward(pixels).slice();
  }

  /* ---------------- backward ---------------- */

  /**
   * Given the gradient of the loss with respect to the logits, push it back
   * through every layer and accumulate into `grad`.
   */
  private backward(target: number): void {
    // d(loss)/d(logit) for softmax + cross-entropy is just p - onehot.
    for (let j = 0; j < this.classes; j++) {
      this.gLogits[j] = this.probs[j] - (j === target ? 1 : 0);
    }

    // dense out: dW = g (x) h, db = g, dh = W^T g
    denseBackward(this.h, this.hidden, this.gLogits, this.classes, this.w4, this.b4, this.gH);
    // relu: the gradient survives only where the activation did.
    for (let i = 0; i < this.hidden; i++) if (this.h[i] <= 0) this.gH[i] = 0;

    // dense in: input is the flattened pooled feature map.
    denseBackward(this.p2, this.flat, this.gH, this.hidden, this.w3, this.b3, this.gP2);

    // pool 2: each gradient goes back to the one cell that won.
    poolBackward(this.gP2, CONV2, this.q1, this.arg2, this.gA2);
    for (let i = 0; i < this.a2.length; i++) if (this.a2[i] <= 0) this.gA2[i] = 0;

    convBackward(this.p1, CONV1, this.q1, this.q1, this.gA2, CONV2, this.w2, this.b2, this.gP1);
    poolBackward(this.gP1, CONV1, this.size, this.arg1, this.gA1);
    for (let i = 0; i < this.a1.length; i++) if (this.a1[i] <= 0) this.gA1[i] = 0;

    convBackward(this.x, 3, this.size, this.size, this.gA1, CONV1, this.w1, this.b1, null);
  }

  private zeroGrads(): void {
    for (const p of this.params) p.zeroGrad();
  }

  /** One Adam step. The bias correction is why `step` is a field. */
  private applyGradients(scale: number, lr: number, beta1 = 0.9, beta2 = 0.999, eps = 1e-8): void {
    this.step += 1;
    const ramp = this.warmupSteps > 0 ? Math.min(1, this.step / this.warmupSteps) : 1;
    const rate = lr * ramp;
    const b1t = 1 - Math.pow(beta1, this.step);
    const b2t = 1 - Math.pow(beta2, this.step);
    for (const p of this.params) {
      const { data, grad, m, v } = p;
      for (let i = 0; i < data.length; i++) {
        const g = grad[i] * scale;
        m[i] = beta1 * m[i] + (1 - beta1) * g;
        v[i] = beta2 * v[i] + (1 - beta2) * g * g;
        data[i] -= (rate * (m[i] / b1t)) / (Math.sqrt(v[i] / b2t) + eps);
      }
    }
  }

  /* ---------------- training ---------------- */

  /**
   * One pass over the training images, in shuffled batches. Returns the mean
   * cross-entropy loss and the fraction the model got right — the two numbers
   * the page plots.
   */
  trainEpoch(samples: TrainSample[], lr: number, useAugment: boolean, rng: () => number): { loss: number; accuracy: number } {
    const order = samples.map((_, i) => i);
    for (let i = order.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [order[i], order[j]] = [order[j], order[i]];
    }

    let lossSum = 0;
    let correct = 0;

    for (let start = 0; start < order.length; start += BATCH) {
      const end = Math.min(start + BATCH, order.length);
      const count = end - start;
      this.zeroGrads();
      for (let k = start; k < end; k++) {
        const sample = samples[order[k]];
        const pixels = useAugment ? augment(sample.pixels, this.size, rng) : sample.pixels;
        const probs = this.forward(pixels);
        lossSum += -Math.log(Math.max(probs[sample.classIndex], 1e-9));
        let best = 0;
        for (let j = 1; j < this.classes; j++) if (probs[j] > probs[best]) best = j;
        if (best === sample.classIndex) correct += 1;
        this.backward(sample.classIndex);
      }
      this.applyGradients(1 / count, lr);
    }

    const n = Math.max(1, order.length);
    return { loss: lossSum / n, accuracy: correct / n };
  }

  /** Loss and accuracy with the weights frozen and no augmentation. */
  evaluate(samples: TrainSample[]): { loss: number; accuracy: number } {
    let lossSum = 0;
    let correct = 0;
    for (const sample of samples) {
      const probs = this.forward(sample.pixels);
      lossSum += -Math.log(Math.max(probs[sample.classIndex], 1e-9));
      let best = 0;
      for (let j = 1; j < this.classes; j++) if (probs[j] > probs[best]) best = j;
      if (best === sample.classIndex) correct += 1;
    }
    const n = Math.max(1, samples.length);
    return { loss: lossSum / n, accuracy: correct / n };
  }

  /* ---------------- things worth looking at ---------------- */

  /**
   * The eight first-layer kernels, as colour. Each 3x3 kernel reaches across
   * three input channels, so the three values at a position are naturally a red,
   * a green and a blue — the picture is literally what the filter looks for.
   */
  conv1Filters(): { tiles: Float32Array[]; max: number } {
    const tiles: Float32Array[] = [];
    let max = 1e-6;
    for (let f = 0; f < CONV1; f++) {
      const tile = new Float32Array(K * K * 3);
      for (let c = 0; c < 3; c++) {
        for (let ky = 0; ky < K; ky++) {
          for (let kx = 0; kx < K; kx++) {
            const v = this.w1.data[((f * 3 + c) * K + ky) * K + kx];
            tile[(ky * K + kx) * 3 + c] = v;
            max = Math.max(max, Math.abs(v));
          }
        }
      }
      tiles.push(tile);
    }
    return { tiles, max };
  }

  /** The eight first-layer activations for one image: what the model noticed. */
  featureMaps(pixels: Uint8Array): { maps: Float32Array[]; max: number } {
    this.forward(pixels);
    const maps: Float32Array[] = [];
    let max = 1e-6;
    const area = this.size * this.size;
    for (let f = 0; f < CONV1; f++) {
      const map = this.a1.slice(f * area, (f + 1) * area);
      for (let i = 0; i < map.length; i++) max = Math.max(max, map[i]);
      maps.push(map);
    }
    return { maps, max };
  }

  /** Mean confidence across a set, for the model card. */
  meanConfidence(samples: TrainSample[]): number {
    if (samples.length === 0) return 0;
    let total = 0;
    for (const sample of samples) {
      const probs = this.predict(sample.pixels);
      let best = 0;
      for (let j = 1; j < this.classes; j++) if (probs[j] > probs[best]) best = j;
      total += probs[best];
    }
    return total / samples.length;
  }

  /* ---------------- save and load ---------------- */

  serialize(classes: string[], history: EpochMetric[], images: number, confusion: number[][], meanConfidence: number, previous?: ModelFile | null): ModelFile {
    const blobs = this.params.map((p) => encodeFloats(p.data));
    const meta = previous ? { ...previous.meta } : emptyMeta();
    meta.updatedAt = new Date().toISOString();
    meta.images = images;
    meta.parameters = this.parameterCount;
    meta.history = history.slice();
    meta.confusion = confusion;
    meta.meanConfidence = meanConfidence;
    return {
      format: 'vision-demo-model',
      version: 1,
      classes: classes.slice(),
      sample: this.size,
      conv1: CONV1,
      conv2: CONV2,
      hidden: this.hidden,
      blobs,
      meta,
    };
  }

  static load(file: ModelFile): Cnn {
    const net = new Cnn(file.classes.length, file.sample, file.hidden);
    const blobs = file.blobs.map(decodeFloats);
    const params = [net.w1, net.b1, net.w2, net.b2, net.w3, net.b3, net.w4, net.b4];
    for (let i = 0; i < params.length; i++) {
      const source = blobs[i];
      if (!source || source.length !== params[i].size) {
        throw new Error(`Model file does not match this network (tensor ${i}).`);
      }
      params[i].data.set(source);
    }
    return net;
  }
}

/* ------------------------------------------------------------------ *
 * Free functions — the parts that read better outside the class
 * ------------------------------------------------------------------ */

function relu(values: Float32Array): void {
  for (let i = 0; i < values.length; i++) if (values[i] < 0) values[i] = 0;
}

function dense(input: Float32Array, inputs: number, out: Float32Array, weight: Param, bias: Param, outputs: number): void {
  const w = weight.data;
  for (let j = 0; j < outputs; j++) {
    let sum = bias.data[j];
    const row = j * inputs;
    for (let i = 0; i < inputs; i++) sum += input[i] * w[row + i];
    out[j] = sum;
  }
}

/**
 * The three gradients a dense layer needs. `gIn` may be null at the very first
 * layer, where there is nothing behind it to update.
 */
function denseBackward(
  input: Float32Array,
  inputs: number,
  gOut: Float32Array,
  outputs: number,
  weight: Param,
  bias: Param,
  gIn: Float32Array,
): void {
  for (let i = 0; i < inputs; i++) gIn[i] = 0;
  for (let j = 0; j < outputs; j++) {
    const g = gOut[j];
    const row = j * inputs;
    bias.grad[j] += g;
    for (let i = 0; i < inputs; i++) {
      weight.grad[row + i] += g * input[i];
      gIn[i] += g * weight.data[row + i];
    }
  }
}

/** Each pooled gradient returns to the single cell that supplied the maximum. */
function poolBackward(gOut: Float32Array, channels: number, size: number, arg: Int32Array, gIn: Float32Array): void {
  gIn.fill(0);
  const half = size / 2;
  for (let c = 0; c < channels; c++) {
    for (let oy = 0; oy < half; oy++) {
      for (let ox = 0; ox < half; ox++) {
        const at = (c * half + oy) * half + ox;
        const g = gOut[at];
        if (g === 0) continue;
        gIn[c * size * size + arg[at]] += g;
      }
    }
  }
}

/**
 * The convolution, backwards. Every weight is credited with the products it
 * contributed to, and every input pixel is charged the sum of the weights that
 * used it — the two loops are the forward loop read in the other direction.
 */
function convBackward(
  input: Float32Array,
  channels: number,
  height: number,
  width: number,
  gOut: Float32Array,
  filters: number,
  weight: Param,
  bias: Param,
  gIn: Float32Array | null,
): void {
  if (gIn) gIn.fill(0);
  const w = weight.data;
  for (let f = 0; f < filters; f++) {
    for (let y = 0; y < height; y++) {
      for (let x = 0; x < width; x++) {
        const g = gOut[(f * height + y) * width + x];
        if (g === 0) continue;
        bias.grad[f] += g;
        for (let c = 0; c < channels; c++) {
          for (let ky = 0; ky < K; ky++) {
            const iy = y + ky - PAD;
            if (iy < 0 || iy >= height) continue;
            for (let kx = 0; kx < K; kx++) {
              const ix = x + kx - PAD;
              if (ix < 0 || ix >= width) continue;
              const wi = ((f * channels + c) * K + ky) * K + kx;
              const xi = (c * height + iy) * width + ix;
              weight.grad[wi] += g * input[xi];
              if (gIn) gIn[xi] += g * w[wi];
            }
          }
        }
      }
    }
  }
}

function softmax(logits: Float32Array, out: Float32Array, count: number): void {
  let max = -Infinity;
  for (let i = 0; i < count; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < count; i++) {
    const e = Math.exp(logits[i] - max);
    out[i] = e;
    sum += e;
  }
  const inv = sum > 0 ? 1 / sum : 0;
  for (let i = 0; i < count; i++) out[i] *= inv;
}

/* ------------------------------------------------------------------ *
 * Base64, so a model file stays a sane size
 * ------------------------------------------------------------------ */

const CHUNK = 8192;

export function encodeFloats(values: Float32Array): string {
  const bytes = new Uint8Array(values.buffer, values.byteOffset, values.byteLength);
  let binary = '';
  for (let i = 0; i < bytes.length; i += CHUNK) {
    binary += String.fromCharCode.apply(null, Array.from(bytes.subarray(i, i + CHUNK)));
  }
  return btoa(binary);
}

export function decodeFloats(text: string): Float32Array {
  const binary = atob(text);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer, 0, bytes.length >> 2);
}
