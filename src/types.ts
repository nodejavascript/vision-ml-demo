/**
 * types.ts — the contract between the page and the trainer.
 *
 * These types are the reason this file exists separately. The page and the
 * worker are two programs that must agree exactly; keeping the message shapes in
 * one place means a field the page reads but the worker never sends is a compile
 * error rather than an `undefined` at runtime.
 */

/** The edge length, in pixels, that every image is reduced to before it is seen. */
export const SAMPLE = 48;

/** How many images one training pass holds before the weights move. */
export const BATCH = 8;

/** Default size of the hidden layer between the convolutions and the classes. */
export const HIDDEN = 32;

/** The two convolution widths. Kept in one place so the model card can print them. */
export const CONV1 = 8;
export const CONV2 = 16;

/* ------------------------------------------------------------------ *
 * The model, in a form that survives a file, a worker and a database
 * ------------------------------------------------------------------ */

/**
 * A trained model as data. The weights travel as base64 of their own buffers,
 * which keeps a 75,000-parameter model to a few hundred kilobytes of JSON
 * instead of a million characters of decimal.
 */
export interface ModelFile {
  format: 'vision-ml-demo-model';
  /**
   * 2 = multi-label. Version 1 was a softmax over one label per picture, so its
   * weights mean something entirely different and must not be loaded.
   */
  version: 2;
  classes: string[];
  sample: number;
  conv1: number;
  conv2: number;
  hidden: number;
  /** Ordered: w1, b1, w2, b2, w3, b3, w4, b4. */
  blobs: string[];
  /** What it was trained on, so a loaded model can say where it came from. */
  meta: ModelMeta;
}

export interface ModelMeta {
  createdAt: string;
  updatedAt: string;
  epochsTrained: number;
  images: number;
  parameters: number;
  /** Per-epoch history, so the charts can be redrawn from a recalled model. */
  history: EpochMetric[];
  /**
   * For each class, how often it was right about it, and how many pictures
   * actually had it. Replaces the old confusion matrix: with a picture allowed to
   * hold several things, there is no single "truth" to put on a row.
   */
  perClassAccuracy: number[];
  perClassCount: number[];
  /** How decided it is, on average, about each thing being there or not. */
  meanConfidence: number;
}

export function emptyMeta(): ModelMeta {
  return {
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    epochsTrained: 0,
    images: 0,
    parameters: 0,
    history: [],
    perClassAccuracy: [],
    perClassCount: [],
    meanConfidence: 0,
  };
}

/* ------------------------------------------------------------------ *
 * Training
 * ------------------------------------------------------------------ */

export interface EpochMetric {
  epoch: number;
  loss: number;
  trainAccuracy: number;
  valAccuracy: number;
  valLoss: number;
}

export interface TrainSample {
  id: string;
  /**
   * Which of the classes this picture contains. One or more — that is the whole
   * point of asking for a list rather than a single name: a photograph of a dog on
   * a beach is honestly both, and it can teach two things at once.
   */
  targets: number[];
  pixels: Uint8Array;
}

export interface TrainPayload {
  samples: TrainSample[];
  classes: string[];
  /** Continue from these weights, or null to start from scratch. */
  weights: ModelFile | null;
  epochs: number;
  learningRate: number;
  /** Fraction held back from training so the accuracy curve means something. */
  validationSplit: number;
  augment: boolean;
  seed: number;
  /**
   * Give up after this many milliseconds, even with epochs left.
   *
   * A fixed number of passes takes longer the more pictures there are, so a page
   * that studies after every single answer would get slower and slower until it
   * felt broken. A budget keeps the wait the same at ten pictures and at two
   * hundred — and because each run continues from the last set of weights, the
   * learning still accumulates across answers rather than being lost.
   */
  budgetMs?: number;
}

export interface TrainStarted {
  type: 'started';
  epochs: number;
  parameters: number;
  trainCount: number;
  valCount: number;
  classes: string[];
}

export interface TrainProgress {
  type: 'progress';
  metric: EpochMetric;
}

export interface TrainFinished {
  type: 'done' | 'stopped';
  metric: EpochMetric;
  weights: ModelFile;
}

export interface TrainError {
  type: 'error';
  message: string;
}

export type HostMessage = TrainStarted | TrainProgress | TrainFinished | TrainError;

export type HostRequest =
  | { type: 'train'; payload: TrainPayload }
  | { type: 'stop' };

/* ------------------------------------------------------------------ *
 * What the page keeps about each image
 * ------------------------------------------------------------------ */

export interface Sample {
  id: string;
  /** Everything the person said is in the picture. Empty until they say. */
  labels: string[];
  /** A data URL, for the grid. Kept beside the pixels so a reload loses nothing. */
  thumb: string;
  /** SAMPLE × SAMPLE × 3, red first, 0–255. The model's actual view of the image. */
  pixels: Uint8Array;
  name: string;
  addedAt: string;
  /** Where it came from: the visitor's disk, or the built-in sample set. */
  origin: 'file' | 'sample';
}
