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
  format: 'vision-demo-model';
  version: 1;
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
  /** Final confusion matrix over the labelled set, row = truth, column = guess. */
  confusion: number[][];
  /** Mean confidence the model had on the set it was trained on. */
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
    confusion: [],
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
  classIndex: number;
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
  /** null until the visitor names it, or the model is confident enough to. */
  label: string | null;
  /** A data URL, for the grid. Kept beside the pixels so a reload loses nothing. */
  thumb: string;
  /** SAMPLE × SAMPLE × 3, red first, 0–255. The model's actual view of the image. */
  pixels: Uint8Array;
  name: string;
  addedAt: string;
  /** Where it came from: the visitor's disk, or the built-in sample set. */
  origin: 'file' | 'sample';
}

/** A prediction for one image. */
export interface Prediction {
  id: string;
  probs: Float32Array;
  topIndex: number;
  topClass: string;
  confidence: number;
  /** True when the model is not confident enough to be worth believing. */
  unsure: boolean;
}
