/**
 * trainer.worker.ts — runs the training off the main thread, so the page keeps
 * painting while the model learns.
 *
 * It is a module worker and imports the same model code the page does. If
 * module workers are unavailable the page runs the identical host on the main
 * thread instead — the arithmetic is the same either way.
 */

import { VisionTrainer } from './trainer-host.js';
import type { HostMessage, HostRequest } from './types.js';

// `lib` in tsconfig includes DOM (the page needs it) and not WebWorker, because
// the two cannot both be loaded — they declare `self` differently and conflict.
// This single cast is the price, and everything below is typed from here on.
const ctx = self as unknown as {
  postMessage(message: HostMessage): void;
  onmessage: ((event: MessageEvent<HostRequest>) => void) | null;
};

const trainer = new VisionTrainer((message: HostMessage): void => {
  ctx.postMessage(message);
});

ctx.onmessage = (event: MessageEvent<HostRequest>): void => {
  trainer.handle(event.data);
};
