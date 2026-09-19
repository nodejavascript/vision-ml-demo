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
export function emptyMeta() {
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
