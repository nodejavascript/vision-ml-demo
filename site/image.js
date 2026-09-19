/**
 * image.ts — from a file on disk to the array of numbers the model sees.
 *
 * Everything here happens on the visitor's own machine. Nothing is uploaded,
 * because there is nowhere to upload it to: this page has no server behind it.
 */
import { SAMPLE } from './types.js';
function makeCanvas(width, height) {
    const canvas = document.createElement('canvas');
    canvas.width = width;
    canvas.height = height;
    return canvas;
}
function context(canvas) {
    const ctx = canvas.getContext('2d', { willReadFrequently: true });
    if (!ctx)
        throw new Error('This browser would not give a 2D canvas context.');
    return ctx;
}
/** Read a canvas back as raw RGB bytes, at the size the model was built for. */
export function pixelsFromCanvas(source, size = SAMPLE) {
    const work = makeCanvas(size, size);
    const ctx = context(work);
    // 'high' matters more than it looks: at 48 pixels the difference between a
    // box filter and a proper one is the difference between a shape and a smear.
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, size, size);
    const data = ctx.getImageData(0, 0, size, size).data;
    const out = new Uint8Array(size * size * 3);
    for (let i = 0, j = 0; i < out.length; i += 3, j += 4) {
        out[i] = data[j];
        out[i + 1] = data[j + 1];
        out[i + 2] = data[j + 2];
    }
    return out;
}
/** A small JPEG for the grid, so a reloaded page shows the images instantly. */
export function thumbFromCanvas(source, max = 240) {
    const scale = Math.min(1, max / Math.max(source.width, source.height));
    const width = Math.max(1, Math.round(source.width * scale));
    const height = Math.max(1, Math.round(source.height * scale));
    const work = makeCanvas(width, height);
    const ctx = context(work);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, width, height);
    return work.toDataURL('image/jpeg', 0.72);
}
/** Decode a chosen file into both the model's view and a displayable thumbnail. */
export async function decodeFile(file, size = SAMPLE) {
    const bitmap = await createImageBitmap(file);
    try {
        const canvas = makeCanvas(bitmap.width, bitmap.height);
        context(canvas).drawImage(bitmap, 0, 0);
        return { pixels: pixelsFromCanvas(canvas, size), thumb: thumbFromCanvas(canvas) };
    }
    finally {
        bitmap.close();
    }
}
/** How wide the face search looks. Wide enough to keep a small face, small enough to be quick. */
const ANALYSIS = 224;
function frameFromCanvas(source) {
    const scale = Math.min(1, ANALYSIS / Math.max(source.width, source.height));
    const width = Math.max(1, Math.round(source.width * scale));
    const height = Math.max(1, Math.round(source.height * scale));
    const work = makeCanvas(width, height);
    const ctx = context(work);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = 'high';
    ctx.drawImage(source, 0, 0, width, height);
    const data = ctx.getImageData(0, 0, width, height).data;
    const rgb = new Uint8Array(width * height * 3);
    for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
        rgb[i] = data[j];
        rgb[i + 1] = data[j + 1];
        rgb[i + 2] = data[j + 2];
    }
    return {
        rgb,
        width,
        height,
        sourceWidth: source.width,
        sourceHeight: source.height,
    };
}
export async function loadPicture(file, size = SAMPLE) {
    const bitmap = await createImageBitmap(file);
    try {
        const source = makeCanvas(bitmap.width, bitmap.height);
        context(source).drawImage(bitmap, 0, 0);
        const frame = frameFromCanvas(source);
        const region = (box) => {
            if (!box)
                return source;
            const cut = makeCanvas(box.w, box.h);
            context(cut).drawImage(source, box.x, box.y, box.w, box.h, 0, 0, box.w, box.h);
            return cut;
        };
        return {
            width: source.width,
            height: source.height,
            kind: file.type || 'image',
            filename: file instanceof File ? file.name : 'a picture',
            thumb: thumbFromCanvas(source),
            analysisWidth: frame.width,
            analysisHeight: frame.height,
            frame,
            whole: () => pixelsFromCanvas(source, size),
            crop: (box) => pixelsFromCanvas(region(box), size),
            // Square-ish so a face is framed the same whatever shape the box came out as.
            cropThumb: (box) => thumbFromCanvas(region(box), 160),
        };
    }
    finally {
        bitmap.close();
    }
}
/** Paint one sample's 48x48 bytes into a canvas, magnified, for inspection. */
export function paintSample(canvas, pixels, size = SAMPLE) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 160;
    const height = canvas.clientHeight || 160;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    const ctx = context(canvas);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.clearRect(0, 0, width, height);
    const off = makeCanvas(size, size);
    const offCtx = context(off);
    const image = offCtx.createImageData(size, size);
    for (let i = 0, j = 0; i < pixels.length; i += 3, j += 4) {
        image.data[j] = pixels[i];
        image.data[j + 1] = pixels[i + 1];
        image.data[j + 2] = pixels[i + 2];
        image.data[j + 3] = 255;
    }
    offCtx.putImageData(image, 0, 0);
    // Nearest-neighbour on purpose: the point is to show the actual 48x48 grid of
    // pixels the network reads, not a smoothed picture that hides it.
    ctx.drawImage(off, 0, 0, width, height);
}
