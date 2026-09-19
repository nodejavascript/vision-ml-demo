/**
 * charts.ts — every picture in the page is drawn here, onto a canvas, by hand.
 *
 * No charting library: the page has no dependencies at all, and a line chart is
 * about forty lines of arithmetic. Each function follows the same two rules,
 * because both were learned the expensive way on a sibling page:
 *
 *  1. An empty chart must INVITE, not sit blank. An empty box on a dark page
 *     reads as something broken, so every chart here draws a sentence when it
 *     has nothing to show.
 *  2. The last point is marked and named. The shape of a curve is the story, and
 *     the end of it is where the thing has actually got to.
 *
 * Every canvas is device-pixel-ratio aware, so nothing is soft on a retina
 * screen, and every colour is read from the stylesheet rather than repeated.
 */

export interface Series {
  values: number[];
  color: string;
  label: string;
}

export interface LineOptions {
  series: Series[];
  /** Pin the axis, for anything that already has a natural range (accuracy: 0–1). */
  floor?: number;
  ceil?: number;
  /** Mark and name the end of the first series. */
  annotate?: boolean;
  emptyTitle?: string;
  emptyHint?: string;
  format?: (value: number) => string;
}

export interface BarItem {
  label: string;
  value: number;
  color?: string;
  caption?: string;
}

interface Theme {
  accent: string;
  yellow: string;
  text: string;
  muted: string;
  line: string;
  surface: string;
}

let cachedTheme: Theme | null = null;

export function theme(): Theme {
  if (cachedTheme) return cachedTheme;
  const styles = getComputedStyle(document.body);
  const read = (name: string, fallback: string): string => styles.getPropertyValue(name).trim() || fallback;
  cachedTheme = {
    accent: read('--accent', '#a78bfa'),
    yellow: read('--yellow', '#fbbf24'),
    text: read('--text', '#ece6f7'),
    muted: read('--muted', '#a89cc4'),
    line: read('--line', 'rgba(255,255,255,0.09)'),
    surface: read('--surface', '#140b22'),
  };
  return cachedTheme;
}

interface Surface {
  ctx: CanvasRenderingContext2D;
  width: number;
  height: number;
}

function prepare(canvas: HTMLCanvasElement, minHeight = 90): Surface | null {
  const ctx = canvas.getContext('2d');
  if (!ctx) return null;
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth || canvas.parentElement?.clientWidth || 320;
  const height = canvas.clientHeight || minHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  ctx.clearRect(0, 0, width, height);
  return { ctx, width, height };
}

function emptyState(surface: Surface, title: string, hint: string): void {
  const { ctx, width, height } = surface;
  const t = theme();
  ctx.fillStyle = t.muted;
  ctx.font = '600 13px system-ui, sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText(title, width / 2, height / 2 - 4);
  ctx.font = '12px system-ui, sans-serif';
  ctx.globalAlpha = 0.8;
  ctx.fillText(hint, width / 2, height / 2 + 17);
  ctx.globalAlpha = 1;
  ctx.textAlign = 'left';
}

/* ------------------------------------------------------------------ *
 * A line chart
 * ------------------------------------------------------------------ */

export function drawLine(canvas: HTMLCanvasElement, options: LineOptions): void {
  const surface = prepare(canvas, 110);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const t = theme();
  const format = options.format ?? ((v: number) => v.toFixed(2));

  const drawn = options.series.filter((s) => s.values.length > 0);
  const longest = drawn.reduce((n, s) => Math.max(n, s.values.length), 0);

  // The grid is drawn first, so a chart with no data still looks like a chart
  // that is waiting rather than a hole in the page.
  ctx.strokeStyle = t.line;
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = Math.round((height / 4) * i) + 0.5;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(width, y);
    ctx.stroke();
  }

  if (longest < 2) {
    emptyState(
      surface,
      options.emptyTitle ?? 'This curve is drawn as it trains.',
      options.emptyHint ?? 'Label a few images and press Train.',
    );
    return;
  }

  let min = options.floor ?? Infinity;
  let max = options.ceil ?? -Infinity;
  if (options.floor === undefined || options.ceil === undefined) {
    for (const s of drawn) {
      for (const v of s.values) {
        if (options.floor === undefined) min = Math.min(min, v);
        if (options.ceil === undefined) max = Math.max(max, v);
      }
    }
  }
  if (!Number.isFinite(min)) min = 0;
  if (!Number.isFinite(max)) max = 1;
  const span = Math.max(1e-9, max - min);

  const x = (i: number, n: number): number => (n === 1 ? 0 : (i / (n - 1)) * width);
  const y = (v: number): number => height - 4 - ((v - min) / span) * (height - 14);

  for (const s of drawn) {
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.values.forEach((v, i) => (i === 0 ? ctx.moveTo(x(i, s.values.length), y(v)) : ctx.lineTo(x(i, s.values.length), y(v))));
    ctx.stroke();
  }

  const primary = options.series[0];
  if (primary && primary.values.length > 1 && options.annotate !== false) {
    const last = primary.values[primary.values.length - 1];
    const first = primary.values[0];
    const px = x(primary.values.length - 1, primary.values.length);
    const py = y(last);
    ctx.fillStyle = primary.color;
    ctx.beginPath();
    ctx.arc(px - 1.5, py, 3, 0, Math.PI * 2);
    ctx.fill();

    const drop = first > 0 ? Math.round((1 - last / first) * 100) : 0;
    const note = options.format
      ? `${format(last)}`
      : `${format(last)} — down ${drop}% from the start`;
    ctx.fillStyle = t.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.fillText(note, width - 6, 12);
    ctx.textAlign = 'left';
  }

  ctx.fillStyle = t.muted;
  ctx.font = '11px system-ui, sans-serif';
  ctx.fillText(format(max), 4, 12);
  ctx.fillText(format(min), 4, height - 4);

  // A legend, because two lines without names is a puzzle.
  if (drawn.length > 1) {
    // Start clear of the axis minimum, which is drawn on the same baseline.
    let offset = 36;
    ctx.font = '11px system-ui, sans-serif';
    for (const s of drawn) {
      ctx.fillStyle = s.color;
      ctx.fillRect(offset, height - 14, 8, 3);
      ctx.fillStyle = t.muted;
      ctx.fillText(s.label, offset + 12, height - 9);
      offset += 12 + ctx.measureText(s.label).width + 14;
    }
  }
}

/* ------------------------------------------------------------------ *
 * Horizontal bars — probabilities, class counts, anything with a label
 * ------------------------------------------------------------------ */

export function drawBars(canvas: HTMLCanvasElement, items: BarItem[], options: { max?: number; format?: (v: number) => string; emptyTitle?: string; emptyHint?: string } = {}): void {
  const surface = prepare(canvas, 90);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const t = theme();
  const format = options.format ?? ((v: number) => v.toFixed(0));

  if (items.length === 0) {
    emptyState(surface, options.emptyTitle ?? 'Nothing to show yet.', options.emptyHint ?? 'Add some images first.');
    return;
  }

  // The font has to be set BEFORE the labels are measured, or the measurement
  // uses the 10px default and the bars start in the wrong place.
  ctx.font = '12px system-ui, sans-serif';
  const labelWidth = Math.min(120, Math.max(56, ...items.map((i) => ctx.measureText(i.label).width + 10)));
  const rowHeight = Math.max(14, Math.min(30, height / items.length));
  const valueWidth = 52;
  const barWidth = Math.max(10, width - labelWidth - valueWidth - 8);
  const max = options.max ?? Math.max(1e-9, ...items.map((i) => i.value));
  // The rows are centred in whatever height the canvas turned out to be, rather
  // than stacked against its ceiling. A tall box with three rows clinging to the
  // top of it reads as a chart that stopped drawing halfway.
  const band = rowHeight * items.length;
  const originY = Math.max(0, (height - band) / 2);

  items.forEach((item, index) => {
    const barHeight = Math.min(16, rowHeight - 4);
    const top = originY + index * rowHeight + (rowHeight - barHeight) / 2;
    const ratio = Math.max(0, Math.min(1, item.value / max));

    ctx.fillStyle = t.muted;
    ctx.font = '12px system-ui, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText(item.label, 0, top + barHeight - 3);

    ctx.fillStyle = t.line;
    roundRect(ctx, labelWidth, top, barWidth, barHeight, 4);
    ctx.fill();

    ctx.fillStyle = item.color ?? t.accent;
    if (ratio > 0) {
      roundRect(ctx, labelWidth, top, Math.max(3, barWidth * ratio), barHeight, 4);
      ctx.fill();
    }

    ctx.fillStyle = item.caption ? t.text : t.muted;
    ctx.font = '11px ui-monospace, monospace';
    ctx.textAlign = 'right';
    ctx.fillText(item.caption ?? format(item.value), width, top + barHeight - 3);
    ctx.textAlign = 'left';
  });
}

function roundRect(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number): void {
  const r = Math.min(radius, height / 2, width / 2);
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + width, y, x + width, y + height, r);
  ctx.arcTo(x + width, y + height, x, y + height, r);
  ctx.arcTo(x, y + height, x, y, r);
  ctx.arcTo(x, y, x + width, y, r);
  ctx.closePath();
}

/* ------------------------------------------------------------------ *
 * A confusion matrix — where the model is wrong, and which way
 * ------------------------------------------------------------------ */

export function drawHeatmap(canvas: HTMLCanvasElement, matrix: number[][], rowLabels: string[], columnLabels: string[], options: { emptyTitle?: string; emptyHint?: string } = {}): void {
  const surface = prepare(canvas, 140);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const t = theme();

  if (matrix.length === 0 || rowLabels.length === 0) {
    emptyState(surface, options.emptyTitle ?? 'Nothing classified yet.', options.emptyHint ?? 'Train, then this fills in.');
    return;
  }

  const size = rowLabels.length;
  ctx.font = '11px system-ui, sans-serif';
  const gutter = Math.min(88, Math.max(52, ...rowLabels.map((l) => ctx.measureText(l).width + 12)));
  const headroom = 20;
  const room = 18;
  const cell = Math.max(10, Math.min((width - gutter - room) / size, (height - headroom) / size));
  const max = Math.max(1, ...matrix.flat());
  // Centre the block horizontally. A square matrix in a wide panel otherwise
  // clings to the left edge with a lake of empty space beside it.
  const originX = gutter + Math.max(0, (width - gutter - room - cell * size) / 2);

  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      const value = matrix[r][c];
      const px = originX + c * cell;
      const py = headroom + r * cell;
      const weight = value / max;
      // The diagonal is agreement, so it gets the accent; a mistake gets the
      // warning colour. Reading the picture should not require the legend.
      const base = r === c ? '167, 139, 250' : '251, 191, 36';
      ctx.fillStyle = value === 0 ? 'rgba(255,255,255,0.04)' : `rgba(${base}, ${0.12 + weight * 0.8})`;
      ctx.fillRect(px + 1, py + 1, cell - 2, cell - 2);
      if (value > 0 && cell > 26) {
        ctx.fillStyle = t.text;
        ctx.textAlign = 'center';
        ctx.fillText(String(value), px + cell / 2, py + cell / 2 + 4);
        ctx.textAlign = 'left';
      }
    }
  }

  ctx.fillStyle = t.muted;
  for (let r = 0; r < size; r++) ctx.fillText(rowLabels[r], 0, headroom + r * cell + cell / 2 + 4);
  ctx.textAlign = 'center';
  for (let c = 0; c < size; c++) ctx.fillText(columnLabels[c], originX + c * cell + cell / 2, 12);
  ctx.textAlign = 'left';

  // "model said" sits along the right edge of the block, where it explains the
  // columns, rather than pinned to the far edge of the canvas.
  ctx.save();
  ctx.translate(originX + cell * size + 12, headroom + cell * size);
  ctx.rotate(-Math.PI / 2);
  ctx.fillStyle = t.muted;
  ctx.font = '10px system-ui, sans-serif';
  ctx.fillText('model said', 0, 0);
  ctx.restore();
}

/* ------------------------------------------------------------------ *
 * Tiles — the kernels, and the feature maps they produce
 * ------------------------------------------------------------------ */

/**
 * Draw a grid of small squares. Used twice: for the eight first-layer kernels,
 * where the three input channels make each tile a colour, and for the eight
 * activations, where one channel makes each tile a grey picture of what the
 * filter found.
 */
export function drawTiles(
  canvas: HTMLCanvasElement,
  tiles: Array<Float32Array | Uint8Array>,
  tileSize: number,
  options: { channels: 1 | 3; max?: number; emptyTitle?: string; emptyHint?: string; caption?: string },
): void {
  const surface = prepare(canvas, 120);
  if (!surface) return;
  const { ctx, width, height } = surface;
  const t = theme();

  if (tiles.length === 0) {
    emptyState(surface, options.emptyTitle ?? 'Nothing to draw yet.', options.emptyHint ?? 'Train the model first.');
    return;
  }

  const columns = Math.ceil(Math.sqrt(tiles.length));
  const rows = Math.ceil(tiles.length / columns);
  const gap = 6;
  const captionRoom = options.caption ? 16 : 0;
  const cell = Math.max(
    6,
    Math.min((width - gap * (columns + 1)) / columns, (height - captionRoom - gap * (rows + 1)) / rows),
  );
  const gridWidth = columns * cell + gap * (columns - 1);
  const gridHeight = rows * cell + gap * (rows - 1);
  const originX = (width - gridWidth) / 2;
  const originY = (height - captionRoom - gridHeight) / 2;

  const scratch = document.createElement('canvas');
  scratch.width = tileSize;
  scratch.height = tileSize;
  const scratchCtx = scratch.getContext('2d');
  if (!scratchCtx) return;

  const max = options.max && options.max > 0 ? options.max : 1;

  tiles.forEach((tile, index) => {
    const column = index % columns;
    const row = Math.floor(index / columns);
    const x = originX + column * (cell + gap);
    const y = originY + row * (cell + gap);

    const image = scratchCtx.createImageData(tileSize, tileSize);
    const count = tileSize * tileSize;
    for (let i = 0; i < count; i++) {
      if (options.channels === 3) {
        // Weights run negative to positive; 128 is zero, so a kernel reads as a
        // colour cast rather than disappearing when it is small.
        image.data[i * 4] = clampByte(128 + (tile[i * 3] / max) * 127);
        image.data[i * 4 + 1] = clampByte(128 + (tile[i * 3 + 1] / max) * 127);
        image.data[i * 4 + 2] = clampByte(128 + (tile[i * 3 + 2] / max) * 127);
      } else {
        // Activations are non-negative: black is nothing, white is a strong hit.
        const v = clampByte((tile[i] / max) * 255);
        image.data[i * 4] = v;
        image.data[i * 4 + 1] = v;
        image.data[i * 4 + 2] = v;
      }
      image.data[i * 4 + 3] = 255;
    }
    scratchCtx.putImageData(image, 0, 0);

    ctx.fillStyle = 'rgba(0,0,0,0.35)';
    ctx.fillRect(x - 1, y - 1, cell + 2, cell + 2);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(scratch, x, y, cell, cell);
  });

  if (options.caption) {
    ctx.fillStyle = t.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(options.caption, width / 2, height - 4);
    ctx.textAlign = 'left';
  }
}

function clampByte(value: number): number {
  if (!Number.isFinite(value)) return 0;
  return value < 0 ? 0 : value > 255 ? 255 : value;
}
