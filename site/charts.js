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
let cachedTheme = null;
export function theme() {
    if (cachedTheme)
        return cachedTheme;
    const styles = getComputedStyle(document.body);
    const read = (name, fallback) => styles.getPropertyValue(name).trim() || fallback;
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
function prepare(canvas, minHeight = 90) {
    const ctx = canvas.getContext('2d');
    if (!ctx)
        return null;
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || canvas.parentElement?.clientWidth || 320;
    const height = canvas.clientHeight || minHeight;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx, width, height };
}
function emptyState(surface, title, hint) {
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
/**
 * One segment per rectangle, so the chart is a picture of the picture.
 *
 * A bar rather than a number because "3 of 8" and a bar showing three filled segments
 * and five hollow ones are the same fact, and only one of them can be taken in without
 * reading. The segment being asked about is the one outlined, so the chart also says
 * WHERE you are, not only how far.
 */
export function drawProgress(canvas, boxes, options = {}) {
    const surface = prepare(canvas, 64);
    if (!surface)
        return;
    const { ctx, width, height } = surface;
    const t = theme();
    if (boxes.length === 0) {
        emptyState(surface, options.emptyTitle ?? 'Nothing to count yet.', options.emptyHint ?? '');
        return;
    }
    const named = boxes.filter((b) => b.state === 'named').length;
    const skipped = boxes.filter((b) => b.state === 'skipped').length;
    const done = named + skipped;
    const current = boxes.findIndex((b) => b.state === 'current');
    // Room for the count above the bar and the words below it.
    const labelHeight = 20;
    const barHeight = Math.min(22, height - labelHeight - 18);
    const gap = boxes.length > 12 ? 2 : 4;
    const segment = (width - gap * (boxes.length - 1)) / boxes.length;
    const top = labelHeight;
    ctx.font = '600 13px system-ui, sans-serif';
    ctx.textBaseline = 'alphabetic';
    const heading = `${done} of ${boxes.length} done`;
    if (current >= 0) {
        ctx.fillStyle = t.muted;
        ctx.fillText(heading, 0, 14);
        const asking = `asking about ${current + 1}`;
        ctx.textAlign = 'right';
        ctx.fillStyle = t.yellow;
        ctx.fillText(asking, width, 14);
        ctx.textAlign = 'left';
    }
    else {
        ctx.fillStyle = done === boxes.length ? t.accent : t.muted;
        ctx.fillText(done === boxes.length ? `all ${boxes.length} done` : heading, 0, 14);
    }
    boxes.forEach((box, index) => {
        const x = index * (segment + gap);
        const radius = Math.min(5, segment / 2, barHeight / 2);
        if (box.state === 'named') {
            ctx.fillStyle = t.accent;
            roundRect(ctx, x, top, segment, barHeight, radius);
            ctx.fill();
        }
        else if (box.state === 'skipped') {
            // Deliberately faint and hatched: dealt with, but not taught anything.
            ctx.fillStyle = 'rgba(255,255,255,0.10)';
            roundRect(ctx, x, top, segment, barHeight, radius);
            ctx.fill();
            ctx.save();
            ctx.beginPath();
            roundRect(ctx, x, top, segment, barHeight, radius);
            ctx.clip();
            ctx.strokeStyle = 'rgba(255,255,255,0.16)';
            ctx.lineWidth = 1.5;
            for (let d = -barHeight; d < segment; d += 6) {
                ctx.beginPath();
                ctx.moveTo(x + d, top + barHeight);
                ctx.lineTo(x + d + barHeight, top);
                ctx.stroke();
            }
            ctx.restore();
        }
        else if (box.state === 'current') {
            ctx.strokeStyle = t.yellow;
            ctx.lineWidth = 2.5;
            roundRect(ctx, x + 1, top + 1, Math.max(2, segment - 2), barHeight - 2, radius);
            ctx.stroke();
        }
        else {
            ctx.strokeStyle = 'rgba(255,255,255,0.16)';
            ctx.lineWidth = 1.5;
            roundRect(ctx, x + 1, top + 1, Math.max(2, segment - 2), barHeight - 2, radius);
            ctx.stroke();
        }
    });
    // The words below, so the colours do not have to be guessed at.
    ctx.font = '11.5px system-ui, sans-serif';
    const parts = [];
    if (named > 0)
        parts.push(`${named} named`);
    if (skipped > 0)
        parts.push(`${skipped} skipped`);
    const left = boxes.length - done;
    if (left > 0)
        parts.push(`${left} to go`);
    ctx.fillStyle = t.muted;
    ctx.globalAlpha = 0.85;
    ctx.fillText(parts.join(' · '), 0, top + barHeight + 14);
    ctx.globalAlpha = 1;
}
/* ------------------------------------------------------------------ *
 * A line chart
 * ------------------------------------------------------------------ */
export function drawLine(canvas, options) {
    const surface = prepare(canvas, 110);
    if (!surface)
        return;
    const { ctx, width, height } = surface;
    const t = theme();
    const format = options.format ?? ((v) => v.toFixed(2));
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
        emptyState(surface, options.emptyTitle ?? 'This curve is drawn as it trains.', options.emptyHint ?? 'Label a few images and press Train.');
        return;
    }
    let min = options.floor ?? Infinity;
    let max = options.ceil ?? -Infinity;
    if (options.floor === undefined || options.ceil === undefined) {
        for (const s of drawn) {
            for (const v of s.values) {
                if (options.floor === undefined)
                    min = Math.min(min, v);
                if (options.ceil === undefined)
                    max = Math.max(max, v);
            }
        }
    }
    if (!Number.isFinite(min))
        min = 0;
    if (!Number.isFinite(max))
        max = 1;
    const span = Math.max(1e-9, max - min);
    // 🔴 THE PLOT IS INSET, AND THE LABELS LIVE IN THE INSET — 23 September 2026. George, looking at
    // this at an 801px window: *"the charts are crowded and have overlaps"*. He was right, and the
    // cause was arithmetic: the plot spanned the whole canvas (`x` from 0 to `width`, `y` from 10 to
    // `height - 4`) and the axis labels were then drawn INTO it — `format(max)` at y=12 sat on the top
    // gridline with the series under it, `format(min)` shared the bottom corner with the legend, and
    // the note was right-aligned on top of the curve. Everything the chart needed to say had nowhere
    // of its own to be.
    //
    // So the gutters are measured first and the plot gets what is left:
    //   · the LEFT gutter holds the two value labels, sized from the widest of them;
    //   · the TOP gutter holds the note;
    //   · the BOTTOM gutter holds the legend, and grows to one row per series when a single row will
    //     not fit — which is what the old `offset = 36` hack was really about: it nudged the legend
    //     right to dodge the axis minimum, and at 344px the two labels ran past the edge instead.
    ctx.font = '11px system-ui, sans-serif';
    const axisWidth = Math.round(Math.max(ctx.measureText(format(max)).width, ctx.measureText(format(min)).width));
    const padLeft = axisWidth + 10;
    const legendRows = drawn.length > 1
        ? (() => {
            const oneRow = drawn.reduce((n, s) => n + 12 + ctx.measureText(s.label).width + 14, 0);
            return oneRow <= width - padLeft ? 1 : drawn.length;
        })()
        : 0;
    const legendHeight = legendRows === 0 ? 0 : legendRows === 1 ? 20 : 16 * legendRows + 4;
    const padTop = 20;
    const plot = {
        x: padLeft,
        y: padTop,
        w: Math.max(10, width - padLeft - 4),
        h: Math.max(10, height - padTop - legendHeight),
    };
    // The grid, inside the plot rather than across the labels.
    ctx.strokeStyle = t.line;
    ctx.lineWidth = 1;
    for (let i = 1; i < 4; i++) {
        const y = Math.round(plot.y + (plot.h / 4) * i) + 0.5;
        ctx.beginPath();
        ctx.moveTo(plot.x, y);
        ctx.lineTo(plot.x + plot.w, y);
        ctx.stroke();
    }
    const x = (i, n) => plot.x + (n === 1 ? 0 : (i / (n - 1)) * plot.w);
    const y = (v) => plot.y + plot.h - ((v - min) / span) * plot.h;
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
        ctx.fillText(note, width - 4, 13);
        ctx.textAlign = 'left';
    }
    // The two value labels, in the gutter and level with the ends of the plot.
    ctx.fillStyle = t.muted;
    ctx.font = '11px system-ui, sans-serif';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(format(max), padLeft - 6, plot.y + 1);
    ctx.fillText(format(min), padLeft - 6, plot.y + plot.h - 1);
    ctx.textAlign = 'left';
    ctx.textBaseline = 'alphabetic';
    // A legend, because two lines without names is a puzzle — and one row per series when one row
    // will not fit, rather than a row that runs off the right-hand edge.
    if (legendRows > 0) {
        ctx.font = '11px system-ui, sans-serif';
        let row = 0;
        let offset = legendRows === 1 ? padLeft : 0;
        for (const s of drawn) {
            const rowY = legendRows === 1
                ? height - 8
                : height - legendHeight + 10 + row * 16;
            ctx.fillStyle = s.color;
            ctx.fillRect(offset, rowY - 4, 8, 3);
            ctx.fillStyle = t.muted;
            ctx.fillText(s.label, offset + 12, rowY + 1);
            if (legendRows === 1) {
                offset += 12 + ctx.measureText(s.label).width + 14;
            }
            else {
                row += 1;
                offset = 0;
            }
        }
    }
}
/* ------------------------------------------------------------------ *
 * Horizontal bars — probabilities, class counts, anything with a label
 * ------------------------------------------------------------------ */
export function drawBars(canvas, items, options = {}) {
    const surface = prepare(canvas, 90);
    if (!surface)
        return;
    const { ctx, width, height } = surface;
    const t = theme();
    const format = options.format ?? ((v) => v.toFixed(0));
    if (items.length === 0) {
        emptyState(surface, options.emptyTitle ?? 'Nothing to show yet.', options.emptyHint ?? 'Add some images first.');
        return;
    }
    // The font has to be set BEFORE the labels are measured, or the measurement
    // uses the 10px default and the bars start in the wrong place.
    ctx.font = '12px system-ui, sans-serif';
    const labelWidth = Math.min(120, Math.max(56, ...items.map((i) => ctx.measureText(i.label).width + 10)));
    const rowHeight = Math.max(14, Math.min(30, height / items.length));
    // 🔴 AND THE CAPTION'S OWN GUTTER IS MEASURED TOO — 23 September 2026, George: *"the charts are
    // crowded and have overlaps"*. This was a flat `52`, and the caption is not a value: it reads
    // `100% · 2 pictures`, about **120px** at 11px monospace. So the bar was laid out to leave 60px for
    // a 120px label and the two landed on top of each other — the number printed over the end of its
    // own bar, and the tail of it clipped by the canvas edge. A gutter is a measurement, not a guess.
    ctx.font = '11px ui-monospace, monospace';
    const valueWidth = Math.round(Math.max(34, ...items.map((i) => ctx.measureText(i.caption ?? format(i.value)).width + 10)));
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
function roundRect(ctx, x, y, width, height, radius) {
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
export function drawHeatmap(canvas, matrix, rowLabels, columnLabels, options = {}) {
    const surface = prepare(canvas, 140);
    if (!surface)
        return;
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
    for (let r = 0; r < size; r++)
        ctx.fillText(rowLabels[r], 0, headroom + r * cell + cell / 2 + 4);
    ctx.textAlign = 'center';
    for (let c = 0; c < size; c++)
        ctx.fillText(columnLabels[c], originX + c * cell + cell / 2, 12);
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
export function drawTiles(canvas, tiles, tileSize, options) {
    const surface = prepare(canvas, 120);
    if (!surface)
        return;
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
    const cell = Math.max(6, Math.min((width - gap * (columns + 1)) / columns, (height - captionRoom - gap * (rows + 1)) / rows));
    const gridWidth = columns * cell + gap * (columns - 1);
    const gridHeight = rows * cell + gap * (rows - 1);
    const originX = (width - gridWidth) / 2;
    const originY = (height - captionRoom - gridHeight) / 2;
    const scratch = document.createElement('canvas');
    scratch.width = tileSize;
    scratch.height = tileSize;
    const scratchCtx = scratch.getContext('2d');
    if (!scratchCtx)
        return;
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
            }
            else {
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
function clampByte(value) {
    if (!Number.isFinite(value))
        return 0;
    return value < 0 ? 0 : value > 255 ? 255 : value;
}
