/**
 * trainer-host.ts — owns the network and drives it one epoch at a time.
 *
 * Exactly the same class runs in the Web Worker (the normal case, so the page
 * keeps painting) and on the main thread (the fallback, if module workers are
 * unavailable). It works a single epoch per turn rather than looping to the end,
 * so a "stop" message is always answered within one epoch rather than never.
 */
import { Cnn, mulberry32 } from './net.js';
/**
 * How long one turn of training may hold the thread before it hands control back.
 *
 * This is a slice rather than a single epoch for a reason that cost a whole
 * afternoon: a `setTimeout` is clamped to roughly one second in a tab the visitor
 * has switched away from, and the original loop yielded after every epoch with a
 * zero-delay timeout. Twelve epochs therefore took eighty seconds in a background
 * tab and eleven seconds in a focused one — the same work, seven times slower,
 * with nothing on screen to say why. A MessageChannel task is not a timer and is
 * not clamped, and a budget of a few dozen milliseconds per turn keeps the work
 * moving however the tab is behaving.
 */
const SLICE_MS = 40;
const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());
function shuffle(items, rng) {
    for (let i = items.length - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [items[i], items[j]] = [items[j], items[i]];
    }
    return items;
}
export class VisionTrainer {
    post;
    net = null;
    payload = null;
    rng = Math.random;
    plan = [];
    watch = [];
    history = [];
    epoch = 0;
    last = null;
    stopRequested = false;
    running = false;
    beganAt = 0;
    channel = null;
    constructor(post) {
        this.post = post;
    }
    handle(request) {
        switch (request.type) {
            case 'train':
                this.start(request.payload);
                return;
            case 'stop':
                this.stop();
                return;
        }
    }
    stop() {
        if (!this.running)
            return;
        this.stopRequested = true;
    }
    start(payload) {
        if (this.running)
            return;
        // One picture of each of two things is the honest minimum. It used to demand
        // four, which quietly broke the moment the page promised somebody they could
        // start with two — the run returned an error nobody was looking for, no model
        // was ever built, and every guess after that came back "I do not know".
        if (payload.samples.length < 2 || payload.classes.length < 2) {
            this.post({
                type: 'error',
                message: 'It needs one picture of two different things before it can tell them apart.',
            });
            return;
        }
        this.stopRequested = false;
        this.running = true;
        this.beganAt = now();
        this.payload = payload;
        this.rng = mulberry32(payload.seed >>> 0);
        this.epoch = 0;
        this.last = null;
        this.history = payload.weights ? payload.weights.meta.history.slice() : [];
        // Continue from the weights we were handed, or start fresh.
        let net;
        if (payload.weights) {
            try {
                net = Cnn.load(payload.weights);
            }
            catch {
                net = new Cnn(payload.classes.length, undefined, undefined, payload.seed);
            }
            // The classes may have grown since those weights were saved.
            net.setClasses(payload.classes.length);
        }
        else {
            net = new Cnn(payload.classes.length, undefined, undefined, payload.seed);
        }
        this.net = net;
        const { train, watch } = this.split(payload);
        this.plan = train;
        this.watch = watch;
        this.post({
            type: 'started',
            epochs: payload.epochs,
            parameters: net.parameterCount,
            trainCount: train.length,
            valCount: watch.length,
            classes: payload.classes.slice(),
        });
        // Hand control back to the message loop rather than recursing, which is what
        // lets a "stop" arrive at all.
        this.schedule();
    }
    /** Ask for the next turn of work. Not a timer, so a hidden tab cannot clamp it. */
    schedule() {
        if (!this.channel) {
            this.channel = new MessageChannel();
            this.channel.port1.onmessage = () => this.pump();
        }
        this.channel.port2.postMessage(null);
    }
    /** Work for up to SLICE_MS, then hand back. */
    pump() {
        if (!this.running)
            return;
        const payload = this.payload;
        const started = now();
        do {
            if (this.stopRequested) {
                this.finish('stopped');
                return;
            }
            // The budget is checked after at least one epoch, so a run with a tiny
            // budget still learns something rather than returning immediately.
            if (payload?.budgetMs && this.epoch > 0 && now() - this.beganAt > payload.budgetMs) {
                this.finish('done');
                return;
            }
            if (!this.runOneEpoch())
                return;
        } while (now() - started < SLICE_MS);
        this.schedule();
    }
    /**
     * Hold back some pictures so the accuracy figure means something.
     *
     * Stratifying gets harder once one picture can hold several things at once: a
     * photo of a dog on a beach belongs to two classes at the same time. So this
     * shuffles, holds back a slice, and then lends a picture back to the training set
     * for any class that ended up entirely on the held-back side. A class with
     * nothing to learn from is a class it can never get right, and the accuracy
     * readout would say so forever.
     */
    split(payload) {
        const all = payload.samples.slice();
        shuffle(all, this.rng);
        const room = Math.max(0, all.length - payload.classes.length);
        const held = Math.min(Math.round(all.length * payload.validationSplit), room);
        const watch = all.slice(0, held);
        const train = all.slice(held);
        for (let c = 0; c < payload.classes.length; c++) {
            if (train.some((s) => s.targets.includes(c)))
                continue;
            const donor = watch.findIndex((s) => s.targets.includes(c));
            if (donor >= 0)
                train.push(watch.splice(donor, 1)[0]);
        }
        return { train, watch };
    }
    /** One epoch. Returns false once training has ended, by finishing or by stopping. */
    runOneEpoch() {
        const net = this.net;
        const payload = this.payload;
        if (!net || !payload)
            return false;
        const { loss, accuracy } = net.trainEpoch(this.plan, payload.learningRate, payload.augment, this.rng);
        let valLoss = loss;
        let valAccuracy = accuracy;
        if (this.watch.length > 0) {
            const measured = net.evaluate(this.watch);
            valLoss = measured.loss;
            valAccuracy = measured.accuracy;
        }
        this.epoch += 1;
        const metric = { epoch: this.epoch, loss, trainAccuracy: accuracy, valAccuracy, valLoss };
        this.history.push(metric);
        this.last = metric;
        this.post({ type: 'progress', metric });
        if (this.epoch >= payload.epochs) {
            this.finish('done');
            return false;
        }
        return true;
    }
    finish(kind) {
        const net = this.net;
        const payload = this.payload;
        if (!net || !payload)
            return;
        this.running = false;
        const all = [...this.plan, ...this.watch];
        const size = payload.classes.length;
        // Per class, read off the finished weights over every picture that named it:
        // how often it was right about that one thing. This replaces the confusion
        // matrix, which no longer has a single "truth" to put on a row now that a
        // picture can be several things at once.
        const perClassAccuracy = [];
        const perClassCount = [];
        for (let c = 0; c < size; c++) {
            let right = 0;
            let positive = 0;
            for (const sample of all) {
                const truth = sample.targets.includes(c);
                if (truth)
                    positive += 1;
                const said = net.predict(sample.pixels)[c] >= 0.5;
                if (said === truth)
                    right += 1;
            }
            perClassAccuracy.push(all.length > 0 ? right / all.length : 0);
            perClassCount.push(positive);
        }
        const weights = net.serialize(payload.classes, this.history, payload.samples.length, perClassAccuracy, perClassCount, net.meanConfidence(all), payload.weights);
        weights.meta.parameters = net.parameterCount;
        weights.meta.epochsTrained = (payload.weights?.meta.epochsTrained ?? 0) + this.epoch;
        const metric = this.last ?? { epoch: this.epoch, loss: 0, trainAccuracy: 0, valAccuracy: 0, valLoss: 0 };
        this.post({ type: kind, metric, weights });
    }
}
