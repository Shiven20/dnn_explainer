/**
 * Tests for the forward-pass animation state machine.
 *
 * The critical property: the animation must only control *reveal*, never the
 * values. If it altered the computation, the diagram would be showing numbers
 * the network never produced.
 *
 * Usage: npm run test:animation
 */

// The module calls requestAnimationFrame at import time only via run(); still,
// stub the globals before importing so nothing touches an undefined symbol.
let rafCallbacks = [];
globalThis.requestAnimationFrame = (fn) => {
  rafCallbacks.push(fn);
  return rafCallbacks.length;
};
globalThis.cancelAnimationFrame = (id) => {
  rafCallbacks[id - 1] = undefined;
};
// Default to motion allowed; individual tests override this.
let reducedMotion = false;
globalThis.window = {
  matchMedia: (q) => ({
    matches: q.includes('reduce') ? reducedMotion : false
  })
};

const {
  animation, STATUS, PHASE, TIMING, run, pause, resume, reset, skipToEnd,
  destroyAnimation, layerReveal, easeInOut, prefersReducedMotion
} = await import('../src/dnn/animation.js');

const { createNetwork } = await import('../src/dnn/engine/network.js');
const { forwardPass } = await import('../src/dnn/engine/forwardPass.js');

let failures = 0;
let checks = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};

const state = () => {
  let s;
  const unsub = animation.subscribe((v) => { s = v; });
  unsub();
  return s;
};

/*
 * Virtual clock. Must be monotonic across calls, because real rAF timestamps
 * are: restarting it per call would feed the loop a negative delta.
 */
let now = 1000;

/** Drive the rAF loop forward by `ms`, in `stepMs` slices. */
const advance = (ms, stepMs = 16) => {
  let elapsed = 0;
  while (elapsed < ms) {
    const pending = rafCallbacks.filter(Boolean);
    rafCallbacks = [];
    if (pending.length === 0) break;
    now += stepMs;
    elapsed += stepMs;
    pending.forEach((fn) => fn(now));
  }
};

const LAYERS = 4;

// ---------------------------------------------------------------- initial
console.log('Initial state');
reset();
check('starts live', state().status === STATUS.LIVE);
check('live means everything is revealed', layerReveal(state(), 3) === 1);

// ---------------------------------------------------------------- run
console.log('\nRunning');
reset();
rafCallbacks = [];
run(LAYERS);
check('status becomes running', state().status === STATUS.RUNNING);
check('starts at the input layer', state().frontier === 0);
check('nothing past the input is revealed yet', state().revealed === 0);
check('input layer is always revealed', layerReveal(state(), 0) === 1);
check('later layers start hidden', layerReveal(state(), 2) === 0);
check('records the layer count', state().layerCount === LAYERS);

// After the lead-in, the first hop should be travelling.
advance(TIMING.lead + 40);
check('advances to the first hop', state().frontier === 1, `frontier=${state().frontier}`);
check('first hop is travelling', state().phase === PHASE.TRAVEL);

// Travel then settle reveals layer 1.
advance(TIMING.travel + 40);
check('hop settles after travelling', state().phase === PHASE.SETTLE,
  `phase=${state().phase}`);
check('layer 1 is revealed once it settles', state().revealed === 1,
  `revealed=${state().revealed}`);
check('layer 2 is still hidden', layerReveal(state(), 2) === 0);

// ---------------------------------------------------------------- ordering
console.log('\nLayer-by-layer ordering');
reset();
rafCallbacks = [];
run(LAYERS);

// Record the reveal order across the whole run.
const revealOrder = [];
let guard = 0;
while (state().status === STATUS.RUNNING && guard < 500) {
  const before = state().revealed;
  advance(50, 16);
  if (state().revealed !== before) revealOrder.push(state().revealed);
  guard++;
}
check('layers reveal in ascending order',
  revealOrder.every((v, i) => i === 0 || v > revealOrder[i - 1]),
  JSON.stringify(revealOrder));
check('every layer gets revealed exactly once',
  JSON.stringify(revealOrder) === JSON.stringify([1, 2, 3]),
  JSON.stringify(revealOrder));
check('finishes after the last layer', state().status === STATUS.FINISHED,
  `status=${state().status}`);
check('all layers revealed at the end', state().revealed === LAYERS - 1);
check('finished reveals everything', layerReveal(state(), 3) === 1);

// ---------------------------------------------------------------- pause
console.log('\nPause and resume');
reset();
rafCallbacks = [];
run(LAYERS);
advance(TIMING.lead + 200);
const atPause = { ...state() };
pause();
check('pausing sets paused', state().status === STATUS.PAUSED);

// A paused animation must not advance, even as frames are pumped.
advance(500);
check('paused animation does not advance',
  state().frontier === atPause.frontier && state().revealed === atPause.revealed,
  `frontier ${atPause.frontier}->${state().frontier}`);

resume();
check('resuming sets running', state().status === STATUS.RUNNING);
advance(TIMING.travel + TIMING.settle + 80);
check('resumed animation makes progress', state().revealed >= 1);

// Pause is a no-op unless running.
reset();
pause();
check('pause on a live animation is a no-op', state().status === STATUS.LIVE);
resume();
check('resume on a live animation is a no-op', state().status === STATUS.LIVE);

// ---------------------------------------------------------------- reset & skip
console.log('\nReset and skip');
reset();
rafCallbacks = [];
run(LAYERS);
advance(TIMING.lead + TIMING.travel + 60);
reset();
check('reset returns to live', state().status === STATUS.LIVE);
check('reset clears the frontier', state().frontier === 0);
check('reset shows all values again', layerReveal(state(), 3) === 1);

// Frames left over from the previous run must not restart it.
advance(300);
check('reset stops the frame loop', state().status === STATUS.LIVE);

reset();
rafCallbacks = [];
run(LAYERS);
advance(TIMING.lead + 40);
skipToEnd();
check('skip jumps to finished', state().status === STATUS.FINISHED);
check('skip reveals every layer', state().revealed === LAYERS - 1);
check('skip stops advancing', (() => {
  const before = { ...state() };
  advance(400);
  return state().revealed === before.revealed;
})());

// ---------------------------------------------------------------- reduced motion
console.log('\nReduced motion');
reducedMotion = true;
reset();
rafCallbacks = [];
check('preference is detected', prefersReducedMotion() === true);
run(LAYERS);
check('skips straight to the result', state().status === STATUS.FINISHED,
  `status=${state().status}`);
check('no animation frames were scheduled', rafCallbacks.filter(Boolean).length === 0);
check('result is fully revealed', layerReveal(state(), 3) === 1);
reducedMotion = false;

// ---------------------------------------------------------------- cleanup
console.log('\nCleanup');
reset();
rafCallbacks = [];
run(LAYERS);
advance(TIMING.lead + 40);
destroyAnimation();
const afterDestroy = { ...state() };
advance(500);
check('destroy stops the frame loop',
  state().frontier === afterDestroy.frontier,
  `frontier ${afterDestroy.frontier}->${state().frontier}`);

// A slow frame must not skip the timeline; the delta is clamped.
reset();
rafCallbacks = [];
run(LAYERS);
const pending = rafCallbacks.filter(Boolean);
rafCallbacks = [];
pending.forEach((fn) => fn(100000)); // a huge jump, as after a backgrounded tab
check('a huge frame delta does not skip to the end',
  state().status === STATUS.RUNNING && state().revealed < LAYERS - 1,
  `revealed=${state().revealed}`);

// ---------------------------------------------------------------- values intact
console.log('\nAnimation does not alter the computation');
const net = forwardPass(
  createNetwork({ layerSizes: [3, 4, 2], seed: 9 }),
  [0.5, -0.25, 0.75]
);
const snapshot = net.layers.map((l) => l.neurons.map((n) => n.output));

reset();
rafCallbacks = [];
run(3);
advance(TIMING.lead + TIMING.travel + 40);

const after = net.layers.map((l) => l.neurons.map((n) => n.output));
check('neuron outputs are untouched mid-animation',
  JSON.stringify(snapshot) === JSON.stringify(after));

// layerReveal gates visibility only, and never reports a value.
const midState = state();
check('reveal is a 0..1 gate',
  [0, 1, 2].every((l) => {
    const r = layerReveal(midState, l);
    return r >= 0 && r <= 1;
  }));
check('unreached layers report zero reveal',
  layerReveal({ ...midState, revealed: 0, frontier: 1, phase: PHASE.TRAVEL,
    status: STATUS.RUNNING }, 2) === 0);

// ---------------------------------------------------------------- easing
console.log('\nEasing');
check('easing is pinned at 0', easeInOut(0) === 0);
check('easing is pinned at 1', easeInOut(1) === 1);
check('easing passes through the midpoint', Math.abs(easeInOut(0.5) - 0.5) < 1e-9);
check('easing is monotonic', (() => {
  for (let t = 0; t < 1; t += 0.05) {
    if (easeInOut(t + 0.05) < easeInOut(t)) return false;
  }
  return true;
})());

destroyAnimation();

console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
