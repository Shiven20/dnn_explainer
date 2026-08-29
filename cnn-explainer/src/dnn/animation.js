/**
 * Forward-pass animation.
 *
 * Important: this controls *reveal*, not computation. The forward pass has
 * always already run, so the values shown are the real ones -- the animation
 * only decides how much of the result the viewer has seen so far. Nothing here
 * invents or interpolates a neuron value.
 *
 * The timeline walks one layer at a time:
 *   travel  -- values move along the connections into the next layer
 *   settle  -- the destination neurons take their computed values
 */

import { writable, derived } from 'svelte/store';

export const STATUS = {
  /** Not animating: the whole network shows live values. */
  LIVE: 'live',
  RUNNING: 'running',
  PAUSED: 'paused',
  /** Animation ran to the end; everything is revealed. */
  FINISHED: 'finished'
};

export const PHASE = {
  TRAVEL: 'travel',
  SETTLE: 'settle'
};

/** Per-layer durations in milliseconds. */
export const TIMING = {
  travel: 620,
  settle: 300,
  /** Pause on the input layer before the first hop, so step 1 registers. */
  lead: 420
};

const initialState = {
  status: STATUS.LIVE,
  /** Layer currently being computed; 0 means only the input is revealed. */
  frontier: 0,
  phase: PHASE.TRAVEL,
  /** Progress through the current phase, 0..1. */
  progress: 0,
  /** Highest layer index whose values have been revealed. */
  revealed: 0,
  /** Total layers in the run, so the UI can show "step 2 of 3". */
  layerCount: 0
};

export const animation = writable({ ...initialState });

/** True when the viewer should see every layer's value. */
export const isLive = derived(
  animation,
  (a) => a.status === STATUS.LIVE || a.status === STATUS.FINISHED
);

// ------------------------------------------------------------------ internals

let frameId;
let lastTimestamp;
/** Elapsed time within the current phase, kept outside the store for speed. */
let phaseElapsed = 0;
let currentState = { ...initialState };

animation.subscribe((value) => {
  currentState = value;
});

export const prefersReducedMotion = () =>
  typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const cancel = () => {
  if (frameId !== undefined) {
    if (typeof cancelAnimationFrame === 'function') cancelAnimationFrame(frameId);
    frameId = undefined;
  }
  lastTimestamp = undefined;
};

const phaseDuration = (state) => {
  if (state.frontier === 0) return TIMING.lead;
  return state.phase === PHASE.TRAVEL ? TIMING.travel : TIMING.settle;
};

/**
 * Advance the timeline. Called once per frame while running.
 * Kept as a pure-ish step so the tick loop stays trivial.
 */
const advance = (state, delta) => {
  phaseElapsed += delta;
  const duration = phaseDuration(state);

  if (phaseElapsed < duration) {
    return { ...state, progress: phaseElapsed / duration };
  }

  // Carry the overflow so a slow frame does not lose time.
  phaseElapsed -= duration;

  // The lead-in only precedes the first hop.
  if (state.frontier === 0) {
    return { ...state, frontier: 1, phase: PHASE.TRAVEL, progress: 0 };
  }

  if (state.phase === PHASE.TRAVEL) {
    // Values have arrived; reveal this layer as they settle.
    return {
      ...state,
      phase: PHASE.SETTLE,
      progress: 0,
      revealed: state.frontier
    };
  }

  // Settled. Move to the next layer, or finish.
  const nextFrontier = state.frontier + 1;
  if (nextFrontier > state.layerCount - 1) {
    phaseElapsed = 0;
    return {
      ...state,
      status: STATUS.FINISHED,
      phase: PHASE.SETTLE,
      progress: 1,
      revealed: state.layerCount - 1
    };
  }

  return { ...state, frontier: nextFrontier, phase: PHASE.TRAVEL, progress: 0 };
};

const tick = (timestamp) => {
  if (currentState.status !== STATUS.RUNNING) {
    cancel();
    return;
  }

  if (lastTimestamp === undefined) lastTimestamp = timestamp;
  // Clamp the delta so a backgrounded tab does not skip the whole timeline on
  // its first frame back. The lower bound guards against a non-monotonic
  // timestamp, which would otherwise wind the timeline backwards.
  const delta = Math.max(0, Math.min(timestamp - lastTimestamp, 80));
  lastTimestamp = timestamp;

  animation.update((state) => advance(state, delta));

  if (currentState.status === STATUS.RUNNING) {
    frameId = requestAnimationFrame(tick);
  } else {
    cancel();
  }
};

// ------------------------------------------------------------------ controls

/**
 * Start (or restart) the walkthrough.
 * @param {number} layerCount Total layers, input included.
 */
export const run = (layerCount) => {
  cancel();
  phaseElapsed = 0;

  // With reduced motion, present the finished result rather than a fast
  // animation: the point is to avoid movement, not to compress it.
  if (prefersReducedMotion()) {
    animation.set({
      ...initialState,
      status: STATUS.FINISHED,
      layerCount,
      frontier: layerCount - 1,
      revealed: layerCount - 1,
      phase: PHASE.SETTLE,
      progress: 1
    });
    return;
  }

  animation.set({
    ...initialState,
    status: STATUS.RUNNING,
    layerCount,
    frontier: 0,
    revealed: 0,
    phase: PHASE.TRAVEL,
    progress: 0
  });

  frameId = requestAnimationFrame(tick);
};

export const pause = () => {
  if (currentState.status !== STATUS.RUNNING) return;
  cancel();
  animation.update((state) => ({ ...state, status: STATUS.PAUSED }));
};

export const resume = () => {
  if (currentState.status !== STATUS.PAUSED) return;
  animation.update((state) => ({ ...state, status: STATUS.RUNNING }));
  frameId = requestAnimationFrame(tick);
};

export const toggle = (layerCount) => {
  if (currentState.status === STATUS.RUNNING) pause();
  else if (currentState.status === STATUS.PAUSED) resume();
  else run(layerCount);
};

/** Stop animating and return to showing live values everywhere. */
export const reset = () => {
  cancel();
  phaseElapsed = 0;
  animation.set({ ...initialState });
};

/** Jump straight to the end without animating the remaining layers. */
export const skipToEnd = () => {
  cancel();
  phaseElapsed = 0;
  animation.update((state) => ({
    ...state,
    status: STATUS.FINISHED,
    frontier: Math.max(0, state.layerCount - 1),
    revealed: Math.max(0, state.layerCount - 1),
    phase: PHASE.SETTLE,
    progress: 1
  }));
};

/**
 * Release the frame loop. Must be called when the host component unmounts,
 * otherwise the callback keeps running against a detached view.
 */
export const destroyAnimation = () => {
  cancel();
};

// ------------------------------------------------------------------ helpers

/** Ease-in-out cubic, used to make travel start and end gently. */
export const easeInOut = (t) =>
  t < 0.5 ? 4 * t * t * t : 1 - Math.pow(-2 * t + 2, 3) / 2;

/**
 * How much of a layer's value to show.
 * Returns 1 for revealed layers, a ramp for the one currently settling, and 0
 * for layers the walkthrough has not reached.
 */
export const layerReveal = (state, layerIndex) => {
  if (state.status === STATUS.LIVE || state.status === STATUS.FINISHED) return 1;
  if (layerIndex === 0) return 1;
  if (layerIndex <= state.revealed) return 1;
  if (layerIndex === state.frontier && state.phase === PHASE.SETTLE) {
    return state.progress;
  }
  return 0;
};
