/**
 * Application state for the DNN Explainer.
 *
 * A single writable store holds the architecture spec and the input vector.
 * Everything the UI draws -- the network, activations, prediction, colour
 * ranges -- is derived from that spec, so there is exactly one source of truth
 * and no possibility of the diagram disagreeing with the maths.
 */

import { writable, derived } from 'svelte/store';

import { createNetwork, resizeNetwork, setWeight, setBias, getWeightMagnitude, countParameters } from './engine/network.js';
import { forwardPass, getPrediction, getActivationRanges } from './engine/forwardPass.js';
import { RELU, SOFTMAX } from './engine/activations.js';
import {
  cloneNetwork, computeGradients, createPatternDataset, evaluateDataset,
  summarizeGradients, summarizeWeightUpdates, trainEpoch
} from './engine/training.js';
import { prepareDataset, DATASET_STATUS } from './engine/dataset.js';

/** Architecture limits, kept modest so the SVG stays readable and fast. */
export const LIMITS = {
  inputNeurons: { min: 2, max: 8 },
  hiddenLayers: { min: 1, max: 5 },
  neuronsPerLayer: { min: 1, max: 12 },
  outputNeurons: { min: 2, max: 6 }
};

export const clamp = (value, { min, max }) => Math.max(min, Math.min(max, value));

const DEFAULT_INPUTS = [0.8, 0.25, 0.6, 0.1];

/*
 * Output labels are deliberately neutral.
 *
 * This network is untrained and has never seen any dataset, so naming the
 * outputs after real categories ("Cat", "Dog") would claim a capability that
 * does not exist: the outputs are just two numbers, and the larger one wins.
 * Concrete labels can be introduced once training makes them mean something.
 */
const DEFAULT_LABELS = ['Output A', 'Output B'];

/**
 * The editable specification. The network itself is derived from this, which
 * means an architecture change is a single store update rather than a manual
 * teardown and rebuild.
 */
export const spec = writable({
  inputSize: 4,
  hiddenSizes: [5, 5],
  outputSize: 2,
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 7,
  inputs: [...DEFAULT_INPUTS],
  labels: [...DEFAULT_LABELS]
});

/**
 * The network, rebuilt whenever the shape changes but weight-preserving so the
 * user's edits and the visual arrangement survive an architecture tweak.
 */
const networkInternal = writable(
  createNetwork({
    layerSizes: [4, 5, 5, 2],
    hiddenActivation: RELU,
    outputActivation: SOFTMAX,
    seed: 7
  })
);

let currentSpec;
let currentNetwork;

spec.subscribe((value) => {
  currentSpec = value;
});

networkInternal.subscribe((value) => {
  currentNetwork = value;
});

const layerSizesOf = (s) => [s.inputSize, ...s.hiddenSizes, s.outputSize];

/** Re-derive the network from the current spec, keeping compatible weights. */
const syncNetwork = () => {
  networkInternal.update((net) =>
    resizeNetwork(net, {
      layerSizes: layerSizesOf(currentSpec),
      hiddenActivation: currentSpec.hiddenActivation,
      outputActivation: currentSpec.outputActivation
    })
  );
};

// ------------------------------------------------------------------ actions

/** Replace one input value. */
export const setInput = (index, value) => {
  spec.update((s) => {
    const inputs = [...s.inputs];
    inputs[index] = value;
    return { ...s, inputs };
  });
};

/** Resize the input layer, padding or trimming the input vector to match. */
export const setInputSize = (size) => {
  const next = clamp(size, LIMITS.inputNeurons);
  spec.update((s) => {
    const inputs = Array.from({ length: next }, (_, i) =>
      s.inputs[i] !== undefined ? s.inputs[i] : Number((Math.random() * 0.9 + 0.05).toFixed(2))
    );
    return { ...s, inputSize: next, inputs };
  });
  syncNetwork();
};

/** Resize the output layer, keeping existing class labels where possible. */
export const setOutputSize = (size) => {
  const next = clamp(size, LIMITS.outputNeurons);
  spec.update((s) => {
    const labels = Array.from({ length: next }, (_, i) =>
      s.labels[i] !== undefined ? s.labels[i] : `Class ${i + 1}`
    );
    return { ...s, outputSize: next, labels };
  });
  syncNetwork();
};

/** Add or remove hidden layers, copying the last layer's width for new ones. */
export const setHiddenLayerCount = (count) => {
  const next = clamp(count, LIMITS.hiddenLayers);
  spec.update((s) => {
    const hiddenSizes = [...s.hiddenSizes];
    while (hiddenSizes.length < next) {
      hiddenSizes.push(hiddenSizes[hiddenSizes.length - 1] ?? 4);
    }
    hiddenSizes.length = next;
    return { ...s, hiddenSizes };
  });
  syncNetwork();
};

/** Set the width of one hidden layer, or all of them when index is omitted. */
export const setNeuronsPerLayer = (size, index) => {
  const next = clamp(size, LIMITS.neuronsPerLayer);
  spec.update((s) => {
    const hiddenSizes =
      index === undefined
        ? s.hiddenSizes.map(() => next)
        : s.hiddenSizes.map((v, i) => (i === index ? next : v));
    return { ...s, hiddenSizes };
  });
  syncNetwork();
};

export const setHiddenActivation = (id) => {
  spec.update((s) => ({ ...s, hiddenActivation: id }));
  syncNetwork();
};

export const setOutputActivation = (id) => {
  spec.update((s) => ({ ...s, outputActivation: id }));
  syncNetwork();
};

/** Edit a single connection weight. */
export const updateWeight = (targetLayerIndex, sourceIndex, targetIndex, weight) => {
  networkInternal.update((net) =>
    setWeight(net, targetLayerIndex, sourceIndex, targetIndex, weight)
  );
};

/** Edit a single neuron bias. */
export const updateBias = (layerIndex, neuronIndex, bias) => {
  networkInternal.update((net) => setBias(net, layerIndex, neuronIndex, bias));
};

/** Draw a fresh set of random weights. */
export const reseedNetwork = (seed = Math.floor(Math.random() * 100000)) => {
  spec.update((s) => ({ ...s, seed }));
  networkInternal.set(
    createNetwork({
      layerSizes: layerSizesOf(currentSpec),
      hiddenActivation: currentSpec.hiddenActivation,
      outputActivation: currentSpec.outputActivation,
      seed
    })
  );
};

/** Restore the initial demo configuration. */
export const resetAll = () => {
  clearTrainingTimer();
  clearWalkthroughTimer();
  walkthroughTransaction = undefined;
  trainingWalkthrough.set(initialWalkthroughState());
  training.set(initialTrainingState());
  spec.set({
    inputSize: 4,
    hiddenSizes: [5, 5],
    outputSize: 2,
    hiddenActivation: RELU,
    outputActivation: SOFTMAX,
    seed: 7,
    inputs: [...DEFAULT_INPUTS],
    labels: [...DEFAULT_LABELS]
  });
  networkInternal.set(
    createNetwork({
      layerSizes: [4, 5, 5, 2],
      hiddenActivation: RELU,
      outputActivation: SOFTMAX,
      seed: 7
    })
  );
};

// ------------------------------------------------------------------ derived

/** Read-only view of the raw network (weights and biases, pre-computation). */
export const network = derived(networkInternal, (n) => n);

/**
 * The network after forward propagation: every neuron carries the values the
 * UI displays. This is the store the visualization reads from.
 */
export const activatedNetwork = derived([networkInternal, spec], ([net, s]) => {
  // Guard against a transient mismatch between the spec and a network that has
  // not yet been resized, which would otherwise throw mid-update.
  const expected = net.layers[0].size;
  const inputs = Array.from({ length: expected }, (_, i) => s.inputs[i] ?? 0);
  return forwardPass(net, inputs);
});

export const prediction = derived([activatedNetwork, spec], ([net, s]) =>
  getPrediction(net, s.labels)
);

export const activationRanges = derived(activatedNetwork, (net) =>
  getActivationRanges(net)
);

export const weightMagnitude = derived(networkInternal, (net) =>
  getWeightMagnitude(net)
);

export const parameterCount = derived(networkInternal, (net) => countParameters(net));

// ------------------------------------------------------------------ custom dataset

const initialDatasetState = () => ({
  status: DATASET_STATUS.EMPTY,
  train: undefined,
  test: undefined,
  classes: undefined,
  stats: undefined,
  size: undefined,
  error: undefined,
  active: false,
  csvText: undefined
});

export const customDataset = writable(initialDatasetState());

let currentDataset = initialDatasetState();
customDataset.subscribe((value) => { currentDataset = value; });

/** Parse and prepare a CSV dataset for training. */
export const loadDataset = (csvText, trainRatio = 0.8) => {
  const result = prepareDataset(csvText, trainRatio);
  customDataset.set({ ...result, csvText, active: currentDataset.active });
};

/** Switch training to use the uploaded custom dataset. */
export const useCustomDataset = () => {
  if (currentDataset.status !== DATASET_STATUS.VALID) return;
  
  // Adjust network architecture to match dataset
  const inputSize = currentDataset.train[0].inputs.length;
  const outputSize = currentDataset.classes.length;
  
  spec.update((s) => {
    const labels = currentDataset.classes.map(c => c);
    return { ...s, inputSize, outputSize, labels };
  });
  syncNetwork();
  
  customDataset.update((state) => ({ ...state, active: true }));
  
  // Reset training state when switching datasets
  clearTrainingTimer();
  resetTrainingWalkthrough();
  const rate = currentTraining.learningRate;
  training.set({ ...initialTrainingState(), learningRate: rate });
};

/** Switch training back to the synthetic pattern dataset. */
export const useSyntheticDataset = () => {
  customDataset.update((state) => ({ ...state, active: false }));
  
  // Reset training state when switching datasets
  clearTrainingTimer();
  resetTrainingWalkthrough();
  const rate = currentTraining.learningRate;
  training.set({ ...initialTrainingState(), learningRate: rate });
};

// ------------------------------------------------------------------ training

export const TRAINING_STATUS = {
  IDLE: 'idle',
  RUNNING: 'running',
  PAUSED: 'paused'
};

const initialTrainingState = () => ({
  status: TRAINING_STATUS.IDLE,
  epoch: 0,
  learningRate: 0.08,
  loss: undefined,
  accuracy: undefined,
  history: [],
  initialNetwork: undefined,
  initialInputs: undefined
});

export const training = writable(initialTrainingState());

let currentTraining = initialTrainingState();
training.subscribe((value) => { currentTraining = value; });

const trainingDataset = () => {
  if (currentDataset.active && currentDataset.train) {
    return currentDataset.train;
  }
  return createPatternDataset(currentSpec.inputSize);
};

export const setLearningRate = (value) => {
  const rate = Math.max(0.001, Math.min(1, Number(value)));
  if (!Number.isFinite(rate)) return;
  training.update((state) => ({ ...state, learningRate: rate }));
};

const rememberTrainingStart = () => {
  if (currentTraining.initialNetwork !== undefined) return;
  training.update((state) => ({
    ...state,
    initialNetwork: cloneNetwork(currentNetwork),
    initialInputs: [...currentSpec.inputs]
  }));
};

/** Apply one real full-batch backpropagation update to the rendered network. */
export const trainOneEpoch = () => {
  rememberTrainingStart();
  const dataset = trainingDataset();
  const result = trainEpoch(currentNetwork, dataset, currentTraining.learningRate);
  
  // Update labels only if using synthetic dataset
  if (!currentDataset.active) {
    spec.update((s) => ({ ...s, labels: ['Pattern A', 'Pattern B'] }));
  }
  
  networkInternal.set(result.network);
  training.update((state) => ({
    ...state,
    epoch: state.epoch + 1,
    loss: result.after.loss,
    accuracy: result.after.accuracy,
    history: [...state.history, result.after.loss].slice(-60)
  }));
  return result.after;
};

let trainingTimer;

const clearTrainingTimer = () => {
  if (trainingTimer !== undefined) {
    clearInterval(trainingTimer);
    trainingTimer = undefined;
  }
};

export const WALKTHROUGH_PHASE = {
  IDLE: 'idle',
  FORWARD: 'forward',
  LOSS: 'loss',
  BACKPROP: 'backprop',
  UPDATE: 'update',
  COMPLETE: 'complete'
};

export const WALKTHROUGH_STATUS = {
  IDLE: 'idle',
  RUNNING: 'running',
  PAUSED: 'paused',
  FINISHED: 'finished'
};

const initialWalkthroughState = () => ({
  status: WALKTHROUGH_STATUS.IDLE,
  phase: WALKTHROUGH_PHASE.IDLE,
  step: 0,
  backwardLayer: undefined,
  target: undefined,
  targetLabel: undefined,
  probabilities: [],
  prediction: undefined,
  sampleLoss: undefined,
  batchLoss: undefined,
  gradientLayers: [],
  updates: []
});

export const trainingWalkthrough = writable(initialWalkthroughState());
let currentWalkthrough = initialWalkthroughState();
let walkthroughTransaction;
let walkthroughTimer;
trainingWalkthrough.subscribe((value) => { currentWalkthrough = value; });

const clearWalkthroughTimer = () => {
  if (walkthroughTimer !== undefined) {
    clearTimeout(walkthroughTimer);
    walkthroughTimer = undefined;
  }
};

const reducedMotion = () => typeof window !== 'undefined' &&
  typeof window.matchMedia === 'function' &&
  window.matchMedia('(prefers-reduced-motion: reduce)').matches;

const applyWalkthroughUpdate = () => {
  if (walkthroughTransaction === undefined || walkthroughTransaction.applied) return;
  walkthroughTransaction.applied = true;
  const result = walkthroughTransaction.result;
  networkInternal.set(result.network);
  training.update((state) => ({
    ...state,
    epoch: state.epoch + 1,
    loss: result.after.loss,
    accuracy: result.after.accuracy,
    history: [...state.history, result.after.loss].slice(-60)
  }));
};

export const nextTrainingStage = () => {
  clearWalkthroughTimer();
  const state = currentWalkthrough;
  if (state.phase === WALKTHROUGH_PHASE.IDLE || state.phase === WALKTHROUGH_PHASE.COMPLETE) return;

  if (state.phase === WALKTHROUGH_PHASE.FORWARD) {
    trainingWalkthrough.set({ ...state, phase: WALKTHROUGH_PHASE.LOSS, step: 2 });
  } else if (state.phase === WALKTHROUGH_PHASE.LOSS) {
    trainingWalkthrough.set({ ...state, phase: WALKTHROUGH_PHASE.BACKPROP, step: 3,
      backwardLayer: currentNetwork.layers.length - 1 });
  } else if (state.phase === WALKTHROUGH_PHASE.BACKPROP && state.backwardLayer > 1) {
    trainingWalkthrough.set({ ...state, backwardLayer: state.backwardLayer - 1 });
  } else if (state.phase === WALKTHROUGH_PHASE.BACKPROP) {
    applyWalkthroughUpdate();
    trainingWalkthrough.set({ ...state, phase: WALKTHROUGH_PHASE.UPDATE, step: 4,
      backwardLayer: undefined });
  } else if (state.phase === WALKTHROUGH_PHASE.UPDATE) {
    trainingWalkthrough.set({ ...state, phase: WALKTHROUGH_PHASE.COMPLETE, step: 4,
      status: WALKTHROUGH_STATUS.FINISHED });
  }
};

const scheduleWalkthrough = () => {
  clearWalkthroughTimer();
  if (currentWalkthrough.status !== WALKTHROUGH_STATUS.RUNNING) return;
  walkthroughTimer = setTimeout(() => {
    nextTrainingStage();
    if (currentWalkthrough.phase !== WALKTHROUGH_PHASE.COMPLETE) scheduleWalkthrough();
  }, 1150);
};

export const startTrainingWalkthrough = () => {
  if (currentSpec.outputSize < 2) return;
  clearTrainingTimer();
  clearWalkthroughTimer();
  rememberTrainingStart();

  const dataset = trainingDataset();
  const sample = dataset[0];
  const sampleResult = computeGradients(currentNetwork, sample.inputs, sample.target);
  const result = trainEpoch(currentNetwork, dataset, currentTraining.learningRate);
  walkthroughTransaction = { result, applied: false };
  spec.update((s) => ({ ...s, inputs: [...sample.inputs] }));
  
  // Update labels only if using synthetic dataset
  if (!currentDataset.active) {
    spec.update((s) => ({ ...s, labels: ['Pattern A', 'Pattern B'] }));
  }
  
  const targetLabel = currentDataset.active && currentDataset.classes
    ? currentDataset.classes[sample.target]
    : (sample.target === 0 ? 'Pattern A' : 'Pattern B');

  trainingWalkthrough.set({
    ...initialWalkthroughState(),
    status: reducedMotion() ? WALKTHROUGH_STATUS.PAUSED : WALKTHROUGH_STATUS.RUNNING,
    phase: WALKTHROUGH_PHASE.FORWARD,
    step: 1,
    target: sample.target,
    targetLabel,
    probabilities: sampleResult.probabilities,
    prediction: sampleResult.prediction,
    sampleLoss: sampleResult.loss,
    batchLoss: result.before.loss,
    gradientLayers: summarizeGradients(result.gradients, result.gradientScale),
    updates: summarizeWeightUpdates(currentNetwork, result.network)
  });
  if (!reducedMotion()) scheduleWalkthrough();
};

export const pauseTrainingWalkthrough = () => {
  if (currentWalkthrough.status !== WALKTHROUGH_STATUS.RUNNING) return;
  clearWalkthroughTimer();
  trainingWalkthrough.update((state) => ({ ...state, status: WALKTHROUGH_STATUS.PAUSED }));
};

export const resumeTrainingWalkthrough = () => {
  if (currentWalkthrough.status !== WALKTHROUGH_STATUS.PAUSED) return;
  trainingWalkthrough.update((state) => ({ ...state, status: WALKTHROUGH_STATUS.RUNNING }));
  scheduleWalkthrough();
};

export const skipTrainingWalkthrough = () => {
  if (currentWalkthrough.phase === WALKTHROUGH_PHASE.IDLE) return;
  clearWalkthroughTimer();
  applyWalkthroughUpdate();
  trainingWalkthrough.update((state) => ({ ...state,
    status: WALKTHROUGH_STATUS.FINISHED, phase: WALKTHROUGH_PHASE.COMPLETE,
    step: 4, backwardLayer: undefined }));
};

export const resetTrainingWalkthrough = () => {
  clearWalkthroughTimer();
  walkthroughTransaction = undefined;
  trainingWalkthrough.set(initialWalkthroughState());
};

export const startTraining = () => {
  if (currentTraining.status === TRAINING_STATUS.RUNNING) return;
  resetTrainingWalkthrough();
  rememberTrainingStart();
  const metrics = evaluateDataset(currentNetwork, trainingDataset());
  training.update((state) => ({ ...state, status: TRAINING_STATUS.RUNNING,
    loss: state.loss === undefined ? metrics.loss : state.loss,
    accuracy: state.accuracy === undefined ? metrics.accuracy : state.accuracy }));
  clearTrainingTimer();
  trainingTimer = setInterval(trainOneEpoch, 120);
};

export const pauseTraining = () => {
  clearTrainingTimer();
  training.update((state) => ({ ...state,
    status: state.epoch > 0 ? TRAINING_STATUS.PAUSED : TRAINING_STATUS.IDLE }));
};

export const resetTraining = () => {
  clearTrainingTimer();
  resetTrainingWalkthrough();
  if (currentTraining.initialNetwork !== undefined) {
    networkInternal.set(cloneNetwork(currentTraining.initialNetwork));
  }
  spec.update((s) => ({ ...s,
    inputs: currentTraining.initialInputs === undefined ? s.inputs : [...currentTraining.initialInputs],
    labels: Array.from({ length: s.outputSize }, (_, i) =>
      DEFAULT_LABELS[i] !== undefined ? DEFAULT_LABELS[i] : `Output ${i + 1}`) }));
  const rate = currentTraining.learningRate;
  training.set({ ...initialTrainingState(), learningRate: rate });
};

export const destroyTraining = () => {
  clearTrainingTimer();
  clearWalkthroughTimer();
};

// ------------------------------------------------------------------ selection

/** Currently inspected neuron: {layerIndex, index} or undefined. */
export const selectedNeuron = writable(undefined);

/** Currently inspected connection, or undefined. */
export const selectedConnection = writable(undefined);

/** Neuron under the pointer, used to highlight its connections. */
export const hoveredNeuron = writable(undefined);

export const selectNeuron = (layerIndex, index) => {
  selectedConnection.set(undefined);
  selectedNeuron.update((current) =>
    current && current.layerIndex === layerIndex && current.index === index
      ? undefined
      : { layerIndex, index }
  );
};

export const selectConnection = (targetLayerIndex, sourceIndex, targetIndex) => {
  selectedNeuron.set(undefined);
  selectedConnection.update((current) =>
    current &&
    current.targetLayerIndex === targetLayerIndex &&
    current.sourceIndex === sourceIndex &&
    current.targetIndex === targetIndex
      ? undefined
      : { targetLayerIndex, sourceIndex, targetIndex }
  );
};

export const clearSelection = () => {
  selectedNeuron.set(undefined);
  selectedConnection.set(undefined);
};
