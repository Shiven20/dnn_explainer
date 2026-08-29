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

/** Architecture limits, kept modest so the SVG stays readable and fast. */
export const LIMITS = {
  inputNeurons: { min: 2, max: 8 },
  hiddenLayers: { min: 1, max: 5 },
  neuronsPerLayer: { min: 1, max: 12 },
  outputNeurons: { min: 2, max: 6 }
};

export const clamp = (value, { min, max }) => Math.max(min, Math.min(max, value));

const DEFAULT_INPUTS = [0.8, 0.25, 0.6, 0.1];
const DEFAULT_LABELS = ['Cat', 'Dog'];

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
