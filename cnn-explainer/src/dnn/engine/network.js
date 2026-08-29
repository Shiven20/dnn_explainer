/**
 * Network construction for the DNN Explainer.
 *
 * The network is a plain data structure -- layers of neurons, with weights held
 * on the incoming connections. Nothing here knows about SVG or Svelte, which
 * keeps the maths independently testable and lets the visualization treat this
 * as a read-only source of truth.
 *
 * Any architecture is supported: a layer-size array defines the shape, so the
 * UI never needs hardcoded coordinates or per-architecture special cases.
 */

import { RELU, SOFTMAX, LINEAR } from './activations.js';

/** Deterministic PRNG, so a given seed always yields the same network. */
export const mulberry32 = (seed) => {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
};

/** Standard normal sample via Box-Muller, drawn from a uniform PRNG. */
const gaussian = (rand) => {
  const u1 = Math.max(rand(), 1e-12);
  const u2 = rand();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
};

export const LAYER_KIND = {
  INPUT: 'input',
  HIDDEN: 'hidden',
  OUTPUT: 'output'
};

/**
 * Human-readable layer name, e.g. "Hidden Layer 2".
 * Hidden layers are numbered from 1 in the order they appear.
 */
export const layerLabel = (kind, hiddenIndex) => {
  if (kind === LAYER_KIND.INPUT) return 'Input';
  if (kind === LAYER_KIND.OUTPUT) return 'Output';
  return `Hidden Layer ${hiddenIndex + 1}`;
};

/**
 * Build a network.
 *
 * @param {object} spec
 * @param {number[]} spec.layerSizes Neuron count per layer, e.g. [4, 5, 5, 2].
 *   Must have at least 2 entries (an input and an output layer).
 * @param {string} spec.hiddenActivation Activation id for every hidden layer.
 * @param {string} spec.outputActivation Activation id for the output layer.
 *   Use 'softmax' for classification.
 * @param {number} spec.seed PRNG seed for weight initialization.
 * @returns {object} network
 */
export const createNetwork = ({
  layerSizes,
  hiddenActivation = RELU,
  outputActivation = SOFTMAX,
  seed = 1
}) => {
  if (!Array.isArray(layerSizes) || layerSizes.length < 2) {
    throw new Error('layerSizes needs at least an input and an output layer');
  }
  if (layerSizes.some((n) => !Number.isInteger(n) || n < 1)) {
    throw new Error('every layer needs at least one neuron');
  }

  const rand = mulberry32(seed);
  const layers = [];
  let hiddenCounter = 0;

  layerSizes.forEach((size, layerIndex) => {
    const isInput = layerIndex === 0;
    const isOutput = layerIndex === layerSizes.length - 1;
    const kind = isInput
      ? LAYER_KIND.INPUT
      : isOutput
        ? LAYER_KIND.OUTPUT
        : LAYER_KIND.HIDDEN;

    // The input layer applies no function; it just holds the values given to it.
    const activation = isInput
      ? LINEAR
      : isOutput
        ? outputActivation
        : hiddenActivation;

    const layer = {
      index: layerIndex,
      kind,
      size,
      activation,
      label: layerLabel(kind, hiddenCounter),
      neurons: []
    };

    if (kind === LAYER_KIND.HIDDEN) hiddenCounter++;

    const fanIn = isInput ? 0 : layerSizes[layerIndex - 1];
    // He-style scaling keeps activations from shrinking or exploding as depth
    // grows, which matters here because the user can stack up to 5 layers.
    const scale = fanIn > 0 ? Math.sqrt(2 / fanIn) : 0;

    for (let i = 0; i < size; i++) {
      layer.neurons.push({
        layerIndex,
        index: i,
        // Stable identity so Svelte keyed each blocks can track neurons across
        // architecture changes instead of rebuilding every node.
        id: `n-${layerIndex}-${i}`,
        activation,
        bias: isInput ? 0 : Number((gaussian(rand) * 0.25).toFixed(4)),
        // Incoming weights, one per neuron in the previous layer.
        weights: isInput
          ? []
          : Array.from({ length: fanIn }, () =>
              Number((gaussian(rand) * scale).toFixed(4))
            ),
        preActivation: 0,
        output: 0
      });
    }

    layers.push(layer);
  });

  return {
    layers,
    layerSizes: [...layerSizes],
    hiddenActivation,
    outputActivation,
    seed
  };
};

/**
 * Rebuild a network with a new shape while keeping weights that still fit.
 *
 * Regenerating from scratch on every architecture tweak would throw away the
 * user's edited weights and make the visualization jump. This preserves any
 * weight whose (layer, neuron, input) position still exists.
 */
export const resizeNetwork = (network, nextSpec) => {
  const next = createNetwork({
    layerSizes: nextSpec.layerSizes,
    hiddenActivation: nextSpec.hiddenActivation ?? network.hiddenActivation,
    outputActivation: nextSpec.outputActivation ?? network.outputActivation,
    seed: nextSpec.seed ?? network.seed
  });

  next.layers.forEach((layer, l) => {
    const prev = network.layers[l];
    if (prev === undefined) return;

    layer.neurons.forEach((neuron, i) => {
      const prevNeuron = prev.neurons[i];
      if (prevNeuron === undefined) return;

      neuron.bias = prevNeuron.bias;
      neuron.weights = neuron.weights.map((w, j) =>
        prevNeuron.weights[j] !== undefined ? prevNeuron.weights[j] : w
      );
    });
  });

  return next;
};

/** Total learnable parameters, shown in the UI as a sense of model size. */
export const countParameters = (network) =>
  network.layers.reduce((total, layer, l) => {
    if (l === 0) return total;
    const fanIn = network.layers[l - 1].size;
    return total + layer.size * fanIn + layer.size;
  }, 0);

/**
 * Every connection as a flat list, which is the form the SVG layer wants.
 * Each entry references its endpoints by index so lookups stay cheap.
 */
export const getConnections = (network) => {
  const connections = [];
  for (let l = 1; l < network.layers.length; l++) {
    const layer = network.layers[l];
    layer.neurons.forEach((neuron) => {
      neuron.weights.forEach((weight, sourceIndex) => {
        connections.push({
          id: `c-${l}-${sourceIndex}-${neuron.index}`,
          targetLayerIndex: l,
          sourceIndex,
          targetIndex: neuron.index,
          weight
        });
      });
    });
  }
  return connections;
};

/** Largest |weight| in the network, used to normalize edge styling. */
export const getWeightMagnitude = (network) => {
  let max = 0;
  for (let l = 1; l < network.layers.length; l++) {
    network.layers[l].neurons.forEach((neuron) => {
      neuron.weights.forEach((w) => {
        const abs = Math.abs(w);
        if (abs > max) max = abs;
      });
    });
  }
  return max === 0 ? 1 : max;
};

/** Read a single weight, or undefined if the indices don't exist. */
export const getWeight = (network, targetLayerIndex, sourceIndex, targetIndex) => {
  const layer = network.layers[targetLayerIndex];
  if (layer === undefined) return undefined;
  const neuron = layer.neurons[targetIndex];
  if (neuron === undefined) return undefined;
  return neuron.weights[sourceIndex];
};

/**
 * Return a copy of the network with one weight replaced.
 *
 * Copying rather than mutating keeps Svelte's change detection honest: stores
 * only notify subscribers when the reference changes.
 */
export const setWeight = (network, targetLayerIndex, sourceIndex, targetIndex, weight) => ({
  ...network,
  layers: network.layers.map((layer, l) => {
    if (l !== targetLayerIndex) return layer;
    return {
      ...layer,
      neurons: layer.neurons.map((neuron, i) => {
        if (i !== targetIndex) return neuron;
        const weights = [...neuron.weights];
        weights[sourceIndex] = weight;
        return { ...neuron, weights };
      })
    };
  })
});

/** Return a copy of the network with one neuron's bias replaced. */
export const setBias = (network, layerIndex, neuronIndex, bias) => ({
  ...network,
  layers: network.layers.map((layer, l) => {
    if (l !== layerIndex) return layer;
    return {
      ...layer,
      neurons: layer.neurons.map((neuron, i) =>
        i === neuronIndex ? { ...neuron, bias } : neuron
      )
    };
  })
});
