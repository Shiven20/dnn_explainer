/**
 * Forward propagation.
 *
 * For each neuron:      z = Σ(xᵢ · wᵢ) + b
 *                  output = f(z)
 *
 * The output of layer N becomes the input of layer N+1. Every number rendered
 * by the UI comes from here, so the visualization can never drift away from
 * the actual computation.
 */

import { getActivation, softmax, SOFTMAX } from './activations.js';

/**
 * Run the network over an input vector.
 *
 * Returns a new network object rather than mutating the one passed in, so
 * Svelte stores see a fresh reference and re-render predictably.
 *
 * @param {object} network
 * @param {number[]} inputs One value per input neuron.
 * @returns {object} A network whose neurons carry preActivation and output.
 */
export const forwardPass = (network, inputs) => {
  const inputLayer = network.layers[0];
  if (inputs.length !== inputLayer.size) {
    throw new Error(
      `Expected ${inputLayer.size} input values but received ${inputs.length}`
    );
  }

  const layers = [];

  // Input layer holds its values directly; there is nothing to compute.
  layers.push({
    ...inputLayer,
    neurons: inputLayer.neurons.map((neuron, i) => ({
      ...neuron,
      preActivation: inputs[i],
      output: inputs[i]
    }))
  });

  for (let l = 1; l < network.layers.length; l++) {
    const layer = network.layers[l];
    const prevOutputs = layers[l - 1].neurons.map((n) => n.output);

    // Weighted sum for every neuron in this layer.
    const preActivations = layer.neurons.map((neuron) => {
      let sum = neuron.bias;
      for (let i = 0; i < neuron.weights.length; i++) {
        sum += neuron.weights[i] * prevOutputs[i];
      }
      return sum;
    });

    // Softmax needs the whole layer at once; every other activation is
    // element-wise.
    let outputs;
    if (layer.activation === SOFTMAX) {
      outputs = softmax(preActivations);
    } else {
      const { fn } = getActivation(layer.activation);
      outputs = preActivations.map(fn);
    }

    layers.push({
      ...layer,
      neurons: layer.neurons.map((neuron, i) => ({
        ...neuron,
        preActivation: preActivations[i],
        output: outputs[i]
      }))
    });
  }

  return { ...network, layers };
};

/**
 * Break one neuron's weighted sum into its individual terms.
 *
 * This is what the neuron inspection panel renders, so the user can trace a
 * value back to the exact products that produced it.
 */
export const explainNeuron = (network, layerIndex, neuronIndex) => {
  const layer = network.layers[layerIndex];
  if (layer === undefined) return undefined;
  const neuron = layer.neurons[neuronIndex];
  if (neuron === undefined) return undefined;

  // Input neurons have no incoming computation to explain.
  if (layerIndex === 0) {
    return {
      neuron,
      layer,
      isInput: true,
      terms: [],
      weightedSum: neuron.output,
      bias: 0,
      preActivation: neuron.output,
      output: neuron.output
    };
  }

  const prevLayer = network.layers[layerIndex - 1];
  const terms = neuron.weights.map((weight, i) => ({
    sourceIndex: i,
    sourceLabel: `${prevLayer.kind === 'input' ? 'x' : 'a'}${i + 1}`,
    input: prevLayer.neurons[i].output,
    weight,
    product: weight * prevLayer.neurons[i].output
  }));

  const weightedSum = terms.reduce((acc, t) => acc + t.product, 0);

  return {
    neuron,
    layer,
    isInput: false,
    terms,
    weightedSum,
    bias: neuron.bias,
    preActivation: weightedSum + neuron.bias,
    output: neuron.output,
    activation: layer.activation
  };
};

/**
 * Output layer read as a prediction.
 * @param {object} network
 * @param {string[]} labels Class names, one per output neuron.
 */
export const getPrediction = (network, labels = []) => {
  const outputLayer = network.layers[network.layers.length - 1];
  const scores = outputLayer.neurons.map((n) => n.output);

  let bestIndex = 0;
  for (let i = 1; i < scores.length; i++) {
    if (scores[i] > scores[bestIndex]) bestIndex = i;
  }

  // With softmax the outputs are already a distribution. For any other
  // activation they are not, so normalize for display while keeping the raw
  // value available.
  const isProbability = outputLayer.activation === SOFTMAX;
  const total = scores.reduce((a, b) => a + Math.max(0, b), 0);

  return {
    index: bestIndex,
    label: labels[bestIndex] ?? `Output ${bestIndex + 1}`,
    confidence: scores[bestIndex],
    isProbability,
    classes: scores.map((score, i) => ({
      index: i,
      label: labels[i] ?? `Output ${i + 1}`,
      score,
      share: isProbability
        ? score
        : total > 0
          ? Math.max(0, score) / total
          : 1 / scores.length
    }))
  };
};

/**
 * Per-layer magnitude used to scale the neuron colour ramp.
 *
 * Bounded activations (sigmoid, tanh) have a known range, so use it directly
 * and keep colours comparable between layers. Unbounded ones (ReLU, linear)
 * are scaled by the largest value actually present.
 */
export const getActivationRanges = (network) =>
  network.layers.map((layer) => {
    if (layer.activation === SOFTMAX) return 1;

    const { unbounded, displayRange } = getActivation(layer.activation);
    if (!unbounded) return Math.max(Math.abs(displayRange[0]), Math.abs(displayRange[1]));

    const max = layer.neurons.reduce((acc, n) => Math.max(acc, Math.abs(n.output)), 0);
    return max === 0 ? 1 : max;
  });
