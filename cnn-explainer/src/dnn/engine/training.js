import { forwardPass } from './forwardPass.js';
import { getActivation, SOFTMAX } from './activations.js';
import { mulberry32 } from './network.js';

const EPSILON = 1e-12;

export const crossEntropy = (probabilities, targetIndex) =>
  -Math.log(Math.max(EPSILON, probabilities[targetIndex]));

const emptyGradients = (network) => network.layers.map((layer, layerIndex) => ({
  biases: layerIndex === 0 ? [] : layer.neurons.map(() => 0),
  weights: layerIndex === 0
    ? []
    : layer.neurons.map((neuron) => neuron.weights.map(() => 0))
}));

export const addGradients = (target, source) => {
  for (let l = 1; l < target.length; l++) {
    for (let j = 0; j < target[l].biases.length; j++) {
      target[l].biases[j] += source[l].biases[j];
      for (let i = 0; i < target[l].weights[j].length; i++) {
        target[l].weights[j][i] += source[l].weights[j][i];
      }
    }
  }
  return target;
};

/** Compute exact dense-network gradients for softmax cross-entropy. */
export const computeGradients = (network, inputs, targetIndex) => {
  const activated = forwardPass(network, inputs);
  const outputLayer = activated.layers[activated.layers.length - 1];
  if (outputLayer.activation !== SOFTMAX) {
    throw new Error('Training requires a softmax output layer');
  }
  if (targetIndex < 0 || targetIndex >= outputLayer.size) {
    throw new Error(`Target ${targetIndex} is outside the output layer`);
  }

  const probabilities = outputLayer.neurons.map((neuron) => neuron.output);
  const gradients = emptyGradients(network);
  let delta = probabilities.map((value, index) =>
    value - (index === targetIndex ? 1 : 0));

  for (let l = activated.layers.length - 1; l > 0; l--) {
    const layer = activated.layers[l];
    const previous = activated.layers[l - 1];

    for (let j = 0; j < layer.size; j++) {
      gradients[l].biases[j] = delta[j];
      for (let i = 0; i < previous.size; i++) {
        gradients[l].weights[j][i] = delta[j] * previous.neurons[i].output;
      }
    }

    if (l > 1) {
      const previousActivation = getActivation(previous.activation);
      const nextDelta = previous.neurons.map((neuron, i) => {
        let upstream = 0;
        for (let j = 0; j < layer.size; j++) {
          upstream += layer.neurons[j].weights[i] * delta[j];
        }
        return upstream * previousActivation.derivative(neuron.preActivation);
      });
      delta = nextDelta;
    }
  }

  return {
    gradients,
    loss: crossEntropy(probabilities, targetIndex),
    prediction: probabilities.indexOf(Math.max(...probabilities)),
    probabilities,
    activated
  };
};

export const applyGradients = (network, gradients, learningRate, scale = 1) => ({
  ...network,
  layers: network.layers.map((layer, l) => {
    if (l === 0) return layer;
    return {
      ...layer,
      neurons: layer.neurons.map((neuron, j) => ({
        ...neuron,
        bias: neuron.bias - learningRate * gradients[l].biases[j] * scale,
        weights: neuron.weights.map((weight, i) =>
          weight - learningRate * gradients[l].weights[j][i] * scale)
      }))
    };
  })
});

export const evaluateDataset = (network, dataset) => {
  if (dataset.length === 0) return { loss: 0, accuracy: 0 };
  let loss = 0;
  let correct = 0;
  dataset.forEach((example) => {
    const activated = forwardPass(network, example.inputs);
    const probabilities = activated.layers[activated.layers.length - 1]
      .neurons.map((neuron) => neuron.output);
    loss += crossEntropy(probabilities, example.target);
    if (probabilities.indexOf(Math.max(...probabilities)) === example.target) correct++;
  });
  return { loss: loss / dataset.length, accuracy: correct / dataset.length };
};

/** One full-batch gradient-descent epoch. */
export const trainEpoch = (network, dataset, learningRate = 0.08) => {
  if (dataset.length === 0) {
    return { network, before: { loss: 0, accuracy: 0 }, after: { loss: 0, accuracy: 0 } };
  }
  const before = evaluateDataset(network, dataset);
  const gradients = emptyGradients(network);
  dataset.forEach((example) => {
    addGradients(gradients, computeGradients(network, example.inputs, example.target).gradients);
  });
  const next = applyGradients(network, gradients, learningRate, 1 / dataset.length);
  return { network: next, before, after: evaluateDataset(next, dataset) };
};

/**
 * A deterministic task the browser can generate for any editable input size.
 * Class A means the first half has the stronger average signal; Class B means
 * the second half does. Noise prevents the examples from being duplicates.
 */
export const createPatternDataset = (inputSize, count = 80, seed = 2026) => {
  const rand = mulberry32(seed + inputSize * 97 + count);
  const split = Math.ceil(inputSize / 2);
  return Array.from({ length: count }, (_, sampleIndex) => {
    const target = sampleIndex % 2;
    const inputs = Array.from({ length: inputSize }, (_, inputIndex) => {
      const inFirstHalf = inputIndex < split;
      const isStrong = target === 0 ? inFirstHalf : !inFirstHalf;
      const base = isStrong ? 0.72 : 0.18;
      const value = base + (rand() - 0.5) * 0.28;
      return Number(Math.max(0, Math.min(1, value)).toFixed(4));
    });
    return { inputs, target };
  });
};

export const cloneNetwork = (network) => ({
  ...network,
  layerSizes: [...network.layerSizes],
  layers: network.layers.map((layer) => ({
    ...layer,
    neurons: layer.neurons.map((neuron) => ({
      ...neuron,
      weights: [...neuron.weights]
    }))
  }))
});