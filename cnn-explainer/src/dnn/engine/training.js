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

/**
 * Compute per-class precision, recall, F1-score, and confusion matrix.
 * 
 * @param {object} network
 * @param {array} dataset
 * @param {number} numClasses
 * @returns {object} perClassMetrics and confusionMatrix
 */
export const computeDetailedMetrics = (network, dataset, numClasses) => {
  if (dataset.length === 0) {
    return {
      perClass: Array(numClasses).fill(null).map(() => ({
        precision: 0, recall: 0, f1: 0, support: 0
      })),
      confusionMatrix: Array(numClasses).fill(null).map(() => Array(numClasses).fill(0)),
      totalCorrect: 0,
      totalSamples: 0
    };
  }

  // Initialize confusion matrix: rows = actual, cols = predicted
  const confusion = Array(numClasses).fill(null).map(() => Array(numClasses).fill(0));

  dataset.forEach((example) => {
    const activated = forwardPass(network, example.inputs);
    const probabilities = activated.layers[activated.layers.length - 1]
      .neurons.map((neuron) => neuron.output);
    const predicted = probabilities.indexOf(Math.max(...probabilities));
    const actual = example.target;
    confusion[actual][predicted]++;
  });

  // Compute per-class metrics
  const perClass = Array(numClasses).fill(null).map((_, classIndex) => {
    const truePositive = confusion[classIndex][classIndex];
    const falsePositive = confusion.reduce((sum, row, actualClass) =>
      sum + (actualClass !== classIndex ? row[classIndex] : 0), 0);
    const falseNegative = confusion[classIndex].reduce((sum, count, predictedClass) =>
      sum + (predictedClass !== classIndex ? count : 0), 0);
    const support = confusion[classIndex].reduce((sum, count) => sum + count, 0);

    const precision = truePositive + falsePositive === 0
      ? 0
      : truePositive / (truePositive + falsePositive);
    const recall = truePositive + falseNegative === 0
      ? 0
      : truePositive / (truePositive + falseNegative);
    const f1 = precision + recall === 0
      ? 0
      : (2 * precision * recall) / (precision + recall);

    return { precision, recall, f1, support };
  });

  const totalCorrect = confusion.reduce((sum, row, i) => sum + row[i], 0);
  const totalSamples = dataset.length;

  return {
    perClass,
    confusionMatrix: confusion,
    totalCorrect,
    totalSamples,
    accuracy: totalCorrect / totalSamples
  };
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
  return {
    network: next,
    before,
    after: evaluateDataset(next, dataset),
    gradients,
    gradientScale: 1 / dataset.length
  };
};

/** Condense a full gradient tensor into honest per-layer values for the UI. */
export const summarizeGradients = (gradients, scale = 1) =>
  gradients.map((layer, layerIndex) => {
    if (layerIndex === 0) return { layerIndex, meanAbsolute: 0, maxAbsolute: 0, count: 0 };
    const values = [];
    layer.biases.forEach((value) => values.push(value * scale));
    layer.weights.forEach((row) => row.forEach((value) => values.push(value * scale)));
    const absolute = values.map(Math.abs);
    return {
      layerIndex,
      meanAbsolute: absolute.reduce((sum, value) => sum + value, 0) / absolute.length,
      maxAbsolute: Math.max(...absolute),
      count: values.length
    };
  });

/**
 * Extract per-connection gradient magnitudes for visualization.
 * Returns a map keyed by "layerIndex-sourceIndex-targetIndex" with absolute gradient values.
 */
export const getConnectionGradients = (gradients, scale = 1) => {
  const connectionGradients = {};
  for (let l = 1; l < gradients.length; l++) {
    gradients[l].weights.forEach((targetWeights, targetIndex) => {
      targetWeights.forEach((grad, sourceIndex) => {
        const key = `${l}-${sourceIndex}-${targetIndex}`;
        connectionGradients[key] = Math.abs(grad * scale);
      });
    });
  }
  return connectionGradients;
};

/**
 * Compute statistics for gradient magnitudes across all connections.
 * Used to normalize gradient visualization intensity.
 */
export const computeGradientStats = (connectionGradients) => {
  const values = Object.values(connectionGradients);
  if (values.length === 0) return { min: 0, max: 0, mean: 0, median: 0 };
  
  const sorted = [...values].sort((a, b) => a - b);
  const sum = values.reduce((acc, v) => acc + v, 0);
  
  return {
    min: sorted[0],
    max: sorted[sorted.length - 1],
    mean: sum / values.length,
    median: sorted[Math.floor(sorted.length / 2)],
    percentile90: sorted[Math.floor(sorted.length * 0.9)]
  };
};

/** Largest real parameter changes, useful for explaining the update step. */
export const summarizeWeightUpdates = (before, after, limit = 4) => {
  const updates = [];
  for (let l = 1; l < before.layers.length; l++) {
    before.layers[l].neurons.forEach((neuron, targetIndex) => {
      neuron.weights.forEach((weight, sourceIndex) => {
        const next = after.layers[l].neurons[targetIndex].weights[sourceIndex];
        updates.push({ layerIndex: l, sourceIndex, targetIndex, before: weight,
          after: next, change: next - weight });
      });
    });
  }
  return updates.sort((a, b) => Math.abs(b.change) - Math.abs(a.change)).slice(0, limit);
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