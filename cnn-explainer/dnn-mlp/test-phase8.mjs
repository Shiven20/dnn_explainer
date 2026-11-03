/**
 * Phase 8 regression suite: gradient visualization on connections.
 */

import { createNetwork } from '../src/dnn/engine/network.js';
import { RELU, SOFTMAX } from '../src/dnn/engine/activations.js';
import { trainEpoch, getConnectionGradients, computeGradientStats } from '../src/dnn/engine/training.js';

let passed = 0;
let failed = 0;

const check = (label, condition) => {
  if (condition) {
    passed++;
  } else {
    failed++;
    console.error(`✗ ${label}`);
  }
};

const approx = (a, b, tol = 1e-6) => Math.abs(a - b) < tol;

console.log('\n=== Phase 8: Gradient Visualization ===\n');

// ------------------------------------------------------------------ setup

const network = createNetwork({
  layerSizes: [3, 4, 2],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 888
});

const dataset = [
  { inputs: [0.8, 0.2, 0.1], target: 0 },
  { inputs: [0.1, 0.9, 0.8], target: 1 },
  { inputs: [0.7, 0.3, 0.2], target: 0 },
  { inputs: [0.2, 0.8, 0.9], target: 1 }
];

// ------------------------------------------------------------------ gradient extraction

const result = trainEpoch(network, dataset, 0.1);
const connectionGradients = getConnectionGradients(result.gradients, result.gradientScale);

check('getConnectionGradients returns an object', typeof connectionGradients === 'object');
check('connectionGradients is not an array', !Array.isArray(connectionGradients));

// Count expected connections: (3 inputs → 4 hidden) + (4 hidden → 2 output) = 12 + 8 = 20
const keys = Object.keys(connectionGradients);
check('connection count matches network structure', keys.length === 20);

// Check key format: "layerIndex-sourceIndex-targetIndex"
const keyPattern = /^\d+-\d+-\d+$/;
check('all keys match expected format', keys.every(k => keyPattern.test(k)));

// Gradients should be absolute values (non-negative)
const values = Object.values(connectionGradients);
check('all gradients are non-negative', values.every(v => v >= 0));
check('all gradients are finite', values.every(v => Number.isFinite(v)));

// At least some gradients should be non-zero after training
check('at least one gradient is non-zero', values.some(v => v > 0));

// Check specific connection exists
check('layer 1 connection exists', connectionGradients['1-0-0'] !== undefined);
check('layer 2 connection exists', connectionGradients['2-0-0'] !== undefined);

// Gradients for hidden layer
const layer1Gradients = keys.filter(k => k.startsWith('1-')).map(k => connectionGradients[k]);
check('layer 1 has 12 connection gradients', layer1Gradients.length === 12);

// Gradients for output layer
const layer2Gradients = keys.filter(k => k.startsWith('2-')).map(k => connectionGradients[k]);
check('layer 2 has 8 connection gradients', layer2Gradients.length === 8);

// ------------------------------------------------------------------ gradient statistics

const stats = computeGradientStats(connectionGradients);

check('stats has min', typeof stats.min === 'number');
check('stats has max', typeof stats.max === 'number');
check('stats has mean', typeof stats.mean === 'number');
check('stats has median', typeof stats.median === 'number');
check('stats has percentile90', typeof stats.percentile90 === 'number');

check('min is non-negative', stats.min >= 0);
check('max is non-negative', stats.max >= 0);
check('mean is non-negative', stats.mean >= 0);
check('median is non-negative', stats.median >= 0);
check('percentile90 is non-negative', stats.percentile90 >= 0);

check('min <= median', stats.min <= stats.median);
check('median <= max', stats.median <= stats.max);
check('median <= percentile90', stats.median <= stats.percentile90);
check('percentile90 <= max', stats.percentile90 <= stats.max);

// Mean should be within [min, max]
check('mean within range', stats.mean >= stats.min && stats.mean <= stats.max);

// All values are finite
check('all stats are finite', Object.values(stats).every(v => Number.isFinite(v)));

// ------------------------------------------------------------------ empty gradients

const emptyGradients = {};
const emptyStats = computeGradientStats(emptyGradients);

check('empty stats returns zeros', emptyStats.min === 0 && emptyStats.max === 0);
check('empty stats mean is zero', emptyStats.mean === 0);
check('empty stats median is zero', emptyStats.median === 0);

// ------------------------------------------------------------------ single value

const singleGradient = { '1-0-0': 0.5 };
const singleStats = computeGradientStats(singleGradient);

check('single value min equals max', singleStats.min === singleStats.max);
check('single value equals 0.5', approx(singleStats.min, 0.5));
check('single value mean equals median', approx(singleStats.mean, singleStats.median));

// ------------------------------------------------------------------ scaling

// Scale = 0.5 should halve all gradient values
const resultScaled = trainEpoch(network, dataset, 0.1);
const gradientsUnscaled = getConnectionGradients(resultScaled.gradients, 1.0);
const gradientsScaled = getConnectionGradients(resultScaled.gradients, 0.5);

const key = keys[0];
check('scaling affects gradient magnitude', approx(gradientsScaled[key], gradientsUnscaled[key] * 0.5));

const unscaledStats = computeGradientStats(gradientsUnscaled);
const scaledStats = computeGradientStats(gradientsScaled);
check('scaling affects max statistic', approx(scaledStats.max, unscaledStats.max * 0.5, 1e-5));

// ------------------------------------------------------------------ multiple epochs

let trained = network;
const gradientHistory = [];

for (let i = 0; i < 3; i++) {
  const epochResult = trainEpoch(trained, dataset, 0.1);
  trained = epochResult.network;
  const grads = getConnectionGradients(epochResult.gradients, epochResult.gradientScale);
  gradientHistory.push(grads);
}

check('gradients computed for each epoch', gradientHistory.length === 3);
check('each epoch has same connection count', gradientHistory.every(g => Object.keys(g).length === 20));

// Gradients should generally decrease as network converges (loss decreases)
const firstEpochMax = computeGradientStats(gradientHistory[0]).max;
const lastEpochMax = computeGradientStats(gradientHistory[2]).max;
check('gradients exist in first epoch', firstEpochMax > 0);
check('gradients exist in last epoch', lastEpochMax >= 0);

// ------------------------------------------------------------------ normalization for visualization

// Simulate visualization normalization using 90th percentile
const testGrads = {
  'weak': 0.001,
  'medium': 0.05,
  'strong': 0.1,
  'outlier': 1.0
};
const testStats = computeGradientStats(testGrads);

const normalize = (value, stats) => Math.min(1, value / stats.percentile90);

check('weak gradient normalized < 1', normalize(0.001, testStats) < 1);
check('strong gradient can reach 1', normalize(testStats.percentile90, testStats) === 1);
check('outlier capped at 1', normalize(1000, testStats) === 1);

// ------------------------------------------------------------------ larger network

const largeNetwork = createNetwork({
  layerSizes: [5, 8, 6, 3],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 999
});

const largeDataset = Array.from({ length: 10 }, (_, i) => ({
  inputs: [0.1, 0.2, 0.3, 0.4, 0.5],
  target: i % 3
}));

const largeResult = trainEpoch(largeNetwork, largeDataset, 0.05);
const largeGradients = getConnectionGradients(largeResult.gradients, largeResult.gradientScale);

// Expected connections: (5→8) + (8→6) + (6→3) = 40 + 48 + 18 = 106
check('large network connection count', Object.keys(largeGradients).length === 106);

const largeStats = computeGradientStats(largeGradients);
check('large network stats are valid', largeStats.max >= largeStats.min);
check('large network gradients are finite', Object.values(largeGradients).every(v => Number.isFinite(v)));

// ------------------------------------------------------------------ summary

console.log(`\n${passed} checks passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
