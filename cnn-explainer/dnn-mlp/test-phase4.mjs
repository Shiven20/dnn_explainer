/**
 * Tests for Phase 4: activation switching and architecture changes.
 *
 * Both features must change the *computation*, not just the labels. A switcher
 * that relabels the UI without altering neuron values would be a lie, so these
 * tests assert the numbers move and stay internally consistent.
 *
 * Usage: npm run test:phase4
 */

import { get } from 'svelte/store';

import * as stores from '../src/dnn/stores.js';
import { getActivation, RELU, SIGMOID, TANH, LINEAR, SOFTMAX } from '../src/dnn/engine/activations.js';
import { computeLayout } from '../src/dnn/engine/layout.js';
import { explainNeuron } from '../src/dnn/engine/forwardPass.js';

let failures = 0;
let checks = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};
const near = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;

const net = () => get(stores.activatedNetwork);
const spec = () => get(stores.spec);
const pred = () => get(stores.prediction);

const hiddenOutputs = () => {
  const n = net();
  const out = [];
  for (let l = 1; l < n.layers.length - 1; l++) {
    n.layers[l].neurons.forEach((x) => out.push(x.output));
  }
  return out;
};

stores.resetAll();

// ---------------------------------------------------------------- baseline
console.log('Baseline');
check('default shape is 4/5/5/2',
  JSON.stringify(net().layers.map((l) => l.size)) === '[4,5,5,2]',
  JSON.stringify(net().layers.map((l) => l.size)));
check('hidden layers use ReLU by default', net().layers[1].activation === RELU);
check('output layer uses softmax', net().layers[3].activation === SOFTMAX);

// ---------------------------------------------------------------- activations
console.log('\nSwitching activations changes the maths');

const reluOutputs = hiddenOutputs();
const reluPrediction = pred().classes.map((c) => c.share);

for (const id of [SIGMOID, TANH, LINEAR]) {
  stores.setHiddenActivation(id);

  check(`${id}: applied to every hidden layer`, (() => {
    const n = net();
    for (let l = 1; l < n.layers.length - 1; l++) {
      if (n.layers[l].activation !== id) return false;
    }
    return true;
  })());

  check(`${id}: output layer still softmax`,
    net().layers[net().layers.length - 1].activation === SOFTMAX);

  // Every hidden neuron's output must equal f(z) for the *new* function.
  const fn = getActivation(id).fn;
  check(`${id}: hidden outputs equal f(z)`, (() => {
    const n = net();
    for (let l = 1; l < n.layers.length - 1; l++) {
      for (const neuron of n.layers[l].neurons) {
        if (!near(neuron.output, fn(neuron.preActivation))) return false;
      }
    }
    return true;
  })());

  // The values must actually differ from ReLU, or the switch did nothing.
  const now = hiddenOutputs();
  check(`${id}: hidden values differ from ReLU`,
    now.some((v, i) => !near(v, reluOutputs[i])));

  // And the change must reach the prediction.
  check(`${id}: prediction changes`,
    pred().classes.some((c, i) => !near(c.share, reluPrediction[i])));

  // Bounded activations must respect their range.
  const meta = getActivation(id);
  if (!meta.unbounded) {
    check(`${id}: outputs stay within ${meta.displayRange.join('..')}`,
      now.every((v) => v >= meta.displayRange[0] - 1e-9 &&
                       v <= meta.displayRange[1] + 1e-9));
  }

  // Softmax must remain a valid distribution regardless of hidden activation.
  check(`${id}: output still sums to 1`,
    near(pred().classes.reduce((a, c) => a + c.share, 0), 1, 1e-9));
}

// Sigmoid is strictly positive, so nothing should be switched off.
stores.setHiddenActivation(SIGMOID);
check('sigmoid switches nothing off', hiddenOutputs().every((v) => v > 0));

// ReLU on this network does switch at least one unit off, which the panel reports.
stores.setHiddenActivation(RELU);
check('relu switches at least one unit off',
  hiddenOutputs().some((v) => v === 0));

// Switching back must restore the original values exactly: the operation is
// reversible because weights were never touched.
check('switching back to ReLU restores the original values',
  hiddenOutputs().every((v, i) => near(v, reluOutputs[i])));

// ---------------------------------------------------------------- architecture
console.log('\nArchitecture changes');
stores.resetAll();

const paramsBefore = get(stores.parameterCount);

stores.setHiddenLayerCount(3);
check('added a hidden layer', spec().hiddenSizes.length === 3);
check('network gained a layer', net().layers.length === 5);
check('new layer is labelled', net().layers[3].label === 'Hidden Layer 3');
check('parameters increased', get(stores.parameterCount) > paramsBefore);
check('network still runs',
  net().layers[4].neurons.every((n) => Number.isFinite(n.output)));
check('output still a distribution',
  near(pred().classes.reduce((a, c) => a + c.share, 0), 1, 1e-9));

stores.setHiddenLayerCount(1);
check('removed layers back down to one', net().layers.length === 3);
check('still runs with a single hidden layer',
  net().layers[2].neurons.every((n) => Number.isFinite(n.output)));

stores.resetAll();

// Input resize must keep the input vector and the layer in agreement, otherwise
// the forward pass would throw.
stores.setInputSize(7);
check('input layer resized', net().layers[0].size === 7);
check('input vector matches the layer', spec().inputs.length >= 7);
check('hidden weights match the new fan-in',
  net().layers[1].neurons.every((n) => n.weights.length === 7));
check('runs after widening the input',
  net().layers[3].neurons.every((n) => Number.isFinite(n.output)));

stores.setInputSize(2);
check('input layer shrunk', net().layers[0].size === 2);
check('runs after shrinking the input',
  net().layers[3].neurons.every((n) => Number.isFinite(n.output)));

stores.resetAll();

stores.setOutputSize(5);
check('output layer resized', net().layers[3].size === 5);
check('label list grew with it', spec().labels.length === 5);
check('all outputs are labelled',
  pred().classes.every((c) => typeof c.label === 'string' && c.label.length > 0));
check('softmax still sums to 1 with 5 outputs',
  near(pred().classes.reduce((a, c) => a + c.share, 0), 1, 1e-9));

stores.resetAll();

// Per-layer widths must be independent.
stores.setNeuronsPerLayer(9, 0);
check('one hidden layer widened independently',
  spec().hiddenSizes[0] === 9 && spec().hiddenSizes[1] === 5,
  JSON.stringify(spec().hiddenSizes));
check('next layer fan-in follows',
  net().layers[2].neurons.every((n) => n.weights.length === 9));

// ---------------------------------------------------------------- limits
console.log('\nLimits are enforced');
stores.resetAll();

stores.setHiddenLayerCount(99);
check('hidden layers capped at the maximum',
  spec().hiddenSizes.length === stores.LIMITS.hiddenLayers.max,
  `got ${spec().hiddenSizes.length}`);

stores.setHiddenLayerCount(-5);
check('hidden layers floored at the minimum',
  spec().hiddenSizes.length === stores.LIMITS.hiddenLayers.min);

stores.setNeuronsPerLayer(500, 0);
check('neurons per layer capped',
  spec().hiddenSizes[0] === stores.LIMITS.neuronsPerLayer.max);

stores.setNeuronsPerLayer(0, 0);
check('neurons per layer floored at 1',
  spec().hiddenSizes[0] === stores.LIMITS.neuronsPerLayer.min);

stores.setInputSize(999);
check('input neurons capped',
  spec().inputSize === stores.LIMITS.inputNeurons.max);

stores.setOutputSize(1);
check('output neurons floored',
  spec().outputSize === stores.LIMITS.outputNeurons.min);

// The largest reachable network must still lay out and run.
stores.resetAll();
stores.setInputSize(stores.LIMITS.inputNeurons.max);
stores.setHiddenLayerCount(stores.LIMITS.hiddenLayers.max);
stores.setNeuronsPerLayer(stores.LIMITS.neuronsPerLayer.max);
stores.setOutputSize(stores.LIMITS.outputNeurons.max);

const biggest = net();
check('largest reachable network runs',
  biggest.layers[biggest.layers.length - 1].neurons
    .every((n) => Number.isFinite(n.output)));
check('largest network still sums to 1',
  near(pred().classes.reduce((a, c) => a + c.share, 0), 1, 1e-9));

const bigLayout = computeLayout(biggest, 1100);
check('largest network lays out',
  bigLayout.positions.length === biggest.layers.length &&
  bigLayout.positions.every((p, i) => p.length === biggest.layers[i].size));

// Every neuron in the largest network must still reconcile.
check('every neuron in the largest network reconciles', (() => {
  for (let l = 1; l < biggest.layers.length; l++) {
    for (let i = 0; i < biggest.layers[l].size; i++) {
      const d = explainNeuron(biggest, l, i);
      if (d === undefined) return false;
      const recomputed = d.terms.reduce((a, t) => a + t.product, 0) + d.bias;
      if (!near(recomputed, d.preActivation, 1e-9)) return false;
    }
  }
  return true;
})());

// The smallest reachable network too.
stores.resetAll();
stores.setInputSize(stores.LIMITS.inputNeurons.min);
stores.setHiddenLayerCount(stores.LIMITS.hiddenLayers.min);
stores.setNeuronsPerLayer(stores.LIMITS.neuronsPerLayer.min);
stores.setOutputSize(stores.LIMITS.outputNeurons.min);
check('smallest reachable network runs',
  net().layers[net().layers.length - 1].neurons
    .every((n) => Number.isFinite(n.output)));

// ---------------------------------------------------------------- weight preservation
console.log('\nWeight preservation across resizes');
stores.resetAll();

stores.updateWeight(1, 0, 0, 1.234);
check('weight edit applied',
  near(net().layers[1].neurons[0].weights[0], 1.234));

// Growing elsewhere must not disturb an existing weight.
stores.setNeuronsPerLayer(7, 1);
check('edited weight survives resizing another layer',
  near(net().layers[1].neurons[0].weights[0], 1.234),
  `got ${net().layers[1].neurons[0].weights[0]}`);

stores.setHiddenLayerCount(3);
check('edited weight survives adding a layer',
  near(net().layers[1].neurons[0].weights[0], 1.234));

// Switching activation must not touch weights either.
stores.setHiddenActivation(TANH);
check('edited weight survives an activation change',
  near(net().layers[1].neurons[0].weights[0], 1.234));

// New weights should replace it.
stores.reseedNetwork(12345);
check('reseeding replaces weights',
  !near(net().layers[1].neurons[0].weights[0], 1.234));
check('reseeding is deterministic', (() => {
  const a = net().layers[1].neurons[0].weights[0];
  stores.reseedNetwork(12345);
  return near(net().layers[1].neurons[0].weights[0], a);
})());

// ---------------------------------------------------------------- reset
console.log('\nReset');
stores.setHiddenActivation(SIGMOID);
stores.setHiddenLayerCount(4);
stores.setInputSize(6);
stores.resetAll();

check('reset restores the default shape',
  JSON.stringify(net().layers.map((l) => l.size)) === '[4,5,5,2]',
  JSON.stringify(net().layers.map((l) => l.size)));
check('reset restores ReLU', net().layers[1].activation === RELU);
check('reset restores the default inputs',
  JSON.stringify(spec().inputs) === JSON.stringify([0.8, 0.25, 0.6, 0.1]));

// Labels must stay neutral: this network is untrained.
console.log('\nHonest labelling');
check('default labels are not real categories',
  !spec().labels.join(' ').match(/cat|dog/i),
  JSON.stringify(spec().labels));
check('labels grow neutrally', (() => {
  stores.setOutputSize(4);
  const ok = spec().labels.every((l) => !/cat|dog/i.test(l));
  stores.resetAll();
  return ok;
})());

console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
