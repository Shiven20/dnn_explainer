/**
 * Tests for the DNN Explainer engine.
 *
 * These import the same modules the browser ships, so a pass here means the
 * visualization is reading correct numbers. Run with: npm run test:engine
 */

import {
  createNetwork, resizeNetwork, countParameters, getConnections,
  getWeightMagnitude, setWeight, setBias, getWeight
} from '../src/dnn/engine/network.js';

import {
  forwardPass, explainNeuron, getPrediction, getActivationRanges
} from '../src/dnn/engine/forwardPass.js';

import {
  activations, getActivation, softmax, RELU, SIGMOID, TANH, LINEAR, SOFTMAX
} from '../src/dnn/engine/activations.js';

import { computeLayout, connectionWidth, formatValue } from '../src/dnn/engine/layout.js';

let failures = 0;
let checks = 0;

const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};

const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

// ---------------------------------------------------------------- activations
console.log('Activation functions');
check('ReLU clips negatives', activations[RELU].fn(-3) === 0);
check('ReLU passes positives', activations[RELU].fn(2.5) === 2.5);
check('Sigmoid(0) is 0.5', near(activations[SIGMOID].fn(0), 0.5));
check('Sigmoid stays within (0,1)',
  activations[SIGMOID].fn(-50) >= 0 && activations[SIGMOID].fn(50) <= 1);
check('Tanh(0) is 0', near(activations[TANH].fn(0), 0));
check('Tanh saturates to -1..1',
  activations[TANH].fn(-50) >= -1 && activations[TANH].fn(50) <= 1);
check('Linear is identity', activations[LINEAR].fn(-4.2) === -4.2);
check('unknown id falls back to ReLU', getActivation('nope').id === RELU);

const sm = softmax([2, 1, 0.1]);
check('softmax sums to 1', near(sm.reduce((a, b) => a + b, 0), 1));
check('softmax preserves ordering', sm[0] > sm[1] && sm[1] > sm[2]);
// Large logits would overflow a naive exp(); the max-subtraction must handle it.
const big = softmax([1000, 999]);
check('softmax handles large logits', big.every(Number.isFinite) && near(big[0] + big[1], 1));
check('softmax of a single value is 1', near(softmax([5])[0], 1));

// ---------------------------------------------------------------- construction
console.log('\nNetwork construction');
const net = createNetwork({
  layerSizes: [4, 5, 5, 2],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 7
});

check('layer count', net.layers.length === 4);
check('layer sizes', JSON.stringify(net.layers.map((l) => l.size)) === '[4,5,5,2]');
check('input layer has no weights', net.layers[0].neurons.every((n) => n.weights.length === 0));
check('hidden weights match fan-in',
  net.layers[1].neurons.every((n) => n.weights.length === 4));
check('output weights match fan-in',
  net.layers[3].neurons.every((n) => n.weights.length === 5));
check('layer labels', net.layers.map((l) => l.label).join('|') ===
  'Input|Hidden Layer 1|Hidden Layer 2|Output');
check('output layer uses softmax', net.layers[3].activation === SOFTMAX);
// 4*5+5 + 5*5+5 + 5*2+2 = 25 + 30 + 12 = 67
check('parameter count is 67', countParameters(net) === 67, `got ${countParameters(net)}`);
check('connection count is 55', getConnections(net).length === 55,
  `got ${getConnections(net).length}`);

// A given seed must always produce the same network, or the UI would jump.
const netA = createNetwork({ layerSizes: [3, 4, 2], seed: 42 });
const netB = createNetwork({ layerSizes: [3, 4, 2], seed: 42 });
check('same seed gives identical weights',
  JSON.stringify(netA.layers[1].neurons.map((n) => n.weights)) ===
  JSON.stringify(netB.layers[1].neurons.map((n) => n.weights)));
const netC = createNetwork({ layerSizes: [3, 4, 2], seed: 43 });
check('different seed gives different weights',
  JSON.stringify(netA.layers[1].neurons[0].weights) !==
  JSON.stringify(netC.layers[1].neurons[0].weights));

// Invalid shapes must fail loudly rather than render something nonsensical.
let threw = false;
try { createNetwork({ layerSizes: [4] }); } catch { threw = true; }
check('rejects a single-layer network', threw);
threw = false;
try { createNetwork({ layerSizes: [4, 0, 2] }); } catch { threw = true; }
check('rejects an empty layer', threw);

// ---------------------------------------------------------------- forward pass
console.log('\nForward pass');

// Hand-built network with known weights, so the expected result is checkable
// by hand rather than by rerunning the same code.
const manual = createNetwork({
  layerSizes: [3, 2, 2],
  hiddenActivation: RELU,
  outputActivation: LINEAR,
  seed: 1
});
manual.layers[1].neurons[0].weights = [0.42, -0.18, 0.73];
manual.layers[1].neurons[0].bias = 0.1;
manual.layers[1].neurons[1].weights = [-1, -1, -1];
manual.layers[1].neurons[1].bias = 0;
manual.layers[2].neurons[0].weights = [2, 3];
manual.layers[2].neurons[0].bias = 0.5;
manual.layers[2].neurons[1].weights = [-1, 1];
manual.layers[2].neurons[1].bias = 0;

const ran = forwardPass(manual, [0.8, 0.25, 0.6]);

// 0.8*0.42 + 0.25*(-0.18) + 0.6*0.73 + 0.1 = 0.336 - 0.045 + 0.438 + 0.1 = 0.829
check('weighted sum is correct',
  near(ran.layers[1].neurons[0].preActivation, 0.829, 1e-12),
  `got ${ran.layers[1].neurons[0].preActivation}`);
check('ReLU passes the positive sum', near(ran.layers[1].neurons[0].output, 0.829, 1e-12));

// -0.8 - 0.25 - 0.6 = -1.65, so ReLU must clamp to 0.
check('negative sum clamps to zero', ran.layers[1].neurons[1].output === 0,
  `got ${ran.layers[1].neurons[1].output}`);

// Layer 2 input is [0.829, 0]: 0.829*2 + 0*3 + 0.5 = 2.158
check('second layer consumes first layer outputs',
  near(ran.layers[2].neurons[0].preActivation, 2.158, 1e-12),
  `got ${ran.layers[2].neurons[0].preActivation}`);

check('forward pass does not mutate the source network',
  manual.layers[1].neurons[0].output === 0);

// Input length must match the input layer exactly.
threw = false;
try { forwardPass(manual, [1, 2]); } catch { threw = true; }
check('rejects a wrong-length input', threw);

// Every activation must behave correctly end to end.
for (const id of [RELU, SIGMOID, TANH, LINEAR]) {
  const n = createNetwork({ layerSizes: [3, 4, 2], hiddenActivation: id, seed: 5 });
  const r = forwardPass(n, [0.5, -0.5, 1]);
  const { fn, displayRange, unbounded } = getActivation(id);
  const ok = r.layers[1].neurons.every((neuron) =>
    near(neuron.output, fn(neuron.preActivation), 1e-12));
  check(`${id}: output equals f(preActivation)`, ok);
  if (!unbounded) {
    const inRange = r.layers[1].neurons.every(
      (neuron) => neuron.output >= displayRange[0] - 1e-9 &&
                  neuron.output <= displayRange[1] + 1e-9);
    check(`${id}: outputs stay within ${displayRange.join('..')}`, inRange);
  }
}

// Softmax on the output layer must yield a real distribution.
const clsNet = forwardPass(
  createNetwork({ layerSizes: [4, 6, 3], outputActivation: SOFTMAX, seed: 11 }),
  [0.3, 0.7, 0.1, 0.9]
);
const outs = clsNet.layers[2].neurons.map((n) => n.output);
check('softmax output sums to 1', near(outs.reduce((a, b) => a + b, 0), 1));
check('softmax output matches softmax(logits)',
  softmax(clsNet.layers[2].neurons.map((n) => n.preActivation))
    .every((p, i) => near(p, outs[i], 1e-12)));

// ---------------------------------------------------------------- explanation
console.log('\nNeuron explanation');
const exp = explainNeuron(ran, 1, 0);
check('one term per incoming weight', exp.terms.length === 3);
check('terms use previous layer outputs',
  exp.terms.every((t, i) => near(t.input, ran.layers[0].neurons[i].output)));
check('each product equals weight times input',
  exp.terms.every((t) => near(t.product, t.weight * t.input, 1e-12)));
check('terms plus bias equals preActivation',
  near(exp.terms.reduce((a, t) => a + t.product, 0) + exp.bias, exp.preActivation, 1e-12));
check('explained preActivation matches the neuron',
  near(exp.preActivation, ran.layers[1].neurons[0].preActivation, 1e-12));
check('input neurons are flagged', explainNeuron(ran, 0, 0).isInput === true);
check('out-of-range neuron returns undefined', explainNeuron(ran, 9, 0) === undefined);

// ---------------------------------------------------------------- prediction
console.log('\nPrediction');
const pred = getPrediction(clsNet, ['Cat', 'Dog', 'Bird']);
check('picks the highest output',
  outs[pred.index] === Math.max(...outs));
check('uses the supplied label', pred.label === ['Cat', 'Dog', 'Bird'][pred.index]);
check('flags probabilities for softmax', pred.isProbability === true);
check('class shares sum to 1',
  near(pred.classes.reduce((a, c) => a + c.share, 0), 1));
check('falls back to generic labels',
  getPrediction(clsNet, []).label.startsWith('Output'));

// Non-softmax outputs are normalized for display only.
const linPred = getPrediction(ran, ['A', 'B']);
check('non-softmax output is not called a probability', linPred.isProbability === false);

// ---------------------------------------------------------------- ranges
console.log('\nActivation ranges');
const sigNet = forwardPass(
  createNetwork({ layerSizes: [3, 4, 2], hiddenActivation: SIGMOID, seed: 3 }),
  [1, 2, 3]
);
check('bounded activation uses its known range', getActivationRanges(sigNet)[1] === 1);
check('softmax layer range is 1', getActivationRanges(clsNet)[2] === 1);
check('all ranges are positive', getActivationRanges(ran).every((r) => r > 0));

// ---------------------------------------------------------------- edits
console.log('\nWeight and bias edits');
const edited = setWeight(net, 1, 2, 0, 0.99);
check('weight is updated', getWeight(edited, 1, 2, 0) === 0.99);
check('original network is untouched', getWeight(net, 1, 2, 0) !== 0.99);
check('unrelated weights are preserved',
  getWeight(edited, 1, 1, 0) === getWeight(net, 1, 1, 0));
check('editing returns a new reference', edited !== net);

const biased = setBias(net, 2, 1, -0.4);
check('bias is updated', biased.layers[2].neurons[1].bias === -0.4);
check('original bias is untouched', net.layers[2].neurons[1].bias !== -0.4);

// A weight edit must actually change the downstream output.
const before = forwardPass(manual, [0.8, 0.25, 0.6]).layers[2].neurons[0].output;
const after = forwardPass(
  setWeight(manual, 1, 0, 0, 5), [0.8, 0.25, 0.6]
).layers[2].neurons[0].output;
check('a weight edit changes the prediction', !near(before, after));

check('weight magnitude is positive', getWeightMagnitude(net) > 0);
check('weight magnitude reflects the largest weight',
  near(getWeightMagnitude(setWeight(net, 1, 0, 0, 12)), 12));

// ---------------------------------------------------------------- resize
console.log('\nArchitecture changes');
const grown = resizeNetwork(net, { layerSizes: [4, 5, 5, 5, 2] });
check('layer added', grown.layers.length === 5);
check('surviving weights are preserved',
  getWeight(grown, 1, 0, 0) === getWeight(net, 1, 0, 0));

const shrunk = resizeNetwork(net, { layerSizes: [4, 3, 5, 2] });
check('layer shrunk', shrunk.layers[1].size === 3);
check('remaining neurons keep their weights',
  getWeight(shrunk, 1, 0, 0) === getWeight(net, 1, 0, 0));

const widened = resizeNetwork(net, { layerSizes: [6, 5, 5, 2] });
check('input widened', widened.layers[0].size === 6);
check('hidden neurons gain weights for new inputs',
  widened.layers[1].neurons.every((n) => n.weights.length === 6));
check('existing input weights survive widening',
  getWeight(widened, 1, 0, 0) === getWeight(net, 1, 0, 0));

const swapped = resizeNetwork(net, { layerSizes: [4, 5, 5, 2], hiddenActivation: TANH });
check('activation swap applies to hidden layers',
  swapped.layers[1].activation === TANH && swapped.layers[2].activation === TANH);
check('activation swap leaves the output layer alone',
  swapped.layers[3].activation === SOFTMAX);

// Resized networks must still run.
const resizedRun = forwardPass(grown, [0.1, 0.2, 0.3, 0.4]);
check('a resized network still runs',
  resizedRun.layers[4].neurons.every((n) => Number.isFinite(n.output)));

// ---------------------------------------------------------------- layout
console.log('\nLayout');
const lay = computeLayout(net, 900);
check('one position set per layer', lay.positions.length === 4);
check('positions per layer match neuron counts',
  lay.positions.every((p, i) => p.length === net.layers[i].size));
check('layers advance left to right',
  lay.layerX.every((x, i) => i === 0 || x > lay.layerX[i - 1]));
check('neurons within a layer share an x',
  lay.positions[1].every((p) => p.x === lay.positions[1][0].x));
check('neurons within a layer are vertically separated',
  lay.positions[1][1].y > lay.positions[1][0].y);
check('layers are vertically centred on a common axis', (() => {
  const mid = (p) => (p[0].y + p[p.length - 1].y) / 2;
  return near(mid(lay.positions[1]), mid(lay.positions[3]), 1e-6);
})());
check('canvas is large enough for the tallest layer',
  lay.height > lay.positions[1][lay.positions[1].length - 1].y);

// A narrow viewport must not collapse the layer spacing.
const narrow = computeLayout(net, 320);
check('narrow viewport keeps a minimum layer gap',
  narrow.layerX[1] - narrow.layerX[0] >= 150);
check('narrow layout overflows for scrolling', narrow.width > 320);

// The largest architecture the UI allows must still lay out sanely.
const biggest = createNetwork({ layerSizes: [8, 12, 12, 12, 12, 12, 6], seed: 2 });
const bigLay = computeLayout(biggest, 1100);
check('largest allowed architecture lays out',
  bigLay.positions.length === 7 &&
  bigLay.positions.every((p, i) => p.length === biggest.layers[i].size));
check('largest architecture runs',
  forwardPass(biggest, [1, 0, 1, 0, 1, 0, 1, 0])
    .layers[6].neurons.every((n) => Number.isFinite(n.output)));

check('connection width grows with magnitude',
  connectionWidth(1, 1) > connectionWidth(0.1, 1));
check('formatValue handles zero', formatValue(0) === '0.00');
check('formatValue handles undefined', formatValue(undefined) === '--');

// ---------------------------------------------------------------- summary
console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
