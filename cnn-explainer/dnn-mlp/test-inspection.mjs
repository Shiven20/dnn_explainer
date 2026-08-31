/**
 * Tests for the Phase 3 inspection logic.
 *
 * The property that matters: what the panels display must reconcile exactly
 * with what the network computed. A breakdown whose terms do not add up to the
 * neuron's own value would teach the wrong thing convincingly.
 *
 * Usage: npm run test:inspection
 */

import {
  createNetwork, setWeight, setBias, getWeight, getWeightMagnitude
} from '../src/dnn/engine/network.js';

import {
  forwardPass, explainNeuron, getPrediction, traceUpstream, traceConnectionUpstream
} from '../src/dnn/engine/forwardPass.js';

import { getActivation, RELU, SIGMOID, TANH, SOFTMAX } from '../src/dnn/engine/activations.js';
import { connectionColor, formatValue, formatSigned } from '../src/dnn/engine/layout.js';

let failures = 0;
let checks = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};
const near = (a, b, eps = 1e-12) => Math.abs(a - b) < eps;

// A network with known weights, so expected values can be derived by hand.
const base = createNetwork({
  layerSizes: [3, 3, 2],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 4
});

base.layers[1].neurons[0].weights = [0.42, -0.18, 0.73];
base.layers[1].neurons[0].bias = 0.1;
base.layers[1].neurons[1].weights = [-1, -1, -1];
base.layers[1].neurons[1].bias = 0;
base.layers[1].neurons[2].weights = [0, 0, 0];
base.layers[1].neurons[2].bias = 0.5;

const INPUTS = [0.8, 0.25, 0.6];
const net = forwardPass(base, INPUTS);

// ---------------------------------------------------------------- breakdown
console.log('Neuron breakdown reconciles with the network');

const d = explainNeuron(net, 1, 0);

// The worked example from the brief: 0.8*0.42 + 0.25*-0.18 + 0.6*0.73 + 0.1
check('term count matches incoming connections', d.terms.length === 3);
check('term inputs are the previous layer outputs',
  d.terms.every((t, i) => near(t.input, INPUTS[i])));
check('each product equals weight x input',
  d.terms.every((t) => near(t.product, t.weight * t.input)));
check('sum of products is correct', near(d.weightedSum, 0.729),
  `got ${d.weightedSum}`);
check('bias is reported', near(d.bias, 0.1));
check('z equals sum plus bias', near(d.preActivation, 0.829), `got ${d.preActivation}`);

// The displayed pieces must add up to the neuron's own stored value; otherwise
// the panel and the diagram would disagree.
check('displayed terms reconcile with the neuron preActivation',
  near(d.terms.reduce((a, t) => a + t.product, 0) + d.bias,
       net.layers[1].neurons[0].preActivation));
check('reported output matches the neuron output',
  near(d.output, net.layers[1].neurons[0].output));

// Running totals, as rendered in the panel.
const running = d.terms.reduce((acc, t) => {
  acc.push((acc.length > 0 ? acc[acc.length - 1] : 0) + t.product);
  return acc;
}, []);
check('running total ends at the sum of products',
  near(running[running.length - 1], d.weightedSum));
check('running total is cumulative',
  running.every((v, i) => i === 0 || near(v, running[i - 1] + d.terms[i].product)));

// ---------------------------------------------------------------- activation
console.log('\nActivation reporting');

const dead = explainNeuron(net, 1, 1);
check('negative z is reported', dead.preActivation < 0, `z=${dead.preActivation}`);
check('a dead ReLU neuron outputs zero', dead.output === 0);
check('output still equals f(z)',
  near(dead.output, getActivation(RELU).fn(dead.preActivation)));

const biasOnly = explainNeuron(net, 1, 2);
check('all-zero weights leave only the bias', near(biasOnly.preActivation, 0.5));
check('every product is zero', biasOnly.terms.every((t) => t.product === 0));

// Each activation must be reported consistently with its own function.
for (const id of [RELU, SIGMOID, TANH]) {
  const n = forwardPass(
    createNetwork({ layerSizes: [3, 4, 2], hiddenActivation: id, seed: 6 }),
    INPUTS
  );
  const ex = explainNeuron(n, 1, 0);
  check(`${id}: reported output equals f(z)`,
    near(ex.output, getActivation(id).fn(ex.preActivation)));
}

// ---------------------------------------------------------------- input neuron
console.log('\nInput neurons');
const inp = explainNeuron(net, 0, 1);
check('input neuron is flagged', inp.isInput === true);
check('input neuron has no terms', inp.terms.length === 0);
check('input neuron reports its value', near(inp.output, INPUTS[1]));
check('input neuron has no bias to show', inp.bias === 0);

// ---------------------------------------------------------------- output layer
console.log('\nOutput neurons');
const out0 = explainNeuron(net, 2, 0);
check('output neuron exposes its logit', Number.isFinite(out0.preActivation));
check('output neuron activation is softmax', out0.activation === SOFTMAX);
check('softmax output is a share between 0 and 1',
  out0.output >= 0 && out0.output <= 1);
check('output shares sum to 1',
  near(net.layers[2].neurons.reduce((a, n) => a + n.output, 0), 1, 1e-9));

// ---------------------------------------------------------------- connections
console.log('\nConnection inspection');

const srcLayer = net.layers[0];
const tgtNeuron = net.layers[1].neurons[0];
const sourceIndex = 2;
const weight = tgtNeuron.weights[sourceIndex];
const contribution = weight * srcLayer.neurons[sourceIndex].output;

check('contribution equals weight x source output', near(contribution, 0.73 * 0.6));

// Share of total arriving magnitude, as the panel reports it.
const totalMagnitude = tgtNeuron.weights.reduce(
  (acc, w, i) => acc + Math.abs(w * srcLayer.neurons[i].output), 0);
const share = Math.abs(contribution) / totalMagnitude;
check('share is a fraction between 0 and 1', share > 0 && share <= 1, `share=${share}`);
check('all shares for a neuron sum to 1', (() => {
  const sum = tgtNeuron.weights.reduce(
    (acc, w, i) => acc + Math.abs(w * srcLayer.neurons[i].output) / totalMagnitude, 0);
  return near(sum, 1, 1e-9);
})());

// A zero-valued source carries nothing regardless of weight -- the case the
// panel calls out explicitly.
const zeroInput = forwardPass(base, [0, 0.25, 0.6]);
check('a zero source contributes nothing even with a large weight',
  zeroInput.layers[1].neurons[0].weights[0] !== 0 &&
  zeroInput.layers[0].neurons[0].output === 0 &&
  near(zeroInput.layers[1].neurons[0].weights[0] * zeroInput.layers[0].neurons[0].output, 0));

// ---------------------------------------------------------------- editing
console.log('\nEditing propagates');

const beforePred = getPrediction(net, ['A', 'B']);
const beforeShares = beforePred.classes.map((c) => c.share);

// Editing a weight must change the downstream output.
const wEdited = forwardPass(setWeight(base, 1, 0, 0, 2.5), INPUTS);
check('a weight edit changes the target neuron',
  !near(wEdited.layers[1].neurons[0].preActivation,
        net.layers[1].neurons[0].preActivation));
check('a weight edit reaches the output layer',
  !near(wEdited.layers[2].neurons[0].output, net.layers[2].neurons[0].output));

const afterShares = getPrediction(wEdited, ['A', 'B']).classes.map((c) => c.share);
check('a weight edit changes the prediction shares',
  !near(beforeShares[0], afterShares[0]));

// Editing a bias must do the same.
const bEdited = forwardPass(setBias(base, 1, 0, 3), INPUTS);
check('a bias edit changes z',
  near(bEdited.layers[1].neurons[0].preActivation, 0.729 + 3),
  `got ${bEdited.layers[1].neurons[0].preActivation}`);
check('a bias edit reaches the output',
  !near(bEdited.layers[2].neurons[0].output, net.layers[2].neurons[0].output));

// A bias large enough to overcome a negative sum revives a dead ReLU unit.
const revived = forwardPass(setBias(base, 1, 1, 5), INPUTS);
check('raising the bias revives a dead ReLU neuron',
  revived.layers[1].neurons[1].output > 0,
  `output=${revived.layers[1].neurons[1].output}`);

// Zeroing a weight removes that input's influence entirely.
const zeroed = forwardPass(setWeight(base, 1, 2, 0, 0), INPUTS);
const zeroedDetail = explainNeuron(zeroed, 1, 0);
check('zeroing a weight zeroes its product',
  zeroedDetail.terms[2].product === 0);
check('zeroing a weight lowers the sum accordingly',
  near(zeroedDetail.weightedSum, 0.729 - 0.73 * 0.6));

// Flipping a weight's sign flips its contribution.
const flipped = forwardPass(setWeight(base, 1, 2, 0, -0.73), INPUTS);
check('flipping a weight flips its contribution',
  near(explainNeuron(flipped, 1, 0).terms[2].product, -(0.73 * 0.6)));

// Edits must not mutate the source network.
check('editing leaves the original network untouched',
  getWeight(base, 1, 0, 0) === 0.42 && base.layers[1].neurons[0].bias === 0.1);

// ---------------------------------------------------------------- input edits
console.log('\nInput edits propagate');
const changedInput = forwardPass(base, [0.1, 0.25, 0.6]);
check('changing an input changes the term',
  near(explainNeuron(changedInput, 1, 0).terms[0].product, 0.1 * 0.42));
check('changing an input changes z',
  !near(changedInput.layers[1].neurons[0].preActivation,
        net.layers[1].neurons[0].preActivation));
check('changing an input changes the prediction',
  !near(getPrediction(changedInput, ['A', 'B']).classes[0].share, beforeShares[0]));

// Negative inputs are permitted by the UI range and must behave.
const negative = forwardPass(base, [-1, -1, -1]);
check('negative inputs are handled',
  negative.layers[2].neurons.every((n) => Number.isFinite(n.output)));
check('negative inputs still produce a distribution',
  near(negative.layers[2].neurons.reduce((a, n) => a + n.output, 0), 1, 1e-9));

// ---------------------------------------------------------------- formatting
console.log('\nDisplay helpers');
check('signed format marks positives', formatSigned(0.5).startsWith('+'));
check('signed format marks negatives', formatSigned(-0.5).startsWith('-'));
check('zero formats cleanly', formatValue(0) === '0.00');
check('tiny values are not shown as zero', formatValue(0.001) === '<0.01');
check('positive and negative weights get different colours',
  connectionColor(1, 1) !== connectionColor(-1, 1));
check('weight magnitude is positive', getWeightMagnitude(base) > 0);

// Every neuron in the network must produce a valid breakdown, including edges.
console.log('\nAll neurons explain cleanly');
let allOk = true;
for (let l = 0; l < net.layers.length; l++) {
  for (let i = 0; i < net.layers[l].size; i++) {
    const ex = explainNeuron(net, l, i);
    if (ex === undefined) { allOk = false; break; }
    if (l > 0) {
      const recomputed = ex.terms.reduce((a, t) => a + t.product, 0) + ex.bias;
      if (!near(recomputed, ex.preActivation)) { allOk = false; break; }
    }
  }
}
check('every neuron reconciles', allOk);

// The largest allowed architecture must also explain cleanly.
const big = forwardPass(
  createNetwork({ layerSizes: [8, 12, 12, 6], seed: 3 }),
  [1, 0.5, -0.5, 0, 0.25, -1, 0.75, 0.1]
);
const bigDetail = explainNeuron(big, 2, 11);
check('largest architecture explains a deep neuron',
  bigDetail.terms.length === 12 &&
  near(bigDetail.terms.reduce((a, t) => a + t.product, 0) + bigDetail.bias,
       bigDetail.preActivation));

// ---------------------------------------------------------------- upstream tracing
console.log('\nUpstream connection tracing');
const outputTrace = traceUpstream(net, 2, 0);
check('output trace reaches both preceding connection layers',
  outputTrace.layers.includes(1) && outputTrace.layers.includes(2));
check('trace keys contain no duplicates',
  outputTrace.active.size === new Set(outputTrace.active).size &&
  outputTrace.inactive.size === new Set(outputTrace.inactive).size);
check('zero-weight and dead-source paths are inactive',
  outputTrace.inactive.has('1:0:2') && outputTrace.inactive.has('2:1:0'));

const firstHop = traceConnectionUpstream(net, 1, 2, 0);
check('a first-layer connection includes itself', firstHop.self === '1:2:0' &&
  (firstHop.active.has(firstHop.self) || firstHop.inactive.has(firstHop.self)));
check('a first-layer connection has no earlier connection layer',
  firstHop.layers.length === 1 && firstHop.layers[0] === 1);

const selectedOutputConnection = traceConnectionUpstream(net, 2, 0, 0);
check('selected deep connection traces through to inputs',
  selectedOutputConnection.layers.includes(1) && selectedOutputConnection.layers.includes(2));

console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
