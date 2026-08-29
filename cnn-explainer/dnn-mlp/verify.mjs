/**
 * Verifies the ACTUAL browser engine (src/utils/dnn.js) against the exported
 * weights, by importing that module directly rather than reimplementing it.
 *
 * This is the check that matters: the visualization reads activations straight
 * off the neuron graph, so if constructDNNFromJSON/forwardPass disagreed with
 * the trainer, every number on screen would be wrong.
 *
 * Usage: node dnn-mlp/verify.mjs
 */

import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

import {
  constructDNNFromJSON, forwardPass, getPrediction, getLayerRanges,
  getWeightRange, weightedSum, flattenGrid, unflattenGrid, emptyGrid, softmax
} from '../src/utils/dnn.js';

const here = path.dirname(fileURLToPath(import.meta.url));
const { CLASSES, buildDataset } = await import('./glyphs.js').then((m) => m.default || m);

const model = JSON.parse(
  fs.readFileSync(path.join(here, '..', 'public', 'assets', 'data', 'dnn_model.json'), 'utf8')
);

let failures = 0;
const check = (label, ok, detail = '') => {
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : ` ${detail}`}`);
  if (!ok) failures++;
};

// ---------------------------------------------------------------- graph
console.log('Graph construction (constructDNNFromJSON)');
const network = constructDNNFromJSON(model);

check('4 layers built', network.length === 4, `got ${network.length}`);
check('input layer has 25 neurons', network[0].length === 25);
check('input layer has no incoming links', network[0].every((n) => n.inputLinks.length === 0));

// "Fully connected" means every neuron links to every neuron in the layer before.
for (let l = 1; l < network.length; l++) {
  const prevSize = network[l - 1].length;
  check(
    `layer ${l} is fully connected (${prevSize} links per neuron)`,
    network[l].every((n) => n.inputLinks.length === prevSize)
  );
}

// Link weights must line up with the JSON, including orientation. A transposed
// weight matrix would still run and still produce plausible-looking output.
const w0 = model.layers[1].weights;
check(
  'link weights match the JSON (not transposed)',
  network[1][3].inputLinks[7].weight === w0[3][7],
  `graph=${network[1][3].inputLinks[7].weight} json=${w0[3][7]}`
);
check('biases match the JSON', network[1][2].bias === model.layers[1].biases[2]);

// ---------------------------------------------------------------- forward pass
console.log('\nForward pass (forwardPass / getPrediction)');
for (const cls of Object.keys(model.prototypes)) {
  const grid = model.prototypes[cls];
  forwardPass(network, flattenGrid(grid));
  const predicted = getPrediction(network);
  const confidence = network[3][predicted].output;
  check(
    `prototype ${cls} -> ${CLASSES[predicted]} (${(confidence * 100).toFixed(1)}%)`,
    predicted === CLASSES.indexOf(cls)
  );
}

// The engine's softmax layer must be a real distribution.
forwardPass(network, flattenGrid(model.prototypes.X));
const probs = network[3].map((n) => n.output);
check('output sums to 1', Math.abs(probs.reduce((a, b) => a + b, 0) - 1) < 1e-9);

// ReLU semantics: output is max(0, preActivation), exactly.
let reluOk = true;
for (const l of [1, 2]) {
  for (const n of network[l]) {
    if (n.output !== Math.max(0, n.preActivation)) reluOk = false;
  }
}
check('ReLU output equals max(0, preActivation)', reluOk);

// The engine's softmax must match softmax() applied to the stored logits.
const logits = network[3].map((n) => n.preActivation);
const expected = softmax(logits);
check(
  'output layer equals softmax(logits)',
  probs.every((p, i) => Math.abs(p - expected[i]) < 1e-12)
);

// ---------------------------------------------------------------- detail view math
console.log('\nDetail view arithmetic (weightedSum)');
const neuron = network[2][1];
const detail = weightedSum(neuron);
check(
  'weightedSum total equals the neuron preActivation',
  Math.abs(detail.total - neuron.preActivation) < 1e-12,
  `${detail.total} vs ${neuron.preActivation}`
);
check(
  'terms sum plus bias equals total',
  Math.abs(detail.terms.reduce((a, t) => a + t.product, 0) + detail.bias - detail.total) < 1e-12
);
check('one term per incoming link', detail.terms.length === neuron.inputLinks.length);
check(
  'each term product equals weight times input',
  detail.terms.every((t) => Math.abs(t.product - t.weight * t.input) < 1e-12)
);

// ---------------------------------------------------------------- ranges & helpers
console.log('\nColour ranges and grid helpers');
const ranges = getLayerRanges(network);
check('one range per layer', ranges.length === network.length);
check('softmax layer range is 1', ranges[3] === 1);
check('all ranges are positive', ranges.every((r) => r > 0));
check('weight range is positive', getWeightRange(network) > 0);

const grid = model.prototypes.L;
check(
  'flatten/unflatten round-trips',
  JSON.stringify(unflattenGrid(flattenGrid(grid), 5)) === JSON.stringify(grid)
);
check('emptyGrid is 5x5 of zeros',
  emptyGrid(5).length === 5 && emptyGrid(5).every((r) => r.length === 5 && r.every((v) => v === 0)));

// A blank canvas is a state the user can reach with the Clear button.
forwardPass(network, flattenGrid(emptyGrid(5)));
check(
  'blank input yields a valid distribution',
  Math.abs(network[3].reduce((a, n) => a + n.output, 0) - 1) < 1e-9
);

// Mismatched input length should fail loudly rather than silently mis-render.
let threw = false;
try {
  forwardPass(network, [1, 0, 1]);
} catch {
  threw = true;
}
check('rejects a wrong-length input vector', threw);

// ---------------------------------------------------------------- accuracy
console.log('\nAccuracy over the full dataset');
const { xs, ys } = buildDataset(42, 400, 0.06);
let correct = 0;
for (let n = 0; n < xs.length; n++) {
  forwardPass(network, xs[n]);
  if (getPrediction(network) === ys[n]) correct++;
}
const acc = correct / xs.length;
console.log(`  ${correct}/${xs.length} = ${(acc * 100).toFixed(2)}%`);
check('browser engine scores above 95%', acc > 0.95, `got ${(acc * 100).toFixed(2)}%`);

console.log(failures === 0 ? '\nAll checks passed.' : `\n${failures} check(s) failed.`);
process.exit(failures === 0 ? 0 : 1);
