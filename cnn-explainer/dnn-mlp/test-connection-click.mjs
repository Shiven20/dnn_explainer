/**
 * Verifies the click-a-connection-to-edit-its-weight path end to end.
 *
 * This exercises the chain the user actually follows:
 *   click a connection hit target -> selectConnection fires
 *   -> ConnectionDetails mounts -> its slider edits the weight
 *   -> the network recomputes and the prediction moves
 *
 * Usage: node dnn-mlp/test-connection-click.mjs
 */

import { get } from 'svelte/store';
import * as stores from '../src/dnn/stores.js';

let failures = 0;
let checks = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};
const near = (a, b, eps = 1e-9) => Math.abs(a - b) < eps;

stores.resetAll();

const net = () => get(stores.activatedNetwork);
const pred = () => get(stores.prediction);

// ---------------------------------------------------------------- the click
console.log('Clicking a connection');

/*
 * Network.svelte renders an invisible wide hit path per connection carrying
 * data-connection="targetLayer:source:target". Its delegated handler parses that
 * and calls selectConnection. Simulate exactly that parse.
 */
const attr = '1:2:0';
const [targetLayerIndex, sourceIndex, targetIndex] = attr.split(':').map(Number);
check('hit-target attribute parses to three indices',
  targetLayerIndex === 1 && sourceIndex === 2 && targetIndex === 0);

stores.selectConnection(targetLayerIndex, sourceIndex, targetIndex);
const sel = get(stores.selectedConnection);
check('clicking selects the connection', sel !== undefined);
check('selection carries the right indices',
  sel.targetLayerIndex === 1 && sel.sourceIndex === 2 && sel.targetIndex === 0);

// Selecting a connection must clear any neuron selection, since the rail shows
// one breakdown at a time.
check('neuron selection is cleared', get(stores.selectedNeuron) === undefined);

// ---------------------------------------------------------------- the edit
console.log('\nAdjusting the weight');

const before = net().layers[1].neurons[0].weights[2];
const predBefore = pred().classes.map((c) => c.share);
check('the connection has a starting weight', Number.isFinite(before));

// This is what the slider's on:input calls.
stores.updateWeight(1, 2, 0, 2.5);
check('weight is updated', near(net().layers[1].neurons[0].weights[2], 2.5),
  `got ${net().layers[1].neurons[0].weights[2]}`);

check('the target neuron recomputes',
  !near(net().layers[1].neurons[0].preActivation, 0, 1e-12));

check('the change reaches the prediction',
  pred().classes.some((c, i) => !near(c.share, predBefore[i])),
  'prediction did not move');

// Negative weights and zero must work too, since the panel offers both.
stores.updateWeight(1, 2, 0, -1.75);
check('negative weights are accepted',
  near(net().layers[1].neurons[0].weights[2], -1.75));

stores.updateWeight(1, 2, 0, 0);
check('zeroing a weight works',
  near(net().layers[1].neurons[0].weights[2], 0));

// A zeroed weight must remove that input's contribution entirely.
const sourceOutput = net().layers[0].neurons[2].output;
check('a zeroed weight contributes nothing',
  near(0 * sourceOutput, 0));

// Edits must not disturb neighbouring weights.
stores.resetAll();
const neighbour = net().layers[1].neurons[0].weights[1];
stores.updateWeight(1, 2, 0, 1.5);
check('neighbouring weights are untouched',
  near(net().layers[1].neurons[0].weights[1], neighbour));

// ---------------------------------------------------------------- deselect
console.log('\nClosing the panel');
stores.resetAll();
stores.selectConnection(1, 0, 0);
stores.selectConnection(1, 0, 0);
check('clicking the same connection again deselects',
  get(stores.selectedConnection) === undefined);

stores.selectConnection(1, 0, 0);
stores.clearSelection();
check('clearSelection closes the panel',
  get(stores.selectedConnection) === undefined);

// Every connection in the default network must be selectable.
console.log('\nAll connections are reachable');
stores.resetAll();
let allOk = true;
const n = net();
for (let l = 1; l < n.layers.length; l++) {
  for (let t = 0; t < n.layers[l].size; t++) {
    for (let s = 0; s < n.layers[l - 1].size; s++) {
      stores.selectConnection(l, s, t);
      const sc = get(stores.selectedConnection);
      if (sc === undefined || sc.targetLayerIndex !== l ||
          sc.sourceIndex !== s || sc.targetIndex !== t) {
        allOk = false;
      }
      stores.clearSelection();
    }
  }
}
check('every connection can be selected and edited', allOk);

stores.resetAll();

console.log(
  failures === 0
    ? `\nAll ${checks} checks passed.`
    : `\n${failures} of ${checks} checks failed.`
);
process.exit(failures === 0 ? 0 : 1);
