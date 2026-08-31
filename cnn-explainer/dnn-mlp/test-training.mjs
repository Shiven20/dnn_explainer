import { createNetwork } from '../src/dnn/engine/network.js';
import { forwardPass } from '../src/dnn/engine/forwardPass.js';
import {
  computeGradients, createPatternDataset, evaluateDataset, trainEpoch
} from '../src/dnn/engine/training.js';
import { RELU, SIGMOID, TANH, LINEAR, SOFTMAX } from '../src/dnn/engine/activations.js';

let checks = 0;
let failures = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};
const near = (a, b, epsilon) => Math.abs(a - b) <= epsilon;

console.log('Finite-difference gradients');
const net = createNetwork({
  layerSizes: [3, 4, 2], hiddenActivation: TANH,
  outputActivation: SOFTMAX, seed: 19
});
const example = { inputs: [0.8, 0.2, 0.65], target: 0 };
const analytical = computeGradients(net, example.inputs, example.target);
const epsilon = 1e-5;
const plus = structuredClone(net);
const minus = structuredClone(net);
plus.layers[1].neurons[2].weights[1] += epsilon;
minus.layers[1].neurons[2].weights[1] -= epsilon;
const plusLoss = computeGradients(plus, example.inputs, example.target).loss;
const minusLoss = computeGradients(minus, example.inputs, example.target).loss;
const numerical = (plusLoss - minusLoss) / (2 * epsilon);
const actual = analytical.gradients[1].weights[2][1];
check('weight gradient matches finite differences', near(actual, numerical, 1e-6),
  `${actual} vs ${numerical}`);

const plusBias = structuredClone(net);
const minusBias = structuredClone(net);
plusBias.layers[2].neurons[0].bias += epsilon;
minusBias.layers[2].neurons[0].bias -= epsilon;
const numericalBias = (
  computeGradients(plusBias, example.inputs, example.target).loss -
  computeGradients(minusBias, example.inputs, example.target).loss
) / (2 * epsilon);

check('bias gradient matches finite differences',
  near(analytical.gradients[2].biases[0], numericalBias, 1e-6));

console.log('\nLearning the synthetic task');
const dataset = createPatternDataset(4, 80, 2026);
check('dataset is deterministic',
  JSON.stringify(dataset) === JSON.stringify(createPatternDataset(4, 80, 2026)));
check('dataset contains both classes',
  new Set(dataset.map((item) => item.target)).size === 2);

for (const activation of [RELU, SIGMOID, TANH, LINEAR]) {
  let model = createNetwork({
    layerSizes: [4, 5, 5, 2], hiddenActivation: activation,
    outputActivation: SOFTMAX, seed: 7
  });
  const before = evaluateDataset(model, dataset);
  const firstWeight = model.layers[1].neurons[0].weights[0];
  for (let epoch = 0; epoch < 100; epoch++) {
    model = trainEpoch(model, dataset, 0.08).network;
  }
  const after = evaluateDataset(model, dataset);
  check(`${activation}: loss remains finite`, Number.isFinite(after.loss));
  check(`${activation}: weights change`,
    model.layers[1].neurons[0].weights[0] !== firstWeight);
  check(`${activation}: loss decreases`, after.loss < before.loss,
    `${before.loss} -> ${after.loss}`);
}

console.log('\nForward values remain real');
const trained = trainEpoch(net, createPatternDataset(3, 20), 0.05).network;
const activated = forwardPass(trained, example.inputs);
check('trained network still produces a probability distribution',
  near(activated.layers[2].neurons.reduce((sum, neuron) => sum + neuron.output, 0), 1, 1e-9));

console.log(failures === 0
  ? `\nAll ${checks} checks passed.`
  : `\n${failures} of ${checks} checks failed.`);
process.exit(failures === 0 ? 0 : 1);