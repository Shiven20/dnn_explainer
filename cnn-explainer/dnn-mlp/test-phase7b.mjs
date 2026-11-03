/**
 * Phase 7b regression suite: per-class metrics and confusion matrix.
 */

import { createNetwork } from '../src/dnn/engine/network.js';
import { forwardPass } from '../src/dnn/engine/forwardPass.js';
import { RELU, SOFTMAX } from '../src/dnn/engine/activations.js';
import { computeDetailedMetrics, trainEpoch } from '../src/dnn/engine/training.js';

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

console.log('\n=== Phase 7b: Per-Class Metrics ===\n');

// ------------------------------------------------------------------ setup

const network = createNetwork({
  layerSizes: [4, 5, 5, 3],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 999
});

// Create a synthetic dataset with 3 classes
const dataset = [
  // Class 0 examples
  { inputs: [0.9, 0.1, 0.1, 0.1], target: 0 },
  { inputs: [0.8, 0.2, 0.1, 0.1], target: 0 },
  { inputs: [0.85, 0.15, 0.05, 0.1], target: 0 },
  { inputs: [0.95, 0.05, 0.1, 0.05], target: 0 },
  // Class 1 examples
  { inputs: [0.1, 0.9, 0.1, 0.1], target: 1 },
  { inputs: [0.2, 0.8, 0.15, 0.1], target: 1 },
  { inputs: [0.15, 0.85, 0.1, 0.05], target: 1 },
  // Class 2 examples
  { inputs: [0.1, 0.1, 0.9, 0.8], target: 2 },
  { inputs: [0.1, 0.15, 0.8, 0.85], target: 2 },
  { inputs: [0.05, 0.1, 0.95, 0.9], target: 2 },
  { inputs: [0.1, 0.2, 0.85, 0.75], target: 2 }
];

// ------------------------------------------------------------------ initial metrics

const initialMetrics = computeDetailedMetrics(network, dataset, 3);

check('computeDetailedMetrics returns perClass array', Array.isArray(initialMetrics.perClass));
check('perClass has 3 entries', initialMetrics.perClass.length === 3);
check('confusion matrix is 3x3', initialMetrics.confusionMatrix.length === 3);
check('confusion matrix rows have 3 columns', initialMetrics.confusionMatrix[0].length === 3);

check('each class has precision', initialMetrics.perClass.every(c => typeof c.precision === 'number'));
check('each class has recall', initialMetrics.perClass.every(c => typeof c.recall === 'number'));
check('each class has f1', initialMetrics.perClass.every(c => typeof c.f1 === 'number'));
check('each class has support', initialMetrics.perClass.every(c => typeof c.support === 'number'));

check('support for class 0 is 4', initialMetrics.perClass[0].support === 4);
check('support for class 1 is 3', initialMetrics.perClass[1].support === 3);
check('support for class 2 is 4', initialMetrics.perClass[2].support === 4);

check('total samples matches dataset length', initialMetrics.totalSamples === dataset.length);
check('total correct is non-negative', initialMetrics.totalCorrect >= 0);
check('total correct does not exceed total samples', initialMetrics.totalCorrect <= initialMetrics.totalSamples);

// Confusion matrix structure
const confusion = initialMetrics.confusionMatrix;
let totalPredictions = 0;
for (let actual = 0; actual < 3; actual++) {
  for (let predicted = 0; predicted < 3; predicted++) {
    totalPredictions += confusion[actual][predicted];
    check(`confusion[${actual}][${predicted}] is non-negative`, confusion[actual][predicted] >= 0);
  }
}
check('confusion matrix sums to total samples', totalPredictions === dataset.length);

// Diagonal sum equals total correct
const diagonalSum = confusion[0][0] + confusion[1][1] + confusion[2][2];
check('confusion matrix diagonal equals total correct', diagonalSum === initialMetrics.totalCorrect);

// Accuracy calculation
check('accuracy equals correct / total', approx(initialMetrics.accuracy, initialMetrics.totalCorrect / initialMetrics.totalSamples));

// Metrics are between 0 and 1
check('all precisions in [0,1]', initialMetrics.perClass.every(c => c.precision >= 0 && c.precision <= 1));
check('all recalls in [0,1]', initialMetrics.perClass.every(c => c.recall >= 0 && c.recall <= 1));
check('all f1 scores in [0,1]', initialMetrics.perClass.every(c => c.f1 >= 0 && c.f1 <= 1));

// ------------------------------------------------------------------ after training

let trained = network;
for (let epoch = 0; epoch < 50; epoch++) {
  const result = trainEpoch(trained, dataset, 0.1);
  trained = result.network;
}

const trainedMetrics = computeDetailedMetrics(trained, dataset, 3);

check('trained accuracy improved or stayed same', trainedMetrics.accuracy >= initialMetrics.accuracy);
check('trained total correct increased or stayed same', trainedMetrics.totalCorrect >= initialMetrics.totalCorrect);

// After training, at least some classes should have non-zero precision
const nonZeroPrecision = trainedMetrics.perClass.filter(c => c.precision > 0).length;
check('at least one class has non-zero precision', nonZeroPrecision > 0);

// F1 score relationship: F1 <= min(precision, recall) * 2
trainedMetrics.perClass.forEach((cls, i) => {
  if (cls.precision > 0 || cls.recall > 0) {
    const maxF1 = 2 * Math.min(cls.precision, cls.recall);
    check(`class ${i} F1 is valid`, cls.f1 <= maxF1 + 1e-6);
  }
});

// ------------------------------------------------------------------ perfect predictions

const perfectDataset = [
  { inputs: [1, 0, 0, 0], target: 0 },
  { inputs: [0, 1, 0, 0], target: 1 },
  { inputs: [0, 0, 1, 1], target: 2 }
];

// Create a simple network that can perfectly classify
const simpleNet = createNetwork({
  layerSizes: [4, 3],
  hiddenActivation: RELU,
  outputActivation: SOFTMAX,
  seed: 42
});

// Train to convergence
let perfect = simpleNet;
for (let i = 0; i < 200; i++) {
  const result = trainEpoch(perfect, perfectDataset, 0.2);
  perfect = result.network;
}

const perfectMetrics = computeDetailedMetrics(perfect, perfectDataset, 3);

// With enough training, should achieve perfect accuracy
check('perfect accuracy is 1.0 or very close', perfectMetrics.accuracy > 0.99);
check('all precision scores are 1.0 or very close', perfectMetrics.perClass.every(c => c.precision > 0.99 || c.support === 0));
check('all recall scores are 1.0 or very close', perfectMetrics.perClass.every(c => c.recall > 0.99 || c.support === 0));
check('all F1 scores are 1.0 or very close', perfectMetrics.perClass.every(c => c.f1 > 0.99 || c.support === 0));

// Perfect confusion matrix is diagonal
const perfectConfusion = perfectMetrics.confusionMatrix;
for (let i = 0; i < 3; i++) {
  for (let j = 0; j < 3; j++) {
    if (i === j) {
      check(`perfect diagonal[${i}] is non-zero`, perfectConfusion[i][j] > 0);
    } else {
      check(`perfect off-diagonal[${i}][${j}] is zero`, perfectConfusion[i][j] === 0);
    }
  }
}

// ------------------------------------------------------------------ edge cases

const emptyMetrics = computeDetailedMetrics(network, [], 3);
check('empty dataset returns valid structure', emptyMetrics.perClass.length === 3);
check('empty dataset has zero correct', emptyMetrics.totalCorrect === 0);
check('empty dataset has zero samples', emptyMetrics.totalSamples === 0);

const singleExample = [{ inputs: [1, 0, 0, 0], target: 0 }];
const singleMetrics = computeDetailedMetrics(network, singleExample, 3);
check('single example returns valid metrics', singleMetrics.totalSamples === 1);
check('single example total predictions in confusion matrix is 1',
  singleMetrics.confusionMatrix.flat().reduce((a, b) => a + b, 0) === 1);

// ------------------------------------------------------------------ imbalanced dataset

const imbalanced = [
  { inputs: [0.9, 0.1, 0.1, 0.1], target: 0 },
  { inputs: [0.8, 0.2, 0.1, 0.1], target: 0 },
  { inputs: [0.85, 0.15, 0.05, 0.1], target: 0 },
  { inputs: [0.95, 0.05, 0.1, 0.05], target: 0 },
  { inputs: [0.9, 0.05, 0.15, 0.1], target: 0 },
  { inputs: [0.92, 0.08, 0.1, 0.05], target: 0 },
  { inputs: [0.88, 0.12, 0.05, 0.1], target: 0 },
  { inputs: [0.93, 0.07, 0.1, 0.08], target: 0 }, // 8 examples of class 0
  { inputs: [0.1, 0.9, 0.1, 0.1], target: 1 }, // 1 example of class 1
  { inputs: [0.1, 0.1, 0.9, 0.8], target: 2 }  // 1 example of class 2
];

const imbalancedMetrics = computeDetailedMetrics(network, imbalanced, 3);
check('imbalanced class 0 support is 8', imbalancedMetrics.perClass[0].support === 8);
check('imbalanced class 1 support is 1', imbalancedMetrics.perClass[1].support === 1);
check('imbalanced class 2 support is 1', imbalancedMetrics.perClass[2].support === 1);
check('imbalanced total samples is 10', imbalancedMetrics.totalSamples === 10);

// Support sums to total
const supportSum = imbalancedMetrics.perClass.reduce((sum, c) => sum + c.support, 0);
check('support sum equals total samples', supportSum === imbalancedMetrics.totalSamples);

// ------------------------------------------------------------------ summary

console.log(`\n${passed} checks passed, ${failed} failed\n`);
process.exit(failed > 0 ? 1 : 0);
