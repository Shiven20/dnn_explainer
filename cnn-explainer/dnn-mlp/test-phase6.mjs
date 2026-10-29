globalThis.window = { matchMedia: () => ({ matches: true }) };

const stores = await import('../src/dnn/stores.js');
const { get } = await import('svelte/store');
const { WALKTHROUGH_PHASE, WALKTHROUGH_STATUS } = stores;

let checks = 0;
let failures = 0;
const check = (label, ok, detail = '') => {
  checks++;
  console.log(`  ${ok ? 'PASS' : 'FAIL'}  ${label}${ok ? '' : `  ${detail}`}`);
  if (!ok) failures++;
};
const snapshot = () => JSON.stringify(get(stores.network).layers.map((layer) =>
  layer.neurons.map((neuron) => ({ bias: neuron.bias, weights: neuron.weights }))));

stores.resetAll();
const originalNetwork = snapshot();
const originalInputs = [...get(stores.spec).inputs];

console.log('Staged training walkthrough');
stores.startTrainingWalkthrough();
let walk = get(stores.trainingWalkthrough);
check('starts at the real forward pass', walk.phase === WALKTHROUGH_PHASE.FORWARD);
check('reduced motion starts paused', walk.status === WALKTHROUGH_STATUS.PAUSED);
check('forward stage has real probabilities',
  walk.probabilities.length === 2 && Math.abs(walk.probabilities[0] + walk.probabilities[1] - 1) < 1e-9);
check('forward stage has a real target and prediction',
  Number.isInteger(walk.target) && Number.isInteger(walk.prediction));
check('network is unchanged during forward pass', snapshot() === originalNetwork);

stores.nextTrainingStage();
walk = get(stores.trainingWalkthrough);
check('next stage exposes cross-entropy loss',
  walk.phase === WALKTHROUGH_PHASE.LOSS && walk.sampleLoss > 0 && walk.batchLoss > 0);
check('network is unchanged during loss', snapshot() === originalNetwork);

stores.nextTrainingStage();
walk = get(stores.trainingWalkthrough);
const outputLayer = get(stores.network).layers.length - 1;
check('backpropagation starts at the output layer',
  walk.phase === WALKTHROUGH_PHASE.BACKPROP && walk.backwardLayer === outputLayer);
check('gradient summaries contain real finite magnitudes',
  walk.gradientLayers.slice(1).every((layer) =>
    Number.isFinite(layer.meanAbsolute) && Number.isFinite(layer.maxAbsolute) && layer.count > 0));
check('network is unchanged while gradients travel', snapshot() === originalNetwork);

while (get(stores.trainingWalkthrough).phase === WALKTHROUGH_PHASE.BACKPROP &&
       get(stores.trainingWalkthrough).backwardLayer > 1) {
  const before = get(stores.trainingWalkthrough).backwardLayer;
  stores.nextTrainingStage();
  check(`backpropagation reaches layer ${before - 1}`,
    get(stores.trainingWalkthrough).backwardLayer === before - 1);
}
stores.nextTrainingStage();
walk = get(stores.trainingWalkthrough);
check('final stage applies the weight update', walk.phase === WALKTHROUGH_PHASE.UPDATE);
check('weights change only at update', snapshot() !== originalNetwork);
check('one epoch is recorded', get(stores.training).epoch === 1);
check('loss and accuracy are recorded',
  Number.isFinite(get(stores.training).loss) && Number.isFinite(get(stores.training).accuracy));
check('largest changed weights have honest before/after values',
  walk.updates.length > 0 && walk.updates.every((item) =>
    item.after - item.before === item.change));

stores.nextTrainingStage();
check('walkthrough finishes after update',
  get(stores.trainingWalkthrough).phase === WALKTHROUGH_PHASE.COMPLETE &&
  get(stores.trainingWalkthrough).status === WALKTHROUGH_STATUS.FINISHED);

console.log('\nReset and skip');
stores.resetTraining();
check('training reset restores exact original weights', snapshot() === originalNetwork);
check('training reset restores the original input vector',
  JSON.stringify(get(stores.spec).inputs) === JSON.stringify(originalInputs));
check('walkthrough reset returns to idle',
  get(stores.trainingWalkthrough).phase === WALKTHROUGH_PHASE.IDLE);

stores.startTrainingWalkthrough();
stores.skipTrainingWalkthrough();
check('skip applies exactly one update',
  get(stores.training).epoch === 1 && snapshot() !== originalNetwork);
check('skip finishes the walkthrough',
  get(stores.trainingWalkthrough).phase === WALKTHROUGH_PHASE.COMPLETE);
const onceUpdated = snapshot();
stores.skipTrainingWalkthrough();
check('skip cannot apply the same update twice', snapshot() === onceUpdated && get(stores.training).epoch === 1);

stores.resetAll();
stores.destroyTraining();
console.log(failures === 0
  ? `\nAll ${checks} checks passed.`
  : `\n${failures} of ${checks} checks failed.`);
process.exit(failures === 0 ? 0 : 1);