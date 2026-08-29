/**
 * Trains the DNN that ships with DNN Explainer and exports its weights.
 *
 * Architecture: 25 -> 10 (ReLU) -> 8 (ReLU) -> 4 (softmax)
 *
 * Implemented with plain JavaScript (no dependencies) so it runs anywhere Node
 * runs, and so the forward pass here matches the forward pass in the browser
 * exactly -- the visualization must show the same numbers the model computed.
 *
 * Usage: node dnn-mlp/train.js
 */

const fs = require('fs');
const path = require('path');
const { CLASSES, PROTOTYPES, buildDataset, mulberry32 } = require('./glyphs');

const LAYER_SIZES = [25, 10, 8, 4];
const EPOCHS = 220;
const LEARNING_RATE = 0.06;
const BATCH_SIZE = 32;
const SEED = 7;
const VAL_FRACTION = 0.2;

// ---------------------------------------------------------------- init

/** He-style initialization, scaled by fan-in, using a seeded PRNG. */
function initParams(rand) {
  const layers = [];
  for (let l = 0; l < LAYER_SIZES.length - 1; l++) {
    const fanIn = LAYER_SIZES[l];
    const fanOut = LAYER_SIZES[l + 1];
    const scale = Math.sqrt(2 / fanIn);
    const weights = [];
    for (let j = 0; j < fanOut; j++) {
      const row = [];
      for (let i = 0; i < fanIn; i++) {
        // Box-Muller for a normal sample from two uniforms.
        const u1 = Math.max(rand(), 1e-12);
        const u2 = rand();
        const g = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        row.push(g * scale);
      }
      weights.push(row);
    }
    layers.push({ weights, biases: new Array(fanOut).fill(0) });
  }
  return layers;
}

// ---------------------------------------------------------------- forward

function relu(x) {
  return x > 0 ? x : 0;
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

/**
 * Forward pass. Returns pre-activations and activations for every layer so the
 * backward pass (and the browser visualization) can reuse them.
 */
function forward(params, input) {
  const preActs = [];
  const acts = [input];
  let current = input;

  for (let l = 0; l < params.length; l++) {
    const { weights, biases } = params[l];
    const z = new Array(weights.length).fill(0);
    for (let j = 0; j < weights.length; j++) {
      let sum = biases[j];
      const row = weights[j];
      for (let i = 0; i < row.length; i++) sum += row[i] * current[i];
      z[j] = sum;
    }
    preActs.push(z);
    // Hidden layers use ReLU; the output layer uses softmax.
    current = l === params.length - 1 ? softmax(z) : z.map(relu);
    acts.push(current);
  }

  return { preActs, acts };
}

// ---------------------------------------------------------------- training

function zerosLike(params) {
  return params.map((layer) => ({
    weights: layer.weights.map((row) => new Array(row.length).fill(0)),
    biases: new Array(layer.biases.length).fill(0)
  }));
}

/** Accumulates gradients for one example into `grads`. Returns the loss. */
function backward(params, grads, input, label) {
  const { preActs, acts } = forward(params, input);
  const probs = acts[acts.length - 1];
  const loss = -Math.log(Math.max(probs[label], 1e-12));

  // Softmax + cross-entropy collapse to (p - y) at the output pre-activation.
  let delta = probs.map((p, i) => p - (i === label ? 1 : 0));

  for (let l = params.length - 1; l >= 0; l--) {
    const inputAct = acts[l];
    const gLayer = grads[l];

    for (let j = 0; j < delta.length; j++) {
      gLayer.biases[j] += delta[j];
      const gRow = gLayer.weights[j];
      for (let i = 0; i < inputAct.length; i++) {
        gRow[i] += delta[j] * inputAct[i];
      }
    }

    if (l > 0) {
      // Propagate into the previous layer, then through its ReLU.
      const prevSize = params[l].weights[0].length;
      const nextDelta = new Array(prevSize).fill(0);
      for (let j = 0; j < delta.length; j++) {
        const row = params[l].weights[j];
        for (let i = 0; i < prevSize; i++) nextDelta[i] += delta[j] * row[i];
      }
      const prevZ = preActs[l - 1];
      delta = nextDelta.map((v, i) => (prevZ[i] > 0 ? v : 0));
    }
  }

  return loss;
}

function applyGradients(params, grads, batchSize, lr) {
  for (let l = 0; l < params.length; l++) {
    const p = params[l];
    const g = grads[l];
    for (let j = 0; j < p.weights.length; j++) {
      for (let i = 0; i < p.weights[j].length; i++) {
        p.weights[j][i] -= (lr * g.weights[j][i]) / batchSize;
      }
      p.biases[j] -= (lr * g.biases[j]) / batchSize;
    }
  }
}

function accuracy(params, xs, ys) {
  let correct = 0;
  for (let n = 0; n < xs.length; n++) {
    const probs = forward(params, xs[n]).acts[LAYER_SIZES.length - 1];
    let best = 0;
    for (let i = 1; i < probs.length; i++) if (probs[i] > probs[best]) best = i;
    if (best === ys[n]) correct++;
  }
  return correct / xs.length;
}

function main() {
  const rand = mulberry32(SEED);
  const { xs, ys } = buildDataset(42, 400, 0.06);

  // Shuffle, then split into train / validation.
  const order = xs.map((_, i) => i);
  for (let i = order.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }
  const split = Math.floor(order.length * (1 - VAL_FRACTION));
  const trainIdx = order.slice(0, split);
  const valIdx = order.slice(split);
  const trainX = trainIdx.map((i) => xs[i]);
  const trainY = trainIdx.map((i) => ys[i]);
  const valX = valIdx.map((i) => xs[i]);
  const valY = valIdx.map((i) => ys[i]);

  console.log(`Dataset: ${xs.length} examples (${trainX.length} train / ${valX.length} val)`);
  console.log(`Architecture: ${LAYER_SIZES.join(' -> ')}`);

  const params = initParams(rand);

  for (let epoch = 1; epoch <= EPOCHS; epoch++) {
    // Reshuffle each epoch.
    for (let i = trainX.length - 1; i > 0; i--) {
      const j = Math.floor(rand() * (i + 1));
      [trainX[i], trainX[j]] = [trainX[j], trainX[i]];
      [trainY[i], trainY[j]] = [trainY[j], trainY[i]];
    }

    let epochLoss = 0;
    for (let start = 0; start < trainX.length; start += BATCH_SIZE) {
      const end = Math.min(start + BATCH_SIZE, trainX.length);
      const grads = zerosLike(params);
      for (let n = start; n < end; n++) {
        epochLoss += backward(params, grads, trainX[n], trainY[n]);
      }
      applyGradients(params, grads, end - start, LEARNING_RATE);
    }

    if (epoch % 20 === 0 || epoch === 1) {
      const trainAcc = accuracy(params, trainX, trainY);
      const valAcc = accuracy(params, valX, valY);
      console.log(
        `epoch ${String(epoch).padStart(3)}  loss ${(epochLoss / trainX.length).toFixed(4)}` +
          `  train ${(trainAcc * 100).toFixed(1)}%  val ${(valAcc * 100).toFixed(1)}%`
      );
    }
  }

  const finalTrain = accuracy(params, trainX, trainY);
  const finalVal = accuracy(params, valX, valY);
  console.log(`\nFinal: train ${(finalTrain * 100).toFixed(2)}%  val ${(finalVal * 100).toFixed(2)}%`);

  // Round for a smaller payload; verify accuracy survives the rounding.
  const round = (v) => Math.round(v * 1e6) / 1e6;
  const model = {
    name: 'glyph-mlp',
    description: '5x5 glyph classifier used by DNN Explainer',
    inputShape: [5, 5],
    classes: CLASSES,
    prototypes: PROTOTYPES,
    metrics: {
      trainAccuracy: round(finalTrain),
      valAccuracy: round(finalVal),
      epochs: EPOCHS,
      learningRate: LEARNING_RATE,
      batchSize: BATCH_SIZE,
      trainExamples: trainX.length,
      valExamples: valX.length
    },
    layers: [
      { name: 'input', type: 'input', size: LAYER_SIZES[0] },
      {
        name: 'hidden_1',
        type: 'dense',
        size: LAYER_SIZES[1],
        activation: 'relu',
        weights: params[0].weights.map((row) => row.map(round)),
        biases: params[0].biases.map(round)
      },
      {
        name: 'hidden_2',
        type: 'dense',
        size: LAYER_SIZES[2],
        activation: 'relu',
        weights: params[1].weights.map((row) => row.map(round)),
        biases: params[1].biases.map(round)
      },
      {
        name: 'output',
        type: 'dense',
        size: LAYER_SIZES[3],
        activation: 'softmax',
        weights: params[2].weights.map((row) => row.map(round)),
        biases: params[2].biases.map(round)
      }
    ]
  };

  const roundedParams = model.layers
    .filter((l) => l.type === 'dense')
    .map((l) => ({ weights: l.weights, biases: l.biases }));
  const roundedVal = accuracy(roundedParams, valX, valY);
  console.log(`Rounded weights val accuracy: ${(roundedVal * 100).toFixed(2)}%`);
  model.metrics.roundedValAccuracy = round(roundedVal);

  const outPath = path.join(__dirname, '..', 'public', 'assets', 'data', 'dnn_model.json');
  fs.writeFileSync(outPath, JSON.stringify(model));
  const kb = (fs.statSync(outPath).size / 1024).toFixed(1);
  console.log(`Wrote ${outPath} (${kb} KB)`);
}

main();
