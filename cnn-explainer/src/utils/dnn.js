/**
 * Core engine for DNN Explainer.
 *
 * Builds a node/link graph for a fully-connected network and runs the forward
 * pass. The forward pass here is intentionally written the same way as the
 * trainer in `dnn-mlp/train.js`, so every number the UI shows is a number the
 * model actually computed.
 */

export const nodeType = {
  INPUT: 'input',
  DENSE: 'dense',
  OUTPUT: 'output'
};

export const activationType = {
  NONE: 'none',
  RELU: 'relu',
  SOFTMAX: 'softmax'
};

/**
 * One neuron. A dense neuron owns its bias; its weights live on the incoming
 * links so the UI can highlight a single weight independently.
 */
export class Neuron {
  constructor(layerName, layerIndex, index, type, bias = 0) {
    this.layerName = layerName;
    this.layerIndex = layerIndex;
    this.index = index;
    this.type = type;
    this.bias = bias;

    // Weighted sum before activation, and the value after activation.
    this.preActivation = 0;
    this.output = 0;

    this.inputLinks = [];
    this.outputLinks = [];
  }

  /** True when a ReLU unit is switched off for the current input. */
  get isDead() {
    return this.activation === activationType.RELU && this.output === 0;
  }
}

export class Link {
  constructor(source, dest, weight) {
    this.source = source;
    this.dest = dest;
    this.weight = weight;
  }

  /** Signed contribution this link makes to its destination's weighted sum. */
  get contribution() {
    return this.weight * this.source.output;
  }
}

/**
 * Turn the exported model JSON into a layered graph of Neurons and Links.
 * @param {object} modelJSON Parsed contents of dnn_model.json
 * @returns {Neuron[][]} network[layerIndex][neuronIndex]
 */
export const constructDNNFromJSON = (modelJSON) => {
  const network = [];

  modelJSON.layers.forEach((layer, layerIndex) => {
    const neurons = [];
    const isOutput = layer.name === 'output';
    const type =
      layer.type === 'input' ? nodeType.INPUT : isOutput ? nodeType.OUTPUT : nodeType.DENSE;

    for (let i = 0; i < layer.size; i++) {
      const bias = layer.biases ? layer.biases[i] : 0;
      const neuron = new Neuron(layer.name, layerIndex, i, type, bias);
      neuron.activation = layer.activation || activationType.NONE;

      // Fully connect to the previous layer.
      if (layerIndex > 0) {
        const prevLayer = network[layerIndex - 1];
        for (let j = 0; j < prevLayer.length; j++) {
          const link = new Link(prevLayer[j], neuron, layer.weights[i][j]);
          prevLayer[j].outputLinks.push(link);
          neuron.inputLinks.push(link);
        }
      }

      neurons.push(neuron);
    }

    network.push(neurons);
  });

  return network;
};

/** Load the trained model and build its graph. */
export const constructDNN = async () => {
  const url = 'PUBLIC_URL/assets/data/dnn_model.json';
  const response = await fetch(url);

  if (!response.ok) {
    throw new Error(`Could not load ${url} (HTTP ${response.status})`);
  }

  // A dev server with SPA fallback answers a missing asset with index.html
  // rather than a 404. Detect that here so the message names the real problem
  // instead of surfacing an opaque JSON parse error.
  const body = await response.text();
  if (body.trimStart().startsWith('<')) {
    throw new Error(
      `${url} returned HTML instead of JSON, which usually means the file is missing.`
    );
  }

  let modelJSON;
  try {
    modelJSON = JSON.parse(body);
  } catch (error) {
    throw new Error(`${url} is not valid JSON: ${error.message}`);
  }

  return { network: constructDNNFromJSON(modelJSON), model: modelJSON };
};

// ------------------------------------------------------------------ math

export const relu = (x) => (x > 0 ? x : 0);

/** Numerically stable softmax. */
export const softmax = (logits) => {
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
};

/**
 * Weighted sum for one neuron, returned with its parts so the detail view can
 * show the arithmetic term by term.
 */
export const weightedSum = (neuron) => {
  const terms = neuron.inputLinks.map((link) => ({
    inputIndex: link.source.index,
    input: link.source.output,
    weight: link.weight,
    product: link.weight * link.source.output
  }));
  const sum = terms.reduce((acc, t) => acc + t.product, 0);
  return { terms, sum, bias: neuron.bias, total: sum + neuron.bias };
};

/**
 * Run the network on one input vector, writing preActivation/output onto every
 * neuron so the visualization can read them straight off the graph.
 * @param {Neuron[][]} network
 * @param {number[]} inputVector Flattened 5x5 glyph, length 25
 */
export const forwardPass = (network, inputVector) => {
  if (inputVector.length !== network[0].length) {
    throw new Error(
      `Input length ${inputVector.length} does not match input layer size ${network[0].length}`
    );
  }

  // Input layer passes values straight through.
  network[0].forEach((neuron, i) => {
    neuron.preActivation = inputVector[i];
    neuron.output = inputVector[i];
  });

  for (let l = 1; l < network.length; l++) {
    const layer = network[l];

    layer.forEach((neuron) => {
      let sum = neuron.bias;
      for (let i = 0; i < neuron.inputLinks.length; i++) {
        const link = neuron.inputLinks[i];
        sum += link.weight * link.source.output;
      }
      neuron.preActivation = sum;
    });

    if (layer[0].activation === activationType.SOFTMAX) {
      const probs = softmax(layer.map((n) => n.preActivation));
      layer.forEach((neuron, i) => {
        neuron.output = probs[i];
      });
    } else if (layer[0].activation === activationType.RELU) {
      layer.forEach((neuron) => {
        neuron.output = relu(neuron.preActivation);
      });
    } else {
      layer.forEach((neuron) => {
        neuron.output = neuron.preActivation;
      });
    }
  }

  return network;
};

/** Index of the highest-probability output neuron. */
export const getPrediction = (network) => {
  const outputLayer = network[network.length - 1];
  let best = 0;
  for (let i = 1; i < outputLayer.length; i++) {
    if (outputLayer[i].output > outputLayer[best].output) best = i;
  }
  return best;
};

// ------------------------------------------------------------------ ranges

/**
 * Per-layer symmetric magnitude used to keep colour scales comparable.
 * Softmax layers are left at 1 because they are already 0..1.
 */
export const getLayerRanges = (network) =>
  network.map((layer) => {
    if (layer[0].activation === activationType.SOFTMAX) return 1;
    const max = layer.reduce((acc, n) => Math.max(acc, Math.abs(n.output)), 0);
    return max === 0 ? 1 : max;
  });

/** Symmetric range across every weight in the network, for the weight legend. */
export const getWeightRange = (network) => {
  let max = 0;
  for (let l = 1; l < network.length; l++) {
    network[l].forEach((neuron) => {
      neuron.inputLinks.forEach((link) => {
        max = Math.max(max, Math.abs(link.weight));
      });
    });
  }
  return max === 0 ? 1 : max;
};

// ------------------------------------------------------------------ inputs

/** Flatten a 2D grid row-major into a 1D vector. */
export const flattenGrid = (grid) => {
  const out = [];
  for (let r = 0; r < grid.length; r++) {
    for (let c = 0; c < grid[r].length; c++) out.push(grid[r][c]);
  }
  return out;
};

/** Inverse of flattenGrid. */
export const unflattenGrid = (vector, width) => {
  const grid = [];
  for (let r = 0; r < vector.length / width; r++) {
    grid.push(vector.slice(r * width, (r + 1) * width));
  }
  return grid;
};

/** Empty width x width grid. */
export const emptyGrid = (width) =>
  Array.from({ length: width }, () => new Array(width).fill(0));

/**
 * Add pixel noise to a copy of a grid. Used by the "add noise" control so the
 * user can watch a confident prediction degrade.
 */
export const noisyGrid = (grid, flipProb = 0.08) =>
  grid.map((row) => row.map((v) => (Math.random() < flipProb ? (v === 1 ? 0 : 1) : v)));
