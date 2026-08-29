/**
 * Activation functions.
 *
 * Each entry carries its own metadata (formula, output range, derivative)
 * alongside the function itself, so the UI can explain an activation without
 * keeping a parallel table of strings that could drift out of sync.
 */

export const RELU = 'relu';
export const SIGMOID = 'sigmoid';
export const TANH = 'tanh';
export const LINEAR = 'linear';
export const SOFTMAX = 'softmax';

export const activations = {
  [RELU]: {
    id: RELU,
    label: 'ReLU',
    formula: 'f(x) = max(0, x)',
    // Used to scale the neuron colour ramp. ReLU is unbounded above, so this is
    // a display convention rather than a true maximum.
    displayRange: [0, 1],
    unbounded: true,
    description:
      'Passes positive values through untouched and clamps everything else to zero. ' +
      'Cheap to compute and the usual default for hidden layers.',
    fn: (x) => (x > 0 ? x : 0),
    derivative: (x) => (x > 0 ? 1 : 0)
  },

  [SIGMOID]: {
    id: SIGMOID,
    label: 'Sigmoid',
    formula: 'f(x) = 1 / (1 + e^-x)',
    displayRange: [0, 1],
    unbounded: false,
    description:
      'Squashes any input into the range 0 to 1. Large positive and large negative ' +
      'inputs both flatten out, which is what makes deep sigmoid networks hard to train.',
    fn: (x) => 1 / (1 + Math.exp(-x)),
    derivative: (x) => {
      const s = 1 / (1 + Math.exp(-x));
      return s * (1 - s);
    }
  },

  [TANH]: {
    id: TANH,
    label: 'Tanh',
    formula: 'f(x) = (e^x - e^-x) / (e^x + e^-x)',
    displayRange: [-1, 1],
    unbounded: false,
    description:
      'Like sigmoid but centred on zero, mapping inputs to the range -1 to 1. ' +
      'The zero centring usually makes it train better than sigmoid.',
    fn: (x) => Math.tanh(x),
    derivative: (x) => 1 - Math.tanh(x) ** 2
  },

  [LINEAR]: {
    id: LINEAR,
    label: 'Linear',
    formula: 'f(x) = x',
    displayRange: [-1, 1],
    unbounded: true,
    description:
      'No transformation at all. Stacking linear layers collapses into a single ' +
      'linear layer, which is exactly why non-linear activations matter.',
    fn: (x) => x,
    derivative: () => 1
  }
};

/** Ordered list for building menus. */
export const hiddenActivationOptions = [RELU, SIGMOID, TANH, LINEAR];

export const getActivation = (id) => activations[id] || activations[RELU];

/**
 * Numerically stable softmax, used on the output layer to turn raw scores into
 * probabilities. Subtracting the max first prevents Math.exp from overflowing
 * on large logits without changing the result.
 */
export const softmax = (logits) => {
  if (logits.length === 0) return [];
  const max = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  // A degenerate sum would mean every exp underflowed; fall back to uniform.
  if (!Number.isFinite(sum) || sum === 0) {
    return logits.map(() => 1 / logits.length);
  }
  return exps.map((v) => v / sum);
};
