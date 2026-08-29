/**
 * Geometry and visual encoding for the network diagram.
 *
 * Positions are derived from the layer sizes, so any architecture the engine can
 * express gets a sensible layout with no per-shape special casing.
 *
 * This module is deliberately free of DOM and framework references: it takes
 * numbers in and returns numbers out, which makes the visual encoding easy to
 * reason about and reuse.
 */

export const layoutConfig = {
  neuronRadius: 20,
  minNeuronGap: 12,
  // Below this width the diagram scrolls horizontally rather than compressing
  // into something unreadable.
  minLayerGap: 150,
  padding: { top: 64, right: 40, bottom: 40, left: 40 },
  labelOffset: 34
};

/**
 * Compute neuron positions and the SVG canvas size.
 *
 * @param {object} network
 * @param {number} availableWidth Width the diagram may occupy.
 * @returns {object} layout with positions[layerIndex][neuronIndex]
 */
export const computeLayout = (network, availableWidth) => {
  const { neuronRadius, minNeuronGap, minLayerGap, padding } = layoutConfig;
  const numLayers = network.layers.length;
  const maxLayerSize = Math.max(...network.layers.map((l) => l.size));

  // Vertical: the tallest layer sets the canvas height, so no layer overflows.
  const neuronPitch = 2 * neuronRadius + minNeuronGap;
  const contentHeight = Math.max(neuronPitch, maxLayerSize * neuronPitch);
  const height = contentHeight + padding.top + padding.bottom;

  // Horizontal: spread layers across the available width, but never closer than
  // minLayerGap. If that pushes past the viewport the caller scrolls instead.
  const horizontalSpace = availableWidth - padding.left - padding.right;
  const layerGap = numLayers > 1
    ? Math.max(minLayerGap, horizontalSpace / (numLayers - 1))
    : 0;
  const width = padding.left + padding.right + layerGap * (numLayers - 1);

  const centerY = padding.top + contentHeight / 2;

  const positions = network.layers.map((layer, l) => {
    const x = padding.left + l * layerGap;
    // Centre each column vertically, so layers of different sizes stay aligned
    // on a common axis.
    const columnHeight = (layer.size - 1) * neuronPitch;
    const top = centerY - columnHeight / 2;

    return layer.neurons.map((_, i) => ({
      x,
      y: top + i * neuronPitch
    }));
  });

  return {
    positions,
    layerX: network.layers.map((_, l) => padding.left + l * layerGap),
    width,
    height,
    centerY,
    neuronRadius,
    labelY: padding.top - layoutConfig.labelOffset
  };
};

// ------------------------------------------------------------------ encoding

const clamp01 = (v) => Math.max(0, Math.min(1, v));

/**
 * Neuron fill colour.
 *
 * Signed activations (tanh, linear) diverge from a neutral midpoint so the sign
 * is visible. Non-negative ones ramp from pale to saturated, so brightness maps
 * to magnitude.
 */
export const neuronFill = (value, range, { signed = false } = {}) => {
  if (signed) {
    const t = clamp01((value / range + 1) / 2);
    return mixColor(NEG_COLOR, POS_COLOR, t, NEUTRAL_COLOR);
  }
  const t = clamp01(value / range);
  return mixHex(ACTIVATION_LOW, ACTIVATION_HIGH, t);
};

/** Text colour that stays legible against the computed fill. */
export const neuronTextColor = (value, range, { signed = false } = {}) => {
  const intensity = signed ? Math.abs(value / range) : clamp01(value / range);
  return intensity > 0.55 ? '#ffffff' : '#1f2933';
};

/**
 * Connection stroke colour: positive and negative weights are visually
 * distinct, and near-zero weights fade toward the background.
 */
export const connectionColor = (weight, magnitude) => {
  const t = clamp01(Math.abs(weight) / magnitude);
  const base = weight >= 0 ? POS_COLOR : NEG_COLOR;
  // Blend toward neutral as the weight approaches zero, so weak connections
  // recede without disappearing entirely.
  return mixColor(NEUTRAL_COLOR, base, 0.15 + 0.85 * t);
};

/** Stroke width scaled by |weight|, so influence reads as line weight. */
export const connectionWidth = (weight, magnitude, { min = 0.6, max = 4 } = {}) => {
  const t = clamp01(Math.abs(weight) / magnitude);
  return min + t * (max - min);
};

/** Weak connections are also more transparent, which reduces visual clutter. */
export const connectionOpacity = (weight, magnitude, { min = 0.12, max = 0.75 } = {}) => {
  const t = clamp01(Math.abs(weight) / magnitude);
  return min + t * (max - min);
};

/** Straight edge trimmed to the circle boundaries at both ends. */
export const connectionPath = (source, target, radius) => {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  return `M ${round(source.x + ux * radius)} ${round(source.y + uy * radius)} ` +
         `L ${round(target.x - ux * radius)} ${round(target.y - uy * radius)}`;
};

// ------------------------------------------------------------------ colours

const ACTIVATION_LOW = '#eef2f7';
const ACTIVATION_HIGH = '#2f6df6';
const POS_COLOR = [47, 109, 246];   // blue: positive weight
const NEG_COLOR = [225, 76, 66];    // red: negative weight
const NEUTRAL_COLOR = [203, 213, 224];

function round(v) {
  return Math.round(v * 100) / 100;
}

function hexToRgb(hex) {
  const clean = hex.replace('#', '');
  return [
    parseInt(clean.slice(0, 2), 16),
    parseInt(clean.slice(2, 4), 16),
    parseInt(clean.slice(4, 6), 16)
  ];
}

function mixHex(fromHex, toHex, t) {
  const from = hexToRgb(fromHex);
  const to = hexToRgb(toHex);
  return rgbString(from.map((c, i) => c + (to[i] - c) * t));
}

/**
 * Blend between two RGB triples. With three arguments the mix runs from `from`
 * to `to`; with a `via` colour, t < 0.5 blends from `from` to `via` and t > 0.5
 * from `via` to `to`, giving a diverging scale with a neutral midpoint.
 */
function mixColor(from, to, t, via) {
  if (via === undefined) {
    return rgbString(from.map((c, i) => c + (to[i] - c) * t));
  }
  if (t <= 0.5) {
    const local = t / 0.5;
    return rgbString(from.map((c, i) => c + (via[i] - c) * local));
  }
  const local = (t - 0.5) / 0.5;
  return rgbString(via.map((c, i) => c + (to[i] - c) * local));
}

function rgbString(rgb) {
  return `rgb(${rgb.map((c) => Math.round(Math.max(0, Math.min(255, c)))).join(', ')})`;
}

/** Shared number formatting so values look consistent everywhere. */
export const formatValue = (value, digits = 2) => {
  if (value === undefined || value === null || Number.isNaN(value)) return '--';
  if (value === 0) return '0.00';
  if (Math.abs(value) < 0.005) return value > 0 ? '<0.01' : '>-0.01';
  return value.toFixed(digits);
};

export const formatPercent = (value) => `${(value * 100).toFixed(1)}%`;

export const formatSigned = (value, digits = 2) =>
  `${value >= 0 ? '+' : '-'}${Math.abs(value).toFixed(digits)}`;
