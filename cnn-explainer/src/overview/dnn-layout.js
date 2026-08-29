/* global d3 */

/**
 * Layout + colour helpers for the overview diagram.
 *
 * The diagram is rendered declaratively by Svelte, so this module only computes
 * geometry and colours. Keeping it free of DOM access makes it easy to reason
 * about and reuse from the detail views.
 */

import { overviewConfig } from '../config.js';

const { neuronRadius, inputCellLength, svgPaddings } = overviewConfig;

/**
 * Position every neuron.
 *
 * The input layer is drawn as a 5x5 grid of cells (it is an image, and showing
 * it as a grid preserves the spatial arrangement the glyphs depend on).
 * Every later layer is a vertical column of circles.
 *
 * @param {Neuron[][]} network
 * @param {number} width Available SVG width
 * @param {number} height Available SVG height
 */
export const computeLayout = (network, width, height) => {
  const numLayers = network.length;
  const inputWidth = Math.sqrt(network[0].length); // 5
  const inputBlockSize = inputWidth * inputCellLength;

  const usableWidth = width - svgPaddings.left - svgPaddings.right;
  const centerY = svgPaddings.top + (height - svgPaddings.top - svgPaddings.bottom) / 2;

  // Horizontal position of each layer's centre line.
  const layerGap = usableWidth / (numLayers - 1);
  const layerX = [];
  for (let l = 0; l < numLayers; l++) {
    layerX.push(svgPaddings.left + l * layerGap);
  }

  const coordinates = [];

  // Input layer: a grid, centred on the layer's centre line.
  const inputCoords = [];
  const gridLeft = layerX[0] - inputBlockSize / 2;
  const gridTop = centerY - inputBlockSize / 2;
  for (let i = 0; i < network[0].length; i++) {
    const row = Math.floor(i / inputWidth);
    const col = i % inputWidth;
    inputCoords.push({
      x: gridLeft + col * inputCellLength + inputCellLength / 2,
      y: gridTop + row * inputCellLength + inputCellLength / 2,
      row,
      col,
      cellX: gridLeft + col * inputCellLength,
      cellY: gridTop + row * inputCellLength
    });
  }
  coordinates.push(inputCoords);

  // Dense layers: evenly spaced circles in a column.
  for (let l = 1; l < numLayers; l++) {
    const layerSize = network[l].length;
    const spacing = 2 * neuronRadius + overviewConfig.vSpaceAroundGap;
    const columnHeight = (layerSize - 1) * spacing;
    const top = centerY - columnHeight / 2;

    const layerCoords = [];
    for (let i = 0; i < layerSize; i++) {
      layerCoords.push({ x: layerX[l], y: top + i * spacing });
    }
    coordinates.push(layerCoords);
  }

  return {
    coordinates,
    layerX,
    centerY,
    inputBlock: {
      left: gridLeft,
      top: gridTop,
      size: inputBlockSize,
      cellLength: inputCellLength,
      width: inputWidth
    }
  };
};

/**
 * Colour for an activation value.
 * Softmax outputs get the probability scale; ReLU outputs are non-negative so
 * they map onto a sequential scale; raw inputs use greyscale like an image.
 */
export const activationColor = (value, range, kind = 'activation') => {
  const scales = overviewConfig.colorScales;
  if (kind === 'input') {
    // Ink is dark, background is light: invert so 1 -> dark.
    return scales.input(0.1 + 0.85 * clamp01(value));
  }
  if (kind === 'probability') {
    return scales.probability(0.12 + 0.85 * clamp01(value));
  }
  const normalized = range === 0 ? 0 : clamp01(value / range);
  return scales.activation(0.08 + 0.82 * normalized);
};

/**
 * Colour for a signed weight. Zero maps to the midpoint of the diverging scale,
 * so negative and positive weights are visually distinguishable.
 */
export const weightColor = (weight, range) => {
  const normalized = range === 0 ? 0.5 : 0.5 - weight / (2 * range);
  return overviewConfig.colorScales.weight(clamp01(normalized));
};

/** Stroke width proportional to |weight|, so strong connections read heavier. */
export const weightStrokeWidth = (weight, range) => {
  const { edgeStrokeWidth, edgeMaxStrokeWidth } = overviewConfig;
  if (range === 0) return edgeStrokeWidth;
  const t = Math.min(1, Math.abs(weight) / range);
  return edgeStrokeWidth + t * (edgeMaxStrokeWidth - edgeStrokeWidth);
};

/** Number formatting used throughout the UI. */
export const fmt = (value, digits = 2) => {
  if (value === undefined || value === null || Number.isNaN(value)) return '--';
  if (value !== 0 && Math.abs(value) < 0.005) return value > 0 ? '<0.01' : '>-0.01';
  return d3.format(`.${digits}f`)(value);
};

export const fmtPercent = (value) => d3.format('.1%')(value);

function clamp01(v) {
  return Math.max(0, Math.min(1, v));
}

/**
 * Straight-line path between two neuron positions, trimmed so it starts and
 * ends at the circle edges rather than the centres.
 */
export const edgePath = (source, target, sourceRadius, targetRadius) => {
  const dx = target.x - source.x;
  const dy = target.y - source.y;
  const len = Math.sqrt(dx * dx + dy * dy) || 1;
  const ux = dx / len;
  const uy = dy / len;
  const x1 = source.x + ux * sourceRadius;
  const y1 = source.y + uy * sourceRadius;
  const x2 = target.x - ux * targetRadius;
  const y2 = target.y - uy * targetRadius;
  return `M ${x1} ${y1} L ${x2} ${y2}`;
};
