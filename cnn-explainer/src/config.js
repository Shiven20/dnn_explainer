/* global d3 */

/**
 * Colour scales, one per role.
 *
 * Activations are one-sided (ReLU output is >= 0) so they use a sequential
 * scale. Weights are signed, so they use a diverging scale where the midpoint
 * is zero -- that makes the sign of a weight readable at a glance.
 */
const colorScales = {
  input: d3.interpolateGreys,
  activation: d3.interpolateBlues,
  preActivation: d3.interpolateRdBu,
  weight: d3.interpolateRdBu,
  bias: d3.interpolatePuOr,
  probability: d3.interpolateOranges
};

const neuronRadius = 17;

export const overviewConfig = {
  neuronRadius,
  inputCellLength: 16,
  numLayers: 4,
  layerNames: ['input', 'hidden_1', 'hidden_2', 'output'],
  layerDisplayNames: {
    input: 'Input',
    hidden_1: 'Hidden 1',
    hidden_2: 'Hidden 2',
    output: 'Output'
  },
  layerSubtitles: {
    input: '5 x 5 = 25 values',
    hidden_1: '10 neurons - ReLU',
    hidden_2: '8 neurons - ReLU',
    output: '4 classes - softmax'
  },
  edgeOpacity: 0.32,
  edgeOpacityFaded: 0.06,
  edgeInitColor: 'rgb(200, 200, 200)',
  edgeHoverColor: 'rgb(70, 70, 70)',
  edgeStrokeWidth: 0.9,
  edgeStrokeWidthHover: 2.4,
  edgeMaxStrokeWidth: 3.2,
  svgPaddings: { top: 40, bottom: 30, left: 60, right: 60 },
  vSpaceAroundGap: 14,
  classLists: ['L shape', 'T shape', 'X shape', 'O ring'],
  colorScales,
  // Kept under the old key as well so any shared helper that still reads
  // `layerColorScales` keeps working.
  layerColorScales: colorScales
};

export const detailViewConfig = {
  barMaxWidth: 120,
  maxTermsShown: 25,
  cellLength: 22
};
