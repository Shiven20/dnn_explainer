import { writable } from 'svelte/store';

// The layered Neuron graph, plus the raw model JSON (metrics, class names).
export const dnnStore = writable([]);
export const modelInfoStore = writable(undefined);

export const svgStore = writable(undefined);

// Layout: computed neuron positions, keyed by layer index then neuron index.
export const neuronCoordinateStore = writable([]);

// Colour scaling.
export const layerRangesStore = writable([]);
export const weightRangeStore = writable(1);

// The 5x5 grid currently fed to the network, and its predicted class index.
export const inputGridStore = writable([]);
export const predictionStore = writable(-1);

// Which neuron the detail view is showing, as {layerIndex, index} or undefined.
export const selectedNeuronStore = writable(undefined);
// Which neuron is hovered, used to highlight its incoming/outgoing edges.
export const hoveredNeuronStore = writable(undefined);

// Toggles the numeric annotations on top of the diagram.
export const detailedModeStore = writable(true);

// Softmax walkthrough panel.
export const isInSoftmaxStore = writable(false);
export const softmaxDetailViewStore = writable({});

// Tooltip payload and the explanatory modal.
export const hoverInfoStore = writable({});
export const modalStore = writable({});
