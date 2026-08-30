<script>
  /**
   * One layer: its heading plus its column of neurons.
   *
   * Keeping the layer as its own component means the heading and the neurons
   * stay positioned together, and each layer can describe itself (kind, size,
   * activation) without the parent special-casing anything.
   */
  import Neuron from './Neuron.svelte';
  import { getActivation } from '../engine/activations.js';
  import { SOFTMAX } from '../engine/activations.js';

  export let layer;
  export let positions;
  export let radius = 20;
  export let range = 1;
  export let labelY = 20;
  export let showValues = true;
  export let selectedNeuron = undefined;
  export let hoveredNeuron = undefined;
  export let dimmedNeurons = false;
  /** Walkthrough has not reached this layer: draw it as not yet computed. */
  export let pending = false;
  /** This layer is taking its values right now: emphasise briefly. */
  export let settling = false;
  /**
   * Optional predicate (index) => boolean marking neurons outside the current
   * upstream trace, so the contributing path stands out. Undefined when no
   * trace is open.
   */
  export let isOutsideTrace = undefined;
  export let onSelectNeuron = () => {};
  export let onHoverNeuron = () => {};
  export let onLeaveNeuron = () => {};

  // Signed activations get a diverging colour ramp so the sign stays visible.
  $: activationMeta = getActivation(layer.activation);
  $: signed = layer.activation !== SOFTMAX && activationMeta.displayRange[0] < 0;

  $: subtitle =
    layer.kind === 'input'
      ? `${layer.size} value${layer.size === 1 ? '' : 's'}`
      : layer.activation === SOFTMAX
        ? `${layer.size} classes · softmax`
        : `${layer.size} neuron${layer.size === 1 ? '' : 's'} · ${activationMeta.label}`;

  $: centerX = positions.length > 0 ? positions[0].x : 0;
</script>

<style>
  .layer-title {
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    text-anchor: middle;
    fill: var(--dnn-ink, #1f2933);
    transition: fill 200ms ease-out;
  }

  /* The layer being computed gets its heading emphasised, which reinforces
     that propagation is happening one layer at a time. */
  .layer-title.settling {
    fill: var(--dnn-accent, #2f6df6);
  }

  .layer-subtitle {
    font-size: 10.5px;
    text-anchor: middle;
    fill: var(--dnn-muted, #64748b);
  }
</style>

<g class="layer" data-layer={layer.index}>
  <text class="layer-title" class:settling x={centerX} y={labelY}>{layer.label}</text>
  <text class="layer-subtitle" x={centerX} y={labelY + 14}>{subtitle}</text>

  {#each layer.neurons as neuron, i (neuron.id)}
    <Neuron
      {neuron}
      position={positions[i]}
      {radius}
      {range}
      {signed}
      {pending}
      showValue={showValues}
      dimmed={dimmedNeurons ||
        (isOutsideTrace !== undefined && isOutsideTrace(i))}
      selected={selectedNeuron !== undefined &&
        selectedNeuron.layerIndex === layer.index && selectedNeuron.index === i}
      hovered={hoveredNeuron !== undefined &&
        hoveredNeuron.layerIndex === layer.index && hoveredNeuron.index === i}
      label={`${layer.label}, neuron ${i + 1}, activation ${neuron.output.toFixed(3)}`}
      onSelect={() => onSelectNeuron(layer.index, i)}
      onHover={() => onHoverNeuron(layer.index, i)}
      onLeave={onLeaveNeuron}
    />
  {/each}
</g>
