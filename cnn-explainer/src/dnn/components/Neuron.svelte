<script>
  /**
   * One neuron.
   *
   * Fill intensity encodes the activation value, so a glance across a layer
   * shows which units responded to the current input. Inactive ReLU units get a
   * dashed outline because "output is exactly zero" is a distinct state worth
   * calling out, not just a pale colour.
   */
  import { neuronFill, neuronTextColor, formatValue } from '../engine/layout.js';
  import { RELU, SOFTMAX } from '../engine/activations.js';

  export let neuron;
  export let position;
  export let radius = 20;
  export let range = 1;
  export let signed = false;      // true for activations that can go negative
  export let showValue = true;
  export let selected = false;
  export let hovered = false;
  export let dimmed = false;
  /**
   * The walkthrough has not reached this neuron yet. Drawn empty rather than
   * showing its value, so the animation never displays a number the viewer is
   * meant to believe has not been computed.
   */
  export let pending = false;
  export let label = '';          // accessible description
  export let onSelect = () => {};
  export let onHover = () => {};
  export let onLeave = () => {};

  $: fill = pending
    ? 'var(--dnn-surface-sunken, #f9fafc)'
    : neuronFill(neuron.output, range, { signed });
  $: textColor = neuronTextColor(neuron.output, range, { signed });
  // An un-reached neuron is not "inactive"; suppress that styling until it has
  // actually been computed.
  $: isInactive = !pending && neuron.activation === RELU && neuron.output === 0;
  $: isProbability = neuron.activation === SOFTMAX;
</script>

<style>
  .neuron {
    cursor: pointer;
    transition: opacity 160ms ease-out;
  }

  .neuron:focus {
    outline: none;
  }

  .body {
    transition: fill 200ms ease-out, r 200ms ease-out;
  }

  .ring {
    fill: none;
    stroke: var(--dnn-neuron-ring, #94a3b8);
    stroke-width: 1.25;
    transition: stroke 160ms ease-out, stroke-width 160ms ease-out;
  }

  .ring.inactive {
    stroke: var(--dnn-negative, #e14c42);
    stroke-dasharray: 3 2.5;
  }

  .ring.selected {
    stroke: var(--dnn-accent, #2f6df6);
    stroke-width: 2.75;
  }

  .ring.hovered {
    stroke: var(--dnn-ink, #1f2933);
    stroke-width: 2;
  }

  /* Keyboard focus needs a visible indicator that does not rely on hover. */
  .neuron:focus-visible .ring {
    stroke: var(--dnn-accent, #2f6df6);
    stroke-width: 2.75;
  }

  .value {
    font-size: 10.5px;
    font-variant-numeric: tabular-nums;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
    user-select: none;
  }

  .halo {
    fill: none;
    stroke: var(--dnn-accent, #2f6df6);
    stroke-width: 1;
    opacity: 0.35;
  }
</style>

<g
  class="neuron"
  opacity={dimmed ? 0.3 : 1}
  role="button"
  tabindex="0"
  aria-label={label}
  aria-pressed={selected}
  on:click|stopPropagation={onSelect}
  on:mouseenter={onHover}
  on:mouseleave={onLeave}
  on:focus={onHover}
  on:blur={onLeave}
  on:keydown={(e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      onSelect();
    }
  }}
>
  {#if selected}
    <circle class="halo" cx={position.x} cy={position.y} r={radius + 5} />
  {/if}

  <circle
    class="body"
    cx={position.x}
    cy={position.y}
    r={radius}
    {fill}
  />

  <circle
    class="ring"
    class:inactive={isInactive}
    class:selected
    class:hovered
    cx={position.x}
    cy={position.y}
    r={radius}
  />

  {#if showValue && !pending}
    <text class="value" x={position.x} y={position.y} fill={textColor}>
      {isProbability ? `${Math.round(neuron.output * 100)}%` : formatValue(neuron.output)}
    </text>
  {/if}
</g>
