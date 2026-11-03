<script>
  /**
   * One weighted connection between two neurons.
   *
   * Visual encoding:
   *   colour   -> sign of the weight (blue positive, red negative)
   *   width    -> magnitude, so influential connections read as heavier lines
   *   opacity  -> magnitude, so near-zero weights recede into the background
   *   glow     -> gradient magnitude during backpropagation (Phase 8)
   */
  import {
    connectionPath, connectionColor, connectionWidth, connectionOpacity
  } from '../engine/layout.js';

  export let source;          // {x, y}
  export let target;          // {x, y}
  export let weight;
  export let magnitude = 1;   // largest |weight| in the network
  export let radius = 20;
  export let highlighted = false;
  export let dimmed = false;
  /** The specific connection the user selected, emphasised above its trace. */
  export let selected = false;
  /**
   * Structurally part of the trace but carrying nothing for this input (a zero
   * weight, or a source neuron switched off). Drawn dashed so it is visibly
   * present but distinguishable from a live contribution.
   */
  export let muted = false;
  /** Educational training phase currently emphasizing this connection. */
  export let trainingState = undefined;
  /** Gradient magnitude for this connection (0..1 normalized). Phase 8. */
  export let gradientIntensity = 0;

  $: path = connectionPath(source, target, radius);

  $: stroke = selected
    ? 'var(--dnn-accent, #2f6df6)'
    : trainingState === 'backprop'
      ? '#8b5cf6'
      : trainingState === 'update'
        ? '#16a36a'
        : trainingState === 'loss'
          ? '#d97706'
          : trainingState === 'forward'
            ? 'var(--dnn-accent, #2f6df6)'
            : highlighted
              ? connectionColor(weight, magnitude)
              : connectionColor(weight, magnitude);

  $: strokeWidth = selected
    ? Math.max(3, connectionWidth(weight, magnitude) + 1)
    : trainingState !== undefined
      ? Math.max(2.2, connectionWidth(weight, magnitude) + (gradientIntensity * 1.5))
      : highlighted
        ? Math.max(2, connectionWidth(weight, magnitude))
        : connectionWidth(weight, magnitude);

  $: opacity = dimmed
    ? 0.05
    : muted
      ? 0.28
      : selected || highlighted || trainingState !== undefined
        ? 0.95
        : connectionOpacity(weight, magnitude);

  // Backpropagation travels opposite the forward signal and is dashed.
  $: dashArray = trainingState === 'backprop' ? '6 3' : muted ? '3 3' : null;
  
  // Gradient glow during backprop phase
  $: glowOpacity = trainingState === 'backprop' ? gradientIntensity * 0.6 : 0;
  $: glowWidth = trainingState === 'backprop' ? strokeWidth + (gradientIntensity * 4) : strokeWidth;
</script>

<style>
  path {
    fill: none;
    /* Transition only opacity and width: animating stroke colour on hundreds of
       paths is expensive and adds nothing legible. */
    transition: opacity 160ms ease-out, stroke-width 160ms ease-out;
    pointer-events: none;
  }
  
  .gradient-glow {
    fill: none;
    pointer-events: none;
    filter: blur(3px);
  }
</style>

{#if glowOpacity > 0.05}
  <path
    class="gradient-glow"
    d={path}
    stroke="#a78bfa"
    stroke-width={glowWidth}
    opacity={glowOpacity}
    stroke-dasharray={dashArray}
  />
{/if}

<path
  d={path}
  {stroke}
  stroke-width={strokeWidth}
  {opacity}
  stroke-dasharray={dashArray}
/>

