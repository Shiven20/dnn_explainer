<script>
  /**
   * One weighted connection between two neurons.
   *
   * Visual encoding:
   *   colour   -> sign of the weight (blue positive, red negative)
   *   width    -> magnitude, so influential connections read as heavier lines
   *   opacity  -> magnitude, so near-zero weights recede into the background
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

  $: path = connectionPath(source, target, radius);

  $: stroke = selected
    ? 'var(--dnn-accent, #2f6df6)'
    : highlighted
      ? connectionColor(weight, magnitude)
      : connectionColor(weight, magnitude);

  $: strokeWidth = selected
    ? Math.max(3, connectionWidth(weight, magnitude) + 1)
    : highlighted
      ? Math.max(2, connectionWidth(weight, magnitude))
      : connectionWidth(weight, magnitude);

  $: opacity = dimmed
    ? 0.05
    : muted
      ? 0.28
      : selected || highlighted
        ? 0.95
        : connectionOpacity(weight, magnitude);

  // Dashes mark a connection that exists but is not carrying anything.
  $: dashArray = muted ? '3 3' : null;
</script>

<style>
  path {
    fill: none;
    /* Transition only opacity and width: animating stroke colour on hundreds of
       paths is expensive and adds nothing legible. */
    transition: opacity 160ms ease-out, stroke-width 160ms ease-out;
    pointer-events: none;
  }
</style>

<path
  d={path}
  {stroke}
  stroke-width={strokeWidth}
  {opacity}
  stroke-dasharray={dashArray}
/>
