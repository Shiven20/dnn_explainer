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

  $: path = connectionPath(source, target, radius);
  $: stroke = highlighted ? '#1f2933' : connectionColor(weight, magnitude);
  $: strokeWidth = highlighted
    ? Math.max(2, connectionWidth(weight, magnitude))
    : connectionWidth(weight, magnitude);
  $: opacity = dimmed
    ? 0.05
    : highlighted
      ? 0.95
      : connectionOpacity(weight, magnitude);
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

<path d={path} {stroke} stroke-width={strokeWidth} {opacity} />
