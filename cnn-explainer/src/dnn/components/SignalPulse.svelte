<script>
  /**
   * A dot travelling along one connection during the forward pass.
   *
   * Makes the left-to-right flow of information literal. The dot's colour
   * matches the weight's sign and its size reflects the magnitude of what is
   * actually being carried (weight x source activation), so a connection
   * contributing nothing stays visually quiet.
   */
  import { connectionColor } from '../engine/layout.js';

  export let source;         // {x, y}
  export let target;         // {x, y}
  export let weight;
  export let magnitude = 1;
  export let contribution = 0;  // weight * source activation
  export let progress = 0;      // 0..1 along the path
  export let radius = 20;

  // Start and end at the circle edges, matching the visible connection.
  $: dx = target.x - source.x;
  $: dy = target.y - source.y;
  $: len = Math.sqrt(dx * dx + dy * dy) || 1;
  $: ux = dx / len;
  $: uy = dy / len;
  $: x1 = source.x + ux * radius;
  $: y1 = source.y + uy * radius;
  $: x2 = target.x - ux * radius;
  $: y2 = target.y - uy * radius;

  $: cx = x1 + (x2 - x1) * progress;
  $: cy = y1 + (y2 - y1) * progress;

  $: strength = Math.min(1, Math.abs(contribution) / magnitude);
  // Floor the radius so a near-zero contribution is still faintly visible
  // rather than vanishing, which would read as a rendering gap.
  $: r = 1.6 + strength * 2.6;
  $: fill = connectionColor(weight, magnitude);
  // Fade in and out at the ends so dots do not pop at the neuron boundary.
  $: opacity = 0.25 + 0.75 * strength * Math.sin(Math.PI * Math.min(1, Math.max(0, progress)));
</script>

<style>
  circle {
    pointer-events: none;
  }
</style>

<circle {cx} {cy} {r} {fill} {opacity} />
