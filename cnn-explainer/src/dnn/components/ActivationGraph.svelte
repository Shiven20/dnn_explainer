<script>
  /**
   * Plots an activation function with the current neuron's z marked on it.
   *
   * Seeing where z falls on the curve is what makes the function concrete --
   * particularly for ReLU, where "the value was negative so it became zero" is
   * far clearer as a position on a graph than as a sentence.
   */
  import { getActivation, SOFTMAX } from '../engine/activations.js';

  export let activationId;
  export let value = 0;      // z, the pre-activation
  export let output = 0;     // f(z)
  export let width = 132;
  export let height = 96;

  const pad = 16;

  $: meta = getActivation(activationId);

  /*
   * Axis range adapts to z so the marker is always on screen, but never shrinks
   * below the function's own interesting region.
   */
  $: xMax = Math.max(3, Math.ceil(Math.abs(value) + 0.5));
  $: yBounds = (() => {
    const [lo, hi] = meta.displayRange;
    if (meta.unbounded) {
      // ReLU and linear grow without bound: follow the data.
      const top = Math.max(1, Math.abs(output), xMax);
      return [lo < 0 ? -top : 0, top];
    }
    return [lo, hi];
  })();

  $: xScale = (v) => pad + ((v + xMax) / (2 * xMax)) * (width - 2 * pad);
  $: yScale = (v) => {
    const [lo, hi] = yBounds;
    const t = (v - lo) / (hi - lo || 1);
    return height - pad - t * (height - 2 * pad);
  };

  /** Sample the function across the visible range to draw its curve. */
  $: curve = (() => {
    if (activationId === SOFTMAX) return '';
    const steps = 60;
    const points = [];
    for (let i = 0; i <= steps; i++) {
      const x = -xMax + (2 * xMax * i) / steps;
      const y = meta.fn(x);
      points.push(`${xScale(x).toFixed(1)} ${yScale(y).toFixed(1)}`);
    }
    return `M ${points.join(' L ')}`;
  })();

  // Clamp the marker so an extreme z stays inside the plot area.
  $: markerX = xScale(Math.max(-xMax, Math.min(xMax, value)));
  $: markerY = yScale(Math.max(yBounds[0], Math.min(yBounds[1], output)));
  $: zeroY = yScale(Math.max(yBounds[0], Math.min(yBounds[1], 0)));
  $: isDead = output === 0 && value <= 0;
</script>

<style>
  svg { flex-shrink: 0; }

  .axis {
    stroke: var(--dnn-border);
    stroke-width: 1;
  }

  .curve {
    fill: none;
    stroke: var(--dnn-ink);
    stroke-width: 1.7;
    stroke-linecap: round;
  }

  .guide {
    stroke-width: 1;
    stroke-dasharray: 2 2;
  }

  .tick {
    font-size: 8.5px;
    fill: var(--dnn-muted);
  }
</style>

<svg
  {width}
  {height}
  role="img"
  aria-label={`${meta.label} curve with the current value ${value.toFixed(2)} marked`}
>
  <!-- Axes -->
  <line class="axis" x1={pad} y1={zeroY} x2={width - pad} y2={zeroY} />
  <line class="axis" x1={xScale(0)} y1={pad - 6} x2={xScale(0)} y2={height - pad + 5} />

  <path class="curve" d={curve} />

  <!-- Drop lines from the marker to both axes. -->
  <line
    class="guide"
    x1={markerX} y1={zeroY} x2={markerX} y2={markerY}
    stroke={isDead ? 'var(--dnn-negative)' : 'var(--dnn-accent)'}
  />
  <line
    class="guide"
    x1={xScale(0)} y1={markerY} x2={markerX} y2={markerY}
    stroke={isDead ? 'var(--dnn-negative)' : 'var(--dnn-accent)'}
  />

  <circle
    cx={markerX}
    cy={markerY}
    r="3.8"
    fill={isDead ? 'var(--dnn-negative)' : 'var(--dnn-accent)'}
  />

  <text class="tick" x={xScale(0) + 3} y={height - 4}>0</text>
  <text class="tick" x={width - pad - 5} y={height - 4}>z</text>
</svg>
