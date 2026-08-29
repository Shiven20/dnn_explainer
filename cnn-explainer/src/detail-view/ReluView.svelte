<script>
  /**
   * Detail view for a hidden (ReLU) neuron.
   *
   * Shows the weighted sum, then the ReLU step applied to it. The graph marks
   * where this neuron's z lands on the ReLU curve, which makes the "switched
   * off" case concrete rather than abstract.
   */
  import { createEventDispatcher } from 'svelte';
  import { weightedSum } from '../utils/dnn.js';
  import { weightColor, fmt } from '../overview/dnn-layout.js';

  export let neuron;
  export let weightRange = 1;
  export let layerName = '';

  const dispatch = createEventDispatcher();

  $: detail = weightedSum(neuron);
  $: activeTerms = detail.terms
    .filter((t) => t.product !== 0)
    .sort((a, b) => Math.abs(b.product) - Math.abs(a.product));
  $: zeroTermCount = detail.terms.length - activeTerms.length;
  $: isDead = neuron.output === 0;

  // ReLU graph geometry.
  const graphW = 150;
  const graphH = 96;
  const pad = 16;
  // Axis range adapts to z so the marker is always on screen.
  $: axisMax = Math.max(2, Math.ceil(Math.abs(detail.total) + 0.5));
  $: xScale = (v) => pad + ((v + axisMax) / (2 * axisMax)) * (graphW - 2 * pad);
  $: yScale = (v) => graphH - pad - (v / axisMax) * (graphH - 2 * pad);
</script>

<style>
  .panel {
    width: 452px;
    background: white;
    border: 1px solid var(--middle-gray);
    border-radius: 6px;
    box-shadow: var(--outer-shadow);
    font-size: 12.5px;
  }

  .header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 8px 12px;
    border-bottom: 1px solid rgb(238, 238, 238);
  }

  .title { font-weight: 600; font-size: 13px; }
  .subtitle { color: var(--deep-gray); font-size: 11.5px; }

  .close {
    border: none; background: none; font-size: 16px; line-height: 1;
    color: var(--dark-gray); cursor: pointer; padding: 0 2px;
  }
  .close:hover { color: var(--red); }

  .body {
    display: flex;
    gap: 10px;
    padding: 8px 12px 10px 12px;
  }

  .left { flex: 1; min-width: 0; }
  .right {
    width: 158px;
    border-left: 1px solid rgb(238, 238, 238);
    padding-left: 10px;
  }

  .terms {
    max-height: 120px;
    overflow-y: auto;
  }

  .term {
    display: grid;
    grid-template-columns: 40px 34px 42px 1fr;
    align-items: center;
    gap: 5px;
    padding: 1.5px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11px;
  }

  .term-index { color: var(--deep-gray); }

  .swatch {
    display: inline-block; width: 100%; height: 12px; border-radius: 2px;
    text-align: center; line-height: 12px; font-size: 9.5px;
  }

  .product { text-align: right; }
  .product.pos { color: rgb(20, 110, 40); }
  .product.neg { color: rgb(170, 40, 30); }

  .row {
    display: flex; justify-content: space-between;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    padding: 1.5px 0;
  }

  .row.total {
    font-weight: 700;
    border-top: 1px solid rgb(230, 230, 230);
    margin-top: 3px; padding-top: 4px;
  }

  .relu-formula {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
    padding: 4px 0 6px 0;
  }

  .verdict {
    padding: 5px 7px;
    border-radius: 4px;
    font-size: 11.5px;
    margin-top: 5px;
  }

  .verdict.on { background: rgb(240, 249, 243); border-left: 3px solid var(--green); }
  .verdict.off { background: rgb(255, 246, 246); border-left: 3px solid var(--red); }

  .note { color: var(--deep-gray); font-size: 10.5px; padding-top: 4px; }

  .axis { stroke: rgb(200, 200, 200); stroke-width: 1; }
  .curve { stroke: rgb(60, 60, 60); stroke-width: 1.6; fill: none; }
  .tick-label { font-size: 8.5px; fill: var(--dark-gray); }
  .col-head {
    display: grid;
    grid-template-columns: 40px 34px 42px 1fr;
    gap: 5px;
    font-size: 9.5px;
    text-transform: uppercase;
    color: var(--dark-gray);
    padding-bottom: 2px;
  }
</style>

<div class="panel">
  <div class="header">
    <div>
      <div class="title">{layerName} &middot; neuron {neuron.index + 1}</div>
      <div class="subtitle">weighted sum, then ReLU</div>
    </div>
    <button class="close" on:click={() => dispatch('close')} aria-label="Close detail view">
      &times;
    </button>
  </div>

  <div class="body">
    <div class="left">
      {#if activeTerms.length > 0}
        <div class="col-head">
          <span>input</span><span>x</span><span>w</span>
          <span style="text-align:right;">w &middot; x</span>
        </div>
        <div class="terms">
          {#each activeTerms as term}
            <div class="term">
              <span class="term-index">#{term.inputIndex + 1}</span>
              <span>{fmt(term.input)}</span>
              <span
                class="swatch"
                style="background: {weightColor(term.weight, weightRange)};
                       color: {Math.abs(term.weight) > 0.6 * weightRange ? 'white' : 'rgb(40,40,40)'};"
              >{fmt(term.weight)}</span>
              <span class="product" class:pos={term.product > 0} class:neg={term.product < 0}>
                {fmt(term.product, 3)}
              </span>
            </div>
          {/each}
        </div>
        {#if zeroTermCount > 0}
          <div class="note">
            + {zeroTermCount} term{zeroTermCount === 1 ? '' : 's'} equal to zero
          </div>
        {/if}
      {:else}
        <div class="note">All incoming values are 0, so only the bias remains.</div>
      {/if}

      <div class="row" style="margin-top:5px;">
        <span>sum</span><span>{fmt(detail.sum, 3)}</span>
      </div>
      <div class="row">
        <span>bias b</span><span>{fmt(detail.bias, 3)}</span>
      </div>
      <div class="row total">
        <span>z</span><span>{fmt(detail.total, 3)}</span>
      </div>
    </div>

    <div class="right">
      <div class="relu-formula">ReLU(z) = max(0, z)</div>

      <svg width={graphW} height={graphH} role="img"
        aria-label="ReLU activation curve with this neuron's value marked">
        <!-- axes -->
        <line class="axis" x1={pad} y1={yScale(0)} x2={graphW - pad} y2={yScale(0)} />
        <line class="axis" x1={xScale(0)} y1={pad - 6} x2={xScale(0)} y2={graphH - pad + 4} />

        <!-- ReLU curve: flat at 0 for z<=0, then slope 1 -->
        <path
          class="curve"
          d="M {xScale(-axisMax)} {yScale(0)} L {xScale(0)} {yScale(0)} L {xScale(axisMax)} {yScale(axisMax)}"
        />

        <!-- marker for this neuron's z -->
        <line
          x1={xScale(Math.max(-axisMax, Math.min(axisMax, detail.total)))}
          y1={yScale(0)}
          x2={xScale(Math.max(-axisMax, Math.min(axisMax, detail.total)))}
          y2={yScale(Math.max(0, Math.min(axisMax, neuron.output)))}
          stroke={isDead ? 'var(--red)' : 'var(--blue)'}
          stroke-width="1"
          stroke-dasharray="2 2"
        />
        <circle
          cx={xScale(Math.max(-axisMax, Math.min(axisMax, detail.total)))}
          cy={yScale(Math.max(0, Math.min(axisMax, neuron.output)))}
          r="3.6"
          fill={isDead ? 'var(--red)' : 'var(--blue)'}
        />

        <text class="tick-label" x={xScale(0) + 3} y={graphH - 3}>0</text>
        <text class="tick-label" x={graphW - pad - 4} y={graphH - 3}>z</text>
      </svg>

      <div class="verdict" class:on={!isDead} class:off={isDead}>
        {#if isDead}
          z is negative, so ReLU clips it to <strong>0</strong>. This neuron
          passes nothing forward for this input.
        {:else}
          z is positive, so ReLU passes it through unchanged:
          <strong>{fmt(neuron.output, 3)}</strong>
        {/if}
      </div>
    </div>
  </div>
</div>
