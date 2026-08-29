<script>
  /**
   * Shows how one neuron's weighted sum is built, term by term:
   *   z = sum_i (w_i * x_i) + b
   *
   * Every number displayed is read off the live network, so it always matches
   * what the model computed for the current input.
   */
  import { createEventDispatcher } from 'svelte';
  import { weightedSum, activationType, softmax } from '../utils/dnn.js';
  import { weightColor, fmt, fmtPercent } from '../overview/dnn-layout.js';

  export let neuron;
  export let weightRange = 1;
  export let layerName = '';
  export let className = '';

  const dispatch = createEventDispatcher();

  // Recomputed whenever the selected neuron or the input changes.
  $: detail = weightedSum(neuron);

  // Only the terms that actually move the sum are worth listing. With 25 inputs
  // and a sparse binary image most products are exactly zero, and hiding them
  // makes the real contributions readable.
  $: activeTerms = detail.terms
    .filter((t) => t.product !== 0)
    .sort((a, b) => Math.abs(b.product) - Math.abs(a.product));

  $: zeroTermCount = detail.terms.length - activeTerms.length;
  $: maxAbsProduct = activeTerms.reduce((acc, t) => Math.max(acc, Math.abs(t.product)), 0) || 1;
  $: isSoftmax = neuron.activation === activationType.SOFTMAX;
</script>

<style>
  .panel {
    width: 420px;
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

  .title {
    font-weight: 600;
    font-size: 13px;
  }

  .subtitle {
    color: var(--deep-gray);
    font-size: 11.5px;
  }

  .close {
    border: none;
    background: none;
    font-size: 16px;
    line-height: 1;
    color: var(--dark-gray);
    cursor: pointer;
    padding: 0 2px;
  }

  .close:hover { color: var(--red); }

  .formula {
    padding: 8px 12px;
    background: rgb(250, 250, 250);
    border-bottom: 1px solid rgb(238, 238, 238);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
  }

  .terms {
    max-height: 168px;
    overflow-y: auto;
    padding: 4px 12px;
  }

  .term {
    display: grid;
    grid-template-columns: 52px 46px 46px 1fr 58px;
    align-items: center;
    gap: 6px;
    padding: 2px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
  }

  .term-index { color: var(--deep-gray); }

  .swatch {
    display: inline-block;
    width: 100%;
    height: 13px;
    border-radius: 2px;
    text-align: center;
    line-height: 13px;
    font-size: 10px;
  }

  .bar-track {
    position: relative;
    height: 9px;
    background: rgb(240, 240, 240);
    border-radius: 2px;
  }

  .bar-mid {
    position: absolute;
    left: 50%;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: rgb(200, 200, 200);
  }

  .bar-fill {
    position: absolute;
    top: 0;
    height: 9px;
    border-radius: 2px;
  }

  .product { text-align: right; }
  .product.pos { color: rgb(20, 110, 40); }
  .product.neg { color: rgb(170, 40, 30); }

  .footer {
    border-top: 1px solid rgb(238, 238, 238);
    padding: 7px 12px 9px 12px;
  }

  .row {
    display: flex;
    justify-content: space-between;
    padding: 1.5px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .row.total {
    font-weight: 700;
    border-top: 1px solid rgb(230, 230, 230);
    margin-top: 3px;
    padding-top: 4px;
  }

  .note {
    color: var(--deep-gray);
    font-size: 11px;
    padding: 2px 12px 6px 12px;
  }

  .col-head {
    display: grid;
    grid-template-columns: 52px 46px 46px 1fr 58px;
    gap: 6px;
    padding: 4px 12px 0 12px;
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--dark-gray);
  }
</style>

<div class="panel">
  <div class="header">
    <div>
      <div class="title">
        {layerName} &middot; neuron {neuron.index + 1}
        {#if isSoftmax && className}<span class="subtitle">({className})</span>{/if}
      </div>
      <div class="subtitle">
        {isSoftmax ? 'logit, before softmax' : 'weighted sum'}
      </div>
    </div>
    <button class="close" on:click={() => dispatch('close')} aria-label="Close detail view">
      &times;
    </button>
  </div>

  <div class="formula">
    z = &Sigma; (w<sub>i</sub> &middot; x<sub>i</sub>) + b = <strong>{fmt(detail.total, 3)}</strong>
    {#if isSoftmax}
      &nbsp;&rarr;&nbsp; softmax = <strong>{fmtPercent(neuron.output)}</strong>
    {/if}
  </div>

  {#if activeTerms.length > 0}
    <div class="col-head">
      <span>input</span>
      <span>x</span>
      <span>w</span>
      <span>w &middot; x</span>
      <span style="text-align:right;">value</span>
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
          <span class="bar-track">
            <span class="bar-mid"></span>
            <span
              class="bar-fill"
              style="
                {term.product >= 0
                  ? `left: 50%; width: ${(term.product / maxAbsProduct) * 50}%;`
                  : `right: 50%; width: ${(Math.abs(term.product) / maxAbsProduct) * 50}%;`}
                background: {term.product >= 0 ? 'rgb(70, 150, 90)' : 'rgb(200, 90, 80)'};"
            ></span>
          </span>
          <span class="product" class:pos={term.product > 0} class:neg={term.product < 0}>
            {fmt(term.product, 3)}
          </span>
        </div>
      {/each}
    </div>

    {#if zeroTermCount > 0}
      <div class="note">
        {zeroTermCount} other input{zeroTermCount === 1 ? '' : 's'} contribute nothing
        (the input value is 0, so w &middot; x = 0).
      </div>
    {/if}
  {:else}
    <div class="note" style="padding: 12px;">
      Every incoming value is 0, so the weighted sum is just the bias.
    </div>
  {/if}

  <div class="footer">
    <div class="row">
      <span>sum of terms</span>
      <span>{fmt(detail.sum, 3)}</span>
    </div>
    <div class="row">
      <span>bias b</span>
      <span>{fmt(detail.bias, 3)}</span>
    </div>
    <div class="row total">
      <span>z</span>
      <span>{fmt(detail.total, 3)}</span>
    </div>
  </div>
</div>
