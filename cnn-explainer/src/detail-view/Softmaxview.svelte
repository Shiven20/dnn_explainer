<script>
  /**
   * Walks through the softmax step: logits -> exp(logit) -> normalized share.
   *
   * The intermediate exp() column is the part that usually clears up why one
   * class can dominate: exponentiating amplifies gaps between logits before
   * anything is normalized.
   */
  import { createEventDispatcher } from 'svelte';
  import { fmt, fmtPercent } from '../overview/dnn-layout.js';

  export let logits = [];
  export let probabilities = [];
  export let classNames = [];
  export let prediction = -1;

  const dispatch = createEventDispatcher();

  // Subtracting the max before exponentiating is what the model does too: it
  // avoids overflow and leaves the result unchanged.
  $: maxLogit = logits.length > 0 ? Math.max(...logits) : 0;
  $: exps = logits.map((z) => Math.exp(z - maxLogit));
  $: expSum = exps.reduce((a, b) => a + b, 0);
  $: maxExp = exps.reduce((a, b) => Math.max(a, b), 0) || 1;
</script>

<style>
  .panel {
    width: 468px;
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

  .formula {
    padding: 7px 12px;
    background: rgb(250, 250, 250);
    border-bottom: 1px solid rgb(238, 238, 238);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
  }

  .table { padding: 6px 12px 10px 12px; }

  .head, .row {
    display: grid;
    grid-template-columns: 76px 52px 56px 1fr 56px;
    gap: 7px;
    align-items: center;
  }

  .head {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: var(--dark-gray);
    padding-bottom: 3px;
    border-bottom: 1px solid rgb(238, 238, 238);
  }

  .row {
    padding: 3px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 11.5px;
  }

  .row.winner { font-weight: 700; }

  .class-name { font-family: inherit; font-size: 11.5px; }

  .bar-track {
    height: 10px;
    background: rgb(240, 240, 240);
    border-radius: 2px;
    overflow: hidden;
  }

  .bar-fill { height: 10px; background: rgb(240, 140, 60); }

  .prob { text-align: right; }

  .footer {
    border-top: 1px solid rgb(238, 238, 238);
    padding: 6px 12px 9px 12px;
    font-size: 11.5px;
    color: var(--deep-gray);
  }

  .footer strong { color: rgb(50, 50, 50); }
</style>

<div class="panel">
  <div class="header">
    <div>
      <div class="title">Softmax</div>
      <div class="subtitle">turning 4 raw scores into probabilities</div>
    </div>
    <button class="close" on:click={() => dispatch('close')} aria-label="Close softmax view">
      &times;
    </button>
  </div>

  <div class="formula">
    P(class i) = exp(z<sub>i</sub>) / &Sigma;<sub>j</sub> exp(z<sub>j</sub>)
  </div>

  <div class="table">
    <div class="head">
      <span>class</span>
      <span>logit z</span>
      <span>exp(z)</span>
      <span>share</span>
      <span style="text-align:right;">P</span>
    </div>

    {#each classNames as name, i}
      <div class="row" class:winner={i === prediction}>
        <span class="class-name">{name}</span>
        <span>{fmt(logits[i], 2)}</span>
        <span>{fmt(exps[i], 3)}</span>
        <span class="bar-track">
          <span class="bar-fill" style="width: {(exps[i] / maxExp) * 100}%;"></span>
        </span>
        <span class="prob">{fmtPercent(probabilities[i])}</span>
      </div>
    {/each}
  </div>

  <div class="footer">
    The exp() column sums to <strong>{fmt(expSum, 3)}</strong>; dividing each row by
    that total is what makes the probabilities add up to 100%. Because exp() grows
    quickly, a logit gap of even 2 or 3 turns into a lopsided result.
  </div>
</div>
