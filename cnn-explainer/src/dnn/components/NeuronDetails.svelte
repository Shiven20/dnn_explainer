<script>
  /**
   * Neuron inspection: where a neuron's value came from.
   *
   * Every number is read from the live network via explainNeuron(), so the
   * arithmetic shown is the arithmetic performed. Nothing is hardcoded, and the
   * running total is accumulated from the same term list that is displayed.
   */
  import {
    activatedNetwork, activationRanges, weightMagnitude, updateBias,
    selectNeuron, selectConnection, clearSelection, setInput
  } from '../stores.js';
  import { explainNeuron } from '../engine/forwardPass.js';
  import { getActivation, SOFTMAX, RELU } from '../engine/activations.js';
  import { connectionColor, neuronFill, formatValue, formatSigned } from '../engine/layout.js';
  import ActivationGraph from './ActivationGraph.svelte';

  export let layerIndex;
  export let index;

  $: net = $activatedNetwork;
  $: detail = net ? explainNeuron(net, layerIndex, index) : undefined;
  $: layer = detail ? detail.layer : undefined;
  $: activationMeta = detail && !detail.isInput ? getActivation(detail.activation) : undefined;
  $: isSoftmax = detail && detail.activation === SOFTMAX;

  /*
   * Terms are shown in the network's own order (input 1, 2, 3 ...) rather than
   * sorted by size: the reader is checking a sum, and a reordered sum is harder
   * to follow than a literal one.
   */
  $: terms = detail ? detail.terms : [];
  $: maxAbsProduct = terms.reduce((acc, t) => Math.max(acc, Math.abs(t.product)), 0) || 1;

  /** Running total, so the sum can be followed line by line. */
  $: runningTotals = terms.reduce((acc, t) => {
    acc.push((acc.length > 0 ? acc[acc.length - 1] : 0) + t.product);
    return acc;
  }, []);

  const handleBias = (event) => {
    const parsed = parseFloat(event.target.value);
    if (event.target.value === '' || Number.isNaN(parsed)) return;
    updateBias(layerIndex, index, Math.max(-2, Math.min(2, parsed)));
  };

  /** Jump to the neuron feeding this term, so the reader can walk backwards. */
  const inspectSource = (sourceIndex) => {
    selectNeuron(layerIndex - 1, sourceIndex);
  };

  /**
   * Open this term's connection. Also the keyboard route to connection details,
   * since the connection paths themselves are not focusable.
   */
  const inspectConnection = (sourceIndex) => {
    selectConnection(layerIndex, sourceIndex, index);
  };
</script>

<style>
  .panel {
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 14px;
    overflow: hidden;
  }

  .head {
    display: flex;
    align-items: flex-start;
    justify-content: space-between;
    gap: 10px;
    padding: 14px 16px 12px 16px;
    border-bottom: 1px solid var(--dnn-border);
  }

  .eyebrow {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--dnn-muted);
  }

  .title {
    font-size: 15px;
    font-weight: 600;
    color: var(--dnn-ink);
    margin-top: 2px;
  }

  .swatch {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    border: 1px solid var(--dnn-border);
    flex-shrink: 0;
  }

  .close {
    border: none;
    background: none;
    font-size: 18px;
    line-height: 1;
    color: var(--dnn-muted);
    cursor: pointer;
    padding: 0 2px;
  }

  .close:hover { color: var(--dnn-negative); }

  .head-actions { display: flex; align-items: center; gap: 10px; }

  .section {
    padding: 12px 16px;
    border-bottom: 1px solid var(--dnn-border);
  }

  .section:last-child { border-bottom: none; }

  .section-label {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--dnn-muted);
    margin-bottom: 8px;
  }

  .terms {
    max-height: 210px;
    overflow-y: auto;
    margin: 0 -4px;
    padding: 0 4px;
  }

  .term-head, .term {
    display: grid;
    grid-template-columns: 34px 48px 52px 1fr 62px;
    align-items: center;
    gap: 7px;
    font-variant-numeric: tabular-nums;
  }

  .term-head {
    font-size: 9.5px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--dnn-muted);
    padding-bottom: 4px;
    border-bottom: 1px solid var(--dnn-border);
    margin-bottom: 3px;
  }

  .term {
    font-size: 11.5px;
    padding: 3px 0;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .source-btn {
    border: none;
    background: none;
    padding: 0;
    font: inherit;
    color: var(--dnn-accent);
    cursor: pointer;
    text-align: left;
  }

  .source-btn:hover { text-decoration: underline; }

  .weight-chip {
    display: inline-block;
    width: 100%;
    height: 16px;
    line-height: 16px;
    border-radius: 3px;
    text-align: center;
    font-size: 10px;
    border: none;
    padding: 0;
    font-family: inherit;
    cursor: pointer;
    transition: filter 120ms ease-out;
  }

  .weight-chip:hover { filter: brightness(1.1); }

  .bar-track {
    position: relative;
    height: 9px;
    background: var(--dnn-track);
    border-radius: 3px;
  }

  .bar-mid {
    position: absolute;
    left: 50%;
    top: -2px;
    bottom: -2px;
    width: 1px;
    background: var(--dnn-border);
  }

  .bar-fill {
    position: absolute;
    top: 0;
    height: 9px;
    border-radius: 3px;
  }

  .product { text-align: right; }
  .product.pos { color: var(--dnn-positive); }
  .product.neg { color: var(--dnn-negative); }

  /* Running total sits under its product, greyed out so the products stay
     dominant and the total reads as supporting information. */
  .running {
    display: block;
    font-size: 9.5px;
    color: var(--dnn-muted);
    text-align: right;
    line-height: 1.2;
  }

  .sum-rows { margin-top: 9px; }

  .sum-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    font-size: 12.5px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    padding: 3px 0;
    color: var(--dnn-muted);
  }

  .sum-row.total {
    color: var(--dnn-ink);
    font-weight: 700;
    border-top: 1px solid var(--dnn-border);
    margin-top: 4px;
    padding-top: 6px;
  }

  .bias-field {
    width: 74px;
    padding: 3px 6px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
    text-align: right;
    border: 1px solid var(--dnn-border);
    border-radius: 6px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .bias-field:focus {
    outline: none;
    border-color: var(--dnn-accent);
    box-shadow: 0 0 0 3px rgba(47, 109, 246, 0.14);
  }

  .activation-row {
    display: flex;
    align-items: center;
    gap: 14px;
    flex-wrap: wrap;
  }

  .formula {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: var(--dnn-ink);
  }

  .result {
    font-size: 13px;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
  }

  .verdict {
    margin-top: 9px;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 12px;
    line-height: 1.5;
    color: var(--dnn-muted);
    background: var(--dnn-surface-sunken);
    border-left: 3px solid var(--dnn-border);
  }

  .verdict.dead { border-left-color: var(--dnn-negative); }
  .verdict.live { border-left-color: var(--dnn-positive); }

  .note {
    font-size: 12px;
    line-height: 1.55;
    color: var(--dnn-muted);
    margin: 0;
  }

  .input-edit {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 10px;
  }

  .input-edit input[type='range'] {
    flex: 1;
    accent-color: var(--dnn-accent);
    cursor: pointer;
  }
</style>

{#if detail !== undefined}
  <section class="panel" aria-label="Neuron details">
    <div class="head">
      <div>
        <div class="eyebrow">{layer.label}</div>
        <div class="title">Neuron {index + 1}</div>
      </div>
      <div class="head-actions">
        <span
          class="swatch"
          style="background: {neuronFill(detail.output, $activationRanges[layerIndex] || 1)}"
          aria-hidden="true"
        ></span>
        <button class="close" on:click={clearSelection} aria-label="Close neuron details">×</button>
      </div>
    </div>

    {#if detail.isInput}
      <!-- Input neurons hold a value rather than computing one, so there is no
           weighted sum to show; offer direct editing instead. -->
      <div class="section">
        <div class="section-label">Value</div>
        <p class="note">
          This is an input neuron. It does not compute anything — it holds the
          value you give the network, and passes it to every neuron in the next
          layer.
        </p>
        <div class="input-edit">
          <input
            type="range"
            min="-1"
            max="1"
            step="0.05"
            value={detail.output}
            aria-label={`Input ${index + 1} value`}
            on:input={(e) => setInput(index, parseFloat(e.target.value))}
          />
          <span class="result">{formatValue(detail.output)}</span>
        </div>
      </div>
    {:else}
      <!-- Weighted sum, term by term. -->
      <div class="section">
        <div class="section-label">
          Weighted sum &mdash; {terms.length} incoming connection{terms.length === 1 ? '' : 's'}
        </div>

        <div class="term-head">
          <span>from</span>
          <span>value</span>
          <span>weight</span>
          <span>product</span>
          <span style="text-align:right;">w × x<br /><span style="font-size:8.5px;opacity:0.75;">running</span></span>
        </div>

        <!--
          Each row's weight chip is a button, which is how keyboard users reach
          connection details: the connections themselves are deliberately not in
          the tab order, since hundreds of stops would bury the neurons.
        -->
        <div class="terms">
          {#each terms as term, i}
            <div class="term">
              <button
                class="source-btn"
                on:click={() => inspectSource(term.sourceIndex)}
                title={`Inspect ${layerIndex === 1 ? 'input' : 'neuron'} ${term.sourceIndex + 1}`}
              >{term.sourceLabel}</button>

              <span>{formatValue(term.input)}</span>

              <button
                class="weight-chip"
                style="background: {connectionColor(term.weight, $weightMagnitude)};
                       color: {Math.abs(term.weight) > 0.6 * $weightMagnitude ? '#fff' : 'var(--dnn-ink)'}"
                title="Inspect and edit this weight"
                aria-label={`Weight ${formatValue(term.weight)} from ${term.sourceLabel}, click to edit`}
                on:click={() => inspectConnection(term.sourceIndex)}
              >{formatValue(term.weight)}</button>

              <span class="bar-track">
                <span class="bar-mid"></span>
                <span
                  class="bar-fill"
                  style="
                    {term.product >= 0
                      ? `left: 50%; width: ${(term.product / maxAbsProduct) * 50}%;`
                      : `right: 50%; width: ${(Math.abs(term.product) / maxAbsProduct) * 50}%;`}
                    background: {term.product >= 0 ? 'var(--dnn-positive)' : 'var(--dnn-negative)'};"
                ></span>
              </span>

              <span class="product" class:pos={term.product > 0} class:neg={term.product < 0}>
                {formatSigned(term.product)}
                <!-- Running total, so the sum can be verified line by line
                     rather than only at the end. -->
                <span class="running">{formatValue(runningTotals[i], 2)}</span>
              </span>
            </div>
          {/each}
        </div>

        <div class="sum-rows">
          <div class="sum-row">
            <span>sum of products</span>
            <span>{formatValue(detail.weightedSum, 3)}</span>
          </div>
          <div class="sum-row">
            <span>bias b</span>
            <input
              class="bias-field"
              type="number"
              step="0.05"
              min="-2"
              max="2"
              value={formatValue(detail.bias)}
              aria-label="Bias for this neuron"
              on:input={handleBias}
            />
          </div>
          <div class="sum-row total">
            <span>z</span>
            <span>{formatValue(detail.preActivation, 3)}</span>
          </div>
        </div>
      </div>

      <!-- Activation step. -->
      <div class="section">
        <div class="section-label">Activation</div>

        {#if isSoftmax}
          <p class="note">
            This is an output neuron. Softmax takes the raw score
            <strong>z = {formatValue(detail.preActivation, 3)}</strong> from every
            output neuron at once and turns the set into percentages that sum
            to 100%.
          </p>
          <div class="verdict live">
            This neuron's share: <strong>{(detail.output * 100).toFixed(1)}%</strong>
          </div>
        {:else}
          <div class="activation-row">
            <ActivationGraph
              activationId={detail.activation}
              value={detail.preActivation}
              output={detail.output}
            />
            <div>
              <div class="formula">{activationMeta.formula}</div>
              <div class="result" style="margin-top: 6px;">
                {activationMeta.label}({formatValue(detail.preActivation, 3)})
                = {formatValue(detail.output, 3)}
              </div>
            </div>
          </div>

          {#if detail.activation === RELU && detail.output === 0}
            <div class="verdict dead">
              z is negative, so ReLU clips it to zero. This neuron passes nothing
              to the next layer for the current input.
            </div>
          {:else}
            <div class="verdict live">
              {activationMeta.description}
            </div>
          {/if}
        {/if}
      </div>
    {/if}
  </section>
{/if}
