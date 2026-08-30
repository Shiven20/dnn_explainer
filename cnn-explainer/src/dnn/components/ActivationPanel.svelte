<script>
  /**
   * Activation function switcher plus an explanation of the current choice.
   *
   * Switching genuinely changes the maths: the store update re-runs the forward
   * pass, so the diagram, every neuron value and the prediction all move. The
   * curve shown is sampled from the same function the engine calls, so it cannot
   * drift from the actual behaviour.
   */
  import { spec, setHiddenActivation } from '../stores.js';
  import {
    activations, hiddenActivationOptions, getActivation, RELU
  } from '../engine/activations.js';
  import { activatedNetwork } from '../stores.js';
  import ActivationGraph from './ActivationGraph.svelte';
  import { formatValue } from '../engine/layout.js';

  $: current = getActivation($spec.hiddenActivation);

  /*
   * Show the function working on a real neuron rather than an abstract axis:
   * the first hidden neuron's z is a number the user can also see in the
   * diagram, which ties the curve to something concrete.
   */
  $: sampleNeuron = (() => {
    const net = $activatedNetwork;
    if (net === undefined || net.layers.length < 2) return undefined;
    return net.layers[1].neurons[0];
  })();

  /** How many hidden units this activation switched off for the current input. */
  $: deadCount = (() => {
    const net = $activatedNetwork;
    if (net === undefined) return 0;
    let count = 0;
    for (let l = 1; l < net.layers.length - 1; l++) {
      net.layers[l].neurons.forEach((n) => {
        if (n.output === 0) count++;
      });
    }
    return count;
  })();

  $: hiddenTotal = (() => {
    const net = $activatedNetwork;
    if (net === undefined) return 0;
    let total = 0;
    for (let l = 1; l < net.layers.length - 1; l++) total += net.layers[l].size;
    return total;
  })();
</script>

<style>
  .panel {
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 14px;
    padding: 15px 16px 16px 16px;
  }

  h2 {
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--dnn-muted);
    margin: 0 0 4px 0;
  }

  .hint {
    font-size: 12px;
    line-height: 1.5;
    color: var(--dnn-muted);
    margin: 0 0 12px 0;
  }

  .options {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 6px;
    margin-bottom: 13px;
  }

  .option {
    border: 1px solid var(--dnn-border);
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    border-radius: 8px;
    padding: 7px 9px;
    font-size: 13px;
    cursor: pointer;
    text-align: left;
    transition: border-color 140ms ease-out, background-color 140ms ease-out;
  }

  .option:hover { border-color: var(--dnn-accent); }

  .option.active {
    border-color: var(--dnn-accent);
    background: var(--dnn-accent-wash, rgba(47, 109, 246, 0.08));
    color: var(--dnn-accent);
    font-weight: 600;
  }

  .formula {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12px;
    color: var(--dnn-ink);
    background: var(--dnn-surface-sunken);
    border-radius: 7px;
    padding: 7px 9px;
    margin-bottom: 11px;
    overflow-x: auto;
    white-space: nowrap;
  }

  .graph-row {
    display: flex;
    gap: 12px;
    align-items: center;
    margin-bottom: 11px;
  }

  .sample {
    font-size: 11.5px;
    color: var(--dnn-muted);
    line-height: 1.6;
    min-width: 0;
  }

  .sample-math {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
    display: block;
    margin-top: 3px;
  }

  .description {
    font-size: 12px;
    line-height: 1.55;
    color: var(--dnn-muted);
    margin: 0 0 10px 0;
  }

  .facts {
    display: flex;
    flex-direction: column;
    gap: 5px;
    font-size: 11.5px;
    color: var(--dnn-muted);
    border-top: 1px solid var(--dnn-border);
    padding-top: 10px;
  }

  .fact {
    display: flex;
    justify-content: space-between;
    gap: 8px;
  }

  .fact-value {
    color: var(--dnn-ink);
    font-variant-numeric: tabular-nums;
  }

  .note {
    margin-top: 10px;
    padding: 7px 9px;
    border-left: 3px solid var(--dnn-accent);
    background: var(--dnn-surface-sunken);
    border-radius: 0 6px 6px 0;
    font-size: 11.5px;
    line-height: 1.5;
    color: var(--dnn-muted);
  }
</style>

<section class="panel" aria-label="Activation function">
  <h2>Activation function</h2>
  <p class="hint">
    Applied to every hidden neuron after its weighted sum. Switching this
    recomputes the whole network.
  </p>

  <div class="options" role="group" aria-label="Choose an activation function">
    {#each hiddenActivationOptions as id}
      <button
        class="option"
        class:active={$spec.hiddenActivation === id}
        aria-pressed={$spec.hiddenActivation === id}
        on:click={() => setHiddenActivation(id)}
      >{activations[id].label}</button>
    {/each}
  </div>

  <div class="formula">{current.formula}</div>

  <div class="graph-row">
    <ActivationGraph
      activationId={current.id}
      value={sampleNeuron ? sampleNeuron.preActivation : 0}
      output={sampleNeuron ? sampleNeuron.output : 0}
      width={118}
      height={88}
    />

    {#if sampleNeuron !== undefined}
      <div class="sample">
        Hidden 1, neuron 1 right now:
        <span class="sample-math">
          z = {formatValue(sampleNeuron.preActivation, 3)}
        </span>
        <span class="sample-math">
          {current.label}(z) = {formatValue(sampleNeuron.output, 3)}
        </span>
      </div>
    {/if}
  </div>

  <p class="description">{current.description}</p>

  <div class="facts">
    <div class="fact">
      <span>Output range</span>
      <span class="fact-value">
        {current.unbounded
          ? `${current.displayRange[0] < 0 ? 'any value' : '0 to ∞'}`
          : `${current.displayRange[0]} to ${current.displayRange[1]}`}
      </span>
    </div>
    <div class="fact">
      <span>Hidden neurons at zero</span>
      <span class="fact-value">{deadCount} of {hiddenTotal}</span>
    </div>
  </div>

  {#if current.id === RELU && deadCount > 0}
    <!-- The dead-unit count is the most visible consequence of choosing ReLU,
         so point at it rather than leaving it as a statistic. -->
    <div class="note">
      {deadCount} hidden neuron{deadCount === 1 ? '' : 's'} received a negative
      sum, so ReLU clipped {deadCount === 1 ? 'it' : 'them'} to zero. Those
      neurons pass nothing forward — look for the dashed red outlines.
    </div>
  {:else if current.id === 'sigmoid'}
    <div class="note">
      Every neuron now sits between 0 and 1, so nothing switches off completely.
      Notice how large sums all flatten toward 1 — that squashing is what makes
      deep sigmoid networks hard to train.
    </div>
  {:else if current.id === 'tanh'}
    <div class="note">
      Outputs now span −1 to 1, so a neuron can push its successors in either
      direction. Negative activations are shown in a contrasting colour.
    </div>
  {:else if current.id === 'linear'}
    <div class="note">
      With no non-linearity, the whole stack collapses into one big weighted sum.
      Two hidden layers now do no more than one could — which is exactly why
      activation functions exist.
    </div>
  {/if}
</section>
