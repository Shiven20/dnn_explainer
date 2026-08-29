<script>
  /**
   * Connection inspection with a live-editable weight.
   *
   * Editing the weight recomputes the whole network immediately, which is the
   * clearest demonstration in the app that weights are what determine the
   * output: drag the slider and the prediction moves.
   */
  import {
    activatedNetwork, weightMagnitude, updateWeight, selectNeuron,
    clearSelection, prediction
  } from '../stores.js';

  import { connectionColor, formatValue, formatSigned } from '../engine/layout.js';

  export let targetLayerIndex;
  export let sourceIndex;
  export let targetIndex;

  $: net = $activatedNetwork;
  $: sourceLayer = net ? net.layers[targetLayerIndex - 1] : undefined;
  $: targetLayer = net ? net.layers[targetLayerIndex] : undefined;
  $: sourceNeuron = sourceLayer ? sourceLayer.neurons[sourceIndex] : undefined;
  $: targetNeuron = targetLayer ? targetLayer.neurons[targetIndex] : undefined;
  $: weight = targetNeuron ? targetNeuron.weights[sourceIndex] : 0;

  /*
   * What this connection actually contributes right now. A large weight on a
   * zero-valued source contributes nothing, which is worth making explicit --
   * weight alone does not determine influence.
   */
  $: contribution = sourceNeuron ? weight * sourceNeuron.output : 0;
  $: totalMagnitude = targetNeuron && sourceLayer
    ? targetNeuron.weights.reduce(
        (acc, w, i) => acc + Math.abs(w * sourceLayer.neurons[i].output), 0)
    : 0;
  $: shareOfInput = totalMagnitude > 0
    ? Math.abs(contribution) / totalMagnitude
    : 0;

  const handleWeight = (event) => {
    const parsed = parseFloat(event.target.value);
    if (event.target.value === '' || Number.isNaN(parsed)) return;
    updateWeight(targetLayerIndex, sourceIndex, targetIndex,
      Math.max(-3, Math.min(3, parsed)));
  };

  const nudge = (delta) => {
    updateWeight(targetLayerIndex, sourceIndex, targetIndex,
      Math.max(-3, Math.min(3, Number((weight + delta).toFixed(3)))));
  };

  const zero = () => updateWeight(targetLayerIndex, sourceIndex, targetIndex, 0);
  const flip = () =>
    updateWeight(targetLayerIndex, sourceIndex, targetIndex, Number((-weight).toFixed(3)));

  /*
   * Written without optional chaining. Terser 4.8.1 (pinned by this project)
   * parses `??` but not `?.`, and fails the production build on the latter.
   */
  $: topShare = (() => {
    if ($prediction === undefined) return 0;
    const top = $prediction.classes[$prediction.index];
    return top === undefined ? 0 : top.share;
  })();
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

  .close {
    border: none; background: none; font-size: 18px; line-height: 1;
    color: var(--dnn-muted); cursor: pointer; padding: 0 2px;
  }

  .close:hover { color: var(--dnn-negative); }

  .section {
    padding: 12px 16px;
    border-bottom: 1px solid var(--dnn-border);
  }

  .section:last-child { border-bottom: none; }

  .section-label {
    font-size: 10px; font-weight: 650; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--dnn-muted); margin-bottom: 8px;
  }

  .route {
    display: flex;
    align-items: center;
    gap: 10px;
    font-size: 13px;
  }

  .endpoint {
    border: none;
    background: var(--dnn-surface-sunken);
    border-radius: 8px;
    padding: 7px 10px;
    text-align: left;
    cursor: pointer;
    flex: 1;
    min-width: 0;
    transition: border-color 140ms ease-out;
    border: 1px solid var(--dnn-border);
  }

  .endpoint:hover { border-color: var(--dnn-accent); }

  .endpoint-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dnn-muted);
    display: block;
  }

  .endpoint-name {
    font-size: 13px;
    color: var(--dnn-ink);
    display: block;
    margin-top: 1px;
  }

  .endpoint-value {
    font-size: 11.5px;
    color: var(--dnn-muted);
    font-variant-numeric: tabular-nums;
  }

  .arrow {
    color: var(--dnn-muted);
    font-size: 15px;
    flex-shrink: 0;
  }

  .weight-display {
    display: flex;
    align-items: baseline;
    gap: 9px;
    margin-bottom: 9px;
  }

  .weight-value {
    font-size: 26px;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  .weight-sign {
    font-size: 12px;
    padding: 2px 8px;
    border-radius: 999px;
    color: white;
  }

  .slider-row {
    display: flex;
    align-items: center;
    gap: 9px;
  }

  input[type='range'] {
    flex: 1;
    accent-color: var(--dnn-accent);
    cursor: pointer;
    margin: 0;
  }

  input[type='number'] {
    width: 68px;
    padding: 4px 6px;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
    text-align: right;
    border: 1px solid var(--dnn-border);
    border-radius: 6px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
  }

  input[type='number']:focus {
    outline: none;
    border-color: var(--dnn-accent);
    box-shadow: 0 0 0 3px rgba(47, 109, 246, 0.14);
  }

  .quick {
    display: flex;
    gap: 6px;
    margin-top: 9px;
    flex-wrap: wrap;
  }

  .quick button {
    font-size: 11.5px;
    padding: 4px 9px;
    border: 1px solid var(--dnn-border);
    border-radius: 6px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    cursor: pointer;
  }

  .quick button:hover { border-color: var(--dnn-accent); color: var(--dnn-accent); }

  .math {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 12.5px;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
    line-height: 1.7;
  }

  .math .muted { color: var(--dnn-muted); }

  .share-bar {
    height: 7px;
    border-radius: 999px;
    background: var(--dnn-track);
    overflow: hidden;
    margin-top: 8px;
  }

  .share-fill {
    height: 7px;
    border-radius: 999px;
    background: var(--dnn-accent);
    transition: width 200ms ease-out;
  }

  .note {
    font-size: 12px;
    line-height: 1.55;
    color: var(--dnn-muted);
    margin: 8px 0 0 0;
  }

  .callout {
    font-size: 12px;
    line-height: 1.5;
    color: var(--dnn-muted);
    background: var(--dnn-surface-sunken);
    border-left: 3px solid var(--dnn-border);
    border-radius: 0 6px 6px 0;
    padding: 7px 10px;
    margin-top: 9px;
  }
</style>

{#if targetNeuron !== undefined && sourceNeuron !== undefined}
  <section class="panel" aria-label="Connection details">
    <div class="head">
      <div>
        <div class="eyebrow">Connection</div>
        <div class="title">Weight {formatValue(weight)}</div>
      </div>
      <button class="close" on:click={clearSelection} aria-label="Close connection details">×</button>
    </div>

    <!-- Endpoints, both clickable so the reader can follow the path. -->
    <div class="section">
      <div class="route">
        <button class="endpoint" on:click={() => selectNeuron(targetLayerIndex - 1, sourceIndex)}>
          <span class="endpoint-label">From</span>
          <span class="endpoint-name">{sourceLayer.label} · {sourceIndex + 1}</span>
          <span class="endpoint-value">outputs {formatValue(sourceNeuron.output)}</span>
        </button>

        <span class="arrow" aria-hidden="true">→</span>

        <button class="endpoint" on:click={() => selectNeuron(targetLayerIndex, targetIndex)}>
          <span class="endpoint-label">To</span>
          <span class="endpoint-name">{targetLayer.label} · {targetIndex + 1}</span>
          <span class="endpoint-value">holds {formatValue(targetNeuron.output)}</span>
        </button>
      </div>
    </div>

    <!-- The editable weight. -->
    <div class="section">
      <div class="section-label">Weight</div>

      <div class="weight-display">
        <span class="weight-value">{formatValue(weight, 3)}</span>
        <span
          class="weight-sign"
          style="background: {connectionColor(weight, $weightMagnitude)}"
        >{weight >= 0 ? 'positive' : 'negative'}</span>
      </div>

      <div class="slider-row">
        <input
          type="range"
          min="-3"
          max="3"
          step="0.01"
          value={weight}
          aria-label="Connection weight"
          on:input={handleWeight}
        />
        <input
          type="number"
          min="-3"
          max="3"
          step="0.05"
          value={formatValue(weight, 3)}
          aria-label="Connection weight exact value"
          on:input={handleWeight}
        />
      </div>

      <div class="quick">
        <button on:click={() => nudge(-0.25)}>−0.25</button>
        <button on:click={() => nudge(0.25)}>+0.25</button>
        <button on:click={zero}>Set to 0</button>
        <button on:click={flip}>Flip sign</button>
      </div>

      <p class="note">
        Drag the slider and watch the prediction change. A weight decides how
        strongly the source neuron's value influences the destination.
      </p>
    </div>

    <!-- What it contributes for the current input. -->
    <div class="section">
      <div class="section-label">Contribution right now</div>

      <div class="math">
        <span class="muted">weight</span> {formatValue(weight, 3)}
        <span class="muted">×</span>
        <span class="muted">source</span> {formatValue(sourceNeuron.output, 3)}
        <span class="muted">=</span>
        <strong>{formatSigned(contribution, 3)}</strong>
      </div>

      <div class="share-bar" aria-hidden="true">
        <span class="share-fill" style="width: {shareOfInput * 100}%"></span>
      </div>
      <p class="note">
        {(shareOfInput * 100).toFixed(1)}% of everything arriving at
        {targetLayer.label} · {targetIndex + 1}.
      </p>

      {#if sourceNeuron.output === 0}
        <!-- Worth stating plainly: weight and influence are not the same thing. -->
        <div class="callout">
          The source neuron currently outputs zero, so this connection carries
          nothing regardless of its weight. Change the inputs to wake it up.
        </div>
      {:else if Math.abs(weight) < 0.05}
        <div class="callout">
          This weight is close to zero, so the source barely affects the
          destination — which is how a network learns to ignore an input.
        </div>
      {/if}
    </div>

    {#if $prediction !== undefined}
      <div class="section">
        <div class="section-label">Current prediction</div>
        <div class="math">
          {$prediction.label}
          <span class="muted">at</span>
          {(topShare * 100).toFixed(1)}%
        </div>
      </div>
    {/if}
  </section>
{/if}
