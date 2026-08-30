<script>
  /**
   * Architecture controls: layer counts and widths.
   *
   * Changing the shape rebuilds the network but preserves any weight whose
   * position still exists, so the diagram morphs rather than randomising itself
   * and the user does not lose edits they made deliberately.
   *
   * Limits are enforced in the store, not just disabled here, so the network can
   * never reach a shape the layout cannot draw legibly.
   */
  import {
    spec, LIMITS, setInputSize, setOutputSize, setHiddenLayerCount,
    setNeuronsPerLayer, reseedNetwork, resetAll, parameterCount
  } from '../stores.js';

  export let disabled = false;

  $: hiddenLayers = $spec.hiddenSizes.length;

  /** Shape string, e.g. "4 → 5 → 5 → 2". */
  $: shape = [$spec.inputSize, ...$spec.hiddenSizes, $spec.outputSize].join(' → ');

  const atMin = (value, limit) => value <= limit.min;
  const atMax = (value, limit) => value >= limit.max;
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

  .shape {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
    color: var(--dnn-ink);
    background: var(--dnn-surface-sunken);
    border-radius: 7px;
    padding: 8px 10px;
    margin-bottom: 12px;
    text-align: center;
  }

  .row {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 6px 0;
  }

  .row-label {
    font-size: 12.5px;
    color: var(--dnn-ink);
    min-width: 0;
  }

  .row-sub {
    display: block;
    font-size: 10.5px;
    color: var(--dnn-muted);
  }

  .stepper {
    display: flex;
    align-items: center;
    gap: 2px;
    flex-shrink: 0;
  }

  .stepper button {
    width: 26px;
    height: 26px;
    border: 1px solid var(--dnn-border);
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    border-radius: 6px;
    font-size: 15px;
    line-height: 1;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    transition: border-color 140ms ease-out, color 140ms ease-out;
  }

  .stepper button:hover:not(:disabled) {
    border-color: var(--dnn-accent);
    color: var(--dnn-accent);
  }

  .stepper button:disabled {
    opacity: 0.35;
    cursor: not-allowed;
  }

  .stepper-value {
    min-width: 28px;
    text-align: center;
    font-size: 13.5px;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
  }

  .per-layer {
    border-top: 1px solid var(--dnn-border);
    margin-top: 10px;
    padding-top: 10px;
  }

  .per-layer-title {
    font-size: 10.5px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--dnn-muted);
    margin-bottom: 5px;
  }

  .actions {
    display: flex;
    gap: 6px;
    margin-top: 12px;
    padding-top: 12px;
    border-top: 1px solid var(--dnn-border);
    flex-wrap: wrap;
  }

  .actions button {
    flex: 1;
    min-width: 96px;
    font-size: 12px;
    padding: 6px 10px;
    border: 1px solid var(--dnn-border);
    border-radius: 7px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    cursor: pointer;
  }

  .actions button:hover {
    border-color: var(--dnn-accent);
    color: var(--dnn-accent);
  }

  .params {
    font-size: 11.5px;
    color: var(--dnn-muted);
    text-align: center;
    margin-top: 9px;
  }

  .params strong {
    color: var(--dnn-ink);
    font-variant-numeric: tabular-nums;
  }
</style>

<section class="panel" aria-label="Network architecture">
  <h2>Architecture</h2>
  <p class="hint">
    Add or remove layers and neurons. The diagram recalculates its own layout, and
    weights that still have a place are kept.
  </p>

  <div class="shape" aria-label={`Current shape ${shape}`}>{shape}</div>

  <div class="row">
    <span class="row-label">
      Input neurons
      <span class="row-sub">values you feed in</span>
    </span>
    <div class="stepper">
      <button
        aria-label="Remove an input neuron"
        disabled={disabled || atMin($spec.inputSize, LIMITS.inputNeurons)}
        on:click={() => setInputSize($spec.inputSize - 1)}
      >−</button>
      <span class="stepper-value">{$spec.inputSize}</span>
      <button
        aria-label="Add an input neuron"
        disabled={disabled || atMax($spec.inputSize, LIMITS.inputNeurons)}
        on:click={() => setInputSize($spec.inputSize + 1)}
      >+</button>
    </div>
  </div>

  <div class="row">
    <span class="row-label">
      Hidden layers
      <span class="row-sub">depth of the network</span>
    </span>
    <div class="stepper">
      <button
        aria-label="Remove a hidden layer"
        disabled={disabled || atMin(hiddenLayers, LIMITS.hiddenLayers)}
        on:click={() => setHiddenLayerCount(hiddenLayers - 1)}
      >−</button>
      <span class="stepper-value">{hiddenLayers}</span>
      <button
        aria-label="Add a hidden layer"
        disabled={disabled || atMax(hiddenLayers, LIMITS.hiddenLayers)}
        on:click={() => setHiddenLayerCount(hiddenLayers + 1)}
      >+</button>
    </div>
  </div>

  <div class="row">
    <span class="row-label">
      Output neurons
      <span class="row-sub">one per possible answer</span>
    </span>
    <div class="stepper">
      <button
        aria-label="Remove an output neuron"
        disabled={disabled || atMin($spec.outputSize, LIMITS.outputNeurons)}
        on:click={() => setOutputSize($spec.outputSize - 1)}
      >−</button>
      <span class="stepper-value">{$spec.outputSize}</span>
      <button
        aria-label="Add an output neuron"
        disabled={disabled || atMax($spec.outputSize, LIMITS.outputNeurons)}
        on:click={() => setOutputSize($spec.outputSize + 1)}
      >+</button>
    </div>
  </div>

  <!-- Per-layer widths, so layers can differ rather than all moving together. -->
  <div class="per-layer">
    <div class="per-layer-title">Neurons per hidden layer</div>
    {#each $spec.hiddenSizes as size, i}
      <div class="row">
        <span class="row-label">Hidden layer {i + 1}</span>
        <div class="stepper">
          <button
            aria-label={`Remove a neuron from hidden layer ${i + 1}`}
            disabled={disabled || atMin(size, LIMITS.neuronsPerLayer)}
            on:click={() => setNeuronsPerLayer(size - 1, i)}
          >−</button>
          <span class="stepper-value">{size}</span>
          <button
            aria-label={`Add a neuron to hidden layer ${i + 1}`}
            disabled={disabled || atMax(size, LIMITS.neuronsPerLayer)}
            on:click={() => setNeuronsPerLayer(size + 1, i)}
          >+</button>
        </div>
      </div>
    {/each}
  </div>

  <div class="actions">
    <button on:click={() => reseedNetwork()}>New weights</button>
    <button on:click={resetAll}>Reset all</button>
  </div>

  <p class="params">
    <strong>{$parameterCount}</strong> learnable parameters
  </p>
</section>
