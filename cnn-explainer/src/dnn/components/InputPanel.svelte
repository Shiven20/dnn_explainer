<script>
  /**
   * Editable input vector.
   *
   * Each input gets a slider and a number field bound to the same value: the
   * slider is for exploring, the field for entering something exact. Editing
   * either one re-runs the forward pass immediately, so cause and effect stay
   * obvious.
   */
  import { spec, setInput } from '../stores.js';
  import { activatedNetwork } from '../stores.js';
  import { neuronFill, formatValue } from '../engine/layout.js';

  export let disabled = false;

  const MIN = -1;
  const MAX = 1;
  const STEP = 0.05;

  $: inputs = $spec.inputs.slice(0, $spec.inputSize);
  // Colour each row's chip with the same ramp the diagram uses, so a row and
  // its neuron are recognisably the same thing.
  $: inputRange = Math.max(1, ...inputs.map((v) => Math.abs(v)));

  const clampValue = (value) => {
    if (Number.isNaN(value)) return 0;
    return Math.max(MIN, Math.min(MAX, value));
  };

  const handleSlider = (index, event) => {
    setInput(index, clampValue(parseFloat(event.target.value)));
  };

  const handleNumber = (index, event) => {
    const parsed = parseFloat(event.target.value);
    // Ignore an empty or partial entry mid-typing instead of snapping to 0.
    if (event.target.value === '' || Number.isNaN(parsed)) return;
    setInput(index, clampValue(parsed));
  };

  /** Round-numbered presets, useful for seeing an obvious change. */
  const applyPreset = (kind) => {
    const size = $spec.inputSize;
    for (let i = 0; i < size; i++) {
      if (kind === 'zero') setInput(i, 0);
      else if (kind === 'max') setInput(i, 1);
      else if (kind === 'alternate') setInput(i, i % 2 === 0 ? 1 : -1);
      else if (kind === 'random') {
        setInput(i, Number((Math.random() * 2 - 1).toFixed(2)));
      }
    }
  };
</script>

<style>
  .panel {
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 14px;
    padding: 16px 18px 18px 18px;
  }

  .head {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    gap: 10px;
    margin-bottom: 4px;
  }

  h2 {
    font-size: 11px;
    font-weight: 650;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--dnn-muted);
    margin: 0;
  }

  .hint {
    font-size: 12px;
    color: var(--dnn-muted);
    margin: 0 0 14px 0;
    line-height: 1.5;
  }

  .row {
    display: grid;
    grid-template-columns: 26px 1fr 66px;
    align-items: center;
    gap: 10px;
    padding: 5px 0;
  }

  .chip {
    width: 24px;
    height: 24px;
    border-radius: 7px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 10px;
    font-weight: 600;
    border: 1px solid var(--dnn-border);
    color: var(--dnn-ink);
  }

  input[type='range'] {
    width: 100%;
    accent-color: var(--dnn-accent);
    cursor: pointer;
    margin: 0;
  }

  input[type='number'] {
    width: 100%;
    padding: 5px 7px;
    font-size: 13px;
    font-variant-numeric: tabular-nums;
    text-align: right;
    border: 1px solid var(--dnn-border);
    border-radius: 7px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
  }

  input[type='number']:focus {
    outline: none;
    border-color: var(--dnn-accent);
    box-shadow: 0 0 0 3px rgba(47, 109, 246, 0.14);
  }

  .presets {
    display: flex;
    flex-wrap: wrap;
    gap: 6px;
    margin-top: 14px;
    padding-top: 14px;
    border-top: 1px solid var(--dnn-border);
  }

  button {
    font-size: 12px;
    padding: 5px 10px;
    border: 1px solid var(--dnn-border);
    border-radius: 7px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    cursor: pointer;
    transition: border-color 140ms ease-out, color 140ms ease-out;
  }

  button:hover:not(:disabled) {
    border-color: var(--dnn-accent);
    color: var(--dnn-accent);
  }

  button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  .panel.is-disabled {
    opacity: 0.6;
  }
</style>

<!-- The individual inputs carry their own `disabled`, which is what assistive
     tech reads; this class only handles the visual dimming. -->
<section class="panel" class:is-disabled={disabled}>
  <div class="head">
    <h2>Input values</h2>
  </div>
  <p class="hint">
    These are the numbers entering the network. Change one and watch the effect
    move through every later layer.
  </p>

  {#each inputs as value, i}
    <div class="row">
      <span
        class="chip"
        style="background: {neuronFill(value, inputRange, { signed: true })}"
        aria-hidden="true"
      >x{i + 1}</span>

      <input
        type="range"
        min={MIN}
        max={MAX}
        step={STEP}
        {value}
        {disabled}
        aria-label={`Input ${i + 1}`}
        on:input={(e) => handleSlider(i, e)}
      />

      <input
        type="number"
        min={MIN}
        max={MAX}
        step={STEP}
        value={formatValue(value)}
        {disabled}
        aria-label={`Input ${i + 1} exact value`}
        on:input={(e) => handleNumber(i, e)}
      />
    </div>
  {/each}

  <div class="presets">
    <button {disabled} on:click={() => applyPreset('zero')}>All zero</button>
    <button {disabled} on:click={() => applyPreset('max')}>All one</button>
    <button {disabled} on:click={() => applyPreset('alternate')}>Alternate</button>
    <button {disabled} on:click={() => applyPreset('random')}>Randomize</button>
  </div>
</section>
