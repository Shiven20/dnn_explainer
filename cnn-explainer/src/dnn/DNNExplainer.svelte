<script>
  /**
   * Phase 2 shell: the network diagram, an editable input vector, and an
   * animated forward pass.
   *
   * The animation reveals a result that has already been computed rather than
   * driving the computation, so the values on screen are always the real ones.
   */
  import { onDestroy } from 'svelte';

  import Network from './components/Network.svelte';
  import InputPanel from './components/InputPanel.svelte';
  import PlaybackControls from './components/PlaybackControls.svelte';
  import NeuronDetails from './components/NeuronDetails.svelte';
  import ConnectionDetails from './components/ConnectionDetails.svelte';

  import {
    activatedNetwork, prediction, parameterCount, weightMagnitude, spec,
    selectedNeuron, selectedConnection, reseedNetwork
  } from './stores.js';

  import {
    animation, STATUS, reset as resetAnimation, destroyAnimation
  } from './animation.js';

  import { formatPercent } from './engine/layout.js';

  let showValues = true;

  $: net = $activatedNetwork;
  $: shape = net ? net.layers.map((l) => l.size).join(' → ') : '';
  $: layerLabels = net ? net.layers.map((l) => l.label) : [];
  $: layerCount = net ? net.layers.length : 0;

  $: anim = $animation;
  $: isWalkthrough = anim.status === STATUS.RUNNING || anim.status === STATUS.PAUSED;
  // The output is only meaningful once propagation has reached it.
  $: outputRevealed = !isWalkthrough || anim.revealed >= layerCount - 1;

  /*
   * Editing an input mid-walkthrough would leave the revealed layers showing
   * values from the previous input, so the animation resets to live instead.
   */
  let lastInputSignature = '';
  $: {
    const signature = $spec.inputs.join(',');
    if (lastInputSignature !== '' && signature !== lastInputSignature && isWalkthrough) {
      resetAnimation();
    }
    lastInputSignature = signature;
  }

  // Release the frame loop when the page unmounts.
  onDestroy(destroyAnimation);
</script>

<style>
  .explainer {
    max-width: 1180px;
    margin: 0 auto;
    padding: 0 24px 48px 24px;
  }

  .intro {
    padding: 40px 0 24px 0;
    max-width: 640px;
  }

  h1 {
    font-size: 30px;
    line-height: 1.18;
    letter-spacing: -0.02em;
    margin: 0 0 10px 0;
    color: var(--dnn-ink);
  }

  .lede {
    font-size: 16px;
    line-height: 1.6;
    color: var(--dnn-muted);
    margin: 0;
  }

  .formula-strip {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    margin-top: 16px;
    padding: 9px 14px;
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 10px;
    font-size: 13.5px;
    color: var(--dnn-ink);
    flex-wrap: wrap;
  }

  .formula-strip code {
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px;
  }

  .formula-strip .arrow { color: var(--dnn-muted); }

  /* The rail has to fit the neuron panel's five-column term table without
     cramping it, which sets the 320px floor. */
  .workspace {
    display: grid;
    grid-template-columns: minmax(0, 1fr) 320px;
    gap: 20px;
    align-items: start;
  }

  .stage {
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 16px;
    box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04),
                0 8px 24px -12px rgba(15, 23, 42, 0.12);
    overflow: hidden;
    min-width: 0;
  }

  .stage-header {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    align-items: center;
    justify-content: space-between;
    padding: 14px 18px;
    border-bottom: 1px solid var(--dnn-border);
  }

  .stage-meta {
    display: flex;
    flex-wrap: wrap;
    gap: 20px;
    align-items: center;
  }

  .meta-item {
    display: flex;
    flex-direction: column;
    gap: 2px;
  }

  .meta-label {
    font-size: 10px;
    font-weight: 600;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: var(--dnn-muted);
  }

  .meta-value {
    font-size: 14px;
    font-variant-numeric: tabular-nums;
    color: var(--dnn-ink);
  }

  .badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    padding: 3px 9px;
    border-radius: 999px;
    background: var(--dnn-surface-sunken);
    border: 1px solid var(--dnn-border);
    font-size: 11px;
    color: var(--dnn-muted);
    cursor: help;
  }

  .controls-bar {
    padding: 14px 18px;
    border-bottom: 1px solid var(--dnn-border);
    background: var(--dnn-surface-sunken);
  }

  .stage-body { padding: 6px 10px 12px 10px; }

  .toggle {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    font-size: 13px;
    color: var(--dnn-ink);
    cursor: pointer;
    user-select: none;
  }

  .toggle input {
    width: 15px;
    height: 15px;
    accent-color: var(--dnn-accent);
    cursor: pointer;
  }

  .legend {
    display: flex;
    flex-wrap: wrap;
    gap: 18px;
    padding: 11px 18px 14px 18px;
    border-top: 1px solid var(--dnn-border);
    font-size: 12px;
    color: var(--dnn-muted);
  }

  .legend-item { display: flex; align-items: center; gap: 7px; }

  .swatch-ramp {
    width: 56px; height: 8px; border-radius: 999px;
    background: linear-gradient(to right, #eef2f7, #2f6df6);
  }

  .swatch-weight {
    width: 56px; height: 8px; border-radius: 999px;
    background: linear-gradient(to right, #e14c42, #cbd5e0, #2f6df6);
  }

  .prediction-strip {
    padding: 15px 18px;
    border-top: 1px solid var(--dnn-border);
    background: var(--dnn-surface-sunken);
  }

  .prediction-head {
    display: flex;
    align-items: baseline;
    gap: 12px;
    flex-wrap: wrap;
  }

  .prediction-label {
    font-size: 10px;
    font-weight: 650;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--dnn-muted);
  }

  .prediction-value {
    font-size: 22px;
    font-weight: 600;
    color: var(--dnn-ink);
  }

  .prediction-value.hidden-value {
    color: var(--dnn-muted);
    font-size: 16px;
    font-weight: 400;
  }

  .prediction-note {
    font-size: 12px;
    color: var(--dnn-muted);
  }

  .classes {
    display: flex;
    flex-direction: column;
    gap: 7px;
    margin-top: 12px;
    max-width: 380px;
  }

  .class-row {
    display: grid;
    grid-template-columns: 74px 1fr 52px;
    align-items: center;
    gap: 10px;
    font-size: 13px;
    color: var(--dnn-muted);
    font-variant-numeric: tabular-nums;
  }

  .class-row.top { color: var(--dnn-ink); font-weight: 600; }

  .class-bar {
    height: 8px;
    border-radius: 999px;
    background: var(--dnn-track);
    overflow: hidden;
  }

  .class-fill {
    height: 8px;
    border-radius: 999px;
    background: var(--dnn-accent);
    transition: width 300ms ease-out;
  }

  .side { display: flex; flex-direction: column; gap: 16px; min-width: 0; }

  .reading-guide {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
    gap: 16px;
    margin-top: 26px;
  }

  .guide-card {
    background: var(--dnn-surface);
    border: 1px solid var(--dnn-border);
    border-radius: 12px;
    padding: 15px 16px;
  }

  .guide-step {
    font-size: 10px; font-weight: 650; letter-spacing: 0.09em;
    text-transform: uppercase; color: var(--dnn-accent); margin-bottom: 5px;
  }

  .guide-title {
    font-size: 14.5px; font-weight: 600; color: var(--dnn-ink); margin-bottom: 5px;
  }

  .guide-text {
    font-size: 13px; line-height: 1.55; color: var(--dnn-muted); margin: 0;
  }

  .next-note {
    margin-top: 24px;
    padding: 13px 16px;
    border-left: 3px solid var(--dnn-accent);
    background: var(--dnn-surface-sunken);
    border-radius: 0 8px 8px 0;
    font-size: 13px;
    line-height: 1.55;
    color: var(--dnn-muted);
  }

  /* Stack the input panel above the diagram once the columns get tight. */
  @media (max-width: 940px) {
    .workspace { grid-template-columns: minmax(0, 1fr); }
    .side { order: -1; }
  }

  @media (max-width: 640px) {
    .explainer { padding: 0 16px 36px 16px; }
    h1 { font-size: 24px; }
    .lede { font-size: 15px; }
    .class-row { grid-template-columns: 62px 1fr 46px; }
  }
</style>

<div class="explainer">
  <div class="intro">
    <h1>How a neural network turns numbers into a decision</h1>
    <p class="lede">
      Every circle is a neuron holding one number. Every line is a weight
      deciding how much of that number travels onward. Change an input, run the
      pass, and watch the arithmetic happen.
    </p>

    <div class="formula-strip">
      <code>inputs × weights + bias</code>
      <span class="arrow" aria-hidden="true">→</span>
      <code>activation</code>
    </div>
  </div>

  <div class="workspace">
    <div class="stage">
      <div class="stage-header">
        <div class="stage-meta">
          <div class="meta-item">
            <span class="meta-label">Architecture</span>
            <span class="meta-value">{shape}</span>
          </div>
          <div class="meta-item">
            <span class="meta-label">Parameters</span>
            <span class="meta-value">{$parameterCount}</span>
          </div>
          <!-- This network has never seen a dataset. The badge is explicit about
               that, because a confident-looking percentage otherwise implies a
               competence the model does not have. -->
          <span
            class="badge"
            title="This network has never been trained on any data. Its weights are random numbers, so the outputs are arithmetic on arbitrary values — not recognition of anything."
          >
            Untrained · random weights
          </span>
        </div>

        <label class="toggle">
          <input type="checkbox" bind:checked={showValues} />
          Show values
        </label>
      </div>

      <div class="controls-bar">
        <PlaybackControls {layerCount} {layerLabels} />
      </div>

      <div class="stage-body">
        <Network {showValues} />
      </div>

      <div class="legend">
        <div class="legend-item">
          <span>Activation</span>
          <span class="swatch-ramp"></span>
          <span>low → high</span>
        </div>
        <div class="legend-item">
          <span>Weight</span>
          <span class="swatch-weight"></span>
          <span>negative → positive</span>
        </div>
        <div class="legend-item">
          <svg width="24" height="12" aria-hidden="true">
            <circle cx="8" cy="6" r="5" fill="none" stroke="#e14c42"
              stroke-width="1.2" stroke-dasharray="3 2.5" />
          </svg>
          <span>ReLU output is zero</span>
        </div>
      </div>

      {#if $prediction !== undefined}
        <div class="prediction-strip">
          <div class="prediction-head">
            <span class="prediction-label">Prediction</span>
            {#if outputRevealed}
              <span class="prediction-value">{$prediction.label}</span>
              <span class="prediction-note">
                simply the largest of the {$prediction.classes.length} output
                values — it carries no meaning until the network is trained
              </span>
            {:else}
              <!-- Withheld until propagation reaches the output layer, so the
                   walkthrough cannot spoil its own conclusion. -->
              <span class="prediction-value hidden-value">
                waiting for the signal to reach the output layer…
              </span>
            {/if}
          </div>

          {#if outputRevealed}
            <div class="classes">
              {#each $prediction.classes as cls (cls.index)}
                <div class="class-row" class:top={cls.index === $prediction.index}>
                  <span>{cls.label}</span>
                  <span class="class-bar">
                    <span class="class-fill" style="width: {cls.share * 100}%"></span>
                  </span>
                  <span>{formatPercent(cls.share)}</span>
                </div>
              {/each}
            </div>
          {/if}
        </div>
      {/if}
    </div>

    <div class="side">
      <!--
        The inspection panel replaces the input editor while something is
        selected. They compete for the same column, and showing both at once in
        a 270px rail makes each of them cramped.
      -->
      {#if $selectedNeuron !== undefined}
        <NeuronDetails
          layerIndex={$selectedNeuron.layerIndex}
          index={$selectedNeuron.index}
        />
      {:else if $selectedConnection !== undefined}
        <ConnectionDetails
          targetLayerIndex={$selectedConnection.targetLayerIndex}
          sourceIndex={$selectedConnection.sourceIndex}
          targetIndex={$selectedConnection.targetIndex}
        />
      {:else}
        <InputPanel />
      {/if}
    </div>
  </div>

  <div class="reading-guide">
    <div class="guide-card">
      <div class="guide-step">Step 1</div>
      <div class="guide-title">Neurons hold values</div>
      <p class="guide-text">
        The input layer holds the numbers you set. Every other neuron holds a
        value it computed. Darker fill means a larger value.
      </p>
    </div>
    <div class="guide-card">
      <div class="guide-step">Step 2</div>
      <div class="guide-title">Weights carry influence</div>
      <p class="guide-text">
        Each line multiplies the value it carries by its weight. Blue adds to the
        destination, red subtracts, and thickness shows how much.
      </p>
    </div>
    <div class="guide-card">
      <div class="guide-step">Step 3</div>
      <div class="guide-title">Each neuron sums, then activates</div>
      <p class="guide-text">
        Add up every incoming product, add the bias, then pass the result through
        the activation function. That single number is the neuron's output.
      </p>
    </div>
    <div class="guide-card">
      <div class="guide-step">Step 4</div>
      <div class="guide-title">Output decides</div>
      <p class="guide-text">
        The final layer produces one value per output. Softmax turns those into
        percentages, and the largest becomes the prediction. In a trained network
        each output would correspond to a real category; here they are just
        positions, since nothing has been learned yet.
      </p>
    </div>
  </div>

  <p class="next-note">
    Click any neuron to see exactly how its value was computed, or click a
    connection to inspect and edit its weight. The activation switcher and
    architecture controls arrive in the next phase.
  </p>
</div>
