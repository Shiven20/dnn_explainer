<script>
  /**
   * Run / Pause / Reset for the forward-pass walkthrough, plus a step readout
   * so the viewer can tell where they are in the sequence.
   */
  import {
    animation, STATUS, PHASE, run, pause, resume, reset, skipToEnd,
    prefersReducedMotion
  } from '../animation.js';

  export let layerCount = 4;
  export let layerLabels = [];

  $: state = $animation;
  $: isRunning = state.status === STATUS.RUNNING;
  $: isPaused = state.status === STATUS.PAUSED;
  $: isActive = isRunning || isPaused;
  $: isFinished = state.status === STATUS.FINISHED;

  // Step 1 is the input layer; each later layer is its own step.
  $: currentStep = isActive || isFinished ? state.frontier : 0;
  $: totalSteps = layerCount - 1;

  $: statusText = (() => {
    if (state.status === STATUS.LIVE) {
      return 'Showing live values. Run the pass to watch them being computed.';
    }
    if (isFinished) {
      return prefersReducedMotion()
        ? 'Result shown without animation, matching your reduced-motion setting.'
        : 'Forward pass complete. The output layer holds the prediction.';
    }
    if (state.frontier === 0) {
      return 'Reading the input values.';
    }
    const target = layerLabels[state.frontier] ?? `layer ${state.frontier}`;
    return state.phase === PHASE.TRAVEL
      ? `Carrying values into ${target}.`
      : `${target} applying weights, bias and activation.`;
  })();

  const handlePrimary = () => {
    if (isRunning) pause();
    else if (isPaused) resume();
    else run(layerCount);
  };
</script>

<style>
  .controls {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 12px;
  }

  .primary {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 18px;
    font-size: 14px;
    font-weight: 600;
    border: none;
    border-radius: 10px;
    background: var(--dnn-accent);
    color: white;
    cursor: pointer;
    transition: filter 140ms ease-out, transform 100ms ease-out;
  }

  .primary:hover { filter: brightness(1.06); }
  .primary:active { transform: translateY(1px); }

  .secondary {
    padding: 9px 14px;
    font-size: 13px;
    border: 1px solid var(--dnn-border);
    border-radius: 9px;
    background: var(--dnn-surface);
    color: var(--dnn-ink);
    cursor: pointer;
    transition: border-color 140ms ease-out, color 140ms ease-out;
  }

  .secondary:hover:not(:disabled) {
    border-color: var(--dnn-accent);
    color: var(--dnn-accent);
  }

  .secondary:disabled {
    opacity: 0.45;
    cursor: not-allowed;
  }

  .progress {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-left: auto;
  }

  .steps {
    display: flex;
    gap: 4px;
  }

  .step-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--dnn-track);
    transition: background-color 200ms ease-out;
  }

  .step-dot.done { background: var(--dnn-accent); }

  .step-count {
    font-size: 12px;
    color: var(--dnn-muted);
    font-variant-numeric: tabular-nums;
    white-space: nowrap;
  }

  .status {
    flex-basis: 100%;
    font-size: 12.5px;
    color: var(--dnn-muted);
    line-height: 1.5;
    min-height: 18px;
  }

  .icon { font-size: 11px; }

  @media (max-width: 620px) {
    .progress { margin-left: 0; width: 100%; }
  }
</style>

<div class="controls">
  <button
    class="primary"
    on:click={handlePrimary}
    aria-label={isRunning ? 'Pause the forward pass' : 'Run the forward pass'}
  >
    <span class="icon" aria-hidden="true">{isRunning ? '❚❚' : '▶'}</span>
    {isRunning ? 'Pause' : isPaused ? 'Resume' : 'Run forward pass'}
  </button>

  <button
    class="secondary"
    disabled={!isActive}
    on:click={skipToEnd}
  >Skip to end</button>

  <button
    class="secondary"
    disabled={state.status === STATUS.LIVE}
    on:click={reset}
  >Reset</button>

  <div class="progress">
    <div class="steps" aria-hidden="true">
      {#each Array(totalSteps) as _, i}
        <span class="step-dot" class:done={currentStep > i}></span>
      {/each}
    </div>
    <span class="step-count">
      {#if isActive || isFinished}
        Step {Math.min(currentStep, totalSteps)} of {totalSteps}
      {:else}
        {totalSteps} steps
      {/if}
    </span>
  </div>

  <!-- Announced politely so screen readers follow the walkthrough without
       interrupting whatever the user is doing. -->
  <p class="status" role="status" aria-live="polite">{statusText}</p>
</div>
