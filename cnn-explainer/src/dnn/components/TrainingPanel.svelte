<script>
  import {
    spec, training, TRAINING_STATUS, setLearningRate,
    trainOneEpoch, startTraining, pauseTraining, resetTraining
  } from '../stores.js';

  $: running = $training.status === TRAINING_STATUS.RUNNING;
  $: canTrain = $spec.outputSize === 2;
  $: losses = $training.history;
  $: maxLoss = losses.length > 0 ? Math.max(...losses, 0.01) : 1;
  $: points = losses.map((loss, index) => {
    const x = losses.length < 2 ? 0 : index * 180 / (losses.length - 1);
    const y = 44 - Math.min(1, loss / maxLoss) * 40;
    return `${x},${y}`;
  }).join(' ');

  const step = () => {
    if (!canTrain) return;
    trainOneEpoch();
  };
</script>

<style>
  .panel { background: var(--dnn-surface); border: 1px solid var(--dnn-border); border-radius: 14px; padding: 15px 16px 16px; }
  h2 { font-size: 11px; font-weight: 650; letter-spacing: .09em; text-transform: uppercase; color: var(--dnn-muted); margin: 0 0 4px; }
  .hint, .note { font-size: 12px; line-height: 1.5; color: var(--dnn-muted); margin: 0; }
  .flow { margin: 11px 0; padding: 8px; border-radius: 7px; background: var(--dnn-surface-sunken); color: var(--dnn-ink); font-size: 10.5px; text-align: center; line-height: 1.5; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 10px 0; }
  .metric { background: var(--dnn-surface-sunken); border-radius: 7px; padding: 7px; text-align: center; }
  .metric span { display: block; color: var(--dnn-muted); font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  .metric strong { color: var(--dnn-ink); font-size: 13px; font-variant-numeric: tabular-nums; }
  .rate { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--dnn-ink); font-size: 12px; }
  .rate input { width: 82px; border: 1px solid var(--dnn-border); border-radius: 6px; background: var(--dnn-surface); color: var(--dnn-ink); padding: 4px 6px; }
  .actions { display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-top: 10px; }
  button { border: 1px solid var(--dnn-border); border-radius: 7px; background: var(--dnn-surface); color: var(--dnn-ink); padding: 7px; cursor: pointer; }
  button.primary { background: var(--dnn-accent); border-color: var(--dnn-accent); color: white; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .chart { width: 100%; height: 48px; margin-top: 8px; }
  .warning { margin-top: 9px; color: var(--dnn-negative); font-size: 11.5px; }
</style>

<section class="panel" aria-label="Training and backpropagation">
  <h2>Training · Phase 5</h2>
  <p class="hint">
    Learn a real synthetic task: <strong>Pattern A</strong> has larger values in
    the first half of its inputs; <strong>Pattern B</strong> in the second half.
  </p>

  <div class="flow">
    Input → Forward pass → Prediction → Cross-entropy loss<br />
    → Backpropagation → Weight update
  </div>

  <div class="metrics">
    <div class="metric"><span>Epoch</span><strong>{$training.epoch}</strong></div>
    <div class="metric"><span>Loss</span><strong>{$training.loss === undefined ? '—' : $training.loss.toFixed(3)}</strong></div>
    <div class="metric"><span>Accuracy</span><strong>{$training.accuracy === undefined ? '—' : `${($training.accuracy * 100).toFixed(0)}%`}</strong></div>
  </div>

  {#if losses.length > 1}
    <svg class="chart" viewBox="0 0 180 48" role="img" aria-label="Loss history, lower is better">
      <line x1="0" y1="44" x2="180" y2="44" stroke="var(--dnn-border)" />
      <polyline points={points} fill="none" stroke="var(--dnn-accent)" stroke-width="2" />
    </svg>
  {/if}

  <label class="rate">
    Learning rate
    <input type="number" min="0.001" max="1" step="0.01" value={$training.learningRate}
      on:change={(event) => setLearningRate(event.target.value)} />
  </label>

  <div class="actions">
    <button on:click={step} disabled={!canTrain || running}>Train one epoch</button>
    {#if running}
      <button class="primary" on:click={pauseTraining}>Pause</button>
    {:else}
      <button class="primary" on:click={startTraining} disabled={!canTrain}>Train continuously</button>
    {/if}
    <button on:click={resetTraining} disabled={$training.initialNetwork === undefined}>Reset training</button>
  </div>

  {#if !canTrain}
    <p class="warning">The educational dataset has two classes. Set output neurons to 2 before training.</p>
  {:else}
    <p class="note">Every update changes the same weights drawn as lines above. Reset restores the exact pre-training network.</p>
  {/if}
</section>