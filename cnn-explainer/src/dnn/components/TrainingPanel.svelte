<script>
  import {
    spec, training, trainingWalkthrough, TRAINING_STATUS,
    WALKTHROUGH_PHASE, WALKTHROUGH_STATUS, setLearningRate,
    trainOneEpoch, startTraining, pauseTraining, resetTraining,
    startTrainingWalkthrough, pauseTrainingWalkthrough,
    resumeTrainingWalkthrough, nextTrainingStage, skipTrainingWalkthrough,
    resetTrainingWalkthrough
  } from '../stores.js';

  const stages = [
    { id: WALKTHROUGH_PHASE.FORWARD, label: 'Forward pass' },
    { id: WALKTHROUGH_PHASE.LOSS, label: 'Loss' },
    { id: WALKTHROUGH_PHASE.BACKPROP, label: 'Backpropagation' },
    { id: WALKTHROUGH_PHASE.UPDATE, label: 'Weight update' }
  ];

  $: running = $training.status === TRAINING_STATUS.RUNNING;
  $: canTrain = $spec.outputSize === 2;
  $: walk = $trainingWalkthrough;
  $: walkthroughOpen = walk.phase !== WALKTHROUGH_PHASE.IDLE;
  $: walkthroughActive = walkthroughOpen && walk.phase !== WALKTHROUGH_PHASE.COMPLETE;
  $: losses = $training.history;
  $: maxLoss = losses.length > 0 ? Math.max(...losses, 0.01) : 1;
  $: points = losses.map((loss, index) => {
    const x = losses.length < 2 ? 0 : index * 180 / (losses.length - 1);
    const y = 44 - Math.min(1, loss / maxLoss) * 40;
    return `${x},${y}`;
  }).join(' ');
  $: activeGradient = walk.backwardLayer === undefined
    ? undefined
    : walk.gradientLayers[walk.backwardLayer];

  const step = () => { if (canTrain) trainOneEpoch(); };
  const stageState = (index) => walk.step > index + 1 ? 'done'
    : walk.step === index + 1 ? 'active' : 'pending';
</script>

<style>
  .panel { background: var(--dnn-surface); border: 1px solid var(--dnn-border); border-radius: 14px; padding: 15px 16px 16px; }
  h2 { font-size: 11px; font-weight: 650; letter-spacing: .09em; text-transform: uppercase; color: var(--dnn-muted); margin: 0 0 4px; }
  .hint, .note { font-size: 12px; line-height: 1.5; color: var(--dnn-muted); margin: 0; }
  .flow { display: grid; gap: 5px; margin: 11px 0; }
  .stage { display: grid; grid-template-columns: 19px 1fr; align-items: center; gap: 7px; font-size: 11.5px; color: var(--dnn-muted); }
  .dot { width: 18px; height: 18px; border-radius: 50%; display: grid; place-items: center; background: var(--dnn-surface-sunken); border: 1px solid var(--dnn-border); font-size: 9px; }
  .stage.active { color: var(--dnn-accent); font-weight: 600; }
  .stage.active .dot { border-color: var(--dnn-accent); background: var(--dnn-accent-wash); }
  .stage.done .dot { border-color: #16a36a; background: #16a36a; color: white; }
  .metrics { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 10px 0; }
  .metric { background: var(--dnn-surface-sunken); border-radius: 7px; padding: 7px; text-align: center; }
  .metric span { display: block; color: var(--dnn-muted); font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  .metric strong { color: var(--dnn-ink); font-size: 13px; font-variant-numeric: tabular-nums; }
  .explain { margin: 9px 0; padding: 9px 10px; border-left: 3px solid var(--dnn-accent); background: var(--dnn-surface-sunken); border-radius: 0 7px 7px 0; font-size: 11.5px; line-height: 1.55; color: var(--dnn-muted); }
  .math { display: block; margin-top: 4px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--dnn-ink); font-variant-numeric: tabular-nums; }
  .updates { margin: 6px 0 0; padding: 0; list-style: none; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 10.5px; }
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
  <h2>Training walkthrough · Phase 6</h2>
  <p class="hint">
    Watch one real epoch progress from prediction to changed weights. Pattern A
    has larger values in the first input half; Pattern B in the second.
  </p>

  <div class="flow" aria-label="Training stages">
    {#each stages as stage, index}
      <div class="stage {stageState(index)}">
        <span class="dot">{stageState(index) === 'done' ? '✓' : index + 1}</span>
        <span>{stage.label}</span>
      </div>
    {/each}
  </div>

  {#if walk.phase === WALKTHROUGH_PHASE.FORWARD}
    <div class="explain">
      The selected training example is flowing left-to-right through the actual network.
      <span class="math">target = {walk.targetLabel}</span>
      <span class="math">prediction = Pattern {walk.prediction === 0 ? 'A' : 'B'} ({(walk.probabilities[walk.prediction] * 100).toFixed(1)}%)</span>
    </div>
  {:else if walk.phase === WALKTHROUGH_PHASE.LOSS}
    <div class="explain">
      Cross-entropy compares the correct class with its predicted probability.
      <span class="math">L = −log(p<sub>correct</sub>) = {walk.sampleLoss.toFixed(4)}</span>
      <span class="math">batch average = {walk.batchLoss.toFixed(4)}</span>
    </div>
  {:else if walk.phase === WALKTHROUGH_PHASE.BACKPROP && activeGradient !== undefined}
    <div class="explain">
      The chain rule is moving right-to-left through layer {walk.backwardLayer}.
      <span class="math">mean |gradient| = {activeGradient.meanAbsolute.toFixed(6)}</span>
      <span class="math">max |gradient| = {activeGradient.maxAbsolute.toFixed(6)}</span>
    </div>
  {:else if walk.phase === WALKTHROUGH_PHASE.UPDATE || walk.phase === WALKTHROUGH_PHASE.COMPLETE}
    <div class="explain">
      Gradient descent applied <span class="math">new = old − learning rate × gradient</span>
      <ul class="updates">
        {#each walk.updates as update}
          <li>L{update.layerIndex} · {update.sourceIndex + 1}→{update.targetIndex + 1}: {update.before.toFixed(4)} → {update.after.toFixed(4)}</li>
        {/each}
      </ul>
    </div>
  {/if}

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
      disabled={walkthroughActive || running}
      on:change={(event) => setLearningRate(event.target.value)} />
  </label>

  <div class="actions">
    {#if !walkthroughOpen}
      <button class="primary" on:click={startTrainingWalkthrough} disabled={!canTrain || running}>Walk through one epoch</button>
      <button on:click={step} disabled={!canTrain || running}>Instant epoch</button>
    {:else if walkthroughActive}
      {#if walk.status === WALKTHROUGH_STATUS.RUNNING}
        <button on:click={pauseTrainingWalkthrough}>Pause</button>
      {:else}
        <button on:click={resumeTrainingWalkthrough}>Resume</button>
      {/if}
      <button class="primary" on:click={nextTrainingStage}>Next stage</button>
      <button on:click={skipTrainingWalkthrough}>Skip to update</button>
      <button on:click={resetTrainingWalkthrough}>Close walkthrough</button>
    {:else}
      <button class="primary" on:click={startTrainingWalkthrough}>Walk through next epoch</button>
      <button on:click={resetTrainingWalkthrough}>Close walkthrough</button>
    {/if}

    {#if running}
      <button class="primary" on:click={pauseTraining}>Pause continuous</button>
    {:else}
      <button on:click={startTraining} disabled={!canTrain || walkthroughActive}>Train continuously</button>
    {/if}
    <button on:click={resetTraining} disabled={$training.initialNetwork === undefined}>Reset all training</button>
  </div>

  {#if !canTrain}
    <p class="warning">The educational task has two classes. Set output neurons to 2 before training.</p>
  {:else}
    <p class="note">The network changes only at stage 4. Reduced-motion mode starts paused so each stage can be advanced without animation.</p>
  {/if}
</section>