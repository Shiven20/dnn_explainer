<script>
  import { customDataset, loadDataset, useCustomDataset, useSyntheticDataset, training, TRAINING_STATUS } from '../stores.js';

  let fileInput;
  let uploading = false;
  let splitRatio = 0.8;

  $: dataset = $customDataset;
  $: hasDataset = dataset.status === 'valid';
  $: trainingActive = $training.status === TRAINING_STATUS.RUNNING;

  const handleFileSelect = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    uploading = true;
    try {
      const text = await file.text();
      loadDataset(text, splitRatio);
    } catch (err) {
      console.error('File read error:', err);
    } finally {
      uploading = false;
    }
  };

  const handleSplitChange = (event) => {
    splitRatio = Number(event.target.value);
    if (hasDataset && dataset.csvText) {
      loadDataset(dataset.csvText, splitRatio);
    }
  };

  const selectFile = () => fileInput.click();
  const clear = () => {
    useSyntheticDataset();
    if (fileInput) fileInput.value = '';
  };
</script>

<style>
  .panel { background: var(--dnn-surface); border: 1px solid var(--dnn-border); border-radius: 14px; padding: 15px 16px 16px; }
  h2 { font-size: 11px; font-weight: 650; letter-spacing: .09em; text-transform: uppercase; color: var(--dnn-muted); margin: 0 0 4px; }
  .hint { font-size: 12px; line-height: 1.5; color: var(--dnn-muted); margin: 0 0 10px; }
  .controls { display: grid; gap: 8px; }
  .row { display: flex; align-items: center; gap: 8px; }
  button { border: 1px solid var(--dnn-border); border-radius: 7px; background: var(--dnn-surface); color: var(--dnn-ink); padding: 7px 10px; cursor: pointer; font-size: 12px; }
  button.primary { background: var(--dnn-accent); border-color: var(--dnn-accent); color: white; }
  button:disabled { opacity: .4; cursor: not-allowed; }
  .hidden { display: none; }
  .status { padding: 9px 10px; border-radius: 7px; font-size: 11.5px; line-height: 1.5; }
  .status.error { background: #fee; border: 1px solid #fcc; color: #c33; }
  .status.success { background: #efe; border: 1px solid #cfc; color: #363; }
  .info { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin: 8px 0; }
  .metric { background: var(--dnn-surface-sunken); border-radius: 7px; padding: 7px; text-align: center; }
  .metric span { display: block; color: var(--dnn-muted); font-size: 9px; text-transform: uppercase; letter-spacing: .06em; }
  .metric strong { color: var(--dnn-ink); font-size: 13px; font-variant-numeric: tabular-nums; }
  .classes { margin: 8px 0; padding: 8px; background: var(--dnn-surface-sunken); border-radius: 7px; }
  .classes-label { font-size: 9px; text-transform: uppercase; letter-spacing: .06em; color: var(--dnn-muted); margin-bottom: 4px; }
  .class-list { display: flex; flex-wrap: wrap; gap: 4px; }
  .class-tag { display: inline-block; padding: 2px 6px; background: var(--dnn-accent-wash); color: var(--dnn-accent); border-radius: 4px; font-size: 11px; }
  label { display: flex; align-items: center; justify-content: space-between; gap: 8px; color: var(--dnn-ink); font-size: 12px; }
  input[type="range"] { flex: 1; }
  .range-value { font-variant-numeric: tabular-nums; min-width: 32px; text-align: right; }
</style>

<section class="panel" aria-label="Custom dataset upload">
  <h2>Custom dataset · Phase 7a</h2>
  <p class="hint">
    Upload a CSV file with numeric features and a class column. The last column should be the target class.
  </p>

  <input type="file" accept=".csv" class="hidden" bind:this={fileInput} on:change={handleFileSelect} />

  <div class="controls">
    {#if uploading}
      <div class="status">Loading...</div>
    {:else if dataset.status === 'error'}
      <div class="status error">{dataset.error}</div>
      <button on:click={selectFile}>Try another file</button>
    {:else if hasDataset}
      <div class="status success">
        Dataset loaded: {dataset.size.total} examples, {dataset.classes.length} classes
      </div>

      <div class="info">
        <div class="metric"><span>Total</span><strong>{dataset.size.total}</strong></div>
        <div class="metric"><span>Train</span><strong>{dataset.size.train}</strong></div>
        <div class="metric"><span>Test</span><strong>{dataset.size.test}</strong></div>
      </div>

      {#if dataset.classes && dataset.classes.length > 0}
        <div class="classes">
          <div class="classes-label">Classes</div>
          <div class="class-list">
            {#each dataset.classes as className}
              <span class="class-tag">{className}</span>
            {/each}
          </div>
        </div>
      {/if}

      <label>
        Train/test split
        <input type="range" min="0.5" max="0.9" step="0.05" value={splitRatio}
          disabled={trainingActive}
          on:input={handleSplitChange} />
        <span class="range-value">{Math.round(splitRatio * 100)}%</span>
      </label>

      <div class="row">
        {#if dataset.active}
          <button class="primary" disabled>Using custom dataset</button>
          <button on:click={useSyntheticDataset} disabled={trainingActive}>Switch to synthetic</button>
        {:else}
          <button class="primary" on:click={useCustomDataset} disabled={trainingActive}>Use for training</button>
          <button on:click={clear}>Clear</button>
        {/if}
      </div>
    {:else}
      <button on:click={selectFile}>Choose CSV file</button>
    {/if}
  </div>
</section>
