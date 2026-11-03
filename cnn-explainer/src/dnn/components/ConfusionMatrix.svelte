<script>
  export let confusionMatrix;
  export let classLabels;
  export let perClassMetrics;

  $: maxCount = Math.max(...confusionMatrix.flat(), 1);
  $: numClasses = confusionMatrix.length;
  
  const getIntensity = (count) => {
    if (count === 0) return 0;
    return Math.max(0.15, count / maxCount);
  };

  const formatMetric = (value) => {
    if (value === undefined || isNaN(value)) return '—';
    return (value * 100).toFixed(0) + '%';
  };
</script>

<style>
  .container { background: var(--dnn-surface-sunken); border-radius: 10px; padding: 12px; }
  h3 { font-size: 10px; font-weight: 650; letter-spacing: .09em; text-transform: uppercase; color: var(--dnn-muted); margin: 0 0 8px; }
  
  .matrix-grid {
    display: grid;
    gap: 2px;
    background: var(--dnn-border);
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 12px;
  }
  
  .cell {
    background: var(--dnn-surface);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-variant-numeric: tabular-nums;
    min-height: 32px;
    padding: 4px;
    position: relative;
  }
  
  .header-cell {
    background: var(--dnn-surface-sunken);
    font-weight: 600;
    color: var(--dnn-muted);
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .06em;
  }
  
  .data-cell {
    font-weight: 500;
    color: var(--dnn-ink);
  }
  
  .data-cell.correct {
    background: var(--dnn-accent-wash);
    color: var(--dnn-accent);
    font-weight: 650;
  }
  
  .data-cell.incorrect {
    position: relative;
  }
  
  .data-cell.incorrect::before {
    content: '';
    position: absolute;
    inset: 0;
    background: #fee;
    opacity: var(--intensity);
  }
  
  .cell-value {
    position: relative;
    z-index: 1;
  }
  
  .metrics-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
    font-size: 11px;
    background: var(--dnn-surface);
    border-radius: 6px;
    overflow: hidden;
  }
  
  .metrics-table th {
    background: var(--dnn-surface-sunken);
    color: var(--dnn-muted);
    font-weight: 600;
    text-transform: uppercase;
    font-size: 9px;
    letter-spacing: .06em;
    padding: 6px 8px;
    text-align: left;
  }
  
  .metrics-table td {
    padding: 6px 8px;
    border-top: 1px solid var(--dnn-border);
    color: var(--dnn-ink);
    font-variant-numeric: tabular-nums;
  }
  
  .metrics-table tr:first-child td {
    border-top: none;
  }
  
  .class-name {
    font-weight: 600;
  }
  
  .legend {
    display: flex;
    gap: 12px;
    margin-top: 8px;
    font-size: 10px;
    color: var(--dnn-muted);
    flex-wrap: wrap;
  }
  
  .legend-item {
    display: flex;
    align-items: center;
    gap: 4px;
  }
  
  .legend-box {
    width: 14px;
    height: 14px;
    border-radius: 3px;
  }
  
  .legend-box.correct {
    background: var(--dnn-accent-wash);
  }
  
  .legend-box.incorrect {
    background: #fee;
  }
  
  .axis-label {
    font-size: 9px;
    text-transform: uppercase;
    letter-spacing: .06em;
    color: var(--dnn-muted);
    margin: 4px 0;
    font-weight: 600;
  }
</style>

<div class="container">
  <h3>Confusion Matrix</h3>
  
  <div class="axis-label">Predicted →</div>
  <div
    class="matrix-grid"
    style="grid-template-columns: 80px repeat({numClasses}, 1fr);"
  >
    <!-- Top-left corner -->
    <div class="cell header-cell">Actual ↓</div>
    
    <!-- Column headers (predicted) -->
    {#each classLabels as label, i}
      <div class="cell header-cell" title="Predicted: {label}">{label}</div>
    {/each}
    
    <!-- Data rows -->
    {#each confusionMatrix as row, actualIndex}
      <!-- Row header (actual) -->
      <div class="cell header-cell" title="Actual: {classLabels[actualIndex]}">{classLabels[actualIndex]}</div>
      
      <!-- Data cells -->
      {#each row as count, predictedIndex}
        <div
          class="cell data-cell"
          class:correct={actualIndex === predictedIndex}
          class:incorrect={actualIndex !== predictedIndex && count > 0}
          style="--intensity: {getIntensity(count)}"
          title="{count} examples: actual={classLabels[actualIndex]}, predicted={classLabels[predictedIndex]}"
        >
          <span class="cell-value">{count}</span>
        </div>
      {/each}
    {/each}
  </div>
  
  <div class="legend">
    <div class="legend-item">
      <span class="legend-box correct"></span>
      <span>Correct predictions</span>
    </div>
    <div class="legend-item">
      <span class="legend-box incorrect"></span>
      <span>Errors (darker = more)</span>
    </div>
  </div>
  
  {#if perClassMetrics}
    <h3 style="margin-top: 16px;">Per-Class Metrics</h3>
    <table class="metrics-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>F1</th>
          <th>Support</th>
        </tr>
      </thead>
      <tbody>
        {#each perClassMetrics as metrics, i}
          <tr>
            <td class="class-name">{classLabels[i]}</td>
            <td>{formatMetric(metrics.precision)}</td>
            <td>{formatMetric(metrics.recall)}</td>
            <td>{formatMetric(metrics.f1)}</td>
            <td>{metrics.support}</td>
          </tr>
        {/each}
      </tbody>
    </table>
  {/if}
</div>
