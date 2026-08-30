<script>
  /**
   * Horizontal flow of the forward pass: Input → H1 → H2 → Output.
   *
   * The diagram itself shows the propagation, but at a glance it is hard to tell
   * *which* step is underway. This gives the sequence an explicit, readable form
   * that fills in as the pass advances, and doubles as a legend for the stages.
   *
   * Adapts to the current architecture rather than assuming four layers.
   */
  import { animation, STATUS, PHASE } from '../animation.js';

  export let layers = [];   // [{label, size, kind}]

  $: state = $animation;
  $: isWalkthrough = state.status === STATUS.RUNNING || state.status === STATUS.PAUSED;
  $: isFinished = state.status === STATUS.FINISHED;

  /** Short label so the strip stays compact on narrow screens. */
  const shortLabel = (layer, index, total) => {
    if (index === 0) return 'Input';
    if (index === total - 1) return 'Output';
    // "Hidden Layer 2" -> "H2"
    return `H${index}`;
  };

  /**
   * A layer is 'done' once revealed, 'active' while it is being computed, and
   * 'pending' before the pass reaches it. When not animating everything reads as
   * done, because the live values are all valid.
   */
  const stateOf = (index) => {
    if (!isWalkthrough) return isFinished || state.status === STATUS.LIVE ? 'done' : 'pending';
    if (index === 0) return state.frontier === 0 ? 'active' : 'done';
    if (index < state.frontier) return 'done';
    if (index === state.frontier) return 'active';
    return 'pending';
  };

  /** The connector before a layer fills while values travel into it. */
  const connectorFill = (index) => {
    if (!isWalkthrough) return 1;
    if (index < state.frontier) return 1;
    if (index === state.frontier) {
      return state.phase === PHASE.TRAVEL ? state.progress : 1;
    }
    return 0;
  };
</script>

<style>
  .strip {
    display: flex;
    align-items: center;
    gap: 0;
    flex-wrap: nowrap;
    overflow-x: auto;
    padding: 2px 0;
  }

  .stage {
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 3px;
    flex-shrink: 0;
    min-width: 54px;
  }

  .chip {
    display: flex;
    align-items: center;
    justify-content: center;
    padding: 4px 11px;
    border-radius: 999px;
    border: 1px solid var(--dnn-border);
    background: var(--dnn-surface);
    color: var(--dnn-muted);
    font-size: 12px;
    font-weight: 600;
    white-space: nowrap;
    transition: background-color 200ms ease-out, border-color 200ms ease-out,
                color 200ms ease-out;
  }

  .chip.done {
    border-color: var(--dnn-accent);
    color: var(--dnn-accent);
    background: var(--dnn-accent-wash, rgba(47, 109, 246, 0.08));
  }

  .chip.active {
    border-color: var(--dnn-accent);
    background: var(--dnn-accent);
    color: white;
  }

  .size {
    font-size: 10px;
    color: var(--dnn-muted);
    font-variant-numeric: tabular-nums;
  }

  /* The connector is a track with a fill, so travel progress is visible as the
     value physically crossing the gap. */
  .connector {
    position: relative;
    flex: 1;
    min-width: 22px;
    height: 2px;
    background: var(--dnn-track);
    margin: 0 2px;
    margin-bottom: 14px;
  }

  .connector-fill {
    position: absolute;
    inset: 0 auto 0 0;
    background: var(--dnn-accent);
    transition: width 90ms linear;
  }

  .arrow {
    position: absolute;
    right: -1px;
    top: -4px;
    font-size: 9px;
    line-height: 1;
    color: var(--dnn-muted);
  }
</style>

<div class="strip" role="img"
  aria-label={`Forward pass stages: ${layers.map((l, i) => shortLabel(l, i, layers.length)).join(' then ')}`}>
  {#each layers as layer, i}
    {#if i > 0}
      <div class="connector" aria-hidden="true">
        <span class="connector-fill" style="width: {connectorFill(i) * 100}%"></span>
        <span class="arrow">▶</span>
      </div>
    {/if}

    <div class="stage">
      <span class="chip {stateOf(i)}">{shortLabel(layer, i, layers.length)}</span>
      <span class="size">{layer.size}</span>
    </div>
  {/each}
</div>
