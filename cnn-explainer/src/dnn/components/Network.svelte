<script>
  /**
   * The network diagram: the visual centrepiece.
   *
   * Reads the activated network from the store and draws it as SVG. Layout is
   * computed from the layer sizes, so any architecture the engine supports
   * renders correctly with no hardcoded coordinates.
   *
   * On narrow screens the SVG keeps its natural width and the wrapper scrolls
   * horizontally, rather than squashing the diagram until it is unreadable.
   */
  import { onMount, onDestroy } from 'svelte';

  import Layer from './Layer.svelte';
  import Connection from './Connection.svelte';
  import SignalPulse from './SignalPulse.svelte';

  import {
    activatedNetwork, activationRanges, weightMagnitude,
    selectedNeuron, selectedConnection, hoveredNeuron,
    selectNeuron, selectConnection, clearSelection
  } from '../stores.js';

  import { animation, STATUS, PHASE, easeInOut } from '../animation.js';

  import { computeLayout } from '../engine/layout.js';

  export let showValues = true;

  let container;
  let availableWidth = 900;
  let resizeObserver;

  onMount(() => {
    // ResizeObserver tracks the container itself, which also catches layout
    // changes that a window resize event would miss (e.g. a panel opening).
    if (typeof ResizeObserver !== 'undefined') {
      resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const w = entry.contentRect.width;
          if (w > 0) availableWidth = w;
        }
      });
      resizeObserver.observe(container);
    } else if (container) {
      availableWidth = container.clientWidth;
    }
  });

  onDestroy(() => {
    // Without this the observer keeps a reference to a detached node.
    if (resizeObserver !== undefined) {
      resizeObserver.disconnect();
      resizeObserver = undefined;
    }
  });

  $: net = $activatedNetwork;
  $: layout = net ? computeLayout(net, availableWidth) : undefined;

  /**
   * Connections for the whole network, flattened for rendering.
   * Recomputed only when the network or layout changes, not on hover.
   */
  $: connections = (() => {
    if (net === undefined || layout === undefined) return [];
    const out = [];
    for (let l = 1; l < net.layers.length; l++) {
      const layer = net.layers[l];
      layer.neurons.forEach((neuron) => {
        neuron.weights.forEach((weight, sourceIndex) => {
          out.push({
            id: `c-${l}-${sourceIndex}-${neuron.index}`,
            targetLayerIndex: l,
            sourceIndex,
            targetIndex: neuron.index,
            weight,
            source: layout.positions[l - 1][sourceIndex],
            target: layout.positions[l][neuron.index]
          });
        });
      });
    }
    return out;
  })();

  /** The neuron driving highlight state: an explicit selection beats a hover. */
  $: focus = $selectedNeuron ?? $hoveredNeuron;

  // ---------------------------------------------------------------- animation
  /*
   * The animation reveals results that have already been computed; it never
   * changes them. `anim.revealed` is how far the walkthrough has got, so layers
   * beyond it are drawn as "not yet computed" rather than showing a wrong value.
   */
  $: anim = $animation;
  $: isWalkthrough = anim.status === STATUS.RUNNING || anim.status === STATUS.PAUSED;

  /** Layers past the frontier have not been revealed yet. */
  const isLayerPending = (layerIndex) =>
    isWalkthrough && layerIndex > 0 && layerIndex > anim.revealed;

  /** The layer currently taking its values, used for a brief emphasis. */
  const isLayerSettling = (layerIndex) =>
    isWalkthrough && anim.frontier === layerIndex && anim.phase === PHASE.SETTLE;

  /**
   * Pulses for the hop currently in flight.
   *
   * Only the active layer's connections carry dots, which keeps the count bounded
   * (at most 12x12 = 144) and makes the propagation read as sequential rather
   * than everything moving at once.
   */
  $: pulses = (() => {
    if (!isWalkthrough || anim.phase !== PHASE.TRAVEL || anim.frontier < 1) return [];
    if (net === undefined || layout === undefined) return [];

    const targetLayer = net.layers[anim.frontier];
    if (targetLayer === undefined) return [];

    const sourceNeurons = net.layers[anim.frontier - 1].neurons;
    const eased = easeInOut(anim.progress);
    const out = [];

    targetLayer.neurons.forEach((neuron) => {
      neuron.weights.forEach((weight, sourceIndex) => {
        const sourceOutput = sourceNeurons[sourceIndex].output;
        // A source that contributes nothing sends no dot: showing one would
        // imply information is flowing when none is.
        if (sourceOutput === 0 || weight === 0) return;
        out.push({
          id: `p-${anim.frontier}-${sourceIndex}-${neuron.index}`,
          source: layout.positions[anim.frontier - 1][sourceIndex],
          target: layout.positions[anim.frontier][neuron.index],
          weight,
          contribution: weight * sourceOutput,
          progress: eased
        });
      });
    });

    return out;
  })();

  /** Connections in the hop being animated, so they can be emphasised. */
  const isConnectionActive = (c) =>
    isWalkthrough && anim.phase === PHASE.TRAVEL && c.targetLayerIndex === anim.frontier;

  /**
   * The connection under the pointer.
   *
   * Without this there is no feedback that a line is clickable at all, which is
   * the main reason the weight editor was hard to discover.
   */
  let hoveredConnectionId;

  const connectionAreaMoved = (event) => {
    const target = event.target;
    const attr = target && target.getAttribute && target.getAttribute('data-connection');
    hoveredConnectionId = attr ? attr : undefined;
  };

  const connectionAreaLeft = () => {
    hoveredConnectionId = undefined;
  };

  const connectionKey = (c) => `${c.targetLayerIndex}:${c.sourceIndex}:${c.targetIndex}`;

  const isConnectionHighlighted = (c) => {
    if ($selectedConnection !== undefined) {
      return (
        $selectedConnection.targetLayerIndex === c.targetLayerIndex &&
        $selectedConnection.sourceIndex === c.sourceIndex &&
        $selectedConnection.targetIndex === c.targetIndex
      );
    }
    if (focus === undefined) return false;
    // Highlight connections flowing into or out of the focused neuron.
    if (focus.layerIndex === c.targetLayerIndex && focus.index === c.targetIndex) return true;
    if (focus.layerIndex === c.targetLayerIndex - 1 && focus.index === c.sourceIndex) return true;
    return false;
  };

  const isConnectionDimmed = (c) => {
    // During the walkthrough, dim everything except the hop in flight so the
    // eye follows one step at a time.
    if (isWalkthrough) return !isConnectionActive(c);
    return (focus !== undefined || $selectedConnection !== undefined) &&
      !isConnectionHighlighted(c);
  };

  /**
   * Hit targets are separate, wider, invisible paths layered above the visible
   * strokes. A 1px line is far too thin to click reliably.
   */
  const hitStrokeWidth = 10;

  /**
   * A single delegated handler for every connection, rather than one listener
   * per path. Which connection was hit is read back from a data attribute.
   */
  const connectionAreaClicked = (event) => {
    /*
     * The clicked path carries the attribute itself, so read it directly rather
     * than walking ancestors with closest(): closest() is unreliable on SVG
     * elements in some browsers, and there is no ancestor to find here anyway.
     */
    const target = event.target;
    if (target === undefined || target === null) return;
    const attr = target.getAttribute && target.getAttribute('data-connection');
    if (!attr) return;

    event.stopPropagation();
    const [targetLayerIndex, sourceIndex, targetIndex] = attr.split(':').map(Number);
    selectConnection(targetLayerIndex, sourceIndex, targetIndex);
  };
</script>

<style>
  .network-wrapper {
    width: 100%;
    /* Horizontal scroll preserves legibility on small screens instead of
       compressing the diagram. */
    overflow-x: auto;
    overflow-y: hidden;
    -webkit-overflow-scrolling: touch;
  }

  svg {
    display: block;
    /* Never shrink below the computed layout width; scroll instead. */
    min-width: 100%;
  }

  .backdrop {
    fill: transparent;
  }

  /*
   * Invisible click targets for the connections.
   *
   * `pointer-events: stroke` is essential: the default (visiblePainted) will not
   * reliably hit-test a `transparent` stroke, so without this the clicks are
   * silently swallowed and the connection never opens.
   */
  .hit {
    fill: none;
    stroke: transparent;
    pointer-events: stroke;
    cursor: pointer;
  }

  .flow-band {
    fill: var(--dnn-band, rgba(148, 163, 184, 0.07));
  }

  .scroll-hint {
    font-size: 11px;
    color: var(--dnn-muted, #64748b);
    padding: 4px 2px 0 2px;
    text-align: center;
  }

  /* Tells the user the lines are interactive. Without this the weight editor is
     effectively invisible: nothing suggests a connection can be clicked. */
  .affordance {
    font-size: 12px;
    color: var(--dnn-muted, #64748b);
    text-align: center;
    padding: 7px 2px 0 2px;
    line-height: 1.5;
  }

  .affordance .strong {
    color: var(--dnn-ink, #1f2933);
    font-weight: 600;
  }

  @media (min-width: 900px) {
    .scroll-hint { display: none; }
  }
</style>

<div class="network-wrapper" bind:this={container}>
  {#if net !== undefined && layout !== undefined}
    <svg
      width={layout.width}
      height={layout.height}
      viewBox="0 0 {layout.width} {layout.height}"
      role="group"
      aria-label="Interactive neural network diagram"
    >
      <!-- Clicking empty space clears the current selection. -->
      <rect
        class="backdrop"
        x="0" y="0"
        width={layout.width}
        height={layout.height}
        on:click={clearSelection}
        role="presentation"
      />

      <!-- Faint band behind the hidden layers, separating them from input/output. -->
      {#if net.layers.length > 2}
        <rect
          class="flow-band"
          x={layout.layerX[1] - layout.neuronRadius - 18}
          y={layout.labelY - 16}
          width={layout.layerX[net.layers.length - 2] - layout.layerX[1] +
            2 * layout.neuronRadius + 36}
          height={layout.height - layout.labelY + 4}
          rx="14"
        />
      {/if}

      <!-- Connections sit beneath the neurons. -->
      <g class="connections">
        {#each connections as c (c.id)}
          <Connection
            source={c.source}
            target={c.target}
            weight={c.weight}
            magnitude={$weightMagnitude}
            radius={layout.neuronRadius}
            highlighted={isConnectionHighlighted(c) || isConnectionActive(c) ||
              hoveredConnectionId === connectionKey(c)}
            dimmed={isConnectionDimmed(c) && hoveredConnectionId !== connectionKey(c)}
          />
        {/each}
      </g>

      <!-- Travelling values for the hop currently in flight. -->
      {#if pulses.length > 0}
        <g class="pulses" aria-hidden="true">
          {#each pulses as p (p.id)}
            <SignalPulse
              source={p.source}
              target={p.target}
              weight={p.weight}
              contribution={p.contribution}
              magnitude={$weightMagnitude}
              progress={p.progress}
              radius={layout.neuronRadius}
            />
          {/each}
        </g>
      {/if}

      <!--
        Wider invisible paths make the thin connections clickable.

        These are a pointer affordance only, and are hidden from assistive tech
        on purpose: putting every connection in the tab order would bury the
        neurons under hundreds of stops. Keyboard and screen reader users reach
        the same information through a neuron's incoming-connection list in the
        inspection panel.
      -->
      <g
        class="hit-targets"
        aria-hidden="true"
        on:click={connectionAreaClicked}
        on:mousemove={connectionAreaMoved}
        on:mouseleave={connectionAreaLeft}
      >
        {#each connections as c (`hit-${c.id}`)}
          <path
            class="hit"
            data-connection={connectionKey(c)}
            d={`M ${c.source.x} ${c.source.y} L ${c.target.x} ${c.target.y}`}
            stroke-width={hitStrokeWidth}
          />
        {/each}
      </g>

      <!-- Neurons, drawn last so they sit above the edges. -->
      {#each net.layers as layer, l (layer.index)}
        <Layer
          {layer}
          positions={layout.positions[l]}
          radius={layout.neuronRadius}
          range={$activationRanges[l]}
          labelY={layout.labelY}
          {showValues}
          pending={isLayerPending(l)}
          settling={isLayerSettling(l)}
          selectedNeuron={$selectedNeuron}
          hoveredNeuron={$hoveredNeuron}
          onSelectNeuron={selectNeuron}
          onHoverNeuron={(layerIndex, index) => hoveredNeuron.set({ layerIndex, index })}
          onLeaveNeuron={() => hoveredNeuron.set(undefined)}
        />
      {/each}
    </svg>
  {/if}
</div>

<p class="affordance">
  Click a <span class="strong">neuron</span> to see how its value was computed, or
  click any <span class="strong">connecting line</span> to adjust its weight.
</p>

<div class="scroll-hint">Scroll sideways to see the full network</div>
