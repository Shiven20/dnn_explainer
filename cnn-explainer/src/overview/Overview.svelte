<script>
  import { onMount } from 'svelte';

  import {
    dnnStore, modelInfoStore, neuronCoordinateStore, layerRangesStore,
    weightRangeStore, inputGridStore, predictionStore, selectedNeuronStore,
    hoveredNeuronStore, detailedModeStore, isInSoftmaxStore, modalStore
  } from '../stores.js';

  import {
    constructDNN, forwardPass, getPrediction, getLayerRanges, getWeightRange,
    flattenGrid, emptyGrid, noisyGrid, activationType
  } from '../utils/dnn.js';

  import {
    computeLayout, activationColor, weightColor, weightStrokeWidth, edgePath, fmt, fmtPercent
  } from './dnn-layout.js';

  import { overviewConfig } from '../config.js';

  import WeightedSumView from '../detail-view/WeightedSumView.svelte';
  import ReluView from '../detail-view/ReluView.svelte';
  import SoftmaxView from '../detail-view/Softmaxview.svelte';
  import Modal from './Modal.svelte';
  import Article from '../article/Article.svelte';

  const {
    neuronRadius, numLayers, layerNames, layerDisplayNames, layerSubtitles,
    edgeOpacity, edgeOpacityFaded, edgeInitColor, edgeHoverColor,
    edgeStrokeWidthHover, svgPaddings, classLists
  } = overviewConfig;

  // ---------------------------------------------------------------- state
  let network = undefined;
  let modelInfo = undefined;
  let loadError = undefined;

  let width = 1000;
  let height = 420;
  let layout = undefined;

  let inputGrid = emptyGrid(5);
  let layerRanges = [];
  let weightRange = 1;
  let prediction = -1;

  let selectedNeuron = undefined;   // {layerIndex, index}
  let hoveredNeuron = undefined;
  let hoveredEdge = undefined;      // {layerIndex, sourceIndex, destIndex}
  let isPainting = false;
  let paintValue = 1;
  let detailedMode = true;
  let isInSoftmax = false;
  let modalInfo = { show: false };
  let selectedPreset = 'L';

  detailedModeStore.subscribe((v) => { detailedMode = v; });
  isInSoftmaxStore.subscribe((v) => { isInSoftmax = v; });
  modalStore.subscribe((v) => { modalInfo = v; });

  // ---------------------------------------------------------------- setup
  onMount(async () => {
    try {
      const loaded = await constructDNN();
      network = loaded.network;
      modelInfo = loaded.model;
      modelInfoStore.set(modelInfo);

      // Start from a clean prototype so the first thing the user sees is a
      // confident, easy-to-read prediction.
      inputGrid = modelInfo.prototypes[selectedPreset].map((r) => [...r]);
      runNetwork();
      handleResize();
    } catch (error) {
      console.error('Failed to initialize DNN Explainer', error);
      loadError = error.message;
    }
  });

  /** Recompute activations, prediction and colour ranges for the current input. */
  const runNetwork = () => {
    if (network === undefined) return;
    forwardPass(network, flattenGrid(inputGrid));
    prediction = getPrediction(network);
    layerRanges = getLayerRanges(network);
    weightRange = getWeightRange(network);

    // Republish so the detail views react.
    network = network;
    dnnStore.set(network);
    inputGridStore.set(inputGrid);
    predictionStore.set(prediction);
    layerRangesStore.set(layerRanges);
    weightRangeStore.set(weightRange);
  };

  const handleResize = () => {
    const container = document.querySelector('#dnn-svg-container');
    if (container === null || network === undefined) return;
    width = Math.max(760, container.clientWidth);
    layout = computeLayout(network, width, height);
    neuronCoordinateStore.set(layout.coordinates);
  };

  // ---------------------------------------------------------------- input editing
  const setCell = (row, col, value) => {
    if (inputGrid[row][col] === value) return;
    inputGrid[row][col] = value;
    inputGrid = [...inputGrid];
    selectedPreset = 'custom';
    runNetwork();
  };

  const startPaint = (row, col) => {
    isPainting = true;
    // Painting toggles based on the first cell touched, which feels like a
    // normal drawing tool rather than always setting ink.
    paintValue = inputGrid[row][col] === 1 ? 0 : 1;
    setCell(row, col, paintValue);
  };

  const continuePaint = (row, col) => {
    if (!isPainting) return;
    setCell(row, col, paintValue);
  };

  const stopPaint = () => { isPainting = false; };

  const loadPreset = (name) => {
    selectedPreset = name;
    inputGrid = modelInfo.prototypes[name].map((r) => [...r]);
    runNetwork();
  };

  const clearInput = () => {
    selectedPreset = 'custom';
    inputGrid = emptyGrid(5);
    runNetwork();
  };

  const addNoise = () => {
    selectedPreset = 'custom';
    inputGrid = noisyGrid(inputGrid, 0.12);
    runNetwork();
  };

  // ---------------------------------------------------------------- interaction
  const neuronClicked = (layerIndex, index) => {
    // Clicking the same neuron closes the detail view.
    if (selectedNeuron !== undefined &&
        selectedNeuron.layerIndex === layerIndex &&
        selectedNeuron.index === index) {
      closeDetailView();
      return;
    }
    selectedNeuron = { layerIndex, index };
    selectedNeuronStore.set(selectedNeuron);
    isInSoftmaxStore.set(false);
  };

  const closeDetailView = () => {
    selectedNeuron = undefined;
    selectedNeuronStore.set(undefined);
    isInSoftmaxStore.set(false);
  };

  const neuronMouseOver = (layerIndex, index) => {
    hoveredNeuron = { layerIndex, index };
    hoveredNeuronStore.set(hoveredNeuron);
  };

  const neuronMouseLeave = () => {
    hoveredNeuron = undefined;
    hoveredNeuronStore.set(undefined);
  };

  const openSoftmaxView = () => {
    isInSoftmaxStore.set(!isInSoftmax);
    selectedNeuron = undefined;
    selectedNeuronStore.set(undefined);
  };

  const showModal = (key) => {
    modalStore.set({ show: true, key });
  };

  // ---------------------------------------------------------------- derived
  /** Edges are only drawn for the hovered/selected neuron plus a faint backdrop. */
  const isEdgeHighlighted = (layerIndex, sourceIndex, destIndex) => {
    const focus = selectedNeuron || hoveredNeuron;
    if (focus === undefined) return false;
    if (focus.layerIndex === layerIndex && focus.index === destIndex) return true;
    if (focus.layerIndex === layerIndex - 1 && focus.index === sourceIndex) return true;
    return false;
  };

  const hasFocus = () => (selectedNeuron || hoveredNeuron) !== undefined;

  const isNeuronDimmed = (layerIndex, index) => {
    const focus = selectedNeuron || hoveredNeuron;
    if (focus === undefined) return false;
    // Keep the focused neuron and the layers it connects to fully visible.
    if (focus.layerIndex === layerIndex && focus.index === index) return false;
    if (Math.abs(focus.layerIndex - layerIndex) === 1) return false;
    return focus.layerIndex !== layerIndex;
  };

  $: selectedNeuronData =
    network !== undefined && selectedNeuron !== undefined
      ? network[selectedNeuron.layerIndex][selectedNeuron.index]
      : undefined;

  $: outputLayer = network !== undefined ? network[network.length - 1] : [];
  $: confidence = prediction >= 0 && outputLayer.length > 0 ? outputLayer[prediction].output : 0;
</script>

<svelte:window on:resize={handleResize} on:mouseup={stopPaint} />

<style>
  #overview {
    display: flex;
    flex-direction: column;
    align-items: center;
    width: 100%;
  }

  .control-panel {
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: center;
    gap: 22px;
    padding: 16px 20px 6px 20px;
    max-width: 1100px;
  }

  .control-group {
    display: flex;
    flex-direction: column;
    gap: 6px;
  }

  .control-label {
    font-size: 11px;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--deep-gray);
  }

  .control-row {
    display: flex;
    align-items: center;
    gap: 8px;
  }

  button.preset {
    border: 1px solid var(--middle-gray);
    background: white;
    border-radius: 5px;
    padding: 5px 11px;
    font-size: 13px;
    cursor: pointer;
    transition: all 120ms ease-out;
  }

  button.preset:hover {
    border-color: var(--deep-gray);
    background: rgb(248, 248, 248);
  }

  button.preset.selected {
    border-color: var(--blue);
    color: var(--blue);
    background: rgb(240, 247, 255);
  }

  .prediction-box {
    display: flex;
    flex-direction: column;
    align-items: flex-start;
    padding: 8px 16px;
    border-left: 3px solid var(--blue);
    background: rgb(248, 250, 253);
    min-width: 170px;
  }

  .prediction-label {
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--deep-gray);
  }

  .prediction-value {
    font-size: 20px;
    font-weight: 600;
  }

  .prediction-conf {
    font-size: 12px;
    color: var(--deep-gray);
  }

  .toggle {
    display: flex;
    align-items: center;
    gap: 6px;
    font-size: 13px;
    cursor: pointer;
    user-select: none;
  }

  #dnn-svg-container {
    width: 100%;
    max-width: 1150px;
    position: relative;
  }

  svg#dnn-svg {
    width: 100%;
    overflow: visible;
  }

  .layer-title {
    font-size: 13px;
    font-weight: 600;
    text-anchor: middle;
    fill: rgb(60, 60, 60);
  }

  .layer-subtitle {
    font-size: 10.5px;
    text-anchor: middle;
    fill: var(--deep-gray);
  }

  .neuron {
    cursor: pointer;
  }

  .neuron-outline {
    fill: none;
    stroke: rgb(120, 120, 120);
    stroke-width: 1;
  }

  .neuron-outline.selected {
    stroke: var(--blue);
    stroke-width: 2.5;
  }

  .neuron-outline.dead {
    stroke-dasharray: 3 2;
    stroke: var(--red);
  }

  .neuron-label {
    font-size: 9.5px;
    text-anchor: middle;
    dominant-baseline: middle;
    pointer-events: none;
  }

  .input-cell {
    cursor: crosshair;
    stroke: rgb(210, 210, 210);
    stroke-width: 0.6;
  }

  .class-label {
    font-size: 12px;
    dominant-baseline: middle;
    fill: rgb(60, 60, 60);
  }

  .class-label.predicted {
    font-weight: 700;
    fill: var(--blue);
  }

  .prob-bar-bg {
    fill: rgb(238, 238, 238);
  }

  .softmax-button {
    cursor: pointer;
  }

  .softmax-button rect {
    fill: white;
    stroke: var(--middle-gray);
    rx: 4;
  }

  .softmax-button:hover rect {
    stroke: var(--blue);
  }

  .softmax-button text {
    font-size: 11px;
    text-anchor: middle;
    dominant-baseline: middle;
    fill: rgb(70, 70, 70);
    pointer-events: none;
  }

  .hint {
    font-size: 12.5px;
    color: var(--deep-gray);
    padding: 2px 20px 10px 20px;
    text-align: center;
    max-width: 720px;
  }

  .legend-row {
    display: flex;
    gap: 26px;
    justify-content: center;
    flex-wrap: wrap;
    font-size: 11.5px;
    color: var(--deep-gray);
    padding-bottom: 8px;
  }

  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }

  .gradient-strip {
    width: 74px;
    height: 9px;
    border-radius: 2px;
  }

  .detail-anchor {
    position: absolute;
    z-index: 10;
  }

  .error-box {
    margin: 40px;
    padding: 16px 20px;
    border-left: 3px solid var(--red);
    background: rgb(255, 248, 248);
    font-size: 14px;
  }

  .info-icon {
    cursor: pointer;
    fill: var(--dark-gray);
  }

  .info-icon:hover {
    fill: var(--blue);
  }

  .loading {
    padding: 60px;
    color: var(--deep-gray);
  }
</style>

<div id="overview">
  {#if loadError !== undefined}
    <div class="error-box">
      <strong>Could not load the model.</strong>
      <div>{loadError}</div>
      <div style="margin-top:6px; font-size:13px;">
        Run <code>node dnn-mlp/train.js</code> to generate
        <code>public/assets/data/dnn_model.json</code>, then reload.
      </div>
    </div>
  {:else if network === undefined}
    <div class="loading">Loading model...</div>
  {:else}

    <!-- ------------------------------------------------ controls -->
    <div class="control-panel">
      <div class="control-group">
        <span class="control-label">Example glyphs</span>
        <div class="control-row">
          {#each Object.keys(modelInfo.prototypes) as name}
            <button
              class="preset"
              class:selected={selectedPreset === name}
              on:click={() => loadPreset(name)}
              aria-pressed={selectedPreset === name}
            >{name}</button>
          {/each}
        </div>
      </div>

      <div class="control-group">
        <span class="control-label">Edit input</span>
        <div class="control-row">
          <button class="preset" on:click={clearInput}>Clear</button>
          <button class="preset" on:click={addNoise}>Add noise</button>
        </div>
      </div>

      <div class="control-group">
        <span class="control-label">Display</span>
        <div class="control-row">
          <label class="toggle">
            <input
              type="checkbox"
              checked={detailedMode}
              on:change={(e) => detailedModeStore.set(e.target.checked)}
            />
            Show numbers
          </label>
        </div>
      </div>

      <div class="prediction-box">
        <span class="prediction-label">Prediction</span>
        <span class="prediction-value">
          {prediction >= 0 ? classLists[prediction] : '--'}
        </span>
        <span class="prediction-conf">{fmtPercent(confidence)} confident</span>
      </div>
    </div>

    <div class="hint">
      Click or drag on the 5 x 5 grid to redraw the input. Click any neuron to see
      how its value is computed, or hover to trace its connections.
    </div>

    <div class="legend-row">
      <div class="legend-item">
        <span>Activation</span>
        <div
          class="gradient-strip"
          style="background: linear-gradient(to right, {activationColor(0, 1)}, {activationColor(1, 1)});"
        ></div>
        <span>low &rarr; high</span>
      </div>
      <div class="legend-item">
        <span>Weight</span>
        <div
          class="gradient-strip"
          style="background: linear-gradient(to right, {weightColor(-weightRange, weightRange)}, {weightColor(0, weightRange)}, {weightColor(weightRange, weightRange)});"
        ></div>
        <span>negative &rarr; positive</span>
      </div>
      <div class="legend-item">
        <svg width="26" height="12">
          <circle cx="8" cy="6" r="5" fill="none" stroke="var(--red)"
            stroke-width="1.2" stroke-dasharray="3 2" />
        </svg>
        <span>ReLU output is 0 (inactive)</span>
      </div>
    </div>

    <!-- ------------------------------------------------ diagram -->
    <div id="dnn-svg-container" on:mouseleave={stopPaint}>
      <svg id="dnn-svg" viewBox="0 0 {width} {height}" role="img"
        aria-label="Fully connected neural network diagram">

        {#if layout !== undefined}
          <!-- Layer headings -->
          {#each layerNames as name, l}
            <g>
              <text class="layer-title" x={layout.layerX[l]} y={18}>
                {layerDisplayNames[name]}
              </text>
              <text class="layer-subtitle" x={layout.layerX[l]} y={32}>
                {layerSubtitles[name]}
              </text>
            </g>
          {/each}

          <!-- Edges, drawn first so neurons sit on top -->
          <g class="edge-group">
            {#each network.slice(1) as layer, li}
              {#each layer as neuron}
                {#each neuron.inputLinks as link}
                  <path
                    d={edgePath(
                      layout.coordinates[li][link.source.index],
                      layout.coordinates[li + 1][neuron.index],
                      li === 0 ? 6 : neuronRadius,
                      neuronRadius
                    )}
                    stroke={isEdgeHighlighted(li + 1, link.source.index, neuron.index)
                      ? weightColor(link.weight, weightRange)
                      : edgeInitColor}
                    stroke-width={isEdgeHighlighted(li + 1, link.source.index, neuron.index)
                      ? weightStrokeWidth(link.weight, weightRange)
                      : overviewConfig.edgeStrokeWidth}
                    opacity={isEdgeHighlighted(li + 1, link.source.index, neuron.index)
                      ? 0.95
                      : hasFocus() ? edgeOpacityFaded : edgeOpacity}
                    fill="none"
                  />
                {/each}
              {/each}
            {/each}
          </g>

          <!-- Input layer as an editable grid -->
          <g class="input-layer">
            {#each network[0] as neuron, i}
              <rect
                class="input-cell"
                x={layout.coordinates[0][i].cellX}
                y={layout.coordinates[0][i].cellY}
                width={layout.inputBlock.cellLength}
                height={layout.inputBlock.cellLength}
                fill={activationColor(neuron.output, 1, 'input')}
                on:mousedown={() => startPaint(
                  layout.coordinates[0][i].row, layout.coordinates[0][i].col)}
                on:mouseenter={() => continuePaint(
                  layout.coordinates[0][i].row, layout.coordinates[0][i].col)}
                role="button"
                tabindex="0"
                aria-label={`Input pixel row ${layout.coordinates[0][i].row + 1}, column ${layout.coordinates[0][i].col + 1}, value ${neuron.output}`}
                on:keydown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    setCell(layout.coordinates[0][i].row,
                      layout.coordinates[0][i].col,
                      neuron.output === 1 ? 0 : 1);
                  }
                }}
              />
            {/each}
            <rect
              x={layout.inputBlock.left}
              y={layout.inputBlock.top}
              width={layout.inputBlock.size}
              height={layout.inputBlock.size}
              fill="none"
              stroke="rgb(120,120,120)"
              stroke-width="1.2"
              pointer-events="none"
            />
            <text
              class="layer-subtitle"
              x={layout.layerX[0]}
              y={layout.inputBlock.top + layout.inputBlock.size + 16}
            >
              click to draw
            </text>
          </g>

          <!-- Hidden and output neurons -->
          {#each network.slice(1) as layer, li}
            {#each layer as neuron, i}
              <g
                class="neuron"
                opacity={isNeuronDimmed(li + 1, i) ? 0.35 : 1}
                on:mouseover={() => neuronMouseOver(li + 1, i)}
                on:focus={() => neuronMouseOver(li + 1, i)}
                on:mouseleave={neuronMouseLeave}
                on:blur={neuronMouseLeave}
                on:click={() => neuronClicked(li + 1, i)}
                role="button"
                tabindex="0"
                aria-label={`${layerDisplayNames[layerNames[li + 1]]} neuron ${i + 1}, activation ${fmt(neuron.output)}`}
                on:keydown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    neuronClicked(li + 1, i);
                  }
                }}
              >
                <circle
                  cx={layout.coordinates[li + 1][i].x}
                  cy={layout.coordinates[li + 1][i].y}
                  r={neuronRadius}
                  fill={neuron.activation === activationType.SOFTMAX
                    ? activationColor(neuron.output, 1, 'probability')
                    : activationColor(neuron.output, layerRanges[li + 1])}
                />
                <circle
                  class="neuron-outline"
                  class:selected={selectedNeuron !== undefined &&
                    selectedNeuron.layerIndex === li + 1 && selectedNeuron.index === i}
                  class:dead={neuron.activation === activationType.RELU && neuron.output === 0}
                  cx={layout.coordinates[li + 1][i].x}
                  cy={layout.coordinates[li + 1][i].y}
                  r={neuronRadius}
                />
                {#if detailedMode}
                  <text
                    class="neuron-label"
                    x={layout.coordinates[li + 1][i].x}
                    y={layout.coordinates[li + 1][i].y}
                    fill={neuron.output > 0.6 * (layerRanges[li + 1] || 1)
                      ? 'white' : 'rgb(50,50,50)'}
                  >
                    {fmt(neuron.output)}
                  </text>
                {/if}
              </g>
            {/each}
          {/each}

          <!-- Output class labels and probability bars -->
          {#each outputLayer as neuron, i}
            <g>
              <rect
                class="prob-bar-bg"
                x={layout.coordinates[numLayers - 1][i].x + neuronRadius + 10}
                y={layout.coordinates[numLayers - 1][i].y - 5}
                width="60"
                height="10"
                rx="2"
              />
              <rect
                x={layout.coordinates[numLayers - 1][i].x + neuronRadius + 10}
                y={layout.coordinates[numLayers - 1][i].y - 5}
                width={60 * neuron.output}
                height="10"
                rx="2"
                fill={activationColor(neuron.output, 1, 'probability')}
              />
              <text
                class="class-label"
                class:predicted={i === prediction}
                x={layout.coordinates[numLayers - 1][i].x + neuronRadius + 78}
                y={layout.coordinates[numLayers - 1][i].y}
              >
                {classLists[i]} {fmtPercent(neuron.output)}
              </text>
            </g>
          {/each}

          <!-- Softmax walkthrough entry point -->
          <g
            class="softmax-button"
            on:click={openSoftmaxView}
            role="button"
            tabindex="0"
            aria-label="Explain the softmax step"
            on:keydown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                openSoftmaxView();
              }
            }}
          >
            <rect
              x={layout.layerX[numLayers - 1] - 42}
              y={height - 26}
              width="84"
              height="20"
            />
            <text x={layout.layerX[numLayers - 1]} y={height - 16}>
              {isInSoftmax ? 'hide softmax' : 'softmax'}
            </text>
          </g>

          <!-- Info affordance for the whole diagram -->
          <g
            on:click={() => showModal('overview')}
            role="button"
            tabindex="0"
            aria-label="About this diagram"
            on:keydown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                e.preventDefault();
                showModal('overview');
              }
            }}
          >
            <circle class="info-icon" cx={width - 14} cy={16} r="8" />
            <text x={width - 14} y={16} text-anchor="middle"
              dominant-baseline="middle" font-size="11" fill="white"
              pointer-events="none">i</text>
          </g>
        {/if}
      </svg>

      <!-- ------------------------------------------------ detail views -->
      {#if selectedNeuronData !== undefined && layout !== undefined}
        <div
          class="detail-anchor"
          style="left: {Math.max(10, Math.min(width - 430, layout.coordinates[selectedNeuron.layerIndex][selectedNeuron.index].x - 210))}px;
                 top: {height + 10}px;"
        >
          {#if selectedNeuronData.activation === activationType.RELU}
            <ReluView
              neuron={selectedNeuronData}
              {weightRange}
              layerName={layerDisplayNames[layerNames[selectedNeuron.layerIndex]]}
              on:close={closeDetailView}
            />
          {:else}
            <WeightedSumView
              neuron={selectedNeuronData}
              {weightRange}
              className={classLists[selectedNeuron.index]}
              layerName={layerDisplayNames[layerNames[selectedNeuron.layerIndex]]}
              on:close={closeDetailView}
            />
          {/if}
        </div>
      {/if}

      {#if isInSoftmax}
        <div class="detail-anchor" style="left: 50%; transform: translateX(-50%); top: {height + 10}px;">
          <SoftmaxView
            logits={outputLayer.map((n) => n.preActivation)}
            probabilities={outputLayer.map((n) => n.output)}
            classNames={classLists}
            {prediction}
            on:close={() => isInSoftmaxStore.set(false)}
          />
        </div>
      {/if}
    </div>

    <!-- Spacer so the detail panel never overlaps the article below -->
    {#if selectedNeuronData !== undefined || isInSoftmax}
      <div style="height: 330px;"></div>
    {/if}

    <Modal />
    <Article {modelInfo} />
  {/if}
</div>
