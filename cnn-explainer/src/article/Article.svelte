<script>
  /** Explanatory text below the diagram. */
  export let modelInfo = undefined;

  $: metrics = modelInfo ? modelInfo.metrics : undefined;
</script>

<style>
  #article {
    max-width: 720px;
    margin: 30px auto 80px auto;
    padding: 0 20px;
    line-height: 1.62;
    font-size: 15px;
    color: rgb(60, 60, 60);
  }

  h2 {
    font-size: 21px;
    margin: 34px 0 8px 0;
    color: rgb(40, 40, 40);
  }

  h3 {
    font-size: 16px;
    margin: 22px 0 5px 0;
    color: rgb(40, 40, 40);
  }

  p { margin: 9px 0; }

  code {
    background: rgb(245, 245, 245);
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 13.5px;
  }

  .callout {
    border-left: 3px solid var(--blue);
    background: rgb(248, 250, 253);
    padding: 10px 14px;
    margin: 16px 0;
    font-size: 14px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0;
    font-size: 14px;
  }

  th, td {
    text-align: left;
    padding: 6px 8px;
    border-bottom: 1px solid rgb(235, 235, 235);
  }

  th { font-size: 12px; text-transform: uppercase; color: var(--deep-gray); }

  ul { margin: 9px 0; padding-left: 22px; }
  li { margin: 4px 0; }

  .metrics {
    display: flex;
    gap: 26px;
    flex-wrap: wrap;
    margin: 14px 0;
  }

  .metric-value { font-size: 19px; font-weight: 600; color: rgb(40, 40, 40); }
  .metric-label { font-size: 11.5px; text-transform: uppercase; color: var(--deep-gray); }
</style>

<div id="article">

  <h2>What is a dense neural network?</h2>
  <p>
    A dense (or fully connected) network is the plainest kind of neural network.
    Values enter as a flat list of numbers, and each layer transforms that list
    into a new one. There is no notion of position, adjacency, or sequence built
    in: every neuron simply looks at all of the values from the previous layer.
  </p>

  <p>
    Each neuron does the same three things:
  </p>
  <ul>
    <li>multiply every incoming value by a <strong>weight</strong> it learned during training,</li>
    <li>add those products up and add a single <strong>bias</strong>, giving a score usually written <code>z</code>,</li>
    <li>pass <code>z</code> through an <strong>activation function</strong>.</li>
  </ul>

  <p>
    Click any neuron in the diagram above to see those three steps carried out
    with real numbers for the input currently drawn on the grid.
  </p>

  <h2>The network in this demo</h2>
  <p>
    The model classifies a 5 x 5 black-and-white glyph as one of four shapes:
    an <strong>L</strong>, a <strong>T</strong>, an <strong>X</strong>, or an
    <strong>O</strong> ring. Because the image is only 25 pixels, the entire
    network fits on one screen, weights and all.
  </p>

  <table>
    <thead>
      <tr><th>Layer</th><th>Size</th><th>Activation</th><th>Parameters</th></tr>
    </thead>
    <tbody>
      <tr><td>Input</td><td>25</td><td>--</td><td>0</td></tr>
      <tr><td>Hidden 1</td><td>10</td><td>ReLU</td><td>25 &times; 10 + 10 = 260</td></tr>
      <tr><td>Hidden 2</td><td>8</td><td>ReLU</td><td>10 &times; 8 + 8 = 88</td></tr>
      <tr><td>Output</td><td>4</td><td>Softmax</td><td>8 &times; 4 + 4 = 36</td></tr>
    </tbody>
  </table>

  <p>
    That is 384 learned parameters in total.
  </p>

  {#if metrics}
    <div class="metrics">
      <div>
        <div class="metric-value">{(metrics.valAccuracy * 100).toFixed(1)}%</div>
        <div class="metric-label">Validation accuracy</div>
      </div>
      <div>
        <div class="metric-value">{metrics.trainExamples}</div>
        <div class="metric-label">Training examples</div>
      </div>
      <div>
        <div class="metric-value">{metrics.epochs}</div>
        <div class="metric-label">Epochs</div>
      </div>
    </div>
    <p style="font-size: 13.5px; color: var(--deep-gray);">
      The four glyph classes are visually well separated, so this task is an easy
      one and the accuracy is correspondingly high. That is deliberate: a model
      that behaves predictably is easier to learn from than a model that is
      wrong in confusing ways.
    </p>
  {/if}

  <h2 id="article-weighted-sum">The weighted sum</h2>
  <p>
    For a neuron with incoming values <code>x</code> and weights <code>w</code>:
  </p>
  <p style="text-align:center; font-family: monospace; font-size: 14px;">
    z = w<sub>1</sub>x<sub>1</sub> + w<sub>2</sub>x<sub>2</sub> + ... + w<sub>25</sub>x<sub>25</sub> + b
  </p>
  <p>
    A positive weight means "this input is evidence for me"; a negative weight
    means the opposite. Because the input pixels here are 0 or 1, every product
    is either 0 or the weight itself, which is why so many terms drop out in the
    detail panel. The bias shifts the score up or down regardless of the input,
    which lets a neuron stay quiet unless it sees enough evidence.
  </p>

  <div class="callout">
    Notice what happens when you erase most of the grid: the weighted sums shrink
    toward the biases alone, and the output probabilities drift toward an even
    split. The network has no evidence to work with.
  </div>

  <h3 id="article-relu">ReLU</h3>
  <p>
    <code>ReLU(z) = max(0, z)</code>. If the score is positive it passes through
    untouched; if it is negative the neuron outputs exactly zero and contributes
    nothing to the next layer. Those switched-off neurons are drawn with a dashed
    red outline.
  </p>
  <p>
    Without an activation function, stacking layers would be pointless: a chain of
    plain weighted sums collapses into a single weighted sum. ReLU introduces the
    kink that lets each layer add something the previous one could not express.
  </p>

  <h3 id="article-softmax">Softmax</h3>
  <p>
    The last layer produces four raw scores called logits, which can be any real
    number. Softmax exponentiates each one and divides by their total, so the
    results are all positive and sum to 1, and can be read as probabilities.
    Because <code>exp()</code> grows fast, a modest lead in the logits becomes a
    large lead in probability.
  </p>

  <h2>How this differs from a CNN</h2>
  <p>
    A convolutional network slides a small kernel across the image and reuses the
    same weights at every position. That gives it two things a dense network
    lacks: far fewer parameters, and a built-in assumption that a pattern means
    the same thing wherever it appears.
  </p>
  <p>
    A dense network has no such assumption. Each neuron learns a separate weight
    for pixel 7 and pixel 8, and it has no way of knowing they are neighbours.
    Shift a glyph one pixel to the right and, as far as the input layer is
    concerned, an entirely different set of features fired. The model in this demo
    copes only because its training data included shifted copies of each glyph, so
    it learned each variant explicitly.
  </p>
  <p>
    That tradeoff is the reason dense layers are rarely used alone on images, and
    also the reason they remain the default choice for tabular data, where inputs
    are unrelated quantities rather than neighbouring pixels.
  </p>

  <h2>Train it yourself</h2>
  <p>
    The model is trained by a small dependency-free script. From the project root:
  </p>
  <p><code>node dnn-mlp/train.js</code></p>
  <p>
    It builds the dataset, trains with mini-batch gradient descent, prints its
    accuracy, and writes the weights to
    <code>public/assets/data/dnn_model.json</code>, which is exactly the file the
    page above loads. Change the layer sizes in the script and the diagram will
    redraw itself around the new architecture.
  </p>

  <h2>Credits</h2>
  <p>
    DNN Explainer is adapted from
    <a href="https://github.com/poloclub/cnn-explainer" target="_blank" rel="noreferrer">CNN
    Explainer</a> by Jay Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das,
    Fred Hohman, Minsuk Kahng, and Polo Chau (Georgia Tech and Oregon State),
    released under the MIT License. The visual language and interaction model come
    from their work; the network, dataset, and views here were rebuilt for dense
    networks.
  </p>
</div>
