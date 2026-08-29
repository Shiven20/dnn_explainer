# DNN Explainer

An interactive visualization for learning how dense (fully connected) neural
networks turn raw numbers into a prediction.

Every value on screen is real: the page loads actual trained weights and runs
the forward pass in the browser, so the arithmetic shown in the detail panels is
the arithmetic the model performed.

## What you can do

- **Draw the input.** Click or drag on the 5 x 5 grid to redraw the glyph. The
  network re-runs on every change, so predictions update as you draw.
- **Inspect any neuron.** Click a neuron to see its weighted sum term by term:
  each input, the weight applied to it, and the resulting product.
- **See ReLU decide.** Hidden-neuron panels plot where that neuron's score lands
  on the ReLU curve. Neurons clipped to zero are outlined in dashed red.
- **Follow the softmax.** The softmax panel shows logits, their exponentials, and
  how those normalize into probabilities.
- **Trace connections.** Hover a neuron to highlight its edges, coloured by
  weight sign and weighted by magnitude.

## The model

A 4-layer classifier that sorts a 5 x 5 binary glyph into one of four shapes
(L, T, X, O ring):

| Layer | Size | Activation | Parameters |
| --- | --- | --- | --- |
| Input | 25 | -- | 0 |
| Hidden 1 | 10 | ReLU | 260 |
| Hidden 2 | 8 | ReLU | 88 |
| Output | 4 | Softmax | 36 |

384 parameters in total. The input is deliberately tiny so that the whole
network, weights included, fits on one screen.

The four classes are visually distinct, so the task is easy and the model scores
100% on a held-out split. That is intentional for a teaching tool: predictable
behaviour is easier to learn from than confusing errors.

## Running locally

```bash
npm install
npm run dev
```

Then open [localhost:3000](http://localhost:3000).

## Retraining

The model is trained by a dependency-free Node script:

```bash
npm run train
```

It builds the dataset, trains with mini-batch gradient descent, reports
accuracy, and writes `public/assets/data/dnn_model.json` — the exact file the
page loads. Change the layer sizes in `dnn-mlp/train.js` and the diagram adapts
to the new architecture on reload.

To check that the browser engine agrees with the trainer:

```bash
npm run verify
```

This imports `src/utils/dnn.js` (the module the app actually ships) and asserts
graph construction, weight orientation, ReLU and softmax semantics, the detail
view arithmetic, and end-to-end accuracy.

## Project layout

```
dnn-mlp/
  glyphs.js            dataset definition and augmentation
  train.js             trainer, exports dnn_model.json
  verify.mjs           checks the browser engine against the weights
src/
  utils/dnn.js         neuron graph + forward pass
  overview/
    Overview.svelte    the main diagram
    dnn-layout.js      geometry and colour scales
    Modal.svelte
  detail-view/
    WeightedSumView.svelte   weighted sum, term by term
    ReluView.svelte          weighted sum + ReLU curve
    Softmaxview.svelte       logits -> probabilities
  article/Article.svelte     explanatory text
```

## Credits

Adapted from [CNN Explainer](https://github.com/poloclub/cnn-explainer) by Jay
Wang, Robert Turko, Omar Shaikh, Haekyu Park, Nilaksh Das, Fred Hohman, Minsuk
Kahng, and Polo Chau — a collaboration between Georgia Tech and Oregon State.
The visual language and interaction model come from their work; the network,
dataset, and detail views here were rebuilt for dense networks.

Original paper: [CNN Explainer: Learning Convolutional Neural Networks with
Interactive Visualization](https://arxiv.org/abs/2004.15004) (IEEE TVCG, 2020).

## License

MIT, same as the original project. See [LICENSE](LICENSE).
