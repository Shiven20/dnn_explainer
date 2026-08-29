<script>
  /** Short explanatory pop-ups, keyed by topic. */
  import { modalStore } from '../stores.js';

  let info = { show: false };
  modalStore.subscribe((v) => { info = v; });

  const content = {
    overview: {
      title: 'Reading this diagram',
      body: `Values flow left to right. The 5 x 5 grid on the left is the raw input:
        25 numbers, one per pixel. Each of the following columns is a layer of
        neurons, and every neuron is connected to every neuron in the layer
        before it -- that is what "fully connected" (dense) means.
        Circle fill shows how strongly a neuron activated for the current input.
        A dashed red outline marks a ReLU neuron whose output is exactly zero.`
    },
    dense: {
      title: 'Dense layers',
      body: `A dense neuron multiplies every incoming value by its own weight,
        adds them up, adds a bias, then applies an activation function. Unlike a
        convolution, nothing is shared or reused across positions: each neuron
        learns its own weight for each of the 25 input pixels.`
    }
  };

  const close = () => modalStore.set({ show: false });

  // Close only when the backdrop itself is clicked, not when the click merely
  // bubbled up from inside the dialog.
  const backdropClicked = (event) => {
    if (event.target === event.currentTarget) close();
  };

  $: current = info.show && info.key ? content[info.key] : undefined;
</script>

<style>
  .backdrop {
    position: fixed;
    inset: 0;
    background: rgba(0, 0, 0, 0.32);
    display: flex;
    align-items: center;
    justify-content: center;
    z-index: 100;
  }

  .modal {
    background: white;
    border-radius: 7px;
    box-shadow: var(--outer-shadow-lg);
    max-width: 460px;
    padding: 18px 20px 20px 20px;
  }

  .title {
    font-size: 16px;
    font-weight: 600;
    margin-bottom: 8px;
  }

  .body {
    font-size: 13.5px;
    line-height: 1.55;
    color: rgb(70, 70, 70);
    white-space: pre-line;
  }

  .actions {
    display: flex;
    justify-content: flex-end;
    margin-top: 14px;
  }

  button {
    border: 1px solid var(--middle-gray);
    background: white;
    border-radius: 5px;
    padding: 5px 14px;
    font-size: 13px;
    cursor: pointer;
  }

  button:hover { border-color: var(--blue); color: var(--blue); }
</style>

{#if current !== undefined}
  <div
    class="backdrop"
    on:click={backdropClicked}
    on:keydown={(e) => { if (e.key === 'Escape') close(); }}
    role="presentation"
  >
    <div class="modal" role="dialog" aria-modal="true" aria-label={current.title}>
      <div class="title">{current.title}</div>
      <div class="body">{current.body}</div>
      <div class="actions">
        <button on:click={close}>Got it</button>
      </div>
    </div>
  </div>
{/if}
