<script>
  /** Slim app bar. Kept minimal so the network stays the visual centrepiece. */
</script>

<style>
  header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 16px;
    padding: 0 24px;
    height: 56px;
    background: var(--dnn-surface);
    border-bottom: 1px solid var(--dnn-border);
    position: sticky;
    top: 0;
    z-index: 20;
  }

  .brand {
    display: flex;
    align-items: baseline;
    gap: 10px;
    min-width: 0;
  }

  .title {
    font-size: 16px;
    font-weight: 650;
    letter-spacing: -0.01em;
    color: var(--dnn-ink);
    white-space: nowrap;
  }

  .tagline {
    font-size: 13px;
    color: var(--dnn-muted);
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .mark {
    display: flex;
    align-items: center;
    gap: 9px;
  }

  a {
    font-size: 13px;
    color: var(--dnn-muted);
    text-decoration: none;
    white-space: nowrap;
  }

  a:hover { color: var(--dnn-accent); }

  @media (max-width: 620px) {
    header { padding: 0 16px; }
    .tagline { display: none; }
  }
</style>

<header>
  <div class="mark">
    <!-- Three connected nodes: a miniature of the diagram below. -->
    <svg width="22" height="22" viewBox="0 0 22 22" aria-hidden="true">
      <line x1="5" y1="6" x2="17" y2="11" stroke="var(--dnn-accent)" stroke-width="1.2" opacity="0.55" />
      <line x1="5" y1="16" x2="17" y2="11" stroke="var(--dnn-accent)" stroke-width="1.2" opacity="0.55" />
      <circle cx="5" cy="6" r="3" fill="var(--dnn-accent)" opacity="0.35" />
      <circle cx="5" cy="16" r="3" fill="var(--dnn-accent)" opacity="0.35" />
      <circle cx="17" cy="11" r="3.4" fill="var(--dnn-accent)" />
    </svg>

    <div class="brand">
      <span class="title">DNN Explainer</span>
      <span class="tagline">Learn how a neural network thinks, one neuron at a time</span>
    </div>
  </div>

  <a href="https://github.com/poloclub/cnn-explainer" target="_blank" rel="noreferrer">
    Based on CNN Explainer
  </a>
</header>
