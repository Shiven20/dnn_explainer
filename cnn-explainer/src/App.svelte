<script>
  import Header from './Header.svelte';
  import DNNExplainer from './dnn/DNNExplainer.svelte';
</script>

<style>
  #app-page {
    min-height: 100vh;
    display: flex;
    flex-direction: column;
  }

  main {
    flex: 1;
  }
</style>

<div id="app-page">
  <Header />
  <main>
    <DNNExplainer />
  </main>
</div>
