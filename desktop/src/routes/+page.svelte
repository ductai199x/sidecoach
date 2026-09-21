<script lang="ts">
  // What the window shows until the shell has the sidecar up and navigates to it (Phase 0
  // of docs/app-plan.md). In Phase 1 the panel itself lives here and the sidecar's port
  // arrives through `invoke("sidecar_port")`.
  import { onMount } from "svelte";

  let message = $state("Starting the coach…");
  let detail = $state("");

  onMount(() => {
    const t0 = Date.now();
    const tick = setInterval(() => {
      const s = Math.round((Date.now() - t0) / 1000);
      detail = s > 20 ? `${s} s: the Python sidecar is slow to start, or missing` : `${s} s`;
    }, 1000);
    return () => clearInterval(tick);
  });
</script>

<main>
  <div class="mark">♞</div>
  <h1>Sidecoach</h1>
  <p>{message}</p>
  <p class="detail">{detail}</p>
</main>

<style>
  :global(body) {
    margin: 0;
    background: #262421;
    color: #e6e2d9;
    font-family: system-ui, "Segoe UI", sans-serif;
  }
  main {
    height: 100vh;
    display: grid;
    place-content: center;
    text-align: center;
    gap: 0.25rem;
  }
  .mark {
    font-size: 4rem;
    line-height: 1;
    color: #81b64c;
  }
  h1 {
    margin: 0.5rem 0 0;
    font-weight: 600;
    letter-spacing: 0.02em;
  }
  p {
    margin: 0;
    opacity: 0.85;
  }
  .detail {
    font-size: 0.85rem;
    opacity: 0.55;
  }
</style>
