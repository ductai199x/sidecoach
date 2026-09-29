<script lang="ts">
  // The cross at the foot of the screen a caption box is dropped on to turn captions off (the
  // phone's `Captions.Target`). Its window lets the pointer through to the box being carried
  // (the shell's `open_captions`); the box shows it, places it and says when it is over it.
  import { onDestroy, onMount } from "svelte";
  import { isTauri } from "@tauri-apps/api/core";

  let hot = $state(false);
  let unlisten: (() => void) | null = null;

  onMount(async () => {
    if (!isTauri()) return;
    const { getCurrentWebviewWindow } = await import("@tauri-apps/api/webviewWindow");
    unlisten = await getCurrentWebviewWindow().listen<boolean>("cross-hot", (e) => { hot = !!e.payload; });
  });

  onDestroy(() => unlisten?.());
</script>

<svelte:head><title>Sidecoach captions</title></svelte:head>

<div class="cross" class:hot aria-hidden="true">&#x2715;</div>

<style>
  :global(html), :global(body) { background: transparent !important; overflow: hidden; margin: 0; }
  :global(body::after) { display: none !important; }
  :global(body) { display: flex; align-items: center; justify-content: center; height: 100vh; }

  .cross {
    width: 56px; height: 56px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    background: rgba(0, 0, 0, 0.6); color: #fff; font-size: 22px;
    transition: transform 120ms ease-out, background 120ms ease-out;
  }
  .cross.hot { background: rgba(179, 38, 30, 0.9); transform: scale(1.3); }
</style>
