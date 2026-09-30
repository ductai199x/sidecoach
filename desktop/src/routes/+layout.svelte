<script lang="ts">
  // The panel's stylesheet is global: the pages share its tokens and widgets, and the
  // puzzle page adds its own on top. The boards' look (pieces, squares) follows the settings
  // in every window (lib/look.ts).
  import "$lib/styles/panel.css";
  import { onMount } from "svelte";
  import { page } from "$app/state";
  import { followLook } from "$lib/look";
  let { children } = $props();
  // Not on the splash (it draws its own board before the sidecar is up) nor the captions.
  onMount(() => {
    const path = page.url.pathname;
    if (path.startsWith("/splash") || path.startsWith("/captions")) return;
    return followLook();
  });
</script>

{@render children()}
