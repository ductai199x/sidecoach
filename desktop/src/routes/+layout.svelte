<script lang="ts">
  // The panel's stylesheet is global: the pages share its tokens and widgets, and the
  // puzzle page adds its own on top.
  import "$lib/styles/panel.css";
  let { children } = $props();
</script>

{@render children()}
