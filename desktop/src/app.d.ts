// The page state SvelteKit's shallow routing carries: the drawer is a history entry, so
// Back closes it (lib/Drawer.svelte).
declare global {
  namespace App {
    interface PageState {
      drawer?: boolean;
    }
  }
}

export {};
