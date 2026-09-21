// Copy the SvelteKit build (desktop/build) into the Python package's web folder, so
// `sidecoach panel` in a browser and the server tests serve the same frontend the shell
// bundles. Everything in the target goes first, except the piece sprites' licence, which
// the build carries anyway. `npm run build:web`.
import { cpSync, existsSync, mkdirSync, readdirSync, rmSync, statSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
const build = join(here, "build");
const target = join(here, "..", "src", "sidecoach", "ui", "web");

if (!existsSync(join(build, "index.html"))) {
  console.error("no build/index.html: run `npm run build` first");
  process.exit(1);
}
mkdirSync(target, { recursive: true });
for (const name of readdirSync(target)) rmSync(join(target, name), { recursive: true, force: true });
cpSync(build, target, { recursive: true });

let files = 0, bytes = 0;
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    const st = statSync(p);
    if (st.isDirectory()) walk(p); else { files++; bytes += st.size; }
  }
};
walk(target);
console.log(`web: ${files} files, ${(bytes / 1024).toFixed(0)} KB -> ${target}`);
