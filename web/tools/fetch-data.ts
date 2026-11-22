// Downloads the trained model picture and the brain mesh from the live page
// into public/data/, so the page runs without training a model first.
//
//     bun run data

import { mkdir } from "node:fs/promises";

const SOURCE = "https://scribble.narasim.me/data/";
const TARGET = new URL("../public/data/", import.meta.url);
const FILES = ["brain.png", "mesh.json", "mesh.bin"];

await mkdir(TARGET, { recursive: true });
for (const name of FILES) {
  const response = await fetch(SOURCE + name);
  if (!response.ok) throw new Error(`${name}: ${response.status}`);
  const bytes = await Bun.write(new URL(name, TARGET), response);
  console.log(`${name}  ${(bytes / 1e6).toFixed(1)} MB`);
}
