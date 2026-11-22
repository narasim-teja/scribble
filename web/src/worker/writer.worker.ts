// Runs the model off the main thread so painting and drawing stay smooth.
// The worker owns the weights the model actually uses; the picture on screen
// is a copy that the page keeps in step by sending every change here.

import type { KernelFactory } from "../model/kernel";
import kernelUrl from "../model/kernel.wasm?url";
import { wasmKernel } from "../model/wasm-kernel";
import { Writer } from "../model/writer";
import type { FromWorker, ToWorker } from "./messages";

const REREAD_TOKENS = 16; // tokens re-read between checks for new paint
const PAUSE_AFTER_STORY_MS = 1800;

let writer: Writer | null = null;
let interval = 200;
let playing = true;
let timer: ReturnType<typeof setTimeout> | undefined;
let rereading = false;

const post = (message: FromWorker, transfer: Transferable[] = []) => postMessage(message, { transfer });

function schedule(delay: number): void {
  clearTimeout(timer);
  timer = setTimeout(tick, delay);
}

function tick(): void {
  if (!writer || !playing) return;
  if (writer.stale) {
    if (!rereading) post({ type: "rereading", active: (rereading = true) });
    writer.reread(REREAD_TOKENS);
    return schedule(0); // yield, so a new stroke can restart the reread
  }
  if (rereading) post({ type: "rereading", active: (rereading = false) });

  const { mine, twin, restarted } = writer.write();
  if (restarted) {
    post({ type: "restart" });
    return schedule(interval);
  }
  post({ type: "step", mine, twin }, mine ? [mine.activity.buffer] : []);
  // Leave both finished stories on screen for a moment before the next pair.
  schedule(mine || twin ? interval : PAUSE_AFTER_STORY_MS);
}

/** The Rust kernel if this browser can load it, otherwise the TypeScript one. */
async function loadKernel(): Promise<KernelFactory | undefined> {
  try {
    return wasmKernel(await WebAssembly.compileStreaming(fetch(kernelUrl)));
  } catch {
    return undefined;
  }
}

addEventListener("message", async ({ data }: MessageEvent<ToWorker>) => {
  if (data.type === "load") {
    const kernel = await loadKernel();
    post({ type: "kernel", name: kernel ? "rust" : "typescript" });
    writer = new Writer(data.picture, undefined, data.seed, kernel);
    interval = 1000 / data.tokensPerSecond;
    schedule(0);
  } else if (data.type === "set") {
    writer?.setBits(data.indices, data.bits);
    if (playing) schedule(0);
  } else {
    playing = data.playing;
    if (playing) schedule(0);
  }
});
