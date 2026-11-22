// What the page and the writer worker say to each other.

import type { Picture } from "../model/picture";

export type ToWorker =
  /** Start writing with this model. */
  | { type: "load"; picture: Picture; tokensPerSecond: number; seed: number }
  /** Weights changed: pixel indices (row * side + column) and their new bits. */
  | { type: "set"; indices: Uint32Array; bits: Uint8Array }
  | { type: "play"; playing: boolean };

export type FromWorker =
  | {
      type: "token";
      text: string;
      probability: number;
      /** A healthy copy of the model would almost never have written this. */
      unexpected: boolean;
      /** Two rows per square of the picture, 0..255: input per column, output per row. */
      activity: Uint8Array<ArrayBuffer>;
    }
  | { type: "story-end" }
  /** The model is reading the story again after its weights changed. */
  | { type: "rereading"; active: boolean }
  /** Which build of the inner loop the model is running on. */
  | { type: "kernel"; name: "rust" | "typescript" };
