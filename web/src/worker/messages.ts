// What the page and the writer worker say to each other.

import type { Picture } from "../model/picture";
import type { MyToken, Token } from "../model/writer";

export type ToWorker =
  /** Start writing with this model. */
  | { type: "load"; picture: Picture; tokensPerSecond: number; seed: number }
  /** Weights changed: pixel indices (row * side + column) and their new bits. */
  | { type: "set"; indices: Uint32Array; bits: Uint8Array }
  | { type: "play"; playing: boolean };

export type FromWorker =
  /** One step of both stories. A side is null once its story has ended. */
  | { type: "step"; mine: MyToken | null; twin: Token | null }
  /** Both stories ended and a new pair is starting. */
  | { type: "restart" }
  /** The model is reading its story again after its weights changed. */
  | { type: "rereading"; active: boolean }
  /** Which build of the inner loop the model is running on. */
  | { type: "kernel"; name: "rust" | "typescript" };
