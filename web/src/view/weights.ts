// The weights as the page shows and paints them: one byte per pixel of the
// picture. The model itself runs in a worker on its own copy, so every edit
// here is returned as a `Change` for the caller to send there.

import * as THREE from "three";
import type { Picture } from "../model/picture";
import { byteTexture } from "./textures";

const BIT = 1; // bit 0 of a texel: the weight
const DAMAGED = 2; // bit 1: painted, not yet healed

/** Weights that were just set, by pixel index (row * side + column). */
export interface Change {
  indices: Uint32Array<ArrayBuffer>;
  bits: Uint8Array<ArrayBuffer>;
}

export class Weights {
  readonly texture: THREE.DataTexture;
  /** Number of weights in the picture (pixels that belong to a square). */
  readonly total: number;

  private readonly original: Uint8Array;
  private readonly texels: Uint8Array;
  private readonly side: number;
  private readonly square: number;
  private painted: number[] = []; // pixel indices, in the order they were damaged

  constructor(picture: Picture) {
    const { side, square, pieces } = picture.manifest;
    this.side = side;
    this.square = square;
    this.total = pieces.length * square * square;
    this.original = new Uint8Array(side * side);
    for (let i = 0; i < this.original.length; i++) this.original[i] = (picture.packed[i >> 3] >> (7 - (i & 7))) & 1;
    this.texels = this.original.slice();
    this.texture = byteTexture(this.texels, side, side, THREE.NearestFilter);
  }

  get damaged(): number {
    return this.painted.length;
  }

  /** The weight at a place (0..1 across and down) inside square number `square`. */
  at(square: number, x: number, y: number): { row: number; column: number; positive: boolean; damaged: boolean } {
    const column = Math.floor(x * this.square), row = Math.floor(y * this.square);
    const texel = this.texels[this.pixel(square, column, row)];
    return { row, column, positive: (texel & BIT) === 1, damaged: (texel & DAMAGED) !== 0 };
  }

  /** Give every weight under a round brush a random value and mark it damaged. */
  paint(square: number, x: number, y: number, radius: number): Change {
    const size = this.square;
    const cx = x * size, cy = y * size, reach = radius * size;
    const indices: number[] = [];
    for (let row = Math.max(0, Math.floor(cy - reach)); row <= Math.min(size - 1, cy + reach); row++) {
      for (let column = Math.max(0, Math.floor(cx - reach)); column <= Math.min(size - 1, cx + reach); column++) {
        if ((column + 0.5 - cx) ** 2 + (row + 0.5 - cy) ** 2 > reach * reach) continue;
        const at = this.pixel(square, column, row);
        if (!(this.texels[at] & DAMAGED)) this.painted.push(at);
        this.texels[at] = DAMAGED | (Math.random() < 0.5 ? BIT : 0);
        indices.push(at);
      }
    }
    return this.changed(indices);
  }

  /** Put back up to `count` original weights, most recent damage first. */
  heal(count: number): Change {
    const indices = this.painted.splice(Math.max(0, this.painted.length - count));
    for (const at of indices) this.texels[at] = this.original[at];
    return this.changed(indices);
  }

  private pixel(square: number, column: number, row: number): number {
    const across = this.side / this.square;
    return (Math.floor(square / across) * this.square + row) * this.side + (square % across) * this.square + column;
  }

  private changed(indices: number[]): Change {
    this.texture.needsUpdate = true;
    return { indices: Uint32Array.from(indices), bits: Uint8Array.from(indices, (at) => this.texels[at] & BIT) };
  }
}
