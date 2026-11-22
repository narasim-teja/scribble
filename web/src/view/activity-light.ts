// Turns the model's activity into the light on the picture.
//
// The worker reports, per token, how hard each square of weights worked. A
// token's pass through the model takes time, layer by layer, so each square
// takes up the new values a little after the one before it and is brightest
// just after. That makes every token a wave moving through the layers.

import * as THREE from "three";
import type { Piece } from "../model/picture";
import { byteTexture } from "./textures";

const MIDDLE_WIDTH = 32; // values per row kept for the middle distance
const FAR_WIDTH = 8; // and for the whole-brain view
const PASS_SPREAD = 0.85; // share of a token's time its pass through the layers takes
const SETTLE_SECONDS = 0.09; // how quickly shown values follow new ones
const PULSE_SECONDS = 0.22; // how long a square stays bright after it is computed, in token times
const STEADY = 128; // pulse byte meaning "no extra brightness"

/** The order squares are computed in within a layer. */
const STEPS = [["query", "key", "value"], ["attn out"], ["mlp in"], ["mlp out"]];

interface Frame {
  values: Uint8Array;
  /** When it arrived, in seconds. */
  at: number;
  /** How long this token's pass is drawn out over, in seconds. */
  period: number;
}

export class ActivityLight {
  readonly near: THREE.DataTexture;
  readonly middle: THREE.DataTexture;
  readonly far: THREE.DataTexture;
  readonly pulse: THREE.DataTexture;

  private readonly shown: Float32Array;
  private readonly turn: number[];
  private previous: Frame;
  private latest: Frame;

  constructor(
    pieces: Piece[],
    layers: number,
    private readonly width: number,
  ) {
    const rows = pieces.length * 2;
    this.shown = new Float32Array(rows * width);
    this.near = byteTexture(new Uint8Array(rows * width), width, rows, THREE.LinearFilter);
    this.middle = byteTexture(new Uint8Array(rows * MIDDLE_WIDTH), MIDDLE_WIDTH, rows, THREE.LinearFilter);
    this.far = byteTexture(new Uint8Array(rows * FAR_WIDTH), FAR_WIDTH, rows, THREE.LinearFilter);
    this.pulse = byteTexture(new Uint8Array(pieces.length).fill(STEADY), 1, pieces.length, THREE.NearestFilter);
    // When in a token's pass each square is computed, from 0 to just under 1.
    this.turn = pieces.map((piece) => {
      const step = STEPS.findIndex((names) => names.some((name) => piece.label.startsWith(name)));
      return (piece.layer + step / STEPS.length) / layers;
    });
    const dark: Frame = { values: new Uint8Array(rows * width), at: -Infinity, period: 1 };
    this.previous = this.latest = dark;
  }

  /** A new token's activity has arrived. */
  show(values: Uint8Array, now: number, period: number): void {
    this.previous = this.latest;
    this.latest = { values, at: now, period };
  }

  /** Advance the textures to time `now`. With `calm`, values change at once and nothing pulses. */
  update(now: number, dt: number, calm: boolean): void {
    const blend = calm ? 1 : 1 - Math.exp(-dt / SETTLE_SECONDS);
    const rows = 2 * this.width;
    const pulse = this.pulse.image.data as Uint8Array;
    for (let n = 0; n < this.turn.length; n++) {
      const reached = (frame: Frame) => now - frame.at - (calm ? 0 : this.turn[n] * PASS_SPREAD * frame.period);
      const frame = reached(this.latest) >= 0 ? this.latest : this.previous;
      const from = n * rows;
      for (let k = from; k < from + rows; k++) this.shown[k] += (frame.values[k] - this.shown[k]) * blend;
      const tokensAgo = Math.max(0, reached(frame)) / frame.period;
      pulse[n] = calm ? STEADY : STEADY * (0.78 + 0.6 * Math.exp(-tokensAgo / PULSE_SECONDS));
    }
    (this.near.image.data as Uint8Array).set(this.shown);
    this.average(this.middle, MIDDLE_WIDTH);
    this.average(this.far, FAR_WIDTH);
    this.near.needsUpdate = this.pulse.needsUpdate = true;
  }

  /** Fill a narrower texture with averages of neighbouring shown values. */
  private average(texture: THREE.DataTexture, toWidth: number): void {
    const to = texture.image.data as Uint8Array;
    const group = this.width / toWidth;
    for (let cell = 0; cell < to.length; cell++) {
      let sum = 0;
      for (let k = 0; k < group; k++) sum += this.shown[cell * group + k];
      to[cell] = sum / group;
    }
    texture.needsUpdate = true;
  }
}
