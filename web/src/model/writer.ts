// Writes stories with a model that can be damaged while it writes.
//
// Two engines read the same text. One runs on the weights as painted. The
// other runs on an untouched copy, and is only used to judge the first: a
// token is "unexpected" when the healthy model, shown the same text, would
// almost never have written it. That is the rule for amber letters. Comparing
// the two stories position by position would not work, because once they
// diverge every later letter differs.

import { Engine } from "./engine";
import type { KernelFactory } from "./kernel";
import type { Picture } from "./picture";
import { pick, probabilities, seededRandom } from "./sampler";

const END_OF_STORY = 0; // token id, as in train/vocab.py
const UNEXPECTED_BELOW = 0.08; // the healthy model gives the token less than this...
const UNEXPECTED_RATIO = 3; // ...and less than a third of what the damaged model gave it
const PEAK_DECAY = 0.995; // how slowly each square's brightness scale forgets old peaks

export interface Written {
  text: string;
  /** How likely the damaged model thought this token was. */
  probability: number;
  unexpected: boolean;
  /** `Engine.activity` scaled to 0..255 per square, for the view's textures. */
  activity: Uint8Array<ArrayBuffer>;
}

export class Writer {
  /** True after weights change, until `reread` has finished. */
  stale = false;

  private readonly damaged: Engine;
  private readonly healthy: Engine;
  private readonly random: () => number;
  private readonly peaks: Float32Array;
  private tokens: number[] = [];
  private rereadAt = 0;
  private nextDamaged: Float32Array = new Float32Array(0);
  private nextHealthy: Float32Array = new Float32Array(0);

  constructor(
    private readonly picture: Picture,
    private readonly temperature = 0.8,
    seed = 0,
    kernel?: KernelFactory,
  ) {
    this.damaged = new Engine(picture, { recordActivity: true, kernel });
    this.healthy = new Engine({ ...picture, packed: picture.packed.slice() }, { kernel });
    this.random = seededRandom(seed);
    this.peaks = new Float32Array(picture.manifest.pieces.length * 2);
    this.begin([END_OF_STORY]);
  }

  /** Set weights by pixel index (row * side + column). Call `reread` before writing on. */
  setBits(indices: Uint32Array, bits: Uint8Array): void {
    const packed = this.damaged.weights;
    for (let i = 0; i < indices.length; i++) {
      const mask = 0x80 >> (indices[i] & 7);
      if (bits[i]) packed[indices[i] >> 3] |= mask;
      else packed[indices[i] >> 3] &= ~mask;
    }
    this.stale = true;
    this.rereadAt = 0;
  }

  /**
   * Have the damaged model read the story so far again with its new weights,
   * at most `budget` tokens per call so painting stays responsive.
   */
  reread(budget: number): void {
    if (this.rereadAt === 0) this.damaged.reset();
    const stop = Math.min(this.tokens.length, this.rereadAt + budget);
    for (; this.rereadAt < stop; this.rereadAt++) this.nextDamaged = this.damaged.step(this.tokens[this.rereadAt]);
    this.stale = this.rereadAt < this.tokens.length;
  }

  /** Write the next token. Returns null when the story ended and a new one has begun. */
  write(): Written | null {
    while (this.stale) this.reread(this.tokens.length);
    const probs = probabilities(this.nextDamaged, this.temperature);
    const token = pick(probs, this.random());
    if (token === END_OF_STORY) {
      this.begin([END_OF_STORY]);
      return null;
    }
    const expected = probabilities(this.nextHealthy, this.temperature)[token];
    const unexpected = expected < UNEXPECTED_BELOW && expected * UNEXPECTED_RATIO < probs[token];

    // Out of room: carry on from the second half of the text.
    if (this.tokens.length === this.damaged.context) this.begin(this.tokens.slice(this.tokens.length / 2));
    this.feed(token);
    return {
      text: this.picture.manifest.vocab.tokens[token],
      probability: probs[token],
      unexpected,
      activity: this.scaledActivity(),
    };
  }

  private begin(tokens: number[]): void {
    this.damaged.reset();
    this.healthy.reset();
    this.tokens = [];
    this.stale = false;
    tokens.forEach((token) => this.feed(token));
  }

  private feed(token: number): void {
    this.tokens.push(token);
    this.nextDamaged = this.damaged.step(token);
    this.nextHealthy = this.healthy.step(token);
  }

  /** Each square's two rows divided by that row's recent peak. */
  private scaledActivity(): Uint8Array<ArrayBuffer> {
    const activity = this.damaged.activity!;
    const width = this.damaged.width;
    const out = new Uint8Array(activity.length);
    for (let row = 0; row < this.peaks.length; row++) {
      let peak = this.peaks[row] * PEAK_DECAY;
      for (let i = row * width; i < (row + 1) * width; i++) peak = Math.max(peak, activity[i]);
      this.peaks[row] = peak;
      for (let i = row * width; i < (row + 1) * width; i++) out[i] = peak ? (activity[i] / peak) * 255 : 0;
    }
    return out;
  }
}
