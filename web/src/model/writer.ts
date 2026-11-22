// Writes stories with a model that can be damaged while it writes, next to
// an untouched twin.
//
// Three engines run:
//   mine   the weights as painted, writing "your copy"
//   judge  untouched weights, reading your copy's text
//   twin   untouched weights, writing its own text
//
// The judge decides which of your tokens are "unexpected": ones a healthy
// model, shown the same text, would almost never have written. That is the
// rule for amber letters. Comparing the two stories position by position
// would not work, because once they diverge every later letter differs.
//
// Yours and the twin start each story together and draw the same random
// number at every step, so they write the same text until damage makes them
// differ. A story that ends early waits for the other before both restart.

import { Engine } from "./engine";
import type { KernelFactory } from "./kernel";
import type { Picture } from "./picture";
import { pick, probabilities, seededRandom } from "./sampler";

const END_OF_STORY = 0; // token id, as in train/vocab.py
const UNEXPECTED_BELOW = 0.08; // the healthy model gives the token less than this...
const UNEXPECTED_RATIO = 3; // ...and less than a third of what the damaged model gave it
const PEAK_DECAY = 0.995; // how slowly each square's brightness scale forgets old peaks
const LONGEST_STORY = 400; // tokens; a damaged model may never end its story on its own

export interface Token {
  /** Token id in the model's vocabulary. */
  id: number;
  text: string;
  /** How likely the model that wrote it thought it was. */
  probability: number;
}

export interface MyToken extends Token {
  /** How likely the healthy judge thought it was. */
  expected: number;
  unexpected: boolean;
  /** `Engine.activity` scaled to 0..255 per square, for the view's textures. */
  activity: Uint8Array<ArrayBuffer>;
}

/** One step of both stories. A side is null once its story has ended. */
export interface Step {
  mine: MyToken | null;
  twin: Token | null;
  /** Both stories had ended, and a new pair has begun. Nothing was written. */
  restarted: boolean;
}

/** One engine and the text it has read. */
class Reader {
  tokens: number[] = [];
  /** Scores for the token after the last one read. */
  next: Float32Array = new Float32Array(0);
  ended = false;

  constructor(readonly engine: Engine) {}

  begin(): void {
    this.engine.reset();
    this.tokens = [];
    this.ended = false;
    this.read(END_OF_STORY);
  }

  read(token: number): void {
    // Out of room: carry on from the second half of the text.
    if (this.tokens.length === this.engine.context) {
      const kept = this.tokens.slice(this.tokens.length / 2);
      this.engine.reset();
      this.tokens = [];
      kept.forEach((old) => this.read(old));
    }
    this.tokens.push(token);
    this.next = this.engine.step(token);
  }
}

export class Writer {
  /** True after weights change, until `reread` has finished. */
  stale = false;

  private readonly mine: Reader;
  private readonly judge: Reader;
  private readonly twin: Reader;
  private readonly random: () => number;
  private readonly peaks: Float32Array;
  private readonly vocabulary: string[];
  private rereadAt = 0;
  private written = 0;

  constructor(
    picture: Picture,
    private readonly temperature = 0.8,
    seed = 0,
    kernel?: KernelFactory,
  ) {
    const untouched = () => new Engine({ ...picture, packed: picture.packed.slice() }, { kernel });
    this.mine = new Reader(new Engine(picture, { recordActivity: true, kernel }));
    this.judge = new Reader(untouched());
    this.twin = new Reader(untouched());
    this.random = seededRandom(seed);
    this.peaks = new Float32Array(picture.manifest.pieces.length * 2);
    this.vocabulary = picture.manifest.vocab.tokens;
    this.begin();
  }

  /** Set weights by pixel index (row * side + column). Call `reread` before writing on. */
  setBits(indices: Uint32Array, bits: Uint8Array): void {
    const packed = this.mine.engine.weights;
    for (let i = 0; i < indices.length; i++) {
      const mask = 0x80 >> (indices[i] & 7);
      if (bits[i]) packed[indices[i] >> 3] |= mask;
      else packed[indices[i] >> 3] &= ~mask;
    }
    this.stale = true;
    this.rereadAt = 0;
  }

  /**
   * Have the damaged model read its story so far again with its new weights,
   * at most `budget` tokens per call so painting stays responsive.
   */
  reread(budget: number): void {
    const { engine, tokens } = this.mine;
    if (this.rereadAt === 0) engine.reset();
    const stop = Math.min(tokens.length, this.rereadAt + budget);
    for (; this.rereadAt < stop; this.rereadAt++) this.mine.next = engine.step(tokens[this.rereadAt]);
    this.stale = this.rereadAt < tokens.length;
  }

  /** Write the next token of each story that is still going. */
  write(): Step {
    while (this.stale) this.reread(this.mine.tokens.length);
    if (this.mine.ended && this.twin.ended) {
      this.begin();
      return { mine: null, twin: null, restarted: true };
    }
    const dice = this.random();
    return { mine: this.writeMine(dice), twin: this.writeTwin(dice), restarted: false };
  }

  private begin(): void {
    this.mine.begin();
    this.judge.begin();
    this.twin.begin();
    this.stale = false;
    this.written = 0;
  }

  private writeMine(dice: number): MyToken | null {
    if (this.mine.ended) return null;
    const probs = probabilities(this.mine.next, this.temperature);
    const id = pick(probs, dice);
    if (id === END_OF_STORY || ++this.written > LONGEST_STORY) {
      this.mine.ended = true;
      return null;
    }
    const expected = probabilities(this.judge.next, this.temperature)[id];
    const unexpected = expected < UNEXPECTED_BELOW && expected * UNEXPECTED_RATIO < probs[id];
    this.mine.read(id);
    this.judge.read(id);
    const text = this.vocabulary[id];
    return { id, text, probability: probs[id], expected, unexpected, activity: this.scaledActivity() };
  }

  private writeTwin(dice: number): Token | null {
    if (this.twin.ended) return null;
    const probs = probabilities(this.twin.next, this.temperature);
    const id = pick(probs, dice);
    if (id === END_OF_STORY || this.twin.tokens.length > LONGEST_STORY) {
      this.twin.ended = true;
      return null;
    }
    this.twin.read(id);
    return { id, text: this.vocabulary[id], probability: probs[id] };
  }

  /** Each square's two rows divided by that row's recent peak. */
  private scaledActivity(): Uint8Array<ArrayBuffer> {
    const { activity, width } = this.mine.engine;
    const out = new Uint8Array(activity!.length);
    for (let row = 0; row < this.peaks.length; row++) {
      let peak = this.peaks[row] * PEAK_DECAY;
      for (let i = row * width; i < (row + 1) * width; i++) peak = Math.max(peak, activity![i]);
      this.peaks[row] = peak;
      for (let i = row * width; i < (row + 1) * width; i++) out[i] = peak ? (activity![i] / peak) * 255 : 0;
    }
    return out;
  }
}
