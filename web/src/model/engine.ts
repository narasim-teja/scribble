// Runs the language model from its picture, one token at a time. This file
// mirrors train/model.py exactly: change them together. A test checks the
// logits against Python's for the same picture.
//
// The engine reads weights from `weights` on every step, so flipping a bit
// there changes the model immediately. After a change, call `reset()` and
// feed the tokens again: the attention memory was built with the old weights.

import { type Kernel, type KernelFactory, tableKernel } from "./kernel";
import { type Picture, type Piece, sideValues } from "./picture";

const NORM_EPSILON = 1e-6;

/** Pieces of one weight matrix that read the same slice of the input. */
interface InputGroup {
  from: number;
  pieces: { piece: Piece; index: number }[];
}

/** One linear layer of the model: normalize the input, then a one-bit matrix. */
interface Linear {
  gain: Float32Array;
  scale: number;
  groups: InputGroup[];
  outputs: number;
}

/** Gaussian error function, accurate to about 1e-7 (Abramowitz and Stegun 7.1.26). */
function erf(x: number): number {
  const t = 1 / (1 + 0.3275911 * Math.abs(x));
  const poly = ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t;
  const value = 1 - poly * Math.exp(-x * x);
  return x < 0 ? -value : value;
}

const gelu = (x: number): number => 0.5 * x * (1 + erf(x * Math.SQRT1_2));

export class Engine {
  readonly width: number;
  readonly context: number;
  readonly vocabSize: number;
  /** How many tokens have been fed since the last reset. */
  position = 0;
  /**
   * What each square of the picture did on the last step, if recording was
   * asked for. Two rows of `width` values per square, in manifest order: the
   * size of the input reaching each column of weights, then the size of the
   * output leaving each row. This is what the view lights the picture with.
   */
  readonly activity: Float32Array | null;

  private readonly heads: number;
  private readonly layers: { qkv: Linear; out: Linear; up: Linear; down: Linear }[];
  private readonly tokens: Float32Array;
  private readonly positions: Float32Array;
  private readonly finalGain: Float32Array;
  private readonly keys: Float64Array[];
  private readonly values: Float64Array[];
  private readonly kernel: Kernel;
  private readonly rowSums: Float64Array;

  constructor(
    private readonly picture: Picture,
    options: { recordActivity?: boolean; kernel?: KernelFactory } = {},
  ) {
    const { config, pieces } = picture.manifest;
    this.activity = options.recordActivity ? new Float32Array(pieces.length * 2 * config.width) : null;
    this.width = config.width;
    this.context = config.context;
    this.vocabSize = config.vocab_size;
    this.heads = config.heads;
    this.tokens = sideValues(picture, "tokens.weight");
    this.positions = sideValues(picture, "positions.weight");
    this.finalGain = sideValues(picture, "norm.gain");

    const linear = (name: string, outputs: number): Linear => {
      const groups: InputGroup[] = [];
      pieces.forEach((piece, index) => {
        if (piece.tensor !== `${name}.weight`) return;
        const group = groups.find((g) => g.from === piece.cols[0]);
        if (group) group.pieces.push({ piece, index });
        else groups.push({ from: piece.cols[0], pieces: [{ piece, index }] });
      });
      const scale = picture.manifest.scales[`${name}.weight`];
      return { gain: sideValues(picture, `${name}.norm.gain`), scale, groups, outputs };
    };
    this.layers = Array.from({ length: config.layers }, (_, i) => ({
      qkv: linear(`blocks.${i}.attention.qkv`, 3 * this.width),
      out: linear(`blocks.${i}.attention.out`, this.width),
      up: linear(`blocks.${i}.mlp.up`, 4 * this.width),
      down: linear(`blocks.${i}.mlp.down`, this.width),
    }));
    this.keys = this.layers.map(() => new Float64Array(this.context * this.width));
    this.values = this.layers.map(() => new Float64Array(this.context * this.width));
    this.kernel = (options.kernel ?? tableKernel)(picture.packed, this.width);
    this.rowSums = new Float64Array(this.width);
  }

  /** The packed weights the engine runs on: `stride` bytes per picture row. */
  get weights(): Uint8Array {
    return this.kernel.packed;
  }

  /** Forget the text so far. */
  reset(): void {
    this.position = 0;
  }

  /** Feed one token and return the scores (logits) for the token after it. */
  step(token: number): Float32Array {
    if (this.position >= this.context) throw new RangeError("context is full: reset first");
    const d = this.width;
    const at = this.position;
    const x = new Float64Array(d);
    for (let i = 0; i < d; i++) x[i] = this.tokens[token * d + i] + this.positions[at * d + i];

    this.layers.forEach((layer, index) => {
      const mixed = this.attend(index, this.apply(layer.qkv, x));
      const attention = this.apply(layer.out, mixed);
      for (let i = 0; i < d; i++) x[i] += attention[i];

      const hidden = this.apply(layer.up, x);
      for (let i = 0; i < hidden.length; i++) hidden[i] = gelu(hidden[i]);
      const mlp = this.apply(layer.down, hidden);
      for (let i = 0; i < d; i++) x[i] += mlp[i];
    });
    this.position++;

    // The output layer shares the token embedding.
    const last = this.normalize(x, this.finalGain);
    const logits = new Float32Array(this.vocabSize);
    for (let t = 0; t < this.vocabSize; t++) {
      let sum = 0;
      for (let i = 0; i < d; i++) sum += last[i] * this.tokens[t * d + i];
      logits[t] = sum;
    }
    return logits;
  }

  /** x scaled to unit root-mean-square, times the learned gain. */
  private normalize(x: Float64Array, gain: Float32Array): Float64Array {
    let squares = 0;
    for (let i = 0; i < x.length; i++) squares += x[i] * x[i];
    const factor = 1 / Math.sqrt(squares / x.length + NORM_EPSILON);
    return x.map((value, i) => value * factor * gain[i]);
  }

  /** One linear layer. Every weight is +scale or -scale, chosen by its pixel. */
  private apply(layer: Linear, x: Float64Array): Float64Array {
    const input = this.normalize(x, layer.gain);
    const out = new Float64Array(layer.outputs);
    const { stride } = this.picture;
    const d = this.width;
    for (const group of layer.groups) {
      const total = this.kernel.buildTables(input.subarray(group.from, group.from + d));
      for (const { piece, index } of group.pieces) {
        this.kernel.sumSetBits(piece.y * stride + piece.x / 8, stride, this.rowSums);
        const record = this.activity?.subarray(index * 2 * d, (index + 1) * 2 * d);
        for (let r = 0; r < d; r++) {
          // Set bits add the input and clear bits subtract it.
          const value = layer.scale * (2 * this.rowSums[r] - total);
          out[piece.rows[0] + r] += value;
          if (record) {
            record[r] = Math.abs(input[group.from + r]);
            record[d + r] = Math.abs(value);
          }
        }
      }
    }
    return out;
  }

  /** Attention for the newest token over everything fed so far in this layer. */
  private attend(layer: number, qkv: Float64Array): Float64Array {
    const d = this.width;
    const size = d / this.heads;
    const keys = this.keys[layer], values = this.values[layer];
    keys.set(qkv.subarray(d, 2 * d), this.position * d);
    values.set(qkv.subarray(2 * d, 3 * d), this.position * d);

    const mixed = new Float64Array(d);
    const scores = new Float64Array(this.position + 1);
    for (let head = 0; head < this.heads; head++) {
      const from = head * size;
      let best = -Infinity;
      for (let past = 0; past <= this.position; past++) {
        let dot = 0;
        for (let i = 0; i < size; i++) dot += qkv[from + i] * keys[past * d + from + i];
        scores[past] = dot / Math.sqrt(size);
        best = Math.max(best, scores[past]);
      }
      let sum = 0;
      for (let past = 0; past <= this.position; past++) sum += scores[past] = Math.exp(scores[past] - best);
      for (let past = 0; past <= this.position; past++) {
        const share = scores[past] / sum;
        for (let i = 0; i < size; i++) mixed[from + i] += share * values[past * d + from + i];
      }
    }
    return mixed;
  }
}
