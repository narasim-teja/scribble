// The inner loop of the engine: multiply a vector by a matrix whose entries
// are single bits, read straight from the packed picture rows.
//
// For every group of 8 inputs we first build a table of all 256 possible
// partial sums. A row's sum over its set bits is then one table lookup per
// byte of the row, instead of eight multiply-adds. Measured about ten times
// faster than the plain loop (see bench/).

const TABLE_SIZE = 256;

/** For each byte value, which input (0 is the first, in the high bit) its lowest set bit selects. */
const LOWEST_INPUT = Uint8Array.from({ length: TABLE_SIZE }, (_, byte) =>
  byte === 0 ? 0 : 7 - Math.log2(byte & -byte),
);

/** Scratch space for `buildTables`, for vectors up to `inputs` long. */
export function tableSpace(inputs: number): Float64Array {
  return new Float64Array((inputs / 8) * TABLE_SIZE);
}

/**
 * Fill `tables` so that `tables[g * 256 + byte]` is the sum of the inputs in
 * group `g` selected by the bits of `byte`. Returns the sum of all inputs.
 */
export function buildTables(x: Float64Array, tables: Float64Array): number {
  let total = 0;
  for (let group = 0; group < x.length / 8; group++) {
    const base = group * TABLE_SIZE;
    tables[base] = 0;
    for (let byte = 1; byte < TABLE_SIZE; byte++) {
      const rest = byte & (byte - 1);
      tables[base + byte] = tables[base + rest] + x[group * 8 + LOWEST_INPUT[byte]];
    }
    total += tables[base + TABLE_SIZE - 1];
  }
  return total;
}

/**
 * Sum the inputs under each row's set bits.
 * @param packed the picture's packed rows
 * @param first byte offset of the first row's first input
 * @param stride bytes from one picture row to the next
 * @param groups bytes of each row to read (inputs / 8)
 * @param out one sum per row; `out.length` rows are read
 */
export function sumSetBits(
  packed: Uint8Array,
  first: number,
  stride: number,
  groups: number,
  tables: Float64Array,
  out: Float64Array,
): void {
  for (let row = 0; row < out.length; row++) {
    const start = first + row * stride;
    let sum = 0;
    for (let group = 0; group < groups; group++) sum += tables[group * TABLE_SIZE + packed[start + group]];
    out[row] = sum;
  }
}

/**
 * The engine's inner loop behind one interface, so the same engine can run on
 * the TypeScript loops above or on the Rust build of them.
 */
export interface Kernel {
  /** The weights this kernel reads. Write bits here to change the model. */
  readonly packed: Uint8Array;
  /** Prepare the lookup tables for an input vector. Returns the sum of the inputs. */
  buildTables(x: Float64Array): number;
  /** Sum the prepared inputs under the set bits of `out.length` rows starting at byte `first`. */
  sumSetBits(first: number, stride: number, out: Float64Array): void;
}

/** Makes a kernel for a model's packed weights and its longest input slice. */
export type KernelFactory = (packed: Uint8Array, inputs: number) => Kernel;

/** The kernel in plain TypeScript. It reads `packed` in place. */
export const tableKernel: KernelFactory = (packed, inputs) => {
  const tables = tableSpace(inputs);
  let groups = 0;
  return {
    packed,
    buildTables(x) {
      groups = x.length / 8;
      return buildTables(x, tables);
    },
    sumSetBits: (first, stride, out) => sumSetBits(packed, first, stride, groups, tables, out),
  };
};
