// The kernel compiled from Rust (web/kernel). Same answers as the TypeScript
// one, faster on the large model. The weights live inside the module's
// memory so the Rust code can read them directly.

import type { Kernel, KernelFactory } from "./kernel";

const PAGE = 65536;
const TABLE_SIZE = 256;

interface Exports {
  memory: WebAssembly.Memory;
  build_tables(x: number, groups: number, tables: number): number;
  sum_set_bits(packed: number, stride: number, groups: number, tables: number, out: number, rows: number): void;
}

/**
 * A kernel factory backed by a compiled `kernel.wasm`. Each kernel gets its
 * own instance, and so its own copy of the weights.
 */
export function wasmKernel(module: WebAssembly.Module): KernelFactory {
  return (weights, inputs): Kernel => {
    const { memory, build_tables, sum_set_bits } = new WebAssembly.Instance(module).exports as unknown as Exports;
    // Our data goes after everything the module itself uses.
    const packedAt = memory.buffer.byteLength;
    const inputAt = Math.ceil((packedAt + weights.length) / 8) * 8;
    const tablesAt = inputAt + inputs * 8;
    const outAt = tablesAt + (inputs / 8) * TABLE_SIZE * 8;
    const end = outAt + inputs * 8;
    memory.grow(Math.ceil((end - packedAt) / PAGE));

    const packed = new Uint8Array(memory.buffer, packedAt, weights.length);
    packed.set(weights);
    const input = new Float64Array(memory.buffer, inputAt, inputs);
    const sums = new Float64Array(memory.buffer, outAt, inputs);
    let groups = 0;
    return {
      packed,
      buildTables(x) {
        input.set(x);
        groups = x.length / 8;
        return build_tables(inputAt, groups, tablesAt);
      },
      sumSetBits(first, stride, out) {
        sum_set_bits(packedAt + first, stride, groups, tablesAt, outAt, out.length);
        out.set(sums.subarray(0, out.length));
      },
    };
  };
}
