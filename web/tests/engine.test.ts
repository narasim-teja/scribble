import { existsSync, readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "bun:test";
import { Engine } from "../src/model/engine";
import { buildTables, sumSetBits, tableKernel, tableSpace } from "../src/model/kernel";
import { readPicture } from "../src/model/picture";
import { wasmKernel } from "../src/model/wasm-kernel";

const TOLERANCE = 1e-3; // milestone M3: logits match Python within this
const here = (path: string) => fileURLToPath(new URL(path, import.meta.url));

interface Reference {
  tokens: number[];
  logits: number[][];
}

async function load(picturePath: string) {
  const file = readFileSync(picturePath);
  const picture = await readPicture(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
  const reference: Reference = JSON.parse(readFileSync(picturePath.replace(/\.png$/, ".reference.json"), "utf8"));
  return { picture, reference };
}

/** Largest difference between the engine's logits and Python's over the whole run. */
function worstError(engine: Engine, reference: Reference): number {
  let worst = 0;
  reference.tokens.forEach((token, position) => {
    const logits = engine.step(token);
    reference.logits[position].forEach((want, i) => (worst = Math.max(worst, Math.abs(logits[i] - want))));
  });
  return worst;
}

describe("kernel", () => {
  it("sums the inputs under each row's set bits", () => {
    const inputs = 16, rows = 5, stride = 4;
    const packed = Uint8Array.from({ length: rows * stride }, (_, i) => (i * 73 + 41) % 256);
    const x = Float64Array.from({ length: inputs }, (_, i) => Math.sin(i + 1));
    const tables = tableSpace(inputs);
    const total = buildTables(x, tables);
    const out = new Float64Array(rows);
    sumSetBits(packed, 1, stride, inputs / 8, tables, out);

    expect(total).toBeCloseTo(x.reduce((a, b) => a + b), 12);
    for (let row = 0; row < rows; row++) {
      let want = 0;
      for (let j = 0; j < inputs; j++) {
        if ((packed[row * stride + 1 + (j >> 3)] >> (7 - (j & 7))) & 1) want += x[j];
      }
      expect(out[row]).toBeCloseTo(want, 12);
    }
  });
});

const rust = wasmKernel(new WebAssembly.Module(readFileSync(here("../src/model/kernel.wasm"))));

describe("rust kernel", () => {
  it("gives the same sums as the TypeScript kernel", () => {
    const inputs = 64, rows = 40, stride = 16;
    const weights = Uint8Array.from({ length: rows * stride }, (_, i) => (i * 97 + 13) % 256);
    const x = Float64Array.from({ length: inputs }, (_, i) => Math.cos(i * 1.7));
    const sums = [tableKernel, rust].map((factory) => {
      const kernel = factory(weights, inputs);
      const out = new Float64Array(rows);
      const total = kernel.buildTables(x);
      kernel.sumSetBits(8, stride, out.subarray(0, rows - 1));
      return [total, ...out];
    });
    sums[0].forEach((want, i) => expect(sums[1][i]).toBeCloseTo(want, 10));
  });

  it("matches Python on the tiny model", async () => {
    const { picture, reference } = await load(here("./fixtures/tiny.png"));
    expect(worstError(new Engine(picture, { kernel: rust }), reference)).toBeLessThan(TOLERANCE);
  });

  it("changes its answer when its own copy of a weight is flipped", async () => {
    const { picture, reference } = await load(here("./fixtures/tiny.png"));
    const engine = new Engine(picture, { kernel: rust });
    const run = () => {
      engine.reset();
      return reference.tokens.slice(0, 6).map((token) => engine.step(token)).at(-1)!;
    };
    const before = run();
    engine.weights[0] ^= 0x80;
    const after = run();
    expect(after.some((value, i) => Math.abs(value - before[i]) > 1e-6)).toBe(true);
  });
});

describe("engine", () => {
  it("matches Python on the tiny model, from the picture alone", async () => {
    const { picture, reference } = await load(here("./fixtures/tiny.png"));
    expect(worstError(new Engine(picture), reference)).toBeLessThan(TOLERANCE);
  });

  it("gives the same answers again after a reset", async () => {
    const { picture, reference } = await load(here("./fixtures/tiny.png"));
    const engine = new Engine(picture);
    worstError(engine, reference);
    engine.reset();
    expect(worstError(engine, reference)).toBeLessThan(TOLERANCE);
  });

  it("changes its answer when one pixel is flipped", async () => {
    const { picture, reference } = await load(here("./fixtures/tiny.png"));
    // Several tokens, because the first weight belongs to a query, and a
    // query cannot matter while there is only one token to attend to.
    const run = () => {
      const engine = new Engine(picture);
      return reference.tokens.slice(0, 6).map((token) => engine.step(token)).at(-1)!;
    };
    const before = run();
    picture.packed[0] ^= 0x80; // the first weight of the first square
    const after = run();
    expect(after.some((value, i) => Math.abs(value - before[i]) > 1e-6)).toBe(true);
  });

  // The trained models are large and git-ignored, so these run only where they exist.
  for (const run of ["bpe512-4m", "bpe512-16m"]) {
    const trained = here(`../../train/runs/${run}/brain.png`);
    it.skipIf(!existsSync(trained))(`matches Python on the trained ${run} model`, async () => {
      const { picture, reference } = await load(trained);
      expect(worstError(new Engine(picture), reference)).toBeLessThan(TOLERANCE);
      expect(worstError(new Engine(picture, { kernel: rust }), reference)).toBeLessThan(TOLERANCE);
    });
  }
});
