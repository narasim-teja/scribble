import { readFileSync } from "node:fs";
const dir = new URL(".", import.meta.url).pathname;
const { instance } = await WebAssembly.instantiate(readFileSync(dir + "k.wasm"));
const ex = instance.exports;
const COLS = 1024, BPR = COLS / 8; const HB = ex.memory.buffer.byteLength;
function jsNaive(w8, x, y, rows, cols) {
  for (let r = 0; r < rows; r++) { let s = 0, o = r * cols; for (let c = 0; c < cols; c++) s += w8[o + c] * x[c]; y[r] = s; }
}
function jsLut(bits, x, y, rows, cols, tab) {
  const bpr = cols >> 3;
  for (let g = 0; g < bpr; g++) {
    const t = g << 8, xs = g << 3; let s = 0;
    for (let k = 0; k < 8; k++) s -= x[xs + k];
    tab[t] = s;
    for (let b = 1; b < 256; b++) tab[t + b] = tab[t + (b & (b - 1))] + 2 * x[xs + (31 - Math.clz32(b & -b))];
  }
  for (let r = 0; r < rows; r++) {
    const o = r * bpr; let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
    for (let g = 0; g < bpr; g += 4) {
      a0 += tab[(g << 8) + bits[o + g]]; a1 += tab[((g + 1) << 8) + bits[o + g + 1]];
      a2 += tab[((g + 2) << 8) + bits[o + g + 2]]; a3 += tab[((g + 3) << 8) + bits[o + g + 3]];
    }
    y[r] = (a0 + a1) + (a2 + a3);
  }
}
function time(fn, minMs = 600) { fn(); fn(); let n = 0; const t0 = performance.now(); let t; do { fn(); n++; t = performance.now() - t0; } while (t < minMs); return t / n; }
const rt = typeof Bun !== "undefined" ? "bun/JSC " + Bun.version : "node/V8 " + process.version;
console.log(rt);
for (const M of [8, 32, 64]) {
  const N = M * 1024 * 1024, rows = N / COLS, nbytes = N / 8;
  const need = HB + 65536 + nbytes + COLS * 4 + rows * 4 + BPR * 256 * 4 + 4096;
  const pages = Math.ceil(need / 65536) - ex.memory.buffer.byteLength / 65536;
  if (pages > 0) ex.memory.grow(pages);
  const base = (HB + 65535) & ~65535;
  const pB = base, pX = (pB + nbytes + 15) & ~15, pY = pX + COLS * 4, pT = pY + rows * 4;
  const mem = ex.memory.buffer;
  const bits = new Uint8Array(mem, pB, nbytes), x = new Float32Array(mem, pX, COLS), y = new Float32Array(mem, pY, rows), tab = new Float32Array(mem, pT, BPR * 256);
  let s = 12345; const rnd = () => (s = (s * 1664525 + 1013904223) >>> 0) / 4294967296;
  for (let i = 0; i < nbytes; i++) bits[i] = rnd() * 256;
  for (let i = 0; i < COLS; i++) x[i] = rnd() * 2 - 1;
  const w8 = new Int8Array(N); for (let i = 0; i < N; i++) w8[i] = (bits[i >> 3] >> (i & 7)) & 1 ? 1 : -1;
  const yj = new Float32Array(rows), yl = new Float32Array(rows), tj = new Float32Array(BPR * 256);
  const res = {
    "JS naive Int8": time(() => jsNaive(w8, x, yj, rows, COLS)),
    "JS bit-packed LUT": time(() => jsLut(bits, x, yl, rows, COLS, tj)),
    "WASM SIMD xor-mask": time(() => ex.matvec_simd(pB, pX, pY, rows, COLS)),
  };
  let e1 = 0; for (let r = 0; r < rows; r++) e1 = Math.max(e1, Math.abs(y[r] - yj[r]), Math.abs(yl[r] - yj[r]));
  res["WASM scalar LUT"] = time(() => ex.matvec_lut(pB, pX, pY, rows, COLS, pT));
  let e2 = 0; for (let r = 0; r < rows; r++) e2 = Math.max(e2, Math.abs(y[r] - yj[r]));
  console.log(`\n${M}M weights (max abs diff vs naive: ${Math.max(e1, e2).toExponential(1)})`);
  for (const [k, ms] of Object.entries(res)) console.log(`  ${k.padEnd(20)} ${ms.toFixed(2).padStart(7)} ms/pass  ${(1000 / ms).toFixed(0).padStart(5)} passes/s`);
}
