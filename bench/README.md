# Kernel benchmark

Times one pass over N one-bit weights (a matrix-vector product, 1024 columns)
four ways: a naive JS loop, a JS lookup-table kernel on bit-packed weights, and
two Rust kernels compiled to WASM. Results are in `docs/technical.md`.

```
rustc --target wasm32-unknown-unknown --crate-type cdylib -C opt-level=3 \
  -C target-feature=+simd128 -C panic=abort -o k.wasm k.rs
bun bench.mjs
node bench.mjs
```
