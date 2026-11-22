# Kernel benchmark

Times one pass over N one-bit weights (a matrix-vector product, 1024 columns)
four ways: a naive JS loop, a JS lookup-table kernel on bit-packed weights, and
two Rust kernels compiled to WASM. On an M2 Max the table kernel was about ten
times faster than the naive loop and Rust about twice as fast again; the SIMD
build was no better than the table one.

```
rustc --target wasm32-unknown-unknown --crate-type cdylib -C opt-level=3 \
  -C target-feature=+simd128 -C panic=abort -o k.wasm k.rs
bun bench.mjs
node bench.mjs
```
