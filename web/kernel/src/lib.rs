//! The inner loop of the engine, mirroring `web/src/model/kernel.ts`.
//!
//! A weight matrix is rows of single bits, read straight from the model
//! picture. For every group of 8 inputs we first build a table of all 256
//! possible partial sums; a row's sum over its set bits is then one lookup per
//! byte. The page calls these two functions through plain WebAssembly exports
//! and owns the memory, so there is no allocator and no bindings layer here.

#![no_std]

use core::slice;

const TABLE: usize = 256;

#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! {
    loop {}
}

/// Fill `tables` so that `tables[g * 256 + byte]` is the sum of the inputs in
/// group `g` selected by the bits of `byte` (the first input is the high bit).
/// Returns the sum of all inputs.
///
/// # Safety
/// `x` must hold `groups * 8` values and `tables` room for `groups * 256`.
#[no_mangle]
pub unsafe extern "C" fn build_tables(x: *const f64, groups: usize, tables: *mut f64) -> f64 {
    let x = slice::from_raw_parts(x, groups * 8);
    let tables = slice::from_raw_parts_mut(tables, groups * TABLE);
    let mut total = 0.0;
    for (table, inputs) in tables.chunks_exact_mut(TABLE).zip(x.chunks_exact(8)) {
        table[0] = 0.0;
        for byte in 1..TABLE {
            let rest = byte & (byte - 1);
            let lowest = (byte & byte.wrapping_neg()).trailing_zeros() as usize;
            table[byte] = table[rest] + inputs[7 - lowest];
        }
        total += table[TABLE - 1];
    }
    total
}

/// For each of `rows` picture rows, sum the inputs under its set bits into `out`.
///
/// # Safety
/// `packed` must reach `(rows - 1) * stride + groups` bytes, `tables` hold
/// `groups * 256` values and `out` room for `rows`.
#[no_mangle]
pub unsafe extern "C" fn sum_set_bits(
    packed: *const u8,
    stride: usize,
    groups: usize,
    tables: *const f64,
    out: *mut f64,
    rows: usize,
) {
    let tables = slice::from_raw_parts(tables, groups * TABLE);
    let out = slice::from_raw_parts_mut(out, rows);
    for (row, sum) in out.iter_mut().enumerate() {
        let bytes = slice::from_raw_parts(packed.add(row * stride), groups);
        // Four running sums let the processor work on several lookups at once.
        let mut sums = [0.0f64; 4];
        let mut chunks = bytes.chunks_exact(4);
        let mut group = 0;
        for chunk in &mut chunks {
            for lane in 0..4 {
                sums[lane] += *tables.get_unchecked((group + lane) * TABLE + chunk[lane] as usize);
            }
            group += 4;
        }
        for &byte in chunks.remainder() {
            sums[0] += *tables.get_unchecked(group * TABLE + byte as usize);
            group += 1;
        }
        *sum = (sums[0] + sums[1]) + (sums[2] + sums[3]);
    }
}
