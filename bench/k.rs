#![no_std]
use core::arch::wasm32::*;
#[panic_handler]
fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

const S: u32 = 0x8000_0000;
const fn m(n: u32) -> [u32; 4] {
    [if n & 1 == 0 { S } else { 0 }, if n & 2 == 0 { S } else { 0 }, if n & 4 == 0 { S } else { 0 }, if n & 8 == 0 { S } else { 0 }]
}
#[repr(align(16))]
struct T([[u32; 4]; 16]);
static MASKS: T = T([m(0), m(1), m(2), m(3), m(4), m(5), m(6), m(7), m(8), m(9), m(10), m(11), m(12), m(13), m(14), m(15)]);

/// bits: row-major, 1 bit per weight, LSB first; cols multiple of 16.
#[no_mangle]
pub unsafe extern "C" fn matvec_simd(bits: *const u8, x: *const f32, y: *mut f32, rows: usize, cols: usize) {
    let bpr = cols / 8;
    let mk = MASKS.0.as_ptr() as *const v128;
    for r in 0..rows {
        let row = bits.add(r * bpr);
        let (mut a0, mut a1, mut a2, mut a3) = (f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0), f32x4_splat(0.0));
        let mut g = 0;
        while g < bpr {
            let b0 = *row.add(g) as usize;
            let b1 = *row.add(g + 1) as usize;
            let xp = x.add(g * 8) as *const v128;
            a0 = f32x4_add(a0, v128_xor(v128_load(xp), v128_load(mk.add(b0 & 15))));
            a1 = f32x4_add(a1, v128_xor(v128_load(xp.add(1)), v128_load(mk.add(b0 >> 4))));
            a2 = f32x4_add(a2, v128_xor(v128_load(xp.add(2)), v128_load(mk.add(b1 & 15))));
            a3 = f32x4_add(a3, v128_xor(v128_load(xp.add(3)), v128_load(mk.add(b1 >> 4))));
            g += 2;
        }
        let s = f32x4_add(f32x4_add(a0, a1), f32x4_add(a2, a3));
        *y.add(r) = f32x4_extract_lane::<0>(s) + f32x4_extract_lane::<1>(s) + f32x4_extract_lane::<2>(s) + f32x4_extract_lane::<3>(s);
    }
}

/// LUT: per group of 8 inputs, 256 partial sums. tab: (cols/8)*256 f32 scratch.
#[no_mangle]
pub unsafe extern "C" fn matvec_lut(bits: *const u8, x: *const f32, y: *mut f32, rows: usize, cols: usize, tab: *mut f32) {
    let bpr = cols / 8;
    for g in 0..bpr {
        let t = tab.add(g * 256);
        let xs = x.add(g * 8);
        let mut s = 0.0f32;
        for k in 0..8 { s -= *xs.add(k); }
        *t = s;
        for b in 1usize..256 {
            let k = b.trailing_zeros() as usize;
            *t.add(b) = *t.add(b & (b - 1)) + 2.0 * *xs.add(k);
        }
    }
    for r in 0..rows {
        let row = bits.add(r * bpr);
        let (mut a0, mut a1, mut a2, mut a3) = (0.0f32, 0.0f32, 0.0f32, 0.0f32);
        let mut g = 0;
        while g < bpr {
            a0 += *tab.add(g * 256 + *row.add(g) as usize);
            a1 += *tab.add((g + 1) * 256 + *row.add(g + 1) as usize);
            a2 += *tab.add((g + 2) * 256 + *row.add(g + 2) as usize);
            a3 += *tab.add((g + 3) * 256 + *row.add(g + 3) as usize);
            g += 4;
        }
        *y.add(r) = (a0 + a1) + (a2 + a3);
    }
}
