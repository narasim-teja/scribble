// Reads the model picture written by train/export.py. The PNG bytes are
// decoded here and never through a canvas, because browsers may add noise to
// canvas pixels and one changed pixel is one changed weight.

const MANIFEST_KEY = "scribble";
const SIDE_TABLE_CHUNK = "sbTb";
const FORMAT_VERSION = 2;

/** One square of the picture: a rectangle of one weight matrix. */
export interface Piece {
  tensor: string;
  label: string;
  layer: number;
  /** Half-open ranges into the matrix. */
  rows: [number, number];
  cols: [number, number];
  /** Top-left pixel of the square. */
  x: number;
  y: number;
}

export interface Config {
  vocab_size: number;
  context: number;
  layers: number;
  heads: number;
  width: number;
}

export interface Manifest {
  version: number;
  config: Config;
  side: number;
  square: number;
  pieces: Piece[];
  scales: Record<string, number>;
  /** Where each named block starts in the side table, in bytes; two bytes per number. */
  side_table: { name: string; shape: number[]; offset: number }[];
  bits_crc32: number;
  vocab: { name: string; tokens: string[] };
}

export interface Picture {
  manifest: Manifest;
  /** The weights, one bit each, packed as in the file: `side / 8` bytes per row, first pixel in the high bit. */
  packed: Uint8Array;
  /** Bytes per picture row in `packed`. */
  stride: number;
  /** The numbers that are not one bit (embeddings, gains). */
  sideTable: Float32Array;
}

/**
 * Little-endian 16-bit floats to ordinary numbers. Done by hand because
 * `Float16Array` is not in every browser yet.
 */
function halfFloats(bytes: Uint8Array): Float32Array<ArrayBuffer> {
  const out = new Float32Array(bytes.length / 2);
  for (let i = 0; i < out.length; i++) {
    const half = bytes[2 * i] | (bytes[2 * i + 1] << 8);
    const exponent = (half >> 10) & 31, fraction = half & 1023;
    const size =
      exponent === 0 ? fraction * 2 ** -24 : exponent === 31 ? (fraction ? NaN : Infinity) : (1024 + fraction) * 2 ** (exponent - 25);
    out[i] = half & 0x8000 ? -size : size;
  }
  return out;
}

async function inflate(parts: Uint8Array[]): Promise<Uint8Array> {
  const stream = new Blob(parts as BlobPart[]).stream().pipeThrough(new DecompressionStream("deflate"));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}

export async function readPicture(buffer: ArrayBuffer): Promise<Picture> {
  const raw = new Uint8Array(buffer);
  const view = new DataView(buffer);
  const latin1 = new TextDecoder("latin1");
  let side = 0;
  let manifest: Manifest | null = null;
  let sideTable = new Float32Array(0);
  const data: Uint8Array[] = [];
  for (let at = 8; at < raw.length; ) {
    const length = view.getUint32(at);
    const kind = latin1.decode(raw.subarray(at + 4, at + 8));
    const body = raw.subarray(at + 8, at + 8 + length);
    if (kind === "IHDR") side = view.getUint32(at + 8);
    if (kind === "IDAT") data.push(body);
    if (kind === SIDE_TABLE_CHUNK) sideTable = halfFloats(body);
    if (kind === "tEXt") {
      const text = latin1.decode(body);
      const split = text.indexOf("\0");
      if (text.slice(0, split) === MANIFEST_KEY) manifest = JSON.parse(text.slice(split + 1));
    }
    at += 12 + length;
  }
  if (!manifest) throw new Error("not a scribble picture: no manifest");
  if (manifest.version !== FORMAT_VERSION) throw new Error(`picture format ${manifest.version}, expected ${FORMAT_VERSION}`);

  const rows = await inflate(data);
  const stride = side / 8;
  const packed = new Uint8Array(side * stride);
  for (let y = 0; y < side; y++) {
    const start = y * (stride + 1);
    if (rows[start] !== 0) throw new Error("expected PNG filter type 0");
    packed.set(rows.subarray(start + 1, start + 1 + stride), y * stride);
  }
  return { manifest, packed, stride, sideTable };
}

/** A named block of the side table. */
export function sideValues(picture: Picture, name: string): Float32Array {
  const entry = picture.manifest.side_table.find((e) => e.name === name);
  if (!entry) throw new Error(`side table has no ${name}`);
  const count = entry.shape.reduce((a, b) => a * b, 1);
  return picture.sideTable.subarray(entry.offset / 2, entry.offset / 2 + count);
}
