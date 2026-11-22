// Turns the brain built by build_mesh.py into the geometry the page draws:
// the weight tiles (one per square of the picture, wrapped onto the cortex)
// and the plain tissue under them.

import * as THREE from "three";

export const FLAT_SIZE = 2.1; // world width of the unfolded picture
export const FLAT_GAP = 0.012; // share of a square left empty at each edge when unfolded
const ROW = 7; // floats per vertex in mesh.bin: position, normal, groove

function split(rows: Float32Array, count: number): THREE.BufferGeometry {
  const position = new Float32Array(count * 3), normal = new Float32Array(count * 3);
  const groove = new Float32Array(count);
  for (let v = 0; v < count; v++) {
    position.set(rows.subarray(v * ROW, v * ROW + 3), v * 3);
    normal.set(rows.subarray(v * ROW + 3, v * ROW + 6), v * 3);
    groove[v] = rows[v * ROW + 6];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(position, 3));
  geometry.setAttribute("normal", new THREE.BufferAttribute(normal, 3));
  geometry.setAttribute("groove", new THREE.BufferAttribute(groove, 1));
  return geometry;
}

/** What build_mesh.py writes beside mesh.bin. */
export interface MeshMeta {
  tiles: number;
  segments: number;
  tissue_vertices: number;
  triangles: number;
}

/**
 * @param buffer mesh.bin
 * @param across squares along one edge of the picture
 * @returns `tiles` has extra attributes: `unfolded` (position in the flat
 *   picture), `uv` (position inside the square) and `square` (which square).
 */
export function brainGeometry(meta: MeshMeta, buffer: ArrayBuffer, across: number) {
  const ring = meta.segments + 1;
  const tileVertices = meta.tiles * ring * ring;
  const tileRows = new Float32Array(buffer, 0, tileVertices * ROW);
  const tissueRows = new Float32Array(buffer, tileRows.byteLength, meta.tissue_vertices * ROW);
  const faces = new Uint32Array(buffer, tileRows.byteLength + tissueRows.byteLength, meta.triangles * 3);

  const tissue = split(tissueRows, meta.tissue_vertices);
  tissue.setIndex(new THREE.BufferAttribute(faces, 1));

  const tiles = split(tileRows, tileVertices);
  const unfolded = new Float32Array(tileVertices * 3), uv = new Float32Array(tileVertices * 2);
  const square = new Float32Array(tileVertices);
  const index = new Uint32Array(meta.tiles * meta.segments * meta.segments * 6);
  let v = 0, i = 0;
  for (let n = 0; n < meta.tiles; n++) {
    const first = v;
    for (let r = 0; r < ring; r++) {
      for (let c = 0; c < ring; c++) {
        const a = c / meta.segments, b = r / meta.segments;
        const x = ((n % across) + FLAT_GAP + a * (1 - 2 * FLAT_GAP)) / across;
        const y = (Math.floor(n / across) + FLAT_GAP + b * (1 - 2 * FLAT_GAP)) / across;
        unfolded.set([(x - 0.5) * FLAT_SIZE, (0.5 - y) * FLAT_SIZE, 0], v * 3);
        uv.set([a, b], v * 2);
        square[v++] = n;
      }
    }
    for (let r = 0; r < meta.segments; r++) {
      for (let c = 0; c < meta.segments; c++) {
        const p = first + r * ring + c;
        index.set([p, p + 1, p + ring, p + 1, p + ring + 1, p + ring], i);
        i += 6;
      }
    }
  }
  tiles.setAttribute("unfolded", new THREE.BufferAttribute(unfolded, 3));
  tiles.setAttribute("uv", new THREE.BufferAttribute(uv, 2));
  tiles.setAttribute("square", new THREE.BufferAttribute(square, 1));
  tiles.setIndex(new THREE.BufferAttribute(index, 1));
  return { tiles, tissue };
}

/** Cerebellum and brainstem: plain shapes that complete the silhouette. */
export function hindbrainGeometry(): THREE.BufferGeometry[] {
  const cerebellum = new THREE.SphereGeometry(1, 128, 96);
  const p = cerebellum.attributes.position;
  for (let k = 0; k < p.count; k++) {
    const y = p.getY(k);
    const ridge = 1 + 0.012 * Math.sin(y * 95) * (1 - y * y);
    p.setXYZ(k, p.getX(k) * 0.5 * ridge, y * 0.2 - 0.5, p.getZ(k) * 0.32 * ridge - 0.56);
  }
  const stem = new THREE.CylinderGeometry(0.1, 0.06, 0.36, 40, 1);
  stem.rotateX(-0.25);
  stem.translate(0, -0.66, -0.26);
  const parts = [cerebellum, stem];
  for (const part of parts) {
    part.computeVertexNormals();
    const flat = new Float32Array(part.attributes.position.count);
    part.setAttribute("groove", new THREE.BufferAttribute(flat, 1));
  }
  return parts;
}
