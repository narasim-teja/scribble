import * as THREE from "three";

/** A one-channel byte texture over `data`. Set `needsUpdate` after changing the bytes. */
export function byteTexture(
  data: Uint8Array,
  width: number,
  height: number,
  filter: THREE.MagnificationTextureFilter,
): THREE.DataTexture {
  const texture = new THREE.DataTexture(data, width, height, THREE.RedFormat, THREE.UnsignedByteType);
  texture.minFilter = texture.magFilter = filter;
  texture.unpackAlignment = 1;
  texture.needsUpdate = true;
  return texture;
}
