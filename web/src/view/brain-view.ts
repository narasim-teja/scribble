// The 3D scene: the model's weights wrapped on a brain, lit by its activity,
// and able to unfold into the flat picture file.

import * as THREE from "three";
import type { Picture } from "../model/picture";
import { ActivityLight } from "./activity-light";
import { brainGeometry, FLAT_GAP, FLAT_SIZE, hindbrainGeometry, type MeshMeta } from "./brain-geometry";
import { glowRenderer } from "./glow";
import { OrbitCamera, type Pose } from "./orbit-camera";
import { TILE_FRAGMENT, TILE_VERTEX, TISSUE_FRAGMENT, TISSUE_VERTEX } from "./shaders";
import { Weights } from "./weights";

const BACKGROUND = 0x0b0a09;
const STAMPS = 12; // recent brush marks kept so fresh damage can glow
const NEAREST = 0.06; // closest viewing distance: a single weight is tens of pixels wide
const SWAY = 0.28; // idle swing either side of the profile, radians
const FOLD_SECONDS = 1.5;
const HOME: Pose = { azimuth: -1.32, elevation: 0.16, distance: 4.5, target: [0, -0.1, -0.04] };
const FACING_PICTURE: Pose = { azimuth: 0, elevation: 0, distance: 5.5, target: [-0.3, -0.12, 0] };

/** What sits under the pointer: a square of the picture and a place inside it. */
export interface Hit {
  square: number;
  /** Position inside the square, 0..1 across and down. */
  x: number;
  y: number;
  /** The same place in the scene. */
  point: THREE.Vector3;
  /** Which way the surface faces there. Absent on the flat picture. */
  normal?: THREE.Vector3;
}

export class BrainView {
  readonly orbit = new OrbitCamera(HOME, SWAY, NEAREST);
  readonly weights: Weights;
  readonly light: ActivityLight;

  private readonly renderer: THREE.WebGLRenderer;
  private readonly render: (scene: THREE.Scene, camera: THREE.Camera) => void;
  private readonly scene = new THREE.Scene();
  private readonly tiles: THREE.Mesh;
  private readonly tissue: THREE.Mesh[];
  private readonly uniforms;
  private readonly raycaster = new THREE.Raycaster();
  private readonly across: number;
  private readonly squares: number;
  private unfolded = false;
  private poseBeforeUnfold: Pose | null = null;
  private stamped = 0;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    picture: Picture,
    meshMeta: MeshMeta,
    meshData: ArrayBuffer,
  ) {
    const { side, square, pieces, config } = picture.manifest;
    this.across = side / square;
    this.squares = pieces.length;
    this.weights = new Weights(picture);
    this.light = new ActivityLight(pieces, config.layers, square);

    // The shaders write display-ready colours themselves.
    THREE.ColorManagement.enabled = false;
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    this.renderer.outputColorSpace = THREE.LinearSRGBColorSpace;
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    this.renderer.setClearColor(BACKGROUND);
    this.render = glowRenderer(this.renderer);

    this.uniforms = {
      uBits: { value: this.weights.texture },
      uActivity: { value: this.light.near },
      uActivityLow: { value: this.light.middle },
      uActivityFar: { value: this.light.far },
      uPulse: { value: this.light.pulse },
      uSide: { value: side },
      uSquareSize: { value: square },
      uAcross: { value: this.across },
      uSquares: { value: this.squares },
      uFold: { value: 0 },
      uBrush: { value: new THREE.Vector4(-1, 0, 0, 0) },
      uStamps: { value: Array.from({ length: STAMPS }, () => new THREE.Vector4(-1, 0, 0, -100)) },
      uTime: { value: 0 },
    };
    const geometry = brainGeometry(meshMeta, meshData, this.across);
    this.tiles = new THREE.Mesh(
      geometry.tiles,
      new THREE.ShaderMaterial({
        uniforms: this.uniforms,
        vertexShader: TILE_VERTEX,
        fragmentShader: TILE_FRAGMENT,
        defines: { STAMPS },
        side: THREE.DoubleSide,
      }),
    );
    const tissueMaterial = new THREE.ShaderMaterial({
      uniforms: { uOpacity: { value: 1 } },
      vertexShader: TISSUE_VERTEX,
      fragmentShader: TISSUE_FRAGMENT,
      side: THREE.DoubleSide,
      transparent: true,
    });
    this.tissue = [geometry.tissue, ...hindbrainGeometry()].map((g) => new THREE.Mesh(g, tissueMaterial));
    this.scene.add(...this.tissue, this.tiles);

    new ResizeObserver(() => this.resize()).observe(canvas);
    this.resize();
  }

  /** True while the tiles are laid out as the flat picture, or on their way there. */
  get isUnfolded(): boolean {
    return this.unfolded;
  }

  /** The square and place inside it under a pointer event, or null. */
  pick(event: { clientX: number; clientY: number }): Hit | null {
    const box = this.canvas.getBoundingClientRect();
    const pointer = new THREE.Vector2(
      ((event.clientX - box.left) / box.width) * 2 - 1,
      1 - ((event.clientY - box.top) / box.height) * 2,
    );
    this.raycaster.setFromCamera(pointer, this.orbit.camera);
    const fold = this.uniforms.uFold.value;
    if (fold === 0) {
      const [hit] = this.raycaster.intersectObject(this.tiles);
      if (!hit) return null;
      const { square, normal } = this.tiles.geometry.attributes;
      const corner = hit.face!.a;
      // The stored normal points out of the brain; the triangle's own may not.
      const outward = new THREE.Vector3().fromBufferAttribute(normal as THREE.BufferAttribute, corner);
      return { square: square.array[corner], x: hit.uv!.x, y: hit.uv!.y, point: hit.point, normal: outward };
    }
    if (fold < 1) return null;

    const { origin, direction } = this.raycaster.ray;
    const point = origin.clone().addScaledVector(direction, -origin.z / direction.z);
    const across = (point.x / FLAT_SIZE + 0.5) * this.across;
    const down = (0.5 - point.y / FLAT_SIZE) * this.across;
    const square = Math.floor(down) * this.across + Math.floor(across);
    const x = (across - Math.floor(across) - FLAT_GAP) / (1 - 2 * FLAT_GAP);
    const y = (down - Math.floor(down) - FLAT_GAP) / (1 - 2 * FLAT_GAP);
    const onPicture = across >= 0 && across < this.across && square >= 0 && square < this.squares;
    return onPicture && x >= 0 && x < 1 && y >= 0 && y < 1 ? { square, x, y, point } : null;
  }

  /** Draw the brush ring at `hit`, or hide it with null. `radius` is in square widths. */
  showBrush(hit: Hit | null, radius: number): void {
    this.uniforms.uBrush.value.set(hit ? hit.square : -1, hit?.x ?? 0, hit?.y ?? 0, radius);
  }

  /** Mark where the brush just touched, so that spot glows hot for a moment. */
  stamp(hit: Hit): void {
    this.uniforms.uStamps.value[this.stamped++ % STAMPS].set(hit.square, hit.x, hit.y, this.uniforms.uTime.value);
  }

  /** Unfold the tiles into the flat picture file, or fold them back onto the brain. */
  setUnfolded(unfolded: boolean): void {
    if (unfolded === this.unfolded) return;
    this.unfolded = unfolded;
    if (unfolded) this.poseBeforeUnfold = this.orbit.pose;
    this.orbit.glideTo(unfolded ? FACING_PICTURE : (this.poseBeforeUnfold ?? HOME));
  }

  /** Fold back if needed and return the camera to where the page opened. */
  resetView(): void {
    this.poseBeforeUnfold = null;
    this.unfolded = false;
    this.orbit.glideTo(HOME);
  }

  /** Draw one frame. `now` is in seconds. */
  frame(now: number, dt: number, calm: boolean): void {
    const fold = this.uniforms.uFold;
    const step = dt / (calm ? 0.01 : FOLD_SECONDS);
    fold.value = Math.min(1, Math.max(0, fold.value + (this.unfolded ? step : -step)));
    (this.tissue[0].material as THREE.ShaderMaterial).uniforms.uOpacity.value = 1 - Math.min(1, fold.value * 3);
    for (const mesh of this.tissue) mesh.visible = fold.value < 0.34;

    this.uniforms.uTime.value = now;
    this.light.update(now, dt, calm);
    this.orbit.update(dt, now, calm);
    this.render(this.scene, this.orbit.camera);
  }

  private resize(): void {
    const { clientWidth, clientHeight } = this.canvas;
    this.renderer.setSize(clientWidth, clientHeight, false);
    this.orbit.resize(clientWidth / clientHeight);
  }
}
