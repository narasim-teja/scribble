// A camera that circles a target point: drag to turn, zoom toward what is
// under the pointer, and glide between saved poses.

import * as THREE from "three";

/** Where the camera is: angles around a target point, and how far from it. */
export interface Pose {
  azimuth: number;
  elevation: number;
  distance: number;
  target: [number, number, number];
}

const TURN_PER_PIXEL = 0.006;
const FARTHEST = 9;
const GLIDE_SECONDS = 1.4;
const SWAY_RATE = 0.22;
const FACE_SURFACE_WITHIN = 2.5; // start turning to face the surface inside this distance

export class OrbitCamera {
  readonly camera = new THREE.PerspectiveCamera(30, 1, 0.02, 40);
  /** Set once the visitor moves the view; the idle sway then stops. */
  touched = false;

  private azimuth: number;
  private elevation: number;
  private distance: number;
  private readonly target: THREE.Vector3;
  private glide: { from: Pose; to: Pose; t: number } | null = null;

  /**
   * @param sway how far the idle view swings either side of `home`, in radians
   * @param nearest closest allowed viewing distance
   */
  constructor(
    private readonly home: Pose,
    private readonly sway: number,
    private readonly nearest: number,
  ) {
    ({ azimuth: this.azimuth, elevation: this.elevation, distance: this.distance } = home);
    this.target = new THREE.Vector3(...home.target);
  }

  get pose(): Pose {
    const { azimuth, elevation, distance } = this;
    return { azimuth, elevation, distance, target: this.target.toArray() };
  }

  set pose(pose: Pose) {
    ({ azimuth: this.azimuth, elevation: this.elevation, distance: this.distance } = pose);
    this.target.set(...pose.target);
    this.glide = null;
  }

  resize(aspect: number): void {
    this.camera.aspect = aspect;
    // Widen the lens when the view is narrow so the whole brain stays in it.
    this.camera.fov = 30 / Math.min(1, Math.max(aspect, 0.5) / 1.05);
    this.camera.updateProjectionMatrix();
  }

  /** Turn by a pointer movement in pixels. */
  orbit(dx: number, dy: number): void {
    this.stopGlide();
    this.azimuth -= dx * TURN_PER_PIXEL;
    this.elevation = Math.min(1.45, Math.max(-1.2, this.elevation + dy * TURN_PER_PIXEL));
  }

  /**
   * Scale the viewing distance. Moving in drifts toward `toward.point` and
   * turns to face the surface there squarely, so weights are not seen edge-on.
   */
  zoom(factor: number, toward?: { point: THREE.Vector3; normal?: THREE.Vector3 } | null): void {
    this.stopGlide();
    const next = Math.min(FARTHEST, Math.max(this.nearest, this.distance * factor));
    if (toward && next < this.distance) {
      const share = 1 - next / this.distance;
      this.target.lerp(toward.point, share);
      if (toward.normal && this.distance < FACE_SURFACE_WITHIN) {
        const facing = Math.atan2(toward.normal.x, toward.normal.z);
        const turn = Math.atan2(Math.sin(facing - this.azimuth), Math.cos(facing - this.azimuth));
        this.azimuth += turn * share * 0.6;
        this.elevation += (Math.asin(toward.normal.y) - this.elevation) * share * 0.6;
      }
    }
    if (next > this.distance) this.target.lerp(new THREE.Vector3(...this.home.target), (next - this.distance) / 6);
    this.distance = next;
  }

  /** Move smoothly to `pose`. */
  glideTo(pose: Pose): void {
    this.touched = true;
    this.glide = { from: this.pose, to: pose, t: 0 };
  }

  /** Advance glides and the idle sway, then place the camera. */
  update(dt: number, seconds: number, calm: boolean): void {
    if (this.glide) {
      const { from, to } = this.glide;
      const t = (this.glide.t = Math.min(1, this.glide.t + dt / (calm ? 0.01 : GLIDE_SECONDS)));
      const eased = 1 - (1 - t) ** 4;
      const mix = (a: number, b: number) => a + (b - a) * eased;
      this.azimuth = mix(from.azimuth, to.azimuth);
      this.elevation = mix(from.elevation, to.elevation);
      this.distance = mix(from.distance, to.distance);
      this.target.set(mix(from.target[0], to.target[0]), mix(from.target[1], to.target[1]), mix(from.target[2], to.target[2]));
      if (t === 1) this.glide = null;
    } else if (!this.touched && !calm) {
      this.azimuth = this.home.azimuth + this.sway * Math.sin(seconds * SWAY_RATE);
    }
    const flat = this.distance * Math.cos(this.elevation);
    this.camera.position.set(
      this.target.x + flat * Math.sin(this.azimuth),
      this.target.y + this.distance * Math.sin(this.elevation),
      this.target.z + flat * Math.cos(this.azimuth),
    );
    this.camera.lookAt(this.target);
  }

  private stopGlide(): void {
    this.touched = true;
    this.glide = null;
  }
}
