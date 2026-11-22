// A soft glow around the brightest parts of the frame, so strongly working
// weights and fresh damage read from a distance.

import * as THREE from "three";

const VERTEX = /* glsl */ `
  varying vec2 vUv;
  void main() {
    vUv = uv;
    gl_Position = vec4(position.xy, 0.0, 1.0);
  }
`;

const BRIGHT = /* glsl */ `
  uniform sampler2D uMap;
  varying vec2 vUv;
  void main() {
    vec3 colour = texture2D(uMap, vUv).rgb;
    float level = max(colour.r, max(colour.g, colour.b));
    gl_FragColor = vec4(colour * smoothstep(0.62, 0.95, level), 1.0);
  }
`;

const BLUR = /* glsl */ `
  uniform sampler2D uMap;
  uniform vec2 uStep;
  varying vec2 vUv;
  void main() {
    vec3 sum = texture2D(uMap, vUv).rgb * 0.227;
    sum += (texture2D(uMap, vUv + uStep * 1.38).rgb + texture2D(uMap, vUv - uStep * 1.38).rgb) * 0.316;
    sum += (texture2D(uMap, vUv + uStep * 3.23).rgb + texture2D(uMap, vUv - uStep * 3.23).rgb) * 0.07;
    gl_FragColor = vec4(sum, 1.0);
  }
`;

const COMPOSE = /* glsl */ `
  uniform sampler2D uMap;
  uniform sampler2D uGlow;
  varying vec2 vUv;
  void main() {
    gl_FragColor = vec4(texture2D(uMap, vUv).rgb + texture2D(uGlow, vUv).rgb * 0.9, 1.0);
  }
`;

const material = (fragmentShader: string, uniforms: Record<string, THREE.IUniform>) =>
  new THREE.ShaderMaterial({ vertexShader: VERTEX, fragmentShader, uniforms, depthTest: false });

/** Returns `render(scene, camera)`, which draws the scene with glow added. */
export function glowRenderer(renderer: THREE.WebGLRenderer) {
  const frame = new THREE.WebGLRenderTarget(1, 1, { samples: 4 });
  const small = [new THREE.WebGLRenderTarget(1, 1), new THREE.WebGLRenderTarget(1, 1)];
  const bright = material(BRIGHT, { uMap: { value: frame.texture } });
  const blur = material(BLUR, { uMap: { value: null }, uStep: { value: new THREE.Vector2() } });
  const compose = material(COMPOSE, { uMap: { value: frame.texture }, uGlow: { value: small[0].texture } });
  const quad = new THREE.Mesh(new THREE.PlaneGeometry(2, 2), bright);
  const stage = new THREE.Scene().add(quad);
  const flat = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const size = new THREE.Vector2();

  function pass(shader: THREE.ShaderMaterial, target: THREE.WebGLRenderTarget | null) {
    quad.material = shader;
    renderer.setRenderTarget(target);
    renderer.render(stage, flat);
  }

  return function render(scene: THREE.Scene, camera: THREE.Camera) {
    renderer.getDrawingBufferSize(size);
    if (frame.width !== size.x || frame.height !== size.y) {
      frame.setSize(size.x, size.y);
      for (const target of small) target.setSize(Math.ceil(size.x / 4), Math.ceil(size.y / 4));
    }
    renderer.setRenderTarget(frame);
    renderer.render(scene, camera);
    pass(bright, small[0]);
    for (let round = 0; round < 2; round++) {
      blur.uniforms.uMap.value = small[0].texture;
      blur.uniforms.uStep.value.set(1 / small[0].width, 0);
      pass(blur, small[1]);
      blur.uniforms.uMap.value = small[1].texture;
      blur.uniforms.uStep.value.set(0, 1 / small[0].height);
      pass(blur, small[0]);
    }
    pass(compose, null);
  };
}
