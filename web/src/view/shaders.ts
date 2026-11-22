// GLSL for the two materials. three.js supplies the matrices and converts
// these to WebGL2 syntax.

const LIGHTING = /* glsl */ `
  const vec3 KEY = normalize(vec3(-0.5, 0.75, 0.45));
  const vec3 BONE = vec3(0.86, 0.81, 0.72);
  const vec3 INK = vec3(0.055, 0.05, 0.045);

  // Soft studio light: one key from above, a little fill, a thin rim.
  float shade(vec3 n, vec3 toEye) {
    float key = max(dot(n, KEY), 0.0);
    float fill = 0.5 + 0.5 * dot(n, vec3(0.6, -0.2, 0.5));
    return 0.03 + 1.05 * key * key + 0.1 * fill;
  }
  float rim(vec3 n, vec3 toEye) {
    return pow(1.0 - max(dot(n, toEye), 0.0), 3.0);
  }
  vec3 toScreen(vec3 linear) {
    return pow(clamp(linear, 0.0, 1.0), vec3(1.0 / 2.2));
  }
`;

export const TILE_VERTEX = /* glsl */ `
  attribute vec3 unfolded;
  attribute float square;
  attribute float groove;
  uniform float uFold;
  uniform float uSquares;
  varying vec2 vLocal;
  varying float vSquare;
  varying float vGroove;
  varying float vFold;
  varying vec3 vNormal;
  varying vec3 vWorld;

  void main() {
    // Each tile leaves the brain a little after the one before it.
    float t = clamp(uFold * 1.7 - 0.7 * square / uSquares, 0.0, 1.0);
    float fold = t * t * (3.0 - 2.0 * t);
    vec3 lifted = position + normal * 0.5 * sin(fold * 3.14159);
    vec3 world = mix(lifted, unfolded, fold);
    vLocal = uv;
    vSquare = square;
    vGroove = groove * (1.0 - fold);
    vFold = fold;
    vNormal = normalize(mix(normal, vec3(0.0, 0.0, 1.0), fold));
    vWorld = world;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(world, 1.0);
  }
`;

export const TILE_FRAGMENT = /* glsl */ `
  uniform sampler2D uBits;     // one texel per weight: bit 0 is the weight, bit 1 marks damage
  uniform sampler2D uActivity; // per square, two rows: input per column, output per row
  uniform sampler2D uActivityLow;
  uniform sampler2D uActivityFar;
  uniform sampler2D uPulse; // per square: how recently it was computed for this token
  uniform float uSide;
  uniform float uSquareSize;
  uniform float uAcross;
  uniform float uSquares;
  uniform vec4 uBrush; // square, centre x, centre y, radius (in square units); square < 0 hides it
  uniform vec4 uStamps[STAMPS]; // recent brush marks: square, centre x, centre y, time made
  uniform float uTime;
  varying vec2 vLocal;
  varying float vSquare;
  varying float vGroove;
  varying float vFold;
  varying vec3 vNormal;
  varying vec3 vWorld;
  ${LIGHTING}
  const vec3 AMBER = vec3(1.0, 0.44, 0.0);
  const vec3 GLOW = vec3(1.0, 0.93, 0.8);

  float segment(vec2 p, vec2 a, vec2 b) {
    vec2 pa = p - a, ba = b - a;
    return length(pa - ba * clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0));
  }

  // Distance to the digit drawn in a weight's cell: 1 for +scale, 0 for -scale.
  // p is centred on the cell with y up; negative means inside the stroke.
  float digit(vec2 p, float bit) {
    float one = min(segment(p, vec2(0.02, -0.24), vec2(0.02, 0.24)), segment(p, vec2(0.02, 0.24), vec2(-0.1, 0.12)));
    float zero = abs(length(p / vec2(0.15, 0.24)) - 1.0) * 0.17;
    return mix(zero, one, bit) - 0.05;
  }

  void main() {
    float n = floor(vSquare + 0.5);
    vec2 cell = vec2(mod(n, uAcross), floor(n / uAcross));
    vec2 pixel = (cell + clamp(vLocal, 0.0, 0.9999)) * uSquareSize;
    float texelsPerPixel = max(length(dFdx(pixel)), length(dFdy(pixel)));

    float byte = texture2D(uBits, (floor(pixel) + 0.5) / uSide).r * 255.0;
    float bit = mod(byte, 2.0);
    float damaged = step(1.5, byte);

    // Far away many weights share a pixel and average to grey, which is the
    // truth. Up close each weight is drawn as itself.
    float detail = smoothstep(1.3, 0.45, texelsPerPixel);
    float tone = mix(0.5, bit, detail);

    float rows = uSquares * 2.0;
    vec2 inAt = vec2(vLocal.x, (n * 2.0 + 0.5) / rows);
    vec2 outAt = vec2(vLocal.y, (n * 2.0 + 1.5) / rows);
    // A weight's work right now: the input it multiplies times the output it
    // feeds. From far away it is averaged over a patch of weights, which
    // turns the fine weave into soft areas of light.
    float fine = smoothstep(0.7, 0.3, texelsPerPixel);
    float reaching = mix(texture2D(uActivityLow, inAt).r, texture2D(uActivity, inAt).r, fine);
    float leaving = mix(texture2D(uActivityLow, outAt).r, texture2D(uActivity, outAt).r, fine);
    float near = pow(clamp(reaching * leaving * 3.2, 0.0, 1.0), 1.5);
    float patchWork = texture2D(uActivityFar, inAt).r * texture2D(uActivityFar, outAt).r;
    float far = pow(clamp((patchWork - 0.022) * 18.0, 0.0, 1.0), 1.6);
    // Blur the straight edge where one square's patch of light meets the next.
    vec2 fromEdge = min(vLocal, 1.0 - vLocal);
    far *= mix(0.5, 1.0, smoothstep(0.0, 0.16, min(fromEdge.x, fromEdge.y)));
    float work = mix(far, near, smoothstep(1.3, 0.7, texelsPerPixel));
    // Unfolded, the picture is read as a diagram, so keep the exact pattern.
    float crisp = smoothstep(6.0, 2.0, texelsPerPixel);
    float flatIn = mix(texture2D(uActivityLow, inAt).r, texture2D(uActivity, inAt).r, crisp);
    float flatOut = mix(texture2D(uActivityLow, outAt).r, texture2D(uActivity, outAt).r, crisp);
    work = mix(work, pow(clamp(flatIn * flatOut * 3.2, 0.0, 1.0), 1.5), vFold);
    // A square is brightest just after the model computes it, so each token
    // shows as a wave moving through the layers.
    work *= texture2D(uPulse, vec2(0.5, (n + 0.5) / uSquares)).r * 255.0 / 128.0;

    vec3 toEye = normalize(cameraPosition - vWorld);
    vec3 normal = normalize(vNormal);
    if (dot(normal, toEye) < 0.0) normal = -normal;
    float light = mix(shade(normal, toEye), 0.7, vFold) * (1.0 - 0.92 * pow(vGroove, 0.7));

    vec3 surface = mix(INK, BONE, tone) * light * 0.62;
    // Working weights brighten the surface they sit on, so the folds stay
    // visible inside the light.
    float lift = work * (0.35 + 0.65 * mix(1.0, bit, detail));
    vec3 colour = surface * (1.0 + 1.5 * lift) + GLOW * lift * (0.12 + 0.3 * light);
    colour += BONE * rim(normal, toEye) * 0.07 * (1.0 - vFold);
    if (damaged > 0.5) {
      // Damage keeps the surface's shading and the model's light, in amber.
      colour = AMBER * 0.85 * light * (0.45 + 0.55 * tone) + AMBER * work * 0.9;
      // Fresh damage is hot and cools over about a second.
      float heat = 0.0;
      for (int i = 0; i < STAMPS; i++) {
        vec4 stamp = uStamps[i];
        if (abs(n - stamp.x) > 0.5) continue;
        vec2 away = (vLocal - stamp.yz) / uBrush.w;
        heat += exp(-dot(away, away)) * exp(-(uTime - stamp.w) / 0.9);
      }
      colour += vec3(1.0, 0.72, 0.3) * min(heat, 1.5) * 0.8;
    }

    // Close enough to tell weights apart: separate them and write each one's bit.
    float close = smoothstep(0.11, 0.07, texelsPerPixel);
    if (close > 0.0) {
      vec2 cell = fract(pixel);
      float inset = max(abs(cell.x - 0.5), abs(cell.y - 0.5));
      colour *= mix(1.0, 0.25 + 0.75 * smoothstep(0.5, 0.46, inset), close);
      // Turn the digit to face the screen whichever way the surface runs,
      // and keep its proportions where the surface stretches the cells.
      vec2 acrossCell = vec2(dFdx(pixel.x), dFdy(pixel.x)), downCell = vec2(dFdx(pixel.y), dFdy(pixel.y));
      bool sideways = abs(acrossCell.x) < abs(downCell.x);
      vec2 right = sideways ? downCell : acrossCell, up = sideways ? acrossCell : downCell;
      vec2 along = sideways ? cell.yx : cell;
      vec2 upright = vec2(right.x > 0.0 ? along.x : 1.0 - along.x, up.y > 0.0 ? along.y : 1.0 - along.y) - 0.5;
      float stretch = clamp(length(up) / length(right), 0.4, 2.5); // cell width over height, on screen
      upright *= vec2(max(stretch, 1.0), max(1.0 / stretch, 1.0)) * 1.15;
      float stroke = smoothstep(texelsPerPixel, -texelsPerPixel, digit(upright, bit));
      vec3 ink = mix(BONE * 0.5, vec3(0.0), bit);
      colour = mix(colour, ink, stroke * close * 0.95);
    }

    if (abs(n - uBrush.x) < 0.5) {
      float edge = abs(distance(vLocal, uBrush.yz) - uBrush.w);
      float width = 1.5 * texelsPerPixel / uSquareSize;
      colour = mix(colour, BONE * 0.7, 0.85 * smoothstep(width, 0.0, edge));
    }
    gl_FragColor = vec4(toScreen(colour), 1.0);
  }
`;

export const TISSUE_VERTEX = /* glsl */ `
  attribute float groove;
  varying float vGroove;
  varying vec3 vNormal;
  varying vec3 vWorld;
  void main() {
    vGroove = groove;
    vNormal = normal;
    vWorld = position;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
  }
`;

export const TISSUE_FRAGMENT = /* glsl */ `
  uniform float uOpacity;
  varying float vGroove;
  varying vec3 vNormal;
  varying vec3 vWorld;
  ${LIGHTING}
  void main() {
    vec3 toEye = normalize(cameraPosition - vWorld);
    vec3 normal = normalize(vNormal);
    if (dot(normal, toEye) < 0.0) normal = -normal;
    float light = shade(normal, toEye) * (1.0 - 0.92 * pow(vGroove, 0.7));
    vec3 colour = BONE * 0.12 * light + BONE * rim(normal, toEye) * 0.04;
    gl_FragColor = vec4(toScreen(colour), uOpacity);
  }
`;
