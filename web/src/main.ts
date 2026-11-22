// Wires the page together: the brain view, the worker that runs the model,
// the story panel and the controls.

import "./styles.css";
import { Instrument } from "./audio/instrument";
import { Hum, keys, scratch } from "./audio/sounds";
import { readPicture } from "./model/picture";
import { StoryPanel } from "./ui/story-panel";
import type { MeshMeta } from "./view/brain-geometry";
import { BrainView, type Hit } from "./view/brain-view";
import type { Change } from "./view/weights";
import type { FromWorker, ToWorker } from "./worker/messages";

const TOKENS_PER_SECOND = 5;
const HEAL_FRAMES = 70; // frames a full heal is spread over
const ZOOM_PER_WHEEL_UNIT = 0.0016;
const SMALLEST_BRUSH_WEIGHTS = 1.5; // brush radius never shrinks below this many weights
const TOUCH_SLOP_PIXELS = 8; // a finger must travel this far before it paints
const calm = matchMedia("(prefers-reduced-motion: reduce)").matches;

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

async function fetchOk(path: string): Promise<Response> {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path}: ${response.status}`);
  return response;
}

async function start(): Promise<void> {
  const [picture, meshMeta, meshData] = await Promise.all([
    fetchOk("data/brain.png").then((r) => r.arrayBuffer()).then(readPicture),
    fetchOk("data/mesh.json").then((r) => r.json() as Promise<MeshMeta>),
    fetchOk("data/mesh.bin").then((r) => r.arrayBuffer()),
  ]);
  const canvas = $<HTMLCanvasElement>("view");
  const view = new BrainView(canvas, picture, meshMeta, meshData);
  const mine = new StoryPanel($("story"));
  const twin = new StoryPanel($("twin"));
  const sounds = new Instrument();
  const hum = new Hum();
  const { pieces } = picture.manifest;

  // ---- the model ---------------------------------------------------------
  const worker = new Worker(new URL("./worker/writer.worker.ts", import.meta.url), { type: "module" });
  const send = (message: ToWorker, transfer: Transferable[] = []) => worker.postMessage(message, transfer);
  const seconds = () => performance.now() / 1000;
  worker.addEventListener("message", ({ data }: MessageEvent<FromWorker>) => {
    if (data.type === "step") {
      if (data.twin) twin.add(data.twin.text, data.twin.probability);
      if (data.mine) {
        mine.add(data.mine.text, data.mine.probability, data.mine.unexpected);
        view.light.show(data.mine.activity, seconds(), 1 / TOKENS_PER_SECOND);
        sounds.type(keys(data.mine.text, data.mine.probability, data.mine.unexpected, 1 / TOKENS_PER_SECOND));
        sounds.setHum(hum.level(data.mine.activity));
      }
    } else if (data.type === "restart") {
      mine.finish();
      twin.finish();
    } else if (data.type === "kernel") {
      document.documentElement.dataset.kernel = data.name;
    } else {
      $("status").textContent = data.active ? "Your copy, reading its story again with the changed weights" : "Your copy";
    }
  });
  send({ type: "load", picture, tokensPerSecond: TOKENS_PER_SECOND, seed: Date.now() });

  /** Tell the model about weights the page just changed, by the brush or by healing. */
  function apply(change: Change, healing = false): void {
    sounds.scratch(scratch(change.indices.length), healing);
    if (change.indices.length) send({ type: "set", ...change }, [change.indices.buffer, change.bits.buffer]);
    const { damaged, total } = view.weights;
    $("damage").textContent = `${((100 * damaged) / total).toFixed(2)}% damaged`;
    $("damage").classList.toggle("hurt", damaged > 0);
    $("count").textContent = `${damaged.toLocaleString()} of ${total.toLocaleString()} weights`;
    $<HTMLButtonElement>("heal").disabled = damaged === 0;
  }

  // ---- tools -------------------------------------------------------------
  let tool: "brush" | "move" = "brush";
  let healPerFrame = 0;
  let playing = true;
  /**
   * Brush radius in square widths. The slider sets it for the opening view;
   * zooming in shrinks it in step, so it stays the same size on screen.
   */
  const brushRadius = () => {
    const chosen = Number($<HTMLInputElement>("size").value) / 100;
    const smallest = SMALLEST_BRUSH_WEIGHTS / picture.manifest.square;
    return Math.max(smallest, chosen * Math.min(1, view.orbit.closeness));
  };

  function setTool(next: typeof tool): void {
    tool = next;
    $("brush").setAttribute("aria-pressed", String(tool === "brush"));
    $("move").setAttribute("aria-pressed", String(tool === "move"));
    canvas.classList.toggle("moving", tool === "move");
  }

  function paint(hit: Hit): void {
    healPerFrame = 0;
    view.stamp(hit);
    apply(view.weights.paint(hit.square, hit.x, hit.y, brushRadius()));
    $("hint").hidden = true;
  }

  /** Show the brush ring and name the weight under the pointer. */
  function describe(hit: Hit | null): void {
    view.showBrush(tool === "brush" ? hit : null, brushRadius());
    if (!hit) return void ($("where").textContent = "");
    const piece = pieces[hit.square];
    const weight = view.weights.at(hit.square, hit.x, hit.y);
    const value = `${weight.positive ? "+1" : "−1"}${weight.damaged ? " damaged" : ""}`;
    $("where").textContent =
      `layer ${piece.layer + 1} · ${piece.label} · ` +
      `row ${piece.rows[0] + weight.row}, column ${piece.cols[0] + weight.column} · ${value}`;
  }

  function setUnfolded(unfolded: boolean): void {
    view.setUnfolded(unfolded);
    $("fold").setAttribute("aria-pressed", String(unfolded));
  }

  $("brush").addEventListener("click", () => setTool("brush"));
  $("move").addEventListener("click", () => setTool("move"));
  $("heal").addEventListener("click", () => (healPerFrame = Math.ceil(view.weights.damaged / HEAL_FRAMES)));
  $("fold").addEventListener("click", () => setUnfolded(!view.isUnfolded));
  $("reset").addEventListener("click", () => {
    setUnfolded(false);
    view.resetView();
  });
  $("pause").addEventListener("click", () => {
    playing = !playing;
    send({ type: "play", playing });
    $("pause").setAttribute("aria-label", playing ? "Pause" : "Play");
    $("pause").classList.toggle("paused", !playing);
  });

  $("sound").addEventListener("click", async () => {
    await sounds.setEnabled(!sounds.isEnabled);
    $("sound").setAttribute("aria-pressed", String(sounds.isEnabled));
  });

  window.addEventListener("keydown", (event: KeyboardEvent) => {
    if (event.metaKey || event.ctrlKey || event.target instanceof HTMLInputElement) return;
    const size = $<HTMLInputElement>("size");
    const keys: Record<string, () => void> = {
      b: () => setTool("brush"),
      m: () => setTool("move"),
      h: () => $("heal").click(),
      u: () => $("fold").click(),
      r: () => $("reset").click(),
      s: () => $("sound").click(),
      " ": () => $("pause").click(),
      "[": () => size.stepDown(2),
      "]": () => size.stepUp(2),
    };
    const action = keys[event.key.toLowerCase()];
    if (!action) return;
    event.preventDefault();
    action();
  });

  // ---- pointer -----------------------------------------------------------
  const fingers = new Map<number, PointerEvent>();
  // A finger that has landed but not yet moved is "waiting": it paints only
  // once it travels, so the first finger of a pinch does no damage.
  let drag: "paint" | "orbit" | "waiting" | null = null;
  let landed: PointerEvent | null = null;
  let pinch = 0;
  const spread = () => {
    const [a, b] = [...fingers.values()];
    return Math.hypot(a.clientX - b.clientX, a.clientY - b.clientY);
  };

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    fingers.set(event.pointerId, event);
    if (fingers.size === 2) {
      drag = "orbit";
      pinch = spread();
      return;
    }
    const hit = pick(event);
    drag = tool === "brush" && event.button === 0 && !event.shiftKey && hit ? "paint" : "orbit";
    if (drag === "paint" && event.pointerType === "touch") {
      drag = "waiting";
      landed = event;
    } else if (drag === "paint" && hit) paint(hit);
  });

  canvas.addEventListener("pointermove", (event) => {
    const before = fingers.get(event.pointerId);
    if (before) fingers.set(event.pointerId, event);
    if (fingers.size === 2) {
      const [a, b] = [...fingers.values()];
      const between = { clientX: (a.clientX + b.clientX) / 2, clientY: (a.clientY + b.clientY) / 2 };
      view.orbit.zoom(pinch / spread(), pick(between) ?? view.pickCentre());
      pinch = spread();
    }
    if (drag === "orbit" && before) {
      const dx = event.clientX - before.clientX, dy = event.clientY - before.clientY;
      // Far away a drag turns the brain. Close up, and on the flat picture,
      // it slides the surface instead.
      if (view.isUnfolded || view.orbit.isClose) view.slide(dx, dy);
      else view.orbit.orbit(dx, dy);
    }
    if (drag === "waiting" && landed) {
      const travelled = Math.hypot(event.clientX - landed.clientX, event.clientY - landed.clientY);
      if (travelled > TOUCH_SLOP_PIXELS) drag = "paint";
    }
    const hit = fingers.size < 2 ? pick(event) : null;
    describe(hit);
    if (drag === "paint" && hit) paint(hit);
  });

  for (const type of ["pointerup", "pointercancel", "pointerleave"] as const) {
    canvas.addEventListener(type, (event) => {
      fingers.delete(event.pointerId);
      // A tap that never moved paints one dab where it landed.
      const tapped = type === "pointerup" && drag === "waiting" ? pick(event) : null;
      if (tapped) paint(tapped);
      if (!fingers.size) drag = null;
      if (type === "pointerleave") describe(null);
    });
  }

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      view.orbit.zoom(Math.exp(event.deltaY * ZOOM_PER_WHEEL_UNIT), pick(event) ?? view.pickCentre());
    },
    { passive: false },
  );
  canvas.addEventListener("contextmenu", (event) => event.preventDefault());

  /** `view.pick` for a pointer or wheel event. */
  function pick(event: { clientX: number; clientY: number }): Hit | null {
    return view.pick(event);
  }

  // ---- frames ------------------------------------------------------------
  let last = seconds();
  function frame(): void {
    const now = seconds();
    const dt = Math.min(0.1, now - last);
    last = now;
    if (healPerFrame) {
      apply(view.weights.heal(healPerFrame), true);
      if (!view.weights.damaged) healPerFrame = 0;
    }
    view.frame(now, dt, calm);
    requestAnimationFrame(frame);
  }

  setTool("brush");
  apply({ indices: new Uint32Array(0), bits: new Uint8Array(0) });
  $("loading").hidden = true;
  requestAnimationFrame(frame);
}

start().catch((error: Error) => {
  $("loading").hidden = true;
  $("failure").hidden = false;
  $("failure").textContent = `Could not load the model picture. ${error.message}`;
});
