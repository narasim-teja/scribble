import { describe, expect, it } from "bun:test";
import { Hum, keys, scratch } from "../src/audio/sounds";

describe("keys", () => {
  it("presses one key per letter, inside the time the piece has", () => {
    const pressed = keys(" there", 0.5, false, 0.2);
    expect(pressed.map((key) => key.kind)).toEqual(["space", "letter", "letter", "letter", "letter", "letter"]);
    expect(pressed.at(-1)!.delay).toBeLessThan(0.2);
    expect(pressed[1].delay).toBeGreaterThan(pressed[0].delay);
  });

  it("is crisper and louder the surer the model was", () => {
    const [sure] = keys("a", 0.9, false, 0.2);
    const [unsure] = keys("a", 0.1, false, 0.2);
    expect(sure.tone).toBeGreaterThan(unsure.tone);
    expect(sure.gain).toBeGreaterThan(unsure.gain);
  });

  it("jams every key of a piece a healthy copy would not write", () => {
    expect(keys("clums", 0.5, true, 0.2).every((key) => key.kind === "jam")).toBe(true);
  });

  it("rings at the end of a line", () => {
    expect(keys(".\n", 0.5, false, 0.2).map((key) => key.kind)).toEqual(["letter", "return"]);
  });
});

describe("Hum", () => {
  it("sits at its usual level for steady activity and rises when the network works harder", () => {
    const hum = new Hum();
    const steady = new Float32Array(8).fill(1);
    expect(hum.level(steady)).toBeCloseTo(1);
    expect(hum.level(steady)).toBeCloseTo(1);
    expect(hum.level(new Float32Array(8).fill(1.5))).toBeGreaterThan(1.3);
  });

  it("stays within limits", () => {
    const hum = new Hum();
    hum.level(new Float32Array(8).fill(1));
    expect(hum.level(new Float32Array(8).fill(100))).toBe(2);
    expect(hum.level(new Float32Array(8))).toBe(0.4);
  });
});

describe("scratch", () => {
  it("is silent for no change and louder for more, up to full", () => {
    expect(scratch(0)).toBe(0);
    expect(scratch(1000)).toBeGreaterThan(scratch(100));
    expect(scratch(1e9)).toBe(1);
  });
});
