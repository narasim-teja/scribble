import { describe, expect, it } from "bun:test";
import { voice } from "../src/audio/voice";

describe("voice", () => {
  it("plays in tune when a healthy copy finds the token plausible", () => {
    expect(voice(17, 0.6, 0.5).detune).toBeCloseTo(0);
    expect(voice(17, 0.6, 0.04).detune).toBeCloseTo(0);
  });

  it("goes further out of tune the less plausible the token, up to a limit", () => {
    const mild = Math.abs(voice(17, 0.6, 0.01).detune);
    const strong = Math.abs(voice(17, 0.6, 0.0001).detune);
    expect(mild).toBeGreaterThan(0);
    expect(strong).toBeGreaterThan(mild);
    expect(Math.abs(voice(17, 0.6, 0).detune)).toBe(150);
  });

  it("gives a token the same note every time, within two octaves of the root", () => {
    expect(voice(40, 0.2, 0.5).frequency).toBe(voice(40, 0.9, 0.5).frequency);
    for (let id = 0; id < 512; id++) {
      const { frequency } = voice(id, 0.5, 0.5);
      expect(frequency).toBeGreaterThanOrEqual(220);
      expect(frequency).toBeLessThan(880);
    }
  });

  it("is louder the surer the model was", () => {
    expect(voice(3, 0.9, 0.5).gain).toBeGreaterThan(voice(3, 0.1, 0.5).gain);
  });
});
