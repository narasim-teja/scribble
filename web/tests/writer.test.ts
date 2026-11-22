import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "bun:test";
import { readPicture } from "../src/model/picture";
import { Writer } from "../src/model/writer";

async function tinyPicture() {
  const file = readFileSync(fileURLToPath(new URL("./fixtures/tiny.png", import.meta.url)));
  return readPicture(file.buffer.slice(file.byteOffset, file.byteOffset + file.byteLength));
}

/** The next `count` steps of one side as text, with story ends shown as a marker. */
const write = (writer: Writer, count: number, side: "mine" | "twin" = "mine") =>
  Array.from({ length: count }, () => writer.write()[side]?.text ?? "<end>").join("");

/** Every weight of the picture set to the given values. */
function allBits(side: number, value: (index: number) => number) {
  const indices = Uint32Array.from({ length: side * side }, (_, i) => i);
  return { indices, bits: Uint8Array.from(indices, value) };
}

describe("writer", () => {
  it("writes the same text for the same seed", async () => {
    const first = write(new Writer(await tinyPicture(), 0.8, 7), 40);
    const second = write(new Writer(await tinyPicture(), 0.8, 7), 40);
    expect(second).toBe(first);
    expect(first.length).toBeGreaterThan(0);
  });

  it("keeps writing past the context length", async () => {
    const picture = await tinyPicture();
    const writer = new Writer(picture, 0.8, 1);
    expect(() => write(writer, picture.manifest.config.context * 3)).not.toThrow();
  });

  it("reports activity for every square, scaled to a byte", async () => {
    const picture = await tinyPicture();
    const { pieces, config } = picture.manifest;
    const written = new Writer(picture, 0.8, 3).write().mine!;
    expect(written.activity.length).toBe(pieces.length * 2 * config.width);
    expect(Math.max(...written.activity)).toBe(255);
  });

  it("writes differently once damaged, and rereads in small steps", async () => {
    const picture = await tinyPicture();
    const side = picture.manifest.side;
    const healthy = new Writer(await tinyPicture(), 0.8, 5);
    const damaged = new Writer(picture, 0.8, 5);
    expect(write(damaged, 10)).toBe(write(healthy, 10));

    const noise = allBits(side, (i) => (i * 2654435761) >>> 31);
    damaged.setBits(noise.indices, noise.bits);
    expect(damaged.stale).toBe(true);
    damaged.reread(4);
    expect(damaged.stale).toBe(true);
    while (damaged.stale) damaged.reread(4);
    expect(write(damaged, 30)).not.toBe(write(healthy, 30));
  });

  it("never flags a token while the weights are untouched", async () => {
    const writer = new Writer(await tinyPicture(), 0.8, 9);
    for (let i = 0; i < 60; i++) expect(writer.write().mine?.unexpected ?? false).toBe(false);
  });

  it("writes the same as its twin until damaged, then differently", async () => {
    const picture = await tinyPicture();
    const writer = new Writer(picture, 0.8, 11);
    for (let i = 0; i < 80; i++) {
      const { mine, twin } = writer.write();
      expect(mine?.text).toBe(twin?.text);
    }
    const noise = allBits(picture.manifest.side, (i) => (i * 2654435761) >>> 31);
    writer.setBits(noise.indices, noise.bits);
    const steps = Array.from({ length: 40 }, () => writer.write());
    expect(steps.some(({ mine, twin }) => mine?.text !== twin?.text)).toBe(true);
  });

  it("starts both stories again together once both have ended", async () => {
    const writer = new Writer(await tinyPicture(), 0.8, 2);
    const steps = Array.from({ length: 1500 }, () => writer.write());
    const restart = steps.findIndex((step) => step.restarted);
    expect(restart).toBeGreaterThan(0);
    expect(steps[restart - 1]).toEqual({ mine: null, twin: null, restarted: false });
    expect(steps[restart + 1].mine?.text).toBe(steps[restart + 1].twin?.text);
  });
});
