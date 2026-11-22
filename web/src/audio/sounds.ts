// What the page sounds like, as numbers. Nothing here makes sound; it turns
// model state into instructions for the instrument. Every value is real:
//   keys      one per letter the model just wrote, crisper the surer it was;
//             a jammed key for a piece a healthy copy would almost never write
//   hum       how hard the network worked on this piece, against its usual
//   scratch   how many weights the brush or the heal just changed

/** One key press. */
export interface Key {
  kind: "letter" | "space" | "return" | "jam";
  /** Seconds after the piece arrives. */
  delay: number;
  /** How bright the click is, in hertz. */
  tone: number;
  /** Loudness from 0 to 1. */
  gain: number;
}

const DULL_HZ = 900;
const CRISP_HZ = 3000;
const JAM_HZ = 420;
const KEY_GAP_SECONDS = 0.045; // between letters of one piece, when there is room
const HUM_RANGE = [0.4, 2] as const;
const USUAL_MEMORY = 0.97; // how slowly the idea of "usual" activity moves
const FULL_SCRATCH = 4000; // weights changed in one touch that give the loudest scratch

/**
 * The key presses for one piece of text.
 * @param probability how likely the writing model thought the piece was
 * @param unexpected a healthy copy would almost never have written it
 * @param seconds how long the piece has before the next one arrives
 */
export function keys(text: string, probability: number, unexpected: boolean, seconds: number): Key[] {
  const letters = [...text];
  const gap = Math.min(KEY_GAP_SECONDS, (seconds * 0.8) / Math.max(1, letters.length));
  const tone = DULL_HZ + (CRISP_HZ - DULL_HZ) * probability;
  const gain = 0.35 + 0.65 * probability;
  return letters.map((letter, i): Key => {
    const delay = i * gap;
    if (unexpected) return { kind: "jam", delay, tone: JAM_HZ, gain: 0.9 };
    if (letter === "\n") return { kind: "return", delay, tone, gain };
    if (letter === " ") return { kind: "space", delay, tone: tone / 2, gain: gain * 0.8 };
    return { kind: "letter", delay, tone, gain };
  });
}

/** Follows the network's activity and says how loud the hum should be against its usual level. */
export class Hum {
  private usual = 0;

  /** @param activity the engine's record of the last step, one size per column and row of each square */
  level(activity: ArrayLike<number>): number {
    let sum = 0;
    for (let i = 0; i < activity.length; i++) sum += activity[i];
    const mean = sum / activity.length;
    this.usual = this.usual ? USUAL_MEMORY * this.usual + (1 - USUAL_MEMORY) * mean : mean;
    return Math.min(HUM_RANGE[1], Math.max(HUM_RANGE[0], mean / this.usual));
  }
}

/** Loudness, 0 to 1, of the scratch for a touch that changed `weights` weights. */
export function scratch(weights: number): number {
  return Math.min(1, Math.sqrt(weights / FULL_SCRATCH));
}
