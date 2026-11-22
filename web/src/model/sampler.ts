// Choosing the next token from the model's scores, repeatably.

/** A small seeded random generator (mulberry32). Returns values in [0, 1). */
export function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let t = state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Turn scores into probabilities. Lower temperature favours the top scores. */
export function probabilities(logits: Float32Array, temperature: number): Float64Array {
  let best = -Infinity;
  for (const value of logits) best = Math.max(best, value);
  const out = new Float64Array(logits.length);
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += out[i] = Math.exp((logits[i] - best) / temperature);
  for (let i = 0; i < out.length; i++) out[i] /= sum;
  return out;
}

/** Pick an index with the given probabilities. `random` is a value in [0, 1). */
export function pick(probs: Float64Array, random: number): number {
  let left = random;
  for (let i = 0; i < probs.length; i++) {
    left -= probs[i];
    if (left < 0) return i;
  }
  return probs.length - 1;
}
