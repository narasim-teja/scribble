// How a written token sounds. Every value comes from the model:
//   which note   the token itself, so the same word always plays the same note
//   how loud     how sure the model was of it
//   how in tune  how plausible a healthy copy of the model finds it
// Healthy writing therefore plays in tune, and damage makes it go sour.

const ROOT_HZ = 220;
const SCALE = [0, 2, 4, 7, 9]; // major pentatonic, in semitones: any two notes sound fine together
const OCTAVES = 2;
const IN_TUNE_BITS = 5; // surprise below this (probability above about 3%) stays in tune
const CENTS_PER_BIT = 30;
const MOST_CENTS = 150;

export interface Voice {
  frequency: number;
  /** How far off pitch, in hundredths of a semitone. */
  detune: number;
  /** Loudness from 0 to 1. */
  gain: number;
}

/**
 * @param id the token's id in the vocabulary
 * @param probability how likely the writing model thought the token was
 * @param expected how likely an undamaged copy thought it was
 */
export function voice(id: number, probability: number, expected: number): Voice {
  const note = id % (SCALE.length * OCTAVES);
  const semitones = SCALE[note % SCALE.length] + 12 * Math.floor(note / SCALE.length);
  const surprise = -Math.log2(Math.max(expected, 1e-9));
  const sour = Math.min(MOST_CENTS, Math.max(0, surprise - IN_TUNE_BITS) * CENTS_PER_BIT);
  return {
    frequency: ROOT_HZ * 2 ** (semitones / 12),
    detune: id % 2 ? sour : -sour,
    gain: 0.25 + 0.75 * probability,
  };
}
