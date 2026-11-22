// Plays one short plucked tone per token. Browsers only allow sound after a
// click, so nothing is created until `setEnabled(true)` is called from one.

import type { Voice } from "./voice";

const VOLUME = 0.14;
const ATTACK_SECONDS = 0.006;
const RING_SECONDS = 0.32;

export class TonePlayer {
  private context: AudioContext | null = null;
  private output: GainNode | null = null;
  private enabled = false;

  get isEnabled(): boolean {
    return this.enabled;
  }

  /** Turn sound on or off. Must be called from a click or key press to turn on. */
  async setEnabled(enabled: boolean): Promise<void> {
    this.enabled = enabled;
    if (!enabled) return void this.context?.suspend();
    if (!this.context) {
      // Lets the sound play on an iPhone with the ring switch set to silent.
      const session = (navigator as { audioSession?: { type: string } }).audioSession;
      if (session) session.type = "playback";
      this.context = new AudioContext();
      const soften = new BiquadFilterNode(this.context, { type: "lowpass", frequency: 2400 });
      this.output = new GainNode(this.context, { gain: VOLUME });
      this.output.connect(soften).connect(this.context.destination);
    }
    await this.context.resume();
  }

  play({ frequency, detune, gain }: Voice): void {
    if (!this.enabled || !this.context || !this.output) return;
    const now = this.context.currentTime;
    const tone = new OscillatorNode(this.context, { type: "triangle", frequency, detune });
    const envelope = new GainNode(this.context, { gain: 0 });
    envelope.gain.linearRampToValueAtTime(gain, now + ATTACK_SECONDS);
    envelope.gain.exponentialRampToValueAtTime(0.001, now + RING_SECONDS);
    tone.connect(envelope).connect(this.output);
    tone.start(now);
    tone.stop(now + RING_SECONDS);
  }
}
