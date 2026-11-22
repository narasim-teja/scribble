// Makes the sounds described in sounds.ts with the Web Audio API. Everything
// is built from noise and sine waves here, so there are no sound files to
// load or license. Browsers only allow sound after the visitor touches the
// page, so nothing is created until `setEnabled(true)` is called from a
// click, tap or key press.

import type { Key } from "./sounds";

const VOLUME = 0.5;
const HUM_VOLUME = 0.035;
const HUM_HZ = [55, 82.5, 110]; // a low open fifth: steady, not a tune
const HUM_GLIDE_SECONDS = 0.25;
const SCRATCH_VOLUME = 0.3;
const SCRATCH_SECONDS = 0.09;

export class Instrument {
  private context: BaseAudioContext | null = null;
  private output: AudioNode | null = null;
  private noise: AudioBuffer | null = null;
  private hum: GainNode | null = null;
  private enabled = false;

  /** @param makeContext where the sound goes; a preview can pass an offline one */
  constructor(private readonly makeContext: () => BaseAudioContext = () => new AudioContext()) {}

  /** True once the browser has let the sound start. */
  get isRunning(): boolean {
    return this.enabled && this.context?.state === "running";
  }

  /**
   * Turn sound on or off. Must be called from a click, tap or key press to
   * turn on.
   * @param insist the visitor asked for sound by name, so play it even on an
   *   iPhone with the ring switch set to silent. Sound that starts by itself
   *   leaves this off and stays quiet on a silenced phone.
   */
  async setEnabled(enabled: boolean, insist = false): Promise<void> {
    this.enabled = enabled;
    const live = () => (this.context instanceof AudioContext ? this.context : null);
    if (!enabled) return void (await live()?.suspend());
    const session = (navigator as { audioSession?: { type: string } }).audioSession;
    if (session) session.type = insist ? "playback" : "ambient";
    if (!this.context) this.build(this.makeContext());
    await live()?.resume();
  }

  /** Play the key presses of one piece of text. */
  type(keys: Key[]): void {
    if (!this.enabled || !this.context) return;
    for (const key of keys) {
      const at = this.context.currentTime + key.delay;
      if (key.kind === "return") this.ring(at, key.gain);
      else if (key.kind === "jam") this.jam(at, key);
      else this.click(at, key);
    }
  }

  /** Set the hum to `level` times its usual loudness. */
  setHum(level: number): void {
    if (!this.context || !this.hum) return;
    this.hum.gain.setTargetAtTime(HUM_VOLUME * level, this.context.currentTime, HUM_GLIDE_SECONDS);
  }

  /** A short scrape as weights change: rough for damage, thin and high for healing. */
  scratch(strength: number, healing: boolean): void {
    if (!this.enabled || !this.context || strength <= 0) return;
    const at = this.context.currentTime;
    if (healing) this.burst(at, 5200, 3, SCRATCH_VOLUME * 0.4 * strength, SCRATCH_SECONDS);
    else this.burst(at, 1500, 0.7, SCRATCH_VOLUME * strength, SCRATCH_SECONDS);
  }

  private build(context: BaseAudioContext): void {
    this.context = context;
    // One second of hiss, the raw material of every click and scrape.
    this.noise = context.createBuffer(1, context.sampleRate, context.sampleRate);
    const samples = this.noise.getChannelData(0);
    for (let i = 0; i < samples.length; i++) samples[i] = Math.random() * 2 - 1;

    const volume = new GainNode(context, { gain: VOLUME });
    volume.connect(new DynamicsCompressorNode(context)).connect(context.destination);
    this.output = volume;

    this.hum = new GainNode(context, { gain: HUM_VOLUME });
    this.hum.connect(volume);
    for (const frequency of HUM_HZ) {
      const tone = new OscillatorNode(context, { frequency });
      tone.connect(this.hum);
      tone.start();
    }
  }

  /** A burst of hiss through a filter: `tone` sets its colour, `sharpness` how pitched it is. */
  private burst(at: number, tone: number, sharpness: number, gain: number, seconds: number): void {
    const context = this.context!;
    const source = new AudioBufferSourceNode(context, { buffer: this.noise });
    const filter = new BiquadFilterNode(context, { type: "bandpass", frequency: tone, Q: sharpness });
    const envelope = new GainNode(context, { gain: 0 });
    envelope.gain.setValueAtTime(0, at);
    envelope.gain.linearRampToValueAtTime(gain, at + 0.002);
    envelope.gain.exponentialRampToValueAtTime(0.0005, at + seconds);
    source.connect(filter).connect(envelope).connect(this.output!);
    source.start(at, Math.random() * 0.8);
    source.stop(at + seconds);
  }

  /** A tone that starts at `gain` and dies away. */
  private thump(at: number, frequency: number, gain: number, seconds: number, type: OscillatorType = "sine"): void {
    const context = this.context!;
    const tone = new OscillatorNode(context, { type, frequency });
    const envelope = new GainNode(context, { gain: 0 });
    envelope.gain.setValueAtTime(gain, at);
    envelope.gain.exponentialRampToValueAtTime(0.0005, at + seconds);
    tone.connect(envelope).connect(this.output!);
    tone.start(at);
    tone.stop(at + seconds);
  }

  /** A key: a click with a little body under it. The space bar is the same, lower and longer. */
  private click(at: number, key: Key): void {
    const long = key.kind === "space";
    this.burst(at, key.tone, 1.4, key.gain * 0.5, long ? 0.05 : 0.028);
    this.thump(at, long ? 95 : 150, key.gain * 0.25, long ? 0.07 : 0.035);
  }

  /** A key that sticks: a dull knock and a short buzz. */
  private jam(at: number, key: Key): void {
    this.burst(at, key.tone, 2.5, key.gain * 0.5, 0.07);
    this.thump(at, 62, key.gain * 0.16, 0.09, "sawtooth");
  }

  /** The end of a line: a small bell. */
  private ring(at: number, gain: number): void {
    this.thump(at, 1568, gain * 0.1, 0.7);
    this.thump(at, 1568 * 2.76, gain * 0.03, 0.35);
  }
}
