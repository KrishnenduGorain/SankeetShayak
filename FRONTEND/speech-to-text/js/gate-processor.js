/* Noise gate AudioWorklet. Runs on the audio rendering thread, so it keeps working when the tab is in the
 * background (timers are throttled there, which would freeze a main-thread gate).
 *
 * Messages in : { threshold: <linear amplitude>, enabled: <boolean> }
 * Messages out: { rms: <input level, linear>, open: <gate is passing speech> }  (about every 43 ms at 48 kHz)
 */
  class SpeechGate extends AudioWorkletProcessor {
    constructor() {
      super();
      this.threshold = 0.0025; this.enabled = true; this.floor = 0.06;
      this.env = 0; this.gain = 0; this.hold = 0; this.sum = 0; this.n = 0;
      this.port.onmessage = (e) => { Object.assign(this, e.data); };
    }
    process(inputs, outputs) {
      const input = inputs[0] && inputs[0][0];
      const output = outputs[0] && outputs[0][0];
      if (!input || !output) return true;
      const sr = sampleRate;
      const atk = Math.exp(-1 / (0.004 * sr)), rel = Math.exp(-1 / (0.060 * sr));     // level detector
      const gAtk = Math.exp(-1 / (0.004 * sr)), gRel = Math.exp(-1 / (0.120 * sr));   // gain smoothing (no clicks)
      const holdSamples = 0.25 * sr;                                                    // keep open between words
      for (let i = 0; i < input.length; i++) {
        const x = input[i], a = Math.abs(x);
        this.env = a > this.env ? atk * this.env + (1 - atk) * a : rel * this.env + (1 - rel) * a;
        if (this.env > this.threshold) this.hold = holdSamples; else if (this.hold > 0) this.hold--;
        const target = (!this.enabled || this.hold > 0) ? 1 : this.floor;
        this.gain = target > this.gain ? gAtk * this.gain + (1 - gAtk) * target : gRel * this.gain + (1 - gRel) * target;
        output[i] = x * this.gain;
        this.sum += x * x;
      }
      this.n += input.length;
      if (this.n >= 2048) {
        this.port.postMessage({ rms: Math.sqrt(this.sum / this.n), open: this.gain > 0.5 });
        this.sum = 0; this.n = 0;
      }
      return true;
    }
  }
  registerProcessor('speech-gate', SpeechGate);
