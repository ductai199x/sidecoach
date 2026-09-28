// The board's sounds, as chess.com plays them: a wooden tock for a move, a sharper clack for a
// capture. Made here with Web Audio, no sound files. The context starts on the first sound, which
// is always the player's own move or a click (browsers keep audio off until a gesture), and the
// line's replies play through it after.

let ctx: AudioContext | null = null;
let noise: AudioBuffer | null = null;

function context(): AudioContext | null {
  if (typeof AudioContext === "undefined") return null;
  if (!ctx) {
    ctx = new AudioContext();
    noise = ctx.createBuffer(1, Math.floor(ctx.sampleRate * 0.15), ctx.sampleRate);
    const data = noise.getChannelData(0);
    for (let i = 0; i < data.length; i++) data[i] = Math.random() * 2 - 1;
  }
  if (ctx.state === "suspended") void ctx.resume();
  return ctx;
}

// One knock of wood on wood, `length` seconds long: a short falling tone for the body and a burst
// of band-passed noise for the contact, both dying away fast.
function knock(c: AudioContext, at: number, pitch: number, bright: number, level: number, length: number) {
  const out = c.createGain();
  out.gain.value = level;
  out.connect(c.destination);
  const body = c.createOscillator();
  body.type = "triangle";
  body.frequency.setValueAtTime(pitch, at);
  body.frequency.exponentialRampToValueAtTime(pitch * 0.62, at + length * 0.5);
  const bodyGain = c.createGain();
  bodyGain.gain.setValueAtTime(0.0001, at);
  bodyGain.gain.exponentialRampToValueAtTime(0.9, at + 0.002);
  bodyGain.gain.exponentialRampToValueAtTime(0.0001, at + length);
  body.connect(bodyGain).connect(out);
  body.start(at);
  body.stop(at + length + 0.02);
  const hit = c.createBufferSource();
  hit.buffer = noise;
  const band = c.createBiquadFilter();
  band.type = "bandpass";
  band.frequency.value = bright;
  band.Q.value = 1.2;
  const hitGain = c.createGain();
  hitGain.gain.setValueAtTime(0.0001, at);
  hitGain.gain.exponentialRampToValueAtTime(0.6, at + 0.001);
  hitGain.gain.exponentialRampToValueAtTime(0.0001, at + length * 0.35);
  hit.connect(band).connect(hitGain).connect(out);
  hit.start(at);
  hit.stop(at + length);
}

export function moveSound(): void {
  const c = context();
  if (c) knock(c, c.currentTime, 210, 1800, 0.35, 0.09);
}

// A capture: a brighter, louder knock, and the taken piece's a moment after it.
export function captureSound(): void {
  const c = context();
  if (!c) return;
  const t = c.currentTime;
  knock(c, t, 300, 2800, 0.45, 0.11);
  knock(c, t + 0.028, 190, 1500, 0.25, 0.08);
}
