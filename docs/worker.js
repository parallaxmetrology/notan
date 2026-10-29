// Runs the Notan read off the main thread, in stages so the page can show the fast
// parts first: mass and values, then the colour map, then the construction read.
import { fromRGBA, resizeWork, notanMass, valueMap, notanReadout, dependenceMap, constructionRead, luminance } from "./notan.js";

function p995(a) {
  const s = Float64Array.from(a).sort();
  const idx = 0.995 * (s.length - 1), lo = Math.floor(idx), hi = Math.min(lo + 1, s.length - 1);
  return s[lo] + (s[hi] - s[lo]) * (idx - lo);
}
const f32 = (a) => Float32Array.from(a);

self.onmessage = (e) => {
  const { id, buffer, w, h } = e.data;
  try {
    const t0 = performance.now();
    const small = resizeWork(fromRGBA(new Uint8ClampedArray(buffer), w, h));
    const mass = notanMass(small);
    const s1 = {
      id, stage: "mass", w: small.w, h: small.h,
      rgb: Float32Array.from(small.data), lum: f32(luminance(small)),
      mass: f32(mass), massScale: Math.max(p995(mass), 1e-9),
      v2: f32(valueMap(small, 2, mass)), v3: f32(valueMap(small, 3, mass)), v5: f32(valueMap(small, 5, mass)),
      readout: notanReadout(small, mass),
    };
    self.postMessage(s1, [s1.rgb, s1.lum, s1.mass, s1.v2, s1.v3, s1.v5].map((a) => a.buffer));

    const ref = notanMass(small, true);
    const dep = dependenceMap(small, "grayscale", ref);
    const s2 = { id, stage: "dependence", lost: f32(dep.lost), lostScale: Math.max(p995(ref), 1e-9), depSurvival: dep.survival };
    self.postMessage(s2, [s2.lost.buffer]);

    const construction = constructionRead(small, ref, { grayscale: dep.degraded });
    self.postMessage({ id, stage: "construction", construction, ms: Math.round(performance.now() - t0), done: true });
  } catch (err) {
    self.postMessage({ id, error: String((err && err.message) || err), done: true });
  }
};
