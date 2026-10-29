// parallax-notan in plain JavaScript: notan_mass, value_map, dependence_map, construction_read
// and notan_readout, ported from src/parallax_notan/. No DOM, so it runs in a browser, a
// worker or Node. tools/js_parity.py checks it against the Python package.
//
// The port follows the arithmetic of numpy, scipy.ndimage and scikit-image step by step
// (float32 where they use float32, the same boundary modes, the same summation order),
// so the readings match the Python package to rounding error, not just in spirit.
//
// An image is {w, h, data}: data is a Float32Array of interleaved RGB in [0, 1].

export const WORK_PX = 512;
export const NOTAN_SCALES = [4.0, 8.0, 16.0];
export const CHROMA_WEIGHT = 0.5;
export const TAU_VOID = 0.06;
export const CHROMA_DUST = 1e-6;
const EPS = 1e-9;
const f32 = Math.fround;

// ---------- numpy helpers ----------

// numpy's pairwise summation (the order np.sum uses on a contiguous float64 array).
function pairwise(a, off, n) {
  if (n < 8) {
    let res = 0;
    for (let i = 0; i < n; i++) res += a[off + i];
    return res;
  }
  if (n <= 128) {
    const r = [a[off], a[off + 1], a[off + 2], a[off + 3], a[off + 4], a[off + 5], a[off + 6], a[off + 7]];
    let i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (let j = 0; j < 8; j++) r[j] += a[off + i + j];
    }
    let res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; i++) res += a[off + i];
    return res;
  }
  let n2 = Math.floor(n / 2);
  n2 -= n2 % 8;
  return pairwise(a, off, n2) + pairwise(a, off + n2, n - n2);
}
const npsum = (a) => pairwise(a, 0, a.length);

function sortedCopy(a) {
  const s = Float64Array.from(a);
  s.sort();
  return s;
}

// np.median
function median(a) {
  const s = sortedCopy(a);
  const n = s.length;
  if (n === 0) return NaN;
  const m = n >> 1;
  return n % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// np.percentile, default "linear" method: numpy's virtual index, then its _lerp
function percentile(a, q) {
  const s = sortedCopy(a), n = s.length;
  q = q / 100;
  const vi = n * q + (1 + q * (1 - 1 - 1)) - 1;
  if (vi >= n - 1) return s[n - 1];
  if (vi < 0) return s[0];
  const lo = Math.floor(vi), t = vi - lo, x = s[lo], y = s[lo + 1], d = y - x;
  return t >= 0.5 ? y - d * (1 - t) : x + d * t;
}

// Python's round(v, d): round the exact binary value, ties to even. toFixed breaks
// exact ties upwards instead, which matters for counts over a power of two
// (round(0.03125, 4) is 0.0312 in Python, toFixed gives 0.0313).
export function pyRound(v, d) {
  if (!Number.isFinite(v)) return v;
  const exact = Math.abs(v).toFixed(d + 25);   // enough digits: a double this close to a tie is the tie
  const dot = exact.indexOf(".");
  const tail = exact.slice(dot + 1 + d);
  let r = Number(v.toFixed(d));
  if (/^50*$/.test(tail)) {
    const kept = exact.slice(0, dot + 1 + d).replace(".", "");
    const lastDigit = Number(kept[kept.length - 1]);
    if (lastDigit % 2 === 0) r = Number((Math.sign(v) * Number(exact.slice(0, dot + 1 + d))).toFixed(d));
  }
  return r;
}

// float(f"{v:.{p}g}") in Python: p significant digits, exact ties to even.
export function pySig(v, p) {
  if (!Number.isFinite(v) || v === 0) return v;
  let e = Math.floor(Math.log10(Math.abs(v)));
  if (10 ** e > Math.abs(v)) e--;            // guard log10 rounding at powers of ten
  else if (10 ** (e + 1) <= Math.abs(v)) e++;
  const d = p - 1 - e;
  if (d >= 0 && d <= 75) return pyRound(v, d);
  return Number(v.toPrecision(p));            // huge or tiny: no ties in practice
}

// ---------- scipy.ndimage gaussian filter ----------

function gaussKernel(sigma) {
  const radius = Math.trunc(4.0 * sigma + 0.5);
  const sigma2 = sigma * sigma;
  const c = -0.5 / sigma2;
  const phi = new Float64Array(2 * radius + 1);
  for (let i = 0; i <= 2 * radius; i++) {
    const x = i - radius;
    phi[i] = Math.exp(c * (x * x));
  }
  const tot = npsum(phi);
  for (let i = 0; i < phi.length; i++) phi[i] = phi[i] / tot;
  return { radius, w: phi.subarray(radius) }; // symmetric: w[j] for offset +-j
}

// Index maps for scipy's boundary modes.
function mapReflect(i, n) {            // 'reflect': d c b a | a b c d | d c b a
  const p = 2 * n;
  i = ((i % p) + p) % p;
  return i >= n ? p - 1 - i : i;
}
function mapMirror(i, n) {             // 'mirror': d c b | a b c d | c b a
  if (n === 1) return 0;
  const p = 2 * n - 2;
  i = ((i % p) + p) % p;
  return i >= n ? p - i : i;
}

// Copy a line into buf with r samples of boundary extension on each side.
function padLine(buf, n, r, map, get) {
  for (let i = -r; i < n + r; i++) buf[i + r] = get(i >= 0 && i < n ? i : map(i, n));
}

// One output sample of scipy's correlate1d with a symmetric kernel, in its summation
// order: centre first, then the outermost pair inwards. buf is padded by k.radius.
function tap(buf, i, k) {
  const r = k.radius, w = k.w, c = i + r;
  let acc = buf[c] * w[0];
  for (let j = r; j >= 1; j--) acc += (buf[c - j] + buf[c + j]) * w[j];
  return acc;
}

// gaussian_filter on a 2D float64 field (h x w), mode 'reflect', axis 0 then axis 1.
function gauss2d(src, h, w, sigma) {
  const k = gaussKernel(sigma), r = k.radius;
  const tmp = new Float64Array(h * w);
  const out = new Float64Array(h * w);
  const bc = new Float64Array(h + 2 * r), br = new Float64Array(w + 2 * r);
  for (let x = 0; x < w; x++) {
    padLine(bc, h, r, mapReflect, (i) => src[i * w + x]);
    for (let y = 0; y < h; y++) tmp[y * w + x] = tap(bc, y, k);
  }
  for (let y = 0; y < h; y++) {
    const o = y * w;
    padLine(br, w, r, mapReflect, (i) => tmp[o + i]);
    for (let x = 0; x < w; x++) out[o + x] = tap(br, x, k);
  }
  return out;
}

// ---------- colour (colour.py) ----------

const LC0 = f32(0.2126), LC1 = f32(0.7152), LC2 = f32(0.0722);

// Rec. 709 luma in float32, as numpy computes it on a float32 image; returned as float64.
export function luminance(img) {
  const { w, h, data } = img;
  const out = new Float64Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 3) {
    let v = f32(f32(f32(LC0 * data[p]) + f32(LC1 * data[p + 1])) + f32(LC2 * data[p + 2]));
    out[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return out;
}

const degamma = (c) => (c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4));

export function oklabChroma(img) {
  const { w, h, data } = img;
  const out = new Float64Array(w * h);
  for (let i = 0, p = 0; i < w * h; i++, p += 3) {
    const r = degamma(data[p]), g = degamma(data[p + 1]), b = degamma(data[p + 2]);
    const L = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b;
    const M = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b;
    const S = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b;
    const l_ = Math.cbrt(L > 0 ? L : 0), m_ = Math.cbrt(M > 0 ? M : 0), s_ = Math.cbrt(S > 0 ? S : 0);
    const a = 1.9779984951 * l_ - 2.428592205 * m_ + 0.4505937099 * s_;
    const bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.808675766 * s_;
    out[i] = Math.hypot(a, bb);
  }
  return out;
}

// ensure_rgb01 for float data: cast to float32 and clip.
function toImage(w, h, rgb64) {
  const d = new Float32Array(rgb64.length);
  for (let i = 0; i < d.length; i++) {
    const v = f32(rgb64[i]);
    d[i] = v < 0 ? 0 : v > 1 ? 1 : v;
  }
  return { w, h, data: d };
}

// From canvas ImageData (RGBA bytes): float32 r/255, as Pillow + numpy load it.
export function fromRGBA(rgba, w, h) {
  const d = new Float32Array(w * h * 3);
  for (let i = 0, p = 0; i < w * h; i++, p += 4) {
    d[i * 3] = f32(rgba[p] / 255);
    d[i * 3 + 1] = f32(rgba[p + 1] / 255);
    d[i * 3 + 2] = f32(rgba[p + 2] / 255);
  }
  return { w, h, data: d };
}

// ---------- resize (skimage.transform.resize, order=1, anti_aliasing=True) ----------

// Gaussian anti-alias filter (mode 'mirror', float32 between axes) then ndimage.zoom with
// grid_mode=True and linear interpolation. Only the rows and columns the interpolation
// reads are filtered, which gives the same numbers at a fraction of the cost.
function skResize(img, oh, ow) {
  const { w: W, h: H, data } = img;
  const fH = H / oh, fW = W / ow;
  const sH = Math.max(0, (fH - 1) / 2), sW = Math.max(0, (fW - 1) / 2);
  let lo = Infinity, hi = -Infinity;
  for (let i = 0; i < data.length; i++) { const v = data[i]; if (v < lo) lo = v; if (v > hi) hi = v; }

  const taps = (n, out, f) => {
    const s = new Int32Array(out), t = new Float64Array(out);
    for (let k = 0; k < out; k++) {
      let cc = k;
      cc += 0.5; cc *= f; cc -= 0.5;
      const st = Math.floor(cc);
      s[k] = st; t[k] = cc - st;
    }
    return { s, t };
  };
  const ty = taps(H, oh, fH), tx = taps(W, ow, fW);

  const rows = new Map();               // source row -> filtered row (axis 0 pass), Float32 W*3
  const needRow = (r) => mapMirror(r, H);
  const kH = sH > 1e-15 ? gaussKernel(sH) : null;
  const kW = sW > 1e-15 ? gaussKernel(sW) : null;
  // Axis 0 pass, only at the rows the interpolation reads, all columns.
  const rowSet = new Set();
  for (let k = 0; k < oh; k++) { rowSet.add(needRow(ty.s[k])); rowSet.add(needRow(ty.s[k] + 1)); }
  const rowList = [...rowSet].sort((a, b) => a - b);
  if (!kH) for (const r of rowList) rows.set(r, data.slice(r * W * 3, (r + 1) * W * 3));
  else {
    for (const r of rowList) rows.set(r, new Float32Array(W * 3));
    const bc = new Float64Array(H + 2 * kH.radius);
    for (let x = 0; x < W; x++) for (let c = 0; c < 3; c++) {
      padLine(bc, H, kH.radius, mapMirror, (i) => data[(i * W + x) * 3 + c]);
      for (const r of rowList) rows.get(r)[x * 3 + c] = tap(bc, r, kH);
    }
  }
  const rowA = (r) => rows.get(r);
  const rowsB = new Map();              // axis 1 pass on that row, only at needed columns
  const colsNeeded = new Set();
  for (let k = 0; k < ow; k++) { colsNeeded.add(mapMirror(tx.s[k], W)); colsNeeded.add(mapMirror(tx.s[k] + 1, W)); }
  const rowB = (r) => {
    if (rowsB.has(r)) return rowsB.get(r);
    const a = rowA(r);
    const out = new Float32Array(W * 3);
    if (!kW) out.set(a);
    else {
      const br = new Float64Array(W + 2 * kW.radius);
      for (let c = 0; c < 3; c++) {
        padLine(br, W, kW.radius, mapMirror, (i) => a[i * 3 + c]);
        for (const x of colsNeeded) out[x * 3 + c] = tap(br, x, kW);
      }
    }
    rowsB.set(r, out);
    return out;
  };

  const out = new Float32Array(oh * ow * 3);
  for (let y = 0; y < oh; y++) {
    const r0 = needRow(ty.s[y]), r1 = needRow(ty.s[y] + 1);
    const A = rowB(r0), B = rowB(r1);
    const wy0 = 1 - ty.t[y], wy1 = 1 - wy0;   // scipy: w[1] = 1 - w[0]
    for (let x = 0; x < ow; x++) {
      const c0 = mapMirror(tx.s[x], W), c1 = mapMirror(tx.s[x] + 1, W);
      const wx0 = 1 - tx.t[x], wx1 = 1 - wx0;
      for (let c = 0; c < 3; c++) {
        let v = 0;
        v += A[c0 * 3 + c] * wy0 * wx0;
        v += A[c1 * 3 + c] * wy0 * wx1;
        v += B[c0 * 3 + c] * wy1 * wx0;
        v += B[c1 * 3 + c] * wy1 * wx1;
        v = f32(v);
        out[(y * ow + x) * 3 + c] = v < lo ? lo : v > hi ? hi : v;
      }
    }
  }
  return { w: ow, h: oh, data: out };
}

// resize_work / _work: longest side to workPx, never upscale.
export function resizeWork(img, workPx = WORK_PX) {
  const { w: W, h: H } = img;
  if (Math.max(H, W) <= workPx) return img;
  const s = workPx / Math.max(H, W);
  return skResize(img, Math.max(1, Math.trunc(H * s)), Math.max(1, Math.trunc(W * s)));
}

// ---------- notan.py ----------

function norm01(f) {
  let lo = Infinity, hi = -Infinity;
  for (const v of f) { if (v < lo) lo = v; if (v > hi) hi = v; }
  const out = new Float64Array(f.length);
  if (hi - lo > EPS) for (let i = 0; i < f.length; i++) out[i] = (f[i] - lo) / (hi - lo);
  return out;
}

export function notanMass(img, absolute = false) {
  const { w, h } = img;
  const lum = luminance(img);
  const chroma = oklabChroma(img);
  let L, C;
  if (absolute) {
    L = lum; // already in [0, 1]
    C = chroma.map((v) => (v < 0 ? 0 : v > 1 ? 1 : v));
  } else {
    L = norm01(lum);
    let cmax = -Infinity;
    for (const v of chroma) if (v > cmax) cmax = v;
    C = cmax >= CHROMA_DUST ? norm01(chroma) : new Float64Array(chroma.length);
  }
  const Lb = median(L), Cb = median(C);
  const mass = new Float64Array(w * h);
  for (const s of NOTAN_SCALES) {
    const gl = gauss2d(L, h, w, s), gc = gauss2d(C, h, w, s);
    for (let i = 0; i < mass.length; i++) mass[i] += Math.abs(gl[i] - Lb) + CHROMA_WEIGHT * Math.abs(gc[i] - Cb);
  }
  for (let i = 0; i < mass.length; i++) mass[i] /= NOTAN_SCALES.length;
  return mass;
}

export function fieldVoid(mass, tau = TAU_VOID) {
  let n = 0;
  for (const v of mass) if (v < tau) n++;
  return n / mass.length;
}

function weighted(mass, w, h, fn) {
  const t = new Float64Array(w * h);
  for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) t[i] = fn(x, y) * mass[i];
  return npsum(t);
}

export function fieldDispersion(mass, w, h) {
  const tot = npsum(mass);
  if (tot < EPS) return 0;
  const cx = weighted(mass, w, h, (x) => x) / tot;
  const cy = weighted(mass, w, h, (x, y) => y) / tot;
  const v = weighted(mass, w, h, (x, y) => (x - cx) ** 2 + (y - cy) ** 2) / tot;
  return Math.sqrt(v) / Math.sqrt(h * h + w * w);
}

// value_map: a 2, 3 or 5 value Notan in [0, 1].
export function valueMap(img, n = 3, mass = null) {
  if (![2, 3, 5].includes(n)) throw new Error("n must be 2, 3 or 5");
  const { w, h } = img;
  if (!mass) mass = notanMass(img);
  const lum = luminance(img);
  const sq = gauss2d(lum, h, w, 8.0);
  const med = median(lum);
  const N = w * h;
  const out = new Float64Array(N);
  const darker = new Uint8Array(N);
  for (let i = 0; i < N; i++) darker[i] = sq[i] < med ? 1 : 0;
  if (n === 2) { for (let i = 0; i < N; i++) out[i] = darker[i] ? 0 : 1; return out; }
  for (let i = 0; i < N; i++) out[i] = mass[i] >= TAU_VOID ? (darker[i] ? 0 : 1) : 0.5;
  if (n === 5) {
    for (const [side, extreme, near, deeper] of [[1, 0.0, 0.25, (a, b) => a < b], [0, 1.0, 0.75, (a, b) => a > b]]) {
      const vals = [];
      for (let i = 0; i < N; i++) if (mass[i] >= TAU_VOID && darker[i] === side) vals.push(sq[i]);
      if (!vals.length) continue;
      const cut = median(vals);
      for (let i = 0; i < N; i++) {
        if (mass[i] >= TAU_VOID && darker[i] === side) out[i] = deeper(sq[i], cut) ? extreme : near;
      }
    }
  }
  return out;
}

const round = pyRound;

// notan_readout on an already-resized image (pass the image from resizeWork).
export function notanReadout(small, mass = null) {
  const { w, h } = small;
  if (!mass) mass = notanMass(small);
  const tot = npsum(mass) + EPS;
  const cx = weighted(mass, w, h, (x) => x) / tot / Math.max(w - 1, 1) - 0.5;
  const cy = weighted(mass, w, h, (x, y) => y) / tot / Math.max(h - 1, 1) - 0.5;
  const chroma = oklabChroma(small);
  let cmax = -Infinity;
  for (const v of chroma) if (v > cmax) cmax = v;
  let mf = 0;
  for (const v of mass) if (v >= TAU_VOID) mf++;
  return {
    field_void: round(fieldVoid(mass), 4),
    field_dispersion: round(fieldDispersion(mass, w, h), 4),
    mass_centroid: [round(cx, 4), round(cy, 4)],
    offcenter_force: round(Math.hypot(cx, cy), 4),
    mass_fraction: round(mf / mass.length, 4),
    chroma_support: pySig(percentile(chroma, 98), 3),
    work_px: Math.max(w, h),
    neutral: cmax < CHROMA_DUST,
  };
}

// ---------- construction.py ----------

// np.linspace(a, b, n) as float64
function linspace(a, b, n) {
  const div = n - 1, delta = b - a, step = delta / div;
  const y = new Float64Array(n);
  for (let i = 0; i < n; i++) y[i] = i * step + a;
  y[n - 1] = b;
  return y;
}

// skimage.exposure.equalize_hist on a float64 field (nbins=256), via np.histogram + np.interp.
function equalizeHist(img) {
  const nb = 256;
  let first = Infinity, last = -Infinity;
  for (const v of img) { if (v < first) first = v; if (v > last) last = v; }
  if (first === last) { first -= 0.5; last += 0.5; }
  const edges = linspace(first, last, nb + 1);
  const hist = new Float64Array(nb);
  const denom = last - first;
  for (const v of img) {
    let k = Math.trunc(((v - first) / denom) * nb);
    if (k === nb) k -= 1;
    if (v < edges[k]) k -= 1;
    if (v >= edges[k + 1] && k !== nb - 1) k += 1;
    hist[k] += 1;
  }
  const cdf = new Float64Array(nb);
  let acc = 0;
  for (let i = 0; i < nb; i++) { acc += hist[i]; cdf[i] = acc; }
  for (let i = 0; i < nb; i++) cdf[i] = cdf[i] / acc;
  const xp = new Float64Array(nb);
  for (let i = 0; i < nb; i++) xp[i] = (edges[i] + edges[i + 1]) / 2.0;
  const slopes = new Float64Array(nb - 1);
  for (let j = 0; j < nb - 1; j++) slopes[j] = (cdf[j + 1] - cdf[j]) / (xp[j + 1] - xp[j]);
  const out = new Float64Array(img.length);
  for (let i = 0; i < img.length; i++) {
    const x = img[i];
    if (x < xp[0]) { out[i] = cdf[0]; continue; }
    if (x >= xp[nb - 1]) { out[i] = cdf[nb - 1]; continue; }
    let lo = 0, hi = nb - 1;                      // xp[lo] <= x < xp[hi]
    while (hi - lo > 1) { const m = (lo + hi) >> 1; if (xp[m] <= x) lo = m; else hi = m; }
    out[i] = x === xp[lo] ? cdf[lo] : slopes[lo] * (x - xp[lo]) + cdf[lo];
  }
  return out;
}

export const DEGRADATIONS = ["grayscale", "gamma_up", "gamma_down", "contrast_norm"];

export function degrade(small, name) {
  const { w, h, data } = small;
  const N = w * h;
  const out = new Float64Array(N * 3);
  if (name === "grayscale") {
    const g = luminance(small);
    for (let i = 0; i < N; i++) out[i * 3] = out[i * 3 + 1] = out[i * 3 + 2] = g[i];
  } else if (name === "gamma_up" || name === "gamma_down") {
    const e = name === "gamma_up" ? 1.6 : 0.6;
    for (let i = 0; i < N * 3; i++) {
      const v = data[i] < 0 ? 0 : data[i] > 1 ? 1 : data[i];
      out[i] = Math.pow(v, e);
    }
  } else if (name === "contrast_norm") {
    const g = luminance(small);
    const lum = g.map((v) => v + 1e-6);
    const eq = equalizeHist(lum);
    for (let i = 0; i < N; i++) {
      const ratio = eq[i] / lum[i];
      for (let c = 0; c < 3; c++) {
        let v = data[i * 3 + c];
        v = v < 0 ? 0 : v > 1 ? 1 : v;
        v = v * ratio;
        out[i * 3 + c] = v < 0 ? 0 : v > 1 ? 1 : v;
      }
    }
  } else throw new Error("unknown degradation " + name);
  return toImage(w, h, out);
}

export function survival(ref, deg) {
  const tot = npsum(ref);
  if (tot < EPS) return 1;
  const m = new Float64Array(ref.length);
  for (let i = 0; i < ref.length; i++) m[i] = Math.min(ref[i], deg[i]);
  const s = npsum(m) / tot;
  return s < 0 ? 0 : s > 1 ? 1 : s;
}

// dependence_map on an already-resized image: per-pixel mass lost under a degradation.
export function dependenceMap(small, degradation = "grayscale", ref = null) {
  if (!ref) ref = notanMass(small, true);
  const deg = notanMass(degrade(small, degradation), true);
  const lost = new Float64Array(ref.length);
  for (let i = 0; i < ref.length; i++) lost[i] = Math.max(ref[i] - deg[i], 0);
  return { lost, mass: ref, degraded: deg, survival: round(survival(ref, deg), 4) };
}

// construction_read on an already-resized image. `known` can hold degraded masses that
// were already computed (dependenceMap returns the greyscale one), to skip a read.
export function constructionRead(small, ref = null, known = {}) {
  if (!ref) ref = notanMass(small, true);
  const surv = {};
  for (const name of DEGRADATIONS) surv[name] = round(survival(ref, known[name] || notanMass(degrade(small, name), true)), 4);
  let fragile = DEGRADATIONS[0];
  for (const k of DEGRADATIONS) if (surv[k] < surv[fragile]) fragile = k;
  const gray = surv.grayscale;
  const built = gray < 0.5 ? "chromatic" : fragile === "gamma_up" || fragile === "gamma_down" ? "tonal" : "mixed";
  const vals = DEGRADATIONS.map((k) => surv[k]);
  return {
    built,
    fragile_to: fragile,
    grayscale_survival: gray,
    mass_survival: round(npsum(Float64Array.from(vals)) / vals.length, 4),
    survival_min: round(Math.min(...vals), 4),
    per_degradation: surv,
  };
}

// Everything the demo shows, from a full-size image.
export function readAll(img, { construction = true } = {}) {
  const small = resizeWork(img);
  const mass = notanMass(small);
  const ref = notanMass(small, true);
  const dep = dependenceMap(small, "grayscale", ref);
  return {
    small,
    mass,
    values: { 2: valueMap(small, 2, mass), 3: valueMap(small, 3, mass), 5: valueMap(small, 5, mass) },
    dependence: dep,
    readout: notanReadout(small, mass),
    construction: construction ? constructionRead(small, ref, { grayscale: dep.degraded }) : null,
  };
}
