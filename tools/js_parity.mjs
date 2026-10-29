// Node side of tools/js_parity.py: runs docs/notan.js on the inputs Python wrote and
// prints the differences as JSON.
import { readFileSync } from "node:fs";
import { join } from "node:path";
import * as N from "../docs/notan.js";

const dir = process.argv[2];
const meta = JSON.parse(readFileSync(join(dir, "meta.json"), "utf8"));
const load = (f, T) => { const b = readFileSync(join(dir, f)); return new T(b.buffer, b.byteOffset, b.byteLength / T.BYTES_PER_ELEMENT); };
const maxAbs = (a, b) => { let m = 0; for (let i = 0; i < a.length; i++) m = Math.max(m, Math.abs(a[i] - b[i])); return m; };
const share = (a, b) => { let n = 0; for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) n++; return n / a.length; };
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);

const rows = [];
for (const m of meta) {
  const ref32 = load(`${m.i}.in.f32`, Float32Array);
  // Paintings go through fromRGBA, exactly as the page feeds canvas bytes to the worker.
  const img = m.rgba ? N.fromRGBA(load(`${m.i}.in.u8`, Uint8Array), m.w, m.h) : { w: m.w, h: m.h, data: ref32 };
  const r = N.readAll(img);
  const pyRead = { ...m.readout };
  const jsRead = { ...r.readout };
  const pyCon = { ...m.construction, dep_survival: m.dep_survival };
  const jsCon = { ...r.construction, dep_survival: r.dependence.survival };
  const row = {
    name: m.name,
    input: maxAbs(img.data, ref32),
    small: r.small.w === m.sw && r.small.h === m.sh ? maxAbs(r.small.data, load(`${m.i}.small.f32`, Float32Array)) : Infinity,
    mass: maxAbs(r.mass, load(`${m.i}.mass.f64`, Float64Array)),
    lost: maxAbs(r.dependence.lost, load(`${m.i}.lost.f64`, Float64Array)),
    v2: share(r.values[2], load(`${m.i}.v2.f64`, Float64Array)),
    v3: share(r.values[3], load(`${m.i}.v3.f64`, Float64Array)),
    v5: share(r.values[5], load(`${m.i}.v5.f64`, Float64Array)),
    readout_equal: same(pyRead, jsRead),
    construction_equal: same(pyCon, jsCon),
  };
  if (!row.readout_equal || !row.construction_equal) row.detail = { pyRead, jsRead, pyCon, jsCon };
  rows.push(row);
}
const roundCases = JSON.parse(readFileSync(join(dir, "round.json"), "utf8"));
const round_mismatch = roundCases.filter(([v, d, py]) => N.pyRound(v, d) !== py).map(([v, d, py]) => [v, d, py, N.pyRound(v, d)]);
console.log(JSON.stringify({ rows, round_mismatch }));
