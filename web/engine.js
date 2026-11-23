// loopmm inference engine for onnxruntime-web (browser and the desktop app).
// Four ONNX graphs: stem (pixels -> patch embeddings), prelude (layers 0-5), core (layers 6-17, run once per loop,
// with its own cache per loop) and coda (layers 18-23 + LM head). Python twin: export_hf.py OnnxRunner.

export class Engine {
  constructor(ort, sessions, meta) {
    this.ort = ort; this.s = sessions; this.meta = meta;
    this.reset(meta.num_loops || 3);
  }

  static async load(ort, buffers, meta, opts = {}) {
    const so = { executionProviders: ['wasm'], graphOptimizationLevel: 'all', ...opts };
    const s = {};
    for (const k of ['stem', 'prelude', 'core', 'coda']) s[k] = await ort.InferenceSession.create(buffers[k], so);
    return new Engine(ort, s, meta);
  }

  // ---- state
  reset(nLoops) {
    const T = this.ort.Tensor, m = this.meta;
    this.nLoops = nLoops ?? this.nLoops;
    this.n = 0;                                  // tokens fed so far
    this.pastPos = new T('int64', BigInt64Array.from([-1n]), [1]);
    this.st = {};
    const parts = ['pre', ...Array.from({ length: this.nLoops }, (_, r) => `core${r}`), 'coda'];
    for (const part of parts) {
      const key = part.startsWith('core') ? 'core' : part, d = {};
      for (const j of m.parts[key].attn) {
        d[`past_k${j}`] = new T('float32', new Float32Array(m.n_kv * m.head_dim), [1, m.n_kv, 1, m.head_dim]);
        d[`past_v${j}`] = new T('float32', new Float32Array(m.n_kv * m.head_dim), [1, m.n_kv, 1, m.head_dim]);
      }
      for (const j of m.parts[key].conv)
        d[`past_c${j}`] = new T('float32', new Float32Array((m.conv_kernel - 1) * m.hidden_size), [1, m.conv_kernel - 1, m.hidden_size]);
      this.st[part] = d;
    }
  }

  async _run(name, part, feeds) {
    const o = await this.s[name].run({ ...feeds, ...this.st[part] });
    const next = {};
    for (const k in o) if (k.startsWith('present_')) next['past_' + k.slice(8)] = o[k];
    this.st[part] = next;
    return o;
  }

  // Feed N new tokens. ids: number[]; img: number[] image id per token (0 = text); images: [{pix Float32Array
  // [M*p*p*3] (0..255), rc Int32Array [M*2], slot number[] (index into ids)}]. Returns logits Float32Array [V].
  async feed(ids, img = null, images = []) {
    const T = this.ort.Tensor, m = this.meta, N = ids.length, d = m.hidden_size, p = m.patch_size;
    const big = (a) => BigInt64Array.from(a, (x) => BigInt(x));
    const pos = new T('int64', big(ids.map((_, j) => this.n + j)), [N]);
    const imgT = new T('int64', big(img || ids.map(() => 0)), [N]);
    const emb = new Float32Array(N * d), isImg = new Float32Array(N);
    for (const im of images) {
      const M = im.slot.length;
      const o = await this.s.stem.run({ pix: new T('float32', im.pix, [M, p, p, 3]),
                                        rc: new T('int64', big(im.rc), [M, 2]) });
      const e = o.emb.data;
      for (let i = 0; i < M; i++) { emb.set(e.subarray(i * d, (i + 1) * d), im.slot[i] * d); isImg[im.slot[i]] = 1; }
    }
    const com = { pos, img: imgT, past_pos: this.pastPos };
    let o = await this._run('prelude', 'pre', { ids: new T('int64', big(ids), [1, N]),
      img_emb: new T('float32', emb, [1, N, d]), is_img: new T('float32', isImg, [1, N]), ...com });
    const base = [o.base0, o.base1, o.base2];
    const zero = new T('float32', new Float32Array(N * d), [1, N, d]);
    let bank = [zero, zero, zero, zero];
    for (let r = 0; r < this.nLoops; r++) {
      o = await this._run('core', `core${r}`, { base0: base[0], base1: base[1], base2: base[2], bank0: bank[0],
        bank1: bank[1], bank2: bank[2], bank3: bank[3], use_msg: new T('float32', Float32Array.from([r ? 1 : 0]), [1]), ...com });
      bank = [o.core0, o.core1, o.core2, o.core3];
    }
    o = await this._run('coda', 'coda', { base0: base[0], base1: base[1], base2: base[2], core0: bank[0],
      core1: bank[1], core2: bank[2], core3: bank[3], ...com });
    const pp = new BigInt64Array(this.pastPos.data.length + N);
    pp.set(this.pastPos.data); pp.set(pos.data, this.pastPos.data.length);
    this.pastPos = new T('int64', pp, [pp.length]);
    this.n += N;
    return o.logits.data;
  }
}

// ---------------------------------------------------------------- images
export function imageGrid(h, w, maxTokens = 256, p = 32, minTokens = 16, maxSide = 64) {
  const native = (h / p) * (w / p);
  const target = Math.min(maxTokens, Math.max(native, minTokens));
  const s = Math.sqrt(target / native);
  let gh = Math.max(1, Math.round(h / p * s)), gw = Math.max(1, Math.round(w / p * s));
  gh = Math.min(gh, maxSide); gw = Math.min(gw, maxSide);
  while (gh * gw > maxTokens) { if (gh >= gw) gh--; else gw--; }
  return [gh, gw];
}

// ImageBitmap/HTMLImageElement -> {gh, gw, pix Float32Array [gh*gw, p, p, 3], rc Int32Array}
export function preprocessImage(bitmap, maxTokens = 256, p = 32) {
  const [gh, gw] = imageGrid(bitmap.height, bitmap.width, maxTokens, p);
  const W = gw * p, H = gh * p;
  const cv = (typeof OffscreenCanvas !== 'undefined') ? new OffscreenCanvas(W, H)
    : Object.assign(document.createElement('canvas'), { width: W, height: H });
  const ctx = cv.getContext('2d');
  ctx.imageSmoothingEnabled = true; ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(bitmap, 0, 0, W, H);
  const px = ctx.getImageData(0, 0, W, H).data;
  const M = gh * gw, pix = new Float32Array(M * p * p * 3), rc = new Int32Array(M * 2);
  for (let r = 0; r < gh; r++) for (let c = 0; c < gw; c++) {
    const t = r * gw + c; rc[2 * t] = r; rc[2 * t + 1] = c;
    for (let i = 0; i < p; i++) for (let j = 0; j < p; j++) {
      const src = ((r * p + i) * W + (c * p + j)) * 4, dst = ((t * p + i) * p + j) * 3;
      pix[dst] = px[src]; pix[dst + 1] = px[src + 1]; pix[dst + 2] = px[src + 2];
    }
  }
  return { gh, gw, pix, rc };
}

// ---------------------------------------------------------------- tokens: byte-level BPE pieces
export class Tok {
  constructor(TokenizerCls, tokJson, tokCfg) {
    this.t = new TokenizerCls(tokJson, tokCfg);
    const V = 65536;
    this.pieces = new Array(V); this.special = new Uint8Array(V); this.specialIds = new Map();
    for (const [p, id] of Object.entries(tokJson.model.vocab)) this.pieces[id] = p;
    for (const a of tokJson.added_tokens || []) {
      this.pieces[a.id] = a.content;
      if (a.special) { this.special[a.id] = 1; this.specialIds.set(a.content, a.id); }
    }
    const bs = []; for (let i = 33; i <= 126; i++) bs.push(i); for (let i = 161; i <= 172; i++) bs.push(i); for (let i = 174; i <= 255; i++) bs.push(i);
    const cs = bs.slice(); let n = 0;
    for (let b = 0; b < 256; b++) if (!bs.includes(b)) { bs.push(b); cs.push(256 + n++); }
    this.BYTE = new Map(); bs.forEach((b, i) => this.BYTE.set(String.fromCharCode(cs[i]), b));
    const esc = [...this.specialIds.keys()].sort((a, b) => b.length - a.length).map((s) => s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'));
    this.splitRe = esc.length ? new RegExp(`(${esc.join('|')})`) : null;
    this.utf8 = new TextDecoder('utf-8');
  }
  // text with special tokens written out (e.g. "<|im_start|>user\n") -> ids
  encode(text) {
    if (!text) return [];
    const out = [];
    for (const part of this.splitRe ? text.split(this.splitRe) : [text]) {
      if (!part) continue;
      if (this.specialIds.has(part)) out.push(this.specialIds.get(part));
      else out.push(...this.t.encode(part, { add_special_tokens: false }).ids);
    }
    return out;
  }
  id(s) { return this.specialIds.has(s) ? this.specialIds.get(s) : (this.encode(s).length === 1 ? this.encode(s)[0] : -1); }
  // ids -> text. keepSpecial: show special tokens that are not plain control tokens (e.g. <think>)
  decode(ids, keep = null) {
    const bytes = [];
    for (const id of ids) {
      const p = this.pieces[id];
      if (p == null) continue;
      if (this.special[id]) { if (keep && keep.has(id)) bytes.push(...new TextEncoder().encode(p)); continue; }
      for (const ch of p) { const b = this.BYTE.get(ch); if (b !== undefined) bytes.push(b); }
    }
    return this.utf8.decode(new Uint8Array(bytes)).replace(/�+$/, '');
  }
}

// ---------------------------------------------------------------- sampling
export function sample(logits, { temperature = 0.7, topP = 0.9, topK = 0, repPenalty = 1.0, seen = null, banned = null }) {
  const V = logits.length, l = new Float32Array(logits);
  if (banned) for (const i of banned) l[i] = -Infinity;
  if (repPenalty !== 1 && seen) for (const i of seen) l[i] = l[i] > 0 ? l[i] / repPenalty : l[i] * repPenalty;
  let best = 0; for (let i = 1; i < V; i++) if (l[i] > l[best]) best = i;
  if (temperature <= 0) return best;
  const mx = l[best], cand = [];
  for (let i = 0; i < V; i++) { const z = (l[i] - mx) / temperature; if (z > -30) cand.push([i, Math.exp(z)]); }
  cand.sort((a, b) => b[1] - a[1]);
  let c = topK > 0 ? cand.slice(0, topK) : cand;
  let tot = c.reduce((s, x) => s + x[1], 0);
  if (topP < 1) { let acc = 0, k = 0; while (k < c.length && acc < topP * tot) acc += c[k++][1]; c = c.slice(0, Math.max(1, k)); tot = acc; }
  let r = Math.random() * tot;
  for (const [i, w] of c) { r -= w; if (r <= 0) return i; }
  return c[c.length - 1][0];
}
