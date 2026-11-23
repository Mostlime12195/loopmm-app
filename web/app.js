// loopmm chat UI. Same code in the Hugging Face Space and in the desktop app; config.js says where files live.
import { Engine, Tok, sample, preprocessImage } from './engine.js';

const C = window.LOOPMM;
const $ = (id) => document.getElementById(id);
const CHUNK = 512, MAX_CTX = 8192;
const IMG_TOKENS = [395, 396, 498, 499];
const store = { get(k) { try { return localStorage.getItem(k); } catch { return null; } },
                set(k, v) { try { localStorage.setItem(k, v); } catch {} } };
const setStatus = (t) => { $('status').textContent = t; };
const setBar = (f) => { $('bar').hidden = f == null; if (f != null) $('bar').firstElementChild.style.width = `${Math.round(f * 100)}%`; };

// ---------------------------------------------------------------- settings
const S = {};
for (const [id, key, fmt] of [['loops', 'loops', (v) => v], ['temp', 'temperature', (v) => v], ['topp', 'topP', (v) => v],
                              ['rep', 'repPenalty', (v) => v], ['max', 'maxNew', (v) => v], ['len', 'len', (v) => v]]) {
  const el = $(id), out = $(id + 'v');
  const saved = store.get('loopmm.' + id); if (saved != null) el.value = saved;
  const upd = () => { S[key] = +el.value; out.textContent = fmt(el.value); store.set('loopmm.' + id, el.value); };
  el.addEventListener('input', upd); upd();
  el.addEventListener('input', () => { if (isCanvas()) { CV.ghost = ''; renderCanvas(); schedule(250); } });
}
$('detail').value = store.get('loopmm.detail') || '576';
$('detail').addEventListener('change', () => store.set('loopmm.detail', $('detail').value));
$('menu').onclick = () => $('app').classList.toggle('open');
$('log').addEventListener('click', () => $('app').classList.remove('open'));

// ---------------------------------------------------------------- downloads (Cache API when available)
async function fetchBytes(url, onProgress) {
  let cache = null;
  try { cache = await caches.open('loopmm-v1'); const hit = await cache.match(url); if (hit) return new Uint8Array(await hit.arrayBuffer()); } catch {}
  const r = await fetch(url);
  if (!r.ok) throw new Error(`HTTP ${r.status} for ${url}`);
  const total = +r.headers.get('content-length') || 0, reader = r.body.getReader(), parts = [];
  let got = 0;
  for (;;) { const { done, value } = await reader.read(); if (done) break; parts.push(value); got += value.length; onProgress?.(got, total); }
  const buf = new Uint8Array(got); let o = 0;
  for (const p of parts) { buf.set(p, o); o += p.length; }
  try { if (cache && /^https?:/.test(url)) await cache.put(url, new Response(buf)); } catch {}
  return buf;
}
const fetchJson = async (url) => JSON.parse(new TextDecoder().decode(await fetchBytes(url)));

// ---------------------------------------------------------------- model loading
let ort, TokenizerCls, engine = null, tok = null, model = null, chatCfg = null;
let fed = { keys: [] };          // what the engine holds: one key per token (image tokens keyed by image uid)

async function loadLibs() {
  await new Promise((ok, bad) => { const s = document.createElement('script'); s.src = C.ortBase + 'ort.wasm.min.js'; s.onload = ok; s.onerror = () => bad(new Error('could not load onnxruntime-web')); document.head.appendChild(s); });
  ort = window.ort;
  ort.env.wasm.wasmPaths = C.ortBase;
  ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(C.maxThreads || 8, navigator.hardwareConcurrency || 1) : 1;
  TokenizerCls = (await import(C.tokenizersUrl)).Tokenizer;
}

async function loadModel(m) {
  $('send').disabled = true;
  if (engine) { for (const s of Object.values(engine.s)) try { await s.release(); } catch {} engine = null; }
  model = m; fed = { keys: [] };
  $('model').value = m.id;
  const url = (f) => C.fileUrl(m, f);
  setStatus(`Loading ${m.label}…`);
  const meta = await fetchJson(url('onnx/meta.json'));
  const tj = await fetchJson(url('tokenizer.json')), tc = await fetchJson(url('tokenizer_config.json'));
  tok = new Tok(TokenizerCls, tj, tc);
  chatCfg = m.mode === 'chat' ? await fetchJson(url('chat.json')) : null;
  const names = ['stem', 'prelude', 'core', 'coda'], sizes = meta.file_bytes || {}, got = {}, bufs = {};
  const total = names.reduce((s, n) => s + (sizes[n] || 0), 0);
  for (const n of names) {
    bufs[n] = await fetchBytes(url(`onnx/${n}.onnx`), (g, t) => {
      got[n] = g; const all = Object.values(got).reduce((a, b) => a + b, 0);
      setBar(total ? all / total : g / (t || 1));
      setStatus(`Downloading ${m.label}: ${(all / 1e6).toFixed(0)}${total ? ` of ${(total / 1e6).toFixed(0)}` : ''} MB (once)`);
    });
  }
  setBar(null); setStatus('Starting…');
  engine = await Engine.load(ort, bufs, meta);
  // per-model defaults (the chat model wants a low temperature and no repetition penalty)
  const dft = { ...(m.defaults || {}), ...(chatCfg?.defaults || {}) };
  for (const [id, key] of [['temp', 'temperature'], ['topp', 'topP'], ['rep', 'repPenalty']])
    if (dft[key] != null) { $(id).value = dft[key]; $(id).dispatchEvent(new Event('input')); }
  if (dft.detail) $('detail').value = String(dft.detail);
  // thinking modes
  const modes = chatCfg?.thinking_modes || [];
  $('thinkfield').hidden = !modes.length;
  $('thinking').innerHTML = modes.map((x) => `<option value="${x.id}">${x.label}</option>`).join('');
  const sv = store.get('loopmm.thinking'); if (sv && modes.some((x) => x.id === sv)) $('thinking').value = sv;
  $('input').placeholder = m.mode === 'chat' ? 'Message loopmm' : 'Start a text; the base model continues it';
  renderEmpty();
  setMode();
  setStatus(`Ready · ${m.label} · ${ort.env.wasm.numThreads} thread${ort.env.wasm.numThreads > 1 ? 's' : ''}`);
  if (isCanvas()) { canvasReset(); schedule(0); }
  updateSend();
}
$('thinking').addEventListener('change', () => store.set('loopmm.thinking', $('thinking').value));

// ---------------------------------------------------------------- conversation -> tokens
let messages = [], uid = 0;

function thinkMode() { return (chatCfg?.thinking_modes || []).find((x) => x.id === $('thinking').value) || null; }

// units: {ids: number[]} or {image: imageObj}
function buildUnits(forGen) {
  const U = [], T = (s) => U.push({ ids: tok.encode(s) });
  U.push({ ids: [1] });
  if (model.mode !== 'chat') {
    for (const m of messages) {
      if (m.role === 'user') { for (const im of m.images) U.push({ image: im }); T(m.text); }
      else U.push({ ids: m.ids });
    }
    return U;
  }
  const c = chatCfg, mode = thinkMode();
  const turn = (role) => c.turn_start.replace('{role}', role);
  const sys = [c.system, mode?.system].filter(Boolean).join('\n');
  if (sys) T(turn('system') + sys + c.turn_end);
  const lastA = messages.map((m) => m.role).lastIndexOf('assistant');
  messages.forEach((m, i) => {
    if (m.role === 'user') {
      T(turn('user'));
      if (c.image_placement === 'after_text') { T(m.text); for (const im of m.images) U.push({ image: im }); T(c.turn_end); }
      else { for (const im of m.images) U.push({ image: im }); T(m.text + c.turn_end); }
    } else {
      T(turn('assistant'));
      if (c.think?.history === 'strip' && (forGen || i < lastA)) U.push({ ids: tok.encode(m.answer) });
      else U.push({ ids: m.ids });
      T(c.turn_end);
    }
  });
  if (forGen) { T(c.generation_prompt); if (mode?.prefill) T(mode.prefill); }
  return U;
}

function flatten(U) {
  const ids = [], keys = [], spans = [];
  for (const u of U) {
    if (u.ids) { for (const t of u.ids) { ids.push(t); keys.push(t); } continue; }
    const p = u.image.prep, s = ids.length;
    const span = [498]; for (let r = 0; r < p.gh; r++) { for (let c = 0; c < p.gw; c++) span.push(396); span.push(395); } span.push(499);
    span.forEach((t, k) => { ids.push(t); keys.push(`i${u.image.uid}:${k}`); });
    spans.push({ s, e: s + span.length, p });
  }
  return { ids, keys, spans };
}

async function feedFlat(F, from, onProgress) {
  let a = from, imgId = 0;
  let logits = null;
  while (a < F.ids.length) {
    let b = Math.min(F.ids.length, a + CHUNK);
    for (const sp of F.spans) if (sp.s < b && sp.e > b) b = sp.s > a ? sp.s : sp.e;   // never split an image
    const img = new Array(b - a).fill(0), images = [];
    for (const sp of F.spans) {
      if (sp.s < a || sp.e > b) continue;
      imgId++;
      for (let k = sp.s + 1; k < sp.e; k++) img[k - a] = imgId;
      const slot = [];
      for (let r = 0; r < sp.p.gh; r++) for (let c = 0; c < sp.p.gw; c++) slot.push(sp.s + 1 + r * (sp.p.gw + 1) + c - a);
      images.push({ pix: sp.p.pix, rc: sp.p.rc, slot });
    }
    logits = await engine.feed(F.ids.slice(a, b), img, images);
    fed.keys.push(...F.keys.slice(a, b));
    a = b;
    onProgress?.(a);
  }
  return logits;
}

// ---------------------------------------------------------------- rendering
const esc = (s) => s.replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]));
// minimal markdown for replies: code blocks, inline code, bold, italics (input is escaped first)
function md(s) {
  const parts = s.split(/```/);
  return parts.map((p, i) => {
    if (i % 2) return `<pre><code>${esc(p.replace(/^[\w+-]*\n/, ''))}</code></pre>`;
    return esc(p).replace(/`([^`\n]+)`/g, '<code>$1</code>').replace(/\*\*([^*\n]+)\*\*/g, '<b>$1</b>')
      .replace(/(^|[\s(])\*([^*\n]+)\*(?=[\s).,!?:;]|$)/g, '$1<i>$2</i>');
  }).join('');
}
function renderEmpty() {
  const e = $('empty'); if (!e) return;
  e.innerHTML = model?.mode === 'chat'
    ? '<b>loopmm</b>Ask something, or add an image and ask about it.<br>It is a small model, so expect mistakes.'
    : '<b>loopmm base</b>This is the pretrained model before chat tuning. Type the start of a text<br>(or add an image and write "A photo of") and it continues.';
}
function addMsgEl(m) {
  $('empty')?.remove();
  const el = document.createElement('div');
  el.className = 'msg ' + m.role;
  $('log').appendChild(el);
  m.el = el;
  renderMsg(m);
  return el;
}
function splitThink(text) {
  const c = chatCfg?.think;
  if (!c) return { think: null, answer: text, open: false };
  const i = text.indexOf(c.open);
  if (i < 0) return { think: null, answer: text, open: false };
  const j = text.indexOf(c.close, i);
  if (j < 0) return { think: text.slice(i + c.open.length).trim(), answer: '', open: true };
  return { think: text.slice(i + c.open.length, j).trim(), answer: (text.slice(0, i) + text.slice(j + c.close.length)).trim(), open: false };
}
function renderMsg(m, live = false) {
  const el = m.el;
  if (m.role === 'user') {
    el.innerHTML = (m.images.length ? `<div class="imgs">${m.images.map((im) => `<img src="${im.url}" alt="">`).join('')}</div>` : '') + esc(m.text);
    return;
  }
  if (m.error) { el.className = 'msg assistant err'; el.textContent = m.error; return; }
  const keep = chatCfg?.think ? new Set([tok.id(chatCfg.think.open), tok.id(chatCfg.think.close)].filter((x) => x >= 0)) : null;
  const text = tok.decode(m.ids, keep);
  const { think, answer, open } = splitThink(text);
  m.answer = answer;
  let h = '';
  if (think != null && think.length) {
    const wasOpen = el.querySelector('details')?.open;
    h += `<details${open || wasOpen ? ' open' : ''}><summary>${open ? 'Thinking…' : `Thought for ${think.split(/\s+/).length} words`}</summary>${esc(think)}</details>`;
  }
  h += `<span class="${live && !open ? 'cursor' : ''}">${model?.mode === 'chat' ? md(answer) : esc(answer)}</span>`;
  if (m.stats) h += `<div class="stats">${m.stats}</div>`;
  el.innerHTML = h;
}
const scrollDown = () => { const l = $('log'); if (l.scrollHeight - l.scrollTop - l.clientHeight < 120) l.scrollTop = l.scrollHeight; };

// ---------------------------------------------------------------- generation
let busy = false, stopFlag = false;
function updateSend() {
  const b = $('send');
  b.textContent = busy ? 'Stop' : 'Send';
  b.disabled = !engine || (!busy && !$('input').value.trim() && !pending.length);
}

async function generate() {
  busy = true; stopFlag = false; updateSend();
  const a = { role: 'assistant', ids: [], answer: '' };
  addMsgEl(a);
  try {
    const mode = model.mode === 'chat' ? thinkMode() : null;
    const prefill = mode?.prefill ? tok.encode(mode.prefill) : [];
    let F = flatten(buildUnits(true));
    while (F.ids.length + S.maxNew > MAX_CTX && messages.length > 2) {        // drop the oldest exchange
      for (const x of messages.splice(0, 2)) x.el?.remove();
      F = flatten(buildUnits(true));
    }
    let L = 0;
    if (engine.nLoops === S.loops) while (L < fed.keys.length && L < F.keys.length && fed.keys[L] === F.keys[L]) L++;
    if (engine.nLoops !== S.loops || L < fed.keys.length || L === F.ids.length) { engine.reset(S.loops); fed = { keys: [] }; L = 0; }
    const t0 = performance.now(), nPrompt = F.ids.length - L;
    let logits = await feedFlat(F, L, (k) => setStatus(`Reading ${k - L} of ${nPrompt} tokens…`));
    const tPrompt = performance.now();
    a.ids = prefill.slice();
    const stop = new Set(model.mode === 'chat' ? [2, ...(chatCfg.stop || []).map((s) => tok.id(s)).filter((x) => x >= 0)] : [1, 2]);
    const banned = [...IMG_TOKENS, ...(model.mode === 'chat' ? [1] : [])];
    const seen = new Set();
    let n = 0;
    setStatus(`Writing… · ${S.loops} loop${S.loops > 1 ? 's' : ''}`);
    while (n < S.maxNew && !stopFlag) {
      const t = sample(logits, { temperature: S.temperature, topP: S.topP, repPenalty: S.repPenalty, seen, banned });
      if (stop.has(t)) break;
      a.ids.push(t); seen.add(t); n++;
      renderMsg(a, true); scrollDown();
      logits = await engine.feed([t]);
      fed.keys.push(t);
    }
    const dt = (performance.now() - tPrompt) / 1000;
    a.stats = `${n} tokens · ${(n / Math.max(dt, 1e-3)).toFixed(1)} tok/s · ${S.loops} loop${S.loops > 1 ? 's' : ''}` +
              (nPrompt > 1 ? ` · read ${nPrompt} in ${((tPrompt - t0) / 1000).toFixed(1)} s` : '');
    messages.push(a);
    renderMsg(a);
    setStatus(`Ready · ${model.label}`);
  } catch (e) {
    console.error(e);
    a.error = `Error: ${e.message || e}`; renderMsg(a);
    engine.reset(S.loops); fed = { keys: [] };
    setStatus('Error (see the message)');
  }
  busy = false; updateSend(); scrollDown();
}

// ---------------------------------------------------------------- composer
let pending = [];
async function addImage(file) {
  if (!file?.type?.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const bmp = await createImageBitmap(file);
  const im = { uid: ++uid, url, prep: preprocessImage(bmp, +$('detail').value, 32) };
  if (isCanvas()) { CV.imgs.push(im); renderCanvasImgs(); CV.ghost = ''; renderCanvas(); schedule(0); return; }
  pending.push(im);
  renderAttach(); updateSend();
}
function renderAttach() {
  $('attach').innerHTML = '';
  pending.forEach((im, i) => {
    const d = document.createElement('div'); d.className = 'thumb';
    d.innerHTML = `<img src="${im.url}" alt=""><button type="button" aria-label="Remove">×</button>`;
    d.querySelector('button').onclick = () => { pending.splice(i, 1); renderAttach(); updateSend(); };
    $('attach').appendChild(d);
  });
}
$('file').addEventListener('change', async (e) => { for (const f of e.target.files) await addImage(f); e.target.value = ''; });
document.addEventListener('paste', async (e) => { for (const it of e.clipboardData?.items || []) if (it.type.startsWith('image/')) await addImage(it.getAsFile()); });
document.addEventListener('dragover', (e) => { e.preventDefault(); document.body.classList.add('drag'); });
document.addEventListener('dragleave', () => document.body.classList.remove('drag'));
document.addEventListener('drop', async (e) => { e.preventDefault(); document.body.classList.remove('drag'); for (const f of e.dataTransfer.files) await addImage(f); });
const input = $('input');
input.addEventListener('input', () => { input.style.height = 'auto'; input.style.height = Math.min(200, input.scrollHeight) + 'px'; updateSend(); });
input.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); $('composer').requestSubmit(); } });
$('composer').addEventListener('submit', (e) => {
  e.preventDefault();
  if (busy) { stopFlag = true; return; }
  const text = input.value;
  if (!engine || (!text.trim() && !pending.length)) return;
  const m = { role: 'user', text, images: pending };
  pending = []; renderAttach();
  input.value = ''; input.style.height = 'auto';
  messages.push(m); addMsgEl(m); scrollDown();
  generate();
});
$('newchat').onclick = () => {
  if (busy) stopFlag = true;
  messages = []; $('log').innerHTML = '<div id="empty"></div>'; renderEmpty();
  if (isCanvas()) { $('ed').value = ''; CV.imgs = []; renderCanvasImgs(); onCanvasInput(); }
  $('app').classList.remove('open');
};


// ---------------------------------------------------------------- canvas (base model): ghost-text continuation
// One text area. The model's continuation shows in grey after the caret. Tab takes the next word, → takes all,
// Esc hides it. Images sit before the text. The engine keeps the tokens it has read and only reads what changed:
// typing at the end costs one or two tokens, an edit earlier rolls back to the nearest snapshot.
const isCanvas = () => !!model && model.mode !== 'chat';
const CV = { imgs: [], ghost: '', ghostFor: null, genId: 0, chain: Promise.resolve(), debounce: null,
             keys: [], snaps: new Map(), logits: null, lastText: '', allowCache: new Map(), stats: null };
const ed = $('ed'), mirror = $('mirror');
const MAX_SNAPS = 120, CANVAS_MAX = 8192;

function setMode() {
  const c = isCanvas();
  $('canvas').hidden = $('canvasbar').hidden = !c;
  $('log').hidden = $('composer').hidden = c;
  $('maxfield').hidden = c; $('lenfield').hidden = !c;
  if (c) $('thinkfield').hidden = true;
  $('newchat').textContent = c ? 'Clear' : 'New chat';
}

function canvasReset() {
  engine.reset(S.loops);
  CV.keys = []; CV.snaps = new Map([[0, engine.convSnap()]]); CV.logits = null;
}

function renderCanvasImgs() {
  const box = $('cimgs');
  box.querySelectorAll('.thumb').forEach((x) => x.remove());
  CV.imgs.forEach((im, i) => {
    const d = document.createElement('div'); d.className = 'thumb';
    d.innerHTML = `<img src="${im.url}" alt=""><button type="button" aria-label="Remove">×</button>`;
    d.querySelector('button').onclick = () => { CV.imgs.splice(i, 1); renderCanvasImgs(); CV.ghost = ''; renderCanvas(); schedule(0); };
    box.insertBefore(d, box.lastElementChild);
  });
}

const caretAtEnd = () => ed.selectionStart === ed.value.length && ed.selectionEnd === ed.value.length;
function renderCanvas() {
  const show = CV.ghost && CV.ghostFor === ed.value && caretAtEnd() && document.activeElement === ed;
  mirror.textContent = ed.value;
  if (show) { const sp = document.createElement('span'); sp.className = 'ghost'; sp.textContent = CV.ghost; mirror.appendChild(sp); }
  mirror.appendChild(document.createTextNode('​'));
  $('bWord').disabled = $('bAll').disabled = !show;
}

function schedule(delay = 140) {
  clearTimeout(CV.debounce);
  const id = ++CV.genId;
  CV.debounce = setTimeout(() => { CV.chain = CV.chain.then(() => suggest(id)).catch((e) => { console.error(e); setStatus('Error: ' + e.message); }); }, delay);
}
// for tests: run one suggestion now and wait for it
function suggestNow() { clearTimeout(CV.debounce); const id = ++CV.genId; CV.chain = CV.chain.then(() => suggest(id)); return CV.chain; }

// feed F.ids[a:b] (never splitting an image), record a snapshot after each chunk
async function canvasFeed(F, a, b, alive) {
  let logits = null, imgId = 0;
  while (a < b) {
    let e = Math.min(b, a + CHUNK);
    for (const sp of F.spans) if (sp.s < e && sp.e > e) e = sp.s > a ? sp.s : sp.e;
    const img = new Array(e - a).fill(0), images = [];
    for (const sp of F.spans) {
      if (sp.s < a || sp.e > e) continue;
      imgId++;
      for (let k = sp.s + 1; k < sp.e; k++) img[k - a] = imgId;
      const slot = [];
      for (let r = 0; r < sp.p.gh; r++) for (let c = 0; c < sp.p.gw; c++) slot.push(sp.s + 1 + r * (sp.p.gw + 1) + c - a);
      images.push({ pix: sp.p.pix, rc: sp.p.rc, slot });
    }
    logits = await engine.feed(F.ids.slice(a, e), img, images);
    CV.keys.push(...F.keys.slice(a, e));
    CV.snaps.set(e, engine.convSnap());
    if (CV.snaps.size > MAX_SNAPS) { const ks = [...CV.snaps.keys()].filter((k) => k > 0).sort((x, y) => x - y); CV.snaps.delete(ks[0]); }
    a = e;
    if (!alive()) return null;
  }
  return logits;
}

async function suggest(id) {
  const alive = () => id === CV.genId && isCanvas() && !!engine;
  if (!alive()) return;
  const text = ed.value;
  if (!text.trim() && !CV.imgs.length) { CV.ghost = ''; renderCanvas(); return; }
  const units = [{ ids: [1] }, ...CV.imgs.map((im) => ({ image: im })), { ids: tok.encode(text) }];
  const F = flatten(units);
  if (F.ids.length > CANVAS_MAX) { setStatus(`The text is too long (${F.ids.length} tokens; the limit is ${CANVAS_MAX})`); return; }
  // token healing: re-generate the last text token, constrained to start with its bytes
  const textStart = F.spans.length ? F.spans[F.spans.length - 1].e : 1;
  let heal = '', last = null, C = F.ids.length;
  if (F.ids.length - 1 >= textStart) { last = F.ids[F.ids.length - 1]; heal = tok.pieces[last] || ''; C -= 1; }
  if (engine.nLoops !== S.loops) canvasReset();
  let L = 0;
  while (L < CV.keys.length && L < C && CV.keys[L] === F.keys[L]) L++;
  if (L < CV.keys.length || (L === C && last == null && !CV.logits)) {       // roll back to a snapshot at or before L
    let best = 0;
    for (const k of CV.snaps.keys()) if (k <= L && k > best && !(k === C && last == null)) best = k;
    engine.restore(CV.snaps.get(best));
    CV.keys = CV.keys.slice(0, best); CV.logits = null;
    for (const k of [...CV.snaps.keys()]) if (k > best) CV.snaps.delete(k);
    L = best;
  }
  const t0 = performance.now(), nRead = C - L + (last == null ? 0 : 1);
  if (L < C) { const lg = await canvasFeed(F, L, C, alive); if (!alive()) return; CV.logits = lg; }
  const fk = engine.fork();
  let logits = last == null ? CV.logits : await fk.feed([last]);
  if (!alive()) return;
  let allow = null;
  if (heal) {
    allow = CV.allowCache.get(heal);
    if (!allow) {
      allow = new Uint8Array(tok.pieces.length);
      for (let i = 0; i < tok.pieces.length; i++) if (tok.pieces[i] != null && !tok.special[i] && tok.pieces[i].startsWith(heal)) allow[i] = 1;
      CV.allowCache.set(heal, allow); if (CV.allowCache.size > 200) CV.allowCache.delete(CV.allowCache.keys().next().value);
    }
  }
  const healText = last == null ? '' : tok.decode([last]);
  const gen = [], seen = new Set();
  const t1 = performance.now();
  for (let s = 0; s < S.len; s++) {
    const t = sample(logits, { temperature: S.temperature, topP: S.topP, repPenalty: S.repPenalty, seen, banned: IMG_TOKENS, allow: s === 0 ? allow : null });
    if (t === 1 || t === 2) break;
    gen.push(t); seen.add(t);
    let g = tok.decode(gen);
    if (!g.startsWith(healText)) break;
    g = g.slice(healText.length);
    const cut = g.indexOf('\n\n');
    if (cut >= 0) g = g.slice(0, cut);
    CV.ghost = g; CV.ghostFor = text; renderCanvas();
    if (cut >= 0 || s === S.len - 1) break;
    logits = await fk.feed([t]);
    if (!alive()) return;
  }
  const genMs = performance.now() - t1;
  CV.stats = { read: nRead, readMs: Math.round(t1 - t0), gen: gen.length, tokPerS: gen.length > 1 ? +(1000 * (gen.length - 1) / genMs).toFixed(1) : null, loops: S.loops };
  setStatus(`Ready · ${model.label} · ${S.loops} loop${S.loops > 1 ? 's' : ''}` + (CV.stats.tokPerS ? ` · ${CV.stats.tokPerS} tok/s` : ''));
}

function onCanvasInput() {
  const text = ed.value;
  store.set('loopmm.canvas', text);
  // typed exactly what the ghost said: keep the rest of it
  if (CV.ghost && CV.ghostFor === CV.lastText && text.startsWith(CV.lastText) && text.length > CV.lastText.length &&
      CV.ghost.startsWith(text.slice(CV.lastText.length))) {
    CV.ghost = CV.ghost.slice(text.length - CV.lastText.length); CV.ghostFor = text; CV.lastText = text;
    CV.genId++; clearTimeout(CV.debounce);
    renderCanvas();
    if (!CV.ghost.trim()) schedule(60);
    return;
  }
  CV.lastText = text;
  CV.ghost = ''; renderCanvas();
  schedule();
}

function accept(all) {
  if (!CV.ghost || CV.ghostFor !== ed.value || !caretAtEnd()) return false;
  const m = all ? [CV.ghost] : CV.ghost.match(/^\s*[^\s]+/) || CV.ghost.match(/^\s+/);
  const piece = m ? m[0] : CV.ghost;
  ed.focus();
  if (!document.execCommand('insertText', false, piece)) { ed.setRangeText(piece, ed.value.length, ed.value.length, 'end'); onCanvasInput(); }
  return true;
}

ed.addEventListener('input', onCanvasInput);
ed.addEventListener('keydown', (e) => {
  if (e.key === 'Tab' && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
    if (accept(false)) e.preventDefault();
    else if (caretAtEnd()) e.preventDefault();          // keep focus in the canvas while a suggestion loads
  } else if (e.key === 'ArrowRight' && !e.shiftKey && caretAtEnd()) {
    if (accept(true)) e.preventDefault();
  } else if (e.key === 'Escape') {
    CV.ghost = ''; CV.genId++; renderCanvas();
  }
});
for (const ev of ['click', 'keyup', 'focus', 'blur', 'select']) ed.addEventListener(ev, () => renderCanvas());
document.addEventListener('selectionchange', () => { if (document.activeElement === ed) renderCanvas(); });
$('bWord').addEventListener('mousedown', (e) => { e.preventDefault(); accept(false); });
$('bAll').addEventListener('mousedown', (e) => { e.preventDefault(); accept(true); });
$('cfile').addEventListener('change', async (e) => { for (const f of e.target.files) await addImage(f); e.target.value = ''; });
{ const saved = store.get('loopmm.canvas'); ed.value = saved != null ? saved : 'The old lighthouse keeper climbed the stairs one last time, and'; CV.lastText = ed.value; renderCanvas(); }

// ---------------------------------------------------------------- start
(async () => {
  try {
    $('about').innerHTML = C.about || '';
    const models = C.models.filter((m) => !m.hidden);
    $('model').innerHTML = models.map((m) => `<option value="${m.id}">${m.label}</option>`).join('');
    const sv = store.get('loopmm.model'); if (sv && models.some((m) => m.id === sv)) $('model').value = sv;
    $('model').addEventListener('change', async () => {
      store.set('loopmm.model', $('model').value);
      $('newchat').onclick();
      try { await loadModel(models.find((m) => m.id === $('model').value)); } catch (e) { setStatus(`Could not load: ${e.message}`); }
    });
    await loadLibs();
    await loadModel(models.find((m) => m.id === $('model').value));
    if (new URLSearchParams(location.search).has('smoke')) await smoke(models);
    else if (isCanvas()) $('ed').focus();
  } catch (e) {
    console.error(e); setStatus(`Could not start: ${e.message || e}`);
    if (new URLSearchParams(location.search).has('smoke')) document.title = 'SMOKE:' + JSON.stringify({ ok: false, threads: window.ort?.env?.wasm?.numThreads, isolated: !!self.crossOriginIsolated, error: String(e && e.stack || e) });
  }
})();
// Scripted run for automated tests (?smoke=1): every model, a two-turn chat with an image, greedy. The result goes
// into document.title as 'SMOKE:{json}' (the desktop app writes it to a file; the Space test reads the title).
async function testImage() {
  const cv = document.createElement('canvas'); cv.width = 320; cv.height = 240;
  const g = cv.getContext('2d');
  g.fillStyle = '#3b6fd1'; g.fillRect(0, 0, 320, 240);
  g.fillStyle = '#e53935'; g.beginPath(); g.arc(160, 120, 70, 0, 2 * Math.PI); g.fill();
  return new File([await new Promise((ok) => cv.toBlob(ok, 'image/png'))], 'test.png', { type: 'image/png' });
}
async function smokeCanvas(m) {
  const r = { model: m.id, mode: 'canvas', ok: true };
  $('newchat').onclick();
  Object.assign(S, { temperature: 0, len: 16, loops: 3 });
  const run = async (text) => { ed.value = text; CV.lastText = text; await suggestNow(); return { text, ghost: CV.ghost, stats: CV.stats }; };
  r.text = await run('The capital of France is');
  const w = (CV.ghost.match(/^\s*[^\s]+/) || [''])[0];
  r.typed = await run('The capital of France is' + w);           // one accepted word: should read only a token or two
  CV.imgs = [{ uid: ++uid, url: '', prep: preprocessImage(await createImageBitmap(await testImage()), 256, 32) }];
  r.image = await run('A photo of');
  CV.imgs = [];
  S.loops = 1;
  r.loops1 = await run('The capital of France is');
  for (const k of ['text', 'typed', 'image', 'loops1']) if (!r[k].ghost) r.ok = false;
  if (r.typed.stats.read > 4) r.ok = false;
  S.loops = 3; await run('The old lighthouse keeper climbed the stairs one last time, and');
  ed.focus(); ed.setSelectionRange(ed.value.length, ed.value.length); renderCanvas();
  const key = (k) => ed.dispatchEvent(new KeyboardEvent('keydown', { key: k, bubbles: true, cancelable: true }));
  const before = ed.value, g0 = CV.ghost, w0 = (g0.match(/^\s*[^\s]+/) || [''])[0];
  key('Tab');
  r.tab = { ok: ed.value === before + w0 && w0.length > 0, added: w0 };
  const g1 = CV.ghost, b1 = ed.value;
  key('ArrowRight');
  r.arrow = { ok: g1.length > 0 && ed.value === b1 + g1, added: g1 };
  if (!r.tab.ok || !r.arrow.ok) r.ok = false;
  await suggestNow(); ed.focus(); ed.setSelectionRange(ed.value.length, ed.value.length); renderCanvas();
  return r;
}
async function smoke(models) {
  const out = { ok: true, threads: ort.env.wasm.numThreads, isolated: !!self.crossOriginIsolated, results: [] };
  const saved = { ...S };
  try {
    for (const m of models) {
      if (m !== model) await loadModel(m);
      if (m.mode !== 'chat') { out.results.push(await smokeCanvas(m)); if (!out.results.at(-1).ok) out.ok = false; continue; }
      $('newchat').onclick();
      Object.assign(S, { temperature: 0, maxNew: 24, loops: 3 });
      const turns = m.mode === 'chat' ? ['What is in this image?', 'What color is the circle?'] : ['A photo of', ' The'];
      const r = { model: m.id, turns: [] };
      for (let i = 0; i < turns.length; i++) {
        if (i === 0) await addImage(await testImage());
        const u = { role: 'user', text: turns[i], images: pending }; pending = []; renderAttach();
        messages.push(u); addMsgEl(u);
        await generate();
        const a = messages[messages.length - 1];
        r.turns.push({ user: turns[i], reply: a.role === 'assistant' ? tok.decode(a.ids) : null, stats: a.stats, error: a.error || null });
        if (a.role !== 'assistant' || !a.ids.length) out.ok = false;
      }
      S.loops = 1; $('newchat').onclick();
      const u = { role: 'user', text: m.mode === 'chat' ? 'Name a fruit.' : 'The capital of France is', images: [] };
      messages.push(u); addMsgEl(u); await generate();
      const a = messages[messages.length - 1];
      r.loops1 = { reply: a.role === 'assistant' ? tok.decode(a.ids) : null, stats: a.stats };
      out.results.push(r);
    }
  } catch (e) { out.ok = false; out.error = String(e && e.stack || e); }
  Object.assign(S, saved);
  document.title = 'SMOKE:' + JSON.stringify(out);
}
window.loopmm = { get engine() { return engine; }, get messages() { return messages; }, S };
