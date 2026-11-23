// Copy onnxruntime-web and the tokenizer library into web/, and download the model files at the pinned revisions
// (release.json) into models/<id>/. Run in CI before electron-builder.
import fs from 'node:fs';
import path from 'node:path';

const rel = JSON.parse(fs.readFileSync('release.json', 'utf8'));
fs.mkdirSync('web/ort', { recursive: true });
for (const f of fs.readdirSync('node_modules/onnxruntime-web/dist'))
  if (/^ort\.wasm\.min\.js$|^ort-wasm.*\.(wasm|mjs)$/.test(f)) fs.copyFileSync(`node_modules/onnxruntime-web/dist/${f}`, `web/ort/${f}`);
fs.mkdirSync('web/vendor', { recursive: true });
fs.copyFileSync('node_modules/@huggingface/tokenizers/dist/tokenizers.mjs', 'web/vendor/tokenizers.mjs');
console.log('ort files:', fs.readdirSync('web/ort'));

const FILES = ['onnx/meta.json', 'onnx/stem.onnx', 'onnx/prelude.onnx', 'onnx/core.onnx', 'onnx/coda.onnx',
               'tokenizer.json', 'tokenizer_config.json'];
for (const m of rel.models) {
  if (!m.rev) continue;
  const files = m.mode === 'chat' ? [...FILES, 'chat.json'] : FILES;
  for (const f of files) {
    const out = path.join('models', m.id, f);
    fs.mkdirSync(path.dirname(out), { recursive: true });
    const url = `https://huggingface.co/${m.repo}/resolve/${m.rev}/${f}`;
    const r = await fetch(url);
    if (!r.ok) throw new Error(`${r.status} ${url}`);
    fs.writeFileSync(out, Buffer.from(await r.arrayBuffer()));
    console.log(out, fs.statSync(out).size);
  }
}
