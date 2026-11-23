window.LOOPMM = {
  ortBase: new URL('ort/', location.href).href,
  tokenizersUrl: new URL('vendor/tokenizers.mjs', location.href).href,
  models: [
 {
  "id": "base",
  "label": "Base (pretrained only)",
  "repo": "MostLime/loopmm-base",
  "rev": "029807d250f7d4458ab4d9d7fb22baaa92ceaffd",
  "mode": "complete"
 }
],
  fileUrl: (m, f) => new URL(`models/${m.id}/${f}`, location.href).href,
  about: "A 189M-parameter looped multimodal model. <a href=\"https://huggingface.co/MostLime/loopmm-base\" target=\"_blank\">Base</a> \u00b7 <a href=\"https://github.com/Mostlime12195/loopmm-app/releases/latest\" target=\"_blank\">Mac app</a>",
};
