window.LOOPMM = {
  ortBase: new URL('ort/', location.href).href,
  tokenizersUrl: new URL('vendor/tokenizers.mjs', location.href).href,
  models: [
 {
  "id": "base",
  "label": "Base (pretrained only)",
  "repo": "MostLime/loopmm-base",
  "rev": "de7d56b3820b30f61f4f6a9ded03513a6ade5f6d",
  "mode": "complete",
  "defaults": {
   "temperature": 0.7,
   "topP": 0.9,
   "repPenalty": 1.1,
   "detail": 256
  }
 }
],
  fileUrl: (m, f) => new URL(`models/${m.id}/${f}`, location.href).href,
  about: "A 189M-parameter looped multimodal model. <a href=\"https://huggingface.co/MostLime/loopmm-base\" target=\"_blank\">Base</a> \u00b7 <a href=\"https://github.com/Mostlime12195/loopmm-app/releases/latest\" target=\"_blank\">Mac app</a>",
};
