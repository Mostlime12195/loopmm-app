# loopmm for Mac

A small Mac app for trying [loopmm](https://huggingface.co/MostLime/loopmm-base), a 189M-parameter looped model that takes images and text as input. The model is inside the app, so it works offline. Your text and images stay on your Mac.

## Install

1. Download `loopmm-<version>-arm64.dmg` from the [latest release](https://github.com/Mostlime12195/loopmm-app/releases/latest). Use the `x64` file on an Intel Mac.
2. Open the dmg and drag loopmm to Applications.
3. The app is not notarized by Apple, so macOS blocks it the first time. Run this once in Terminal, then open it as usual:

```bash
xattr -dr com.apple.quarantine /Applications/loopmm.app
```

## Use

- Write in the canvas. The model suggests how your text continues, in grey. Press Tab to take the next word, or the right arrow key to take all of it. Keep typing to ignore it, or press Esc to hide it.
- Add images above the text with the + image button, by pasting, or by dropping them on the window. The model reads them before the text.
- **Loops**: how many times the model runs its 12 middle layers for each token. It was trained with 3. Fewer loops are faster.
- This is the pretrained base model, so it continues text and does not follow instructions. A post-trained chat version is pending; the app already has a chat mode for it.

## How it works

It is an Electron app. The page is the same one as the [playground Space](https://huggingface.co/spaces/MostLime/loopmm-playground). The model runs as ONNX graphs (fp16 weights) on onnxruntime-web (WebAssembly, with threads). The model files come from the Hugging Face repos at the revisions pinned in `release.json`. The source of the page and of the model export is in the loopmm project.

## Build

```bash
npm install
npm run prepare-web     # copies onnxruntime-web and downloads the pinned model files
npm start               # run from source
npm run dist            # build the dmg and zip files
```

## License

App code: Apache 2.0. Model weights: CC BY-NC 4.0. The tokenizer is Liquid AI's, under the LFM Open License v1.0. See the model cards for details.
