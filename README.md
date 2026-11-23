# loopmm for Mac

A small Mac app for chatting with [loopmm](https://huggingface.co/MostLime/loopmm-chat), a 189M-parameter looped model that reads images and text. The model is inside the app, so it works offline. Your messages and images stay on your Mac.

## Install

1. Download `loopmm-<version>-arm64.dmg` from the [latest release](https://github.com/Mostlime12195/loopmm-app/releases/latest). Use the `x64` file on an Intel Mac.
2. Open the dmg and drag loopmm to Applications.
3. The app is not notarized by Apple, so macOS blocks it the first time. Run this once in Terminal, then open it as usual:

```bash
xattr -dr com.apple.quarantine /Applications/loopmm.app
```

## Use

- Type a message, or add images with the + button, by pasting, or by dropping them on the window.
- **Model**: the chat model, or the base model, which only continues text.
- **Thinking**: whether the chat model writes its reasoning first. The reasoning shows in a box that you can open and close.
- **Loops**: how many times the model runs its 12 middle layers for each token. It was trained with 3. Fewer loops are faster.

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
