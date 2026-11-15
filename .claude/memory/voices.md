---
name: voices
description: Read this file before you change a voice source, a voice pin, the voice switches, the voice list or the voice text of the README.
metadata:
  type: project
---

- `engine.load_voices` reads three sources in this order: the voices of the model, the CC0 voices, the files in `/data/voices`. A voice of a later source replaces a voice with the same name.
- `VOICES_PATRIOTYK` and `VOICES_MORKON` turn off the voices of the model and the CC0 voices. A source that is off is not downloaded. The two names are a decision of the user. The model itself still comes from the repository of patriotyk when `VOICES_PATRIOTYK=0`.
- An empty voice list makes `load_voices` raise `ValueError` with the names of both switches. The server does not stop: `__main__._warm_up` logs the error, and each request gets it.
- An unknown voice name gets the default voice and a warning, not HTTP 400. The openai_tts integration of Home Assistant sends the voice "alloy" in a probe. With `VOICES_PATRIOTYK=0`, the default `Speaker_67` is not in the list, so the first voice of the list replaces it.

## Voices of the model

- The 27 voices are `onnx-int8/voices.npz` of `patriotyk/HolosTTS` at `engine.MODEL_REVISION`. Both bundles hold the same voices.
- The speakers of these voices did not agree to the synthesis, and the author of the model does not know whose voices they are. Source: https://huggingface.co/patriotyk/HolosTTS/discussions/5. The README tells the users to set `VOICES_PATRIOTYK=0` for public content.

## CC0 voices of Morkon

- The 29 CC0 voices are `voices.npz` of the dataset `Morkon/holos-voices-uk-cc0`, pinned to `engine.CC0_VOICES_REVISION`. Morkon made them with HolosTTS (float32) from Common Voice recordings with the CC0-1.0 license. `SOURCES.md` of the dataset names the recording of each voice.
- The same file, with the same SHA-256, is `voices/uk-cc0/voices.npz` of the pull request https://huggingface.co/patriotyk/HolosTTS/discussions/6. The server reads the dataset, because the author of the model can close the pull request, and the user can ask Morkon to keep the dataset. When that pull request merges, move the CC0 source to the model repository: `MODEL_REPO`, the file `voices/uk-cc0/voices.npz` and the merge commit, with `repo_type` model. Make sure that the SHA-256 of the file stays the same.
- The `voices.npz` file holds only `names` and `values`. The names are Ukrainian descriptions, for example "Чоловічий баритон · зрілий · теплий". The ASCII ids and the other data are in `names.json` and `manifest.json` of the dataset. The six youth voices have names that start with "Юнацький голос".
- The vectors fit the model of `fdd917a`: the `onnx/` files of the pull request have the same SHA-256 as at `fdd917a`. The int8 bundle speaks with them too. When you move `MODEL_REVISION`, listen to these voices again.
- A failed download of the CC0 voices logs an error, and the other voices stay.
- Before you move `CC0_VOICES_REVISION`, make sure that `voices.npz` keeps the format `names` and `values`, and listen to each voice.
- On 2026-11-14, Morkon wrote in the pull request that the dataset and the commit `53a73a5` stay. HolosTTS Core (https://github.com/Karnemorkon/holos-tts-core), a C++ server, uses the same pack as its built-in voice catalog.
- `manifest.json` of the dataset recommends a post-processing for one voice, "Юнацький голос · дівчина · високий · теплий" (`child-girl-high-warm`): `highpass=f=70,rubberband=tempo=1.05:formant=preserved`. The server does not apply it and plays the raw voice.

## Checks of the speech

- To check new voice vectors without an image, run a script with `uv run` on the host. It builds `worker.Worker(Settings(data_dir=...))`, adds the new vectors to `worker._voices`, and calls `worker.synthesize(text.prepare_sentences(sentence), voice, 1.0)` for each voice. `audio.encode(audio, SAMPLE_RATE, "wav")` gives the file. To run the FP32 bundle on the CPU, set `engine.BUNDLES["cpu"] = "onnx"` before the worker loads the engine. On Windows, first read the trap about `data/huggingface`. For it, read this file: [containers.md](containers.md).

- For a listen by the user, give each variant as its own WAV file, named by the variant, in one folder per voice. Never join the variants into one file: the user does not count positions in a file.
- STT heard a repeated syllable, for example "машишина", in 11 of the 29 CC0 voices. A listen found no repeat, so judge a repeated syllable by ear, not by STT.
- All voices say "Целсія" with a hard "л", also the voices of the model. The phonemizer gives `ˈt͡sɛlʲsʲijɐ` with a soft `lʲ`, and `engine.tokenize` keeps `l` and `ʲ`. So the acoustic model makes this error, not the text pipeline. The quantization is not the cause either: FP32, FP16, int8 and int4 copies of the ONNX model all lose about half of the soft sign, by the ear of the user.
- The Space `patriotyk/HolosTTS` downloads `holos.pt` and `voices.pt` from `main` with no revision. Both files have the same SHA-256 at `MODEL_REVISION` and at `main`. With the same IPA, PyTorch and ONNX FP32 give the same durations.
- To run PyTorch `holos.pt` as the Space does, run `uv run --with holos-tts==0.4 --with "torchaudio==2.11.*" python <script>` from the repository root. PyPI has no torchaudio for the torch version of the project, so give `2.11.*`. The script copies the text steps and `split_to_parts` of `app.py`, and calls `HolosTTS("patriotyk/HolosTTS", device="cpu").generate(ipa, voice)`. A voice is a name of `voices.pt` or a `torch.Tensor` of 256 numbers.
- To make an own quantized copy of the ONNX bundle, run `onnxruntime.quantization.quantize_dynamic` or `MatMulNBitsQuantizer` with `--with onnx --with onnx-ir`. First remove `graph.value_info` from the FP32 graph: the ONNX shape inference refuses its stored shapes. The int8 bundle of the author is static QDQ, and its encoder stays mostly FP32.
- The server has no phonemizer of its own. It uses `ukrainian-word-stress` and `ipa-uk` from the GitHub of patriotyk, as the README of the model tells. `frontend.Phonemizer` does the same text steps as `synthesize()` in `app.py` of the Space `patriotyk/HolosTTS`, and both give the same IPA. The Space runs PyTorch `holos.pt`, and the server runs the ONNX bundles.
