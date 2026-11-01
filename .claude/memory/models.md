---
name: models
description: Read this file before you change a model, a model revision, the verbalizer, the stress model or the text pipeline.
metadata:
  type: project
---

- The server runs the ONNX bundles of HolosTTS: `onnx-int8/` on the CPU and `onnx/` (float32) on CUDA. Each bundle has three stages, `encoder`, `prosody` and `decoder`, and `holos_onnx.json` names them. `engine.HolosEngine` is a port of the MIT package `holos-tts-onnx`. Do not depend on that package: on Linux it needs `onnxruntime-gpu`, which conflicts with the `cpu` extra.
- Do not switch to the torch code of the author. The README of the author gives RTF 0.044 for ONNX INT8 and 0.074 for PyTorch on an i7-13700K.
- Each download has a pinned revision: `engine.MODEL_REVISION`, `verbalizer.MODEL_REVISION` and `verbalizer.TOKENIZER_REVISION`. The author changed the model architecture between revisions, and old code cannot load new weights. `engine.check_manifest` refuses a bundle with another format, decoder mode or vocabulary. Before you move a pin, listen to the speech and compare it with STT.
- A move of the pin can change the voice vectors. The update from upstream to fdd917a retrained all 27 vectors under the same names, and an old vector gives near silence in the new model. A user then must clone the voice again.
- `engine.VOCAB` has U+0027 four times. The tokenizer of the checkpoint keeps the last index of a repeated symbol, and `engine.TOKEN_IDS` does the same.
- The decoder takes at most `max_frames` of the manifest (1000 frames, 25 s) and fails above it. The encoder takes at least 2069 tokens and returns the duration of each token, about 2.27 frames for each token at speed 1.0. The character limits of `text` do not keep a chunk below the limit, because the verbalizer can make a text three times longer. `HolosEngine._speak` encodes a chunk, and when its frames are above the limit, it splits the phonemes with `engine.split_phonemes` in proportion and encodes each part again.
- At speed 2.0 the old and the new weights both give unintelligible speech by STT. The server allows 0.7 to 1.3, the range of the HolosTTS demo.
- The verbalizer is the CTranslate2 build of M2M100. It needs only sentencepiece and `vocab.json`, not transformers. The output is the same, and transformers costs about 200 MB.
- Only a sentence with digits, symbols, Latin letters or acronyms goes to the verbalizer (`text.needs_verbalization`). The verbalizer is slow, and it sometimes rewrites plain words, for example "Невеличкі" into "Невеликі".
- The verbalizer keeps a numeric date such as 15.03.2026 as digits. `text.prenormalize` writes the month as a word first.
- The verbalizer model ends each output after 127 tokens, its language token included, also in float32. A longer output loses its tail, for example "1, 2, …, 30" stops at "двад". `verbalizer.verbalize_in_parts` then verbalizes the two halves of the text, and a sentence that fits stays as it was.
- Alone, the verbalizer reads a part such as "1333, 1370," as one number, or "16, 17," as ordinals. It also loses the case of a number list: after "о 06:15, 07:15," the half "08:15, 08:45" becomes "вісім годин п'ятнадцять хвилин". For this reason, a later half that starts with a number gets the preposition of its list before it (`text.governing_preposition`), or else "і". On 43 cut sentences, the preposition cut the wrong words from 163 to 37, with the same number of passes.
- Decoding past 127 tokens (`min_decoding_length`) gives garbage, so the parts are the only way to verbalize a long output.
- `text.recover_stress` puts the stress marks of the user back after the verbalizer. It cannot do this next to punctuation that the verbalizer changes.
- The stress model is stanza through `ukrainian-word-stress`. A ByT5 CTranslate2 stressifier needs 2.7 s for each sentence and 1.36 GB more memory, against 0.07 s and 0.57 GB.
- An unknown voice name gets the default voice and a warning, not HTTP 400. The openai_tts integration of Home Assistant sends the voice "alloy" in a probe.
- Free-threaded Python gives no gain now. The wheels of onnxruntime, ctranslate2 and sentencepiece for it do not exist, and ctranslate2 turns the GIL on again at import.
