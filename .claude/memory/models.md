---
name: models
description: Read this file before you change a model, a model revision, the verbalizer, the stress model or the text pipeline.
metadata:
  type: project
---

- The server runs the ONNX export of HolosTTS: `holos_cpu_int8.onnx` on the CPU and `holos.onnx` (float32) on CUDA. Do not switch to the torch code of the author. The ONNX graph has a `speed` input, and the torch code has none.
- Each download has a pinned revision: `engine.MODEL_REVISION`, `verbalizer.MODEL_REVISION` and `verbalizer.TOKENIZER_REVISION`. The author changed the model architecture between revisions, and old code cannot load new weights. Before you move a pin, compare `engine.VOCAB` with the symbol table of the new checkpoint, and listen to the speech.
- `engine.VOCAB` has the apostrophe three times. The tokenizer of the checkpoint keeps the last index of a repeated symbol, and `engine.TOKEN_IDS` does the same.
- One pass of the int8 ONNX graph returns at most 600 000 samples (25 s) and drops the rest without an error. It has no 512-token limit: 1200 tokens run without an error. The cap comes at about 480 tokens at speed 1.0 and at about 250 tokens at speed 0.5. The character limits of `text` do not keep a chunk below the cap, because the verbalizer can make a text three times longer. `engine.split_phonemes` splits the phonemes of a chunk into passes of at most `engine.PASS_TOKENS` × speed tokens. When the audio of a pass still reaches the cap, `HolosEngine` logs a warning and speaks that pass again in parts.
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
