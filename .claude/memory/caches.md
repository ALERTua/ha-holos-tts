---
name: caches
description: Read this file before you change a file under /data/cache or the code that writes it.
metadata:
  type: project
---

- `/data/cache` holds only files that the server makes from the downloaded models. A user can delete the folder, and the server makes the files again.
- All three caches write through `cache_files.atomic_path`. The temporary name is `<target>.<pid>.partial`, and a new write removes the old temporary paths of the same target. Two servers on one `/data` folder are not supported.
- HolosTTS has no cache. An optimized graph of the three-stage bundle saves only about 0.2 s for each load: encoder 0.45 → 0.30 s, prosody 0.11 → 0.06 s, decoder 0.31 → 0.28 s. `HolosEngine` removes the folder `onnx/` of the older single-graph model.
- On an x86 CPU without VNNI, the old single int8 graph at `ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL` gave unintelligible speech, because the U8S8 matmul overflows. `session.x64quantprecision=1` fixes it. Under qemu `-cpu Haswell`, the new int8 bundle speaks correctly without the entry, and the old graph shows the fault. The real i5-10400 is not checked yet, so the entry stays. Make sure of the speech with an STT model, because a waveform comparison does not show this error.
- `verbalizer-int8/`: `ct2_int8.convert_to_int8` writes the int8 copy of the CTranslate2 model with numpy. The binary layout comes from `ModelSpec._serialize` of the ctranslate2 package. The loader of CTranslate2 quantizes each variable whose name ends with "weight", with the absolute maximum of each row. A unit test compares the output byte by byte with the int8 file of ctranslate2.
- `stanza-pretrain/`: a word list and a `.npy` file replace `torch.load(..., weights_only=True)` of the stanza vectors, which is slow on 250 000 words. The key has the path, the size and the change time of the source file, so a copied file gets a new key. `meta.json` names the source, so the old copy of the same source goes away.
- `stanza_cache.install` replaces `Pretrain.load` of stanza. On any error, the original method runs.
