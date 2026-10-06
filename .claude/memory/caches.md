---
name: caches
description: Read this file before you change a file under /data/cache or the code that writes it.
metadata:
  type: project
---

- `/data/cache` holds only files that the server makes from the downloaded models. A user can delete the folder, and the server makes the files again.
- All three caches write through `cache_files.atomic_path`. The temporary name is `<target>.<pid>.partial`, and a new write removes the old temporary paths of the same target. Two servers on one `/data` folder are not supported.
- `onnx/`: the optimized graph of HolosTTS, for the CPU only. ONNX Runtime warns that a graph saved with `ORT_ENABLE_ALL` is specific to the hardware. For this reason, the file name has a key of the model revision, the file, the ONNX Runtime version and the CPU flags.
- On an x86 CPU without VNNI, the int8 graph at `ORT_ENABLE_EXTENDED` or `ORT_ENABLE_ALL` gives unintelligible speech, because the U8S8 matmul overflows. Each CPU session there needs `session.x64quantprecision=1`, the saving session too. A graph that a session without this entry saved stays broken, so the entry is part of the key. Make sure of the speech with an STT model, because a waveform comparison does not show this error.
- The session that saves the graph keeps about 281 MB more memory for its life. Drop it, and open a new session from the saved file with `ORT_DISABLE_ALL`.
- Do not open the original graph with `ORT_DISABLE_ALL`. That session is about 60 % slower.
- `engine` remembers a failed save for the life of the process. Else each reload after an idle unload tries the save again and takes twice the time.
- `verbalizer-int8/`: `ct2_int8.convert_to_int8` writes the int8 copy of the CTranslate2 model with numpy. The binary layout comes from `ModelSpec._serialize` of the ctranslate2 package. The loader of CTranslate2 quantizes each variable whose name ends with "weight", with the absolute maximum of each row. A unit test compares the output byte by byte with the int8 file of ctranslate2.
- `stanza-pretrain/`: a word list and a `.npy` file replace `torch.load(..., weights_only=True)` of the stanza vectors, which is slow on 250 000 words. The key has the path, the size and the change time of the source file, so a copied file gets a new key. `meta.json` names the source, so the old copy of the same source goes away.
- `stanza_cache.install` replaces `Pretrain.load` of stanza. On any error, the original method runs.
