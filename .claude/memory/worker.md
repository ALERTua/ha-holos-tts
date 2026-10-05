---
name: worker
description: Read this file before you change the worker process, the lock, the warm-up, the idle unload or a cancel path.
metadata:
  type: project
---

- One spawned worker process holds the verbalizer, the stress model and HolosTTS. The main process never imports torch, stanza, ONNX Runtime or CTranslate2. Keep these imports inside the functions of the worker modules.
- `Synthesizer._request` holds the lock for one exchange only, which is one chunk of text. A slow client then cannot stop the other requests.
- The warm-up sends one `load_part` request for each model. `asyncio.Lock` wakes the waiters in order, so a synthesis that waits goes between two loads. A text with digits needs all three models, so its request loads the missing models itself.
- The warm-up and `PRELOAD` skip the verbalizer when `VERBALIZER_UNLOAD_AFTER_SECONDS` is above 0. Else each Wyoming `synthesize-start` loads the verbalizer again.
- A cancelled caller still waits for its reply, because the next request would read that reply from the pipe. After a second cancel, `_send` keeps the exchange in `_abandoned`. The next exchange kills the worker and waits until the abandoned thread ends, and only then opens a new pipe. If the new pipe opens first, the old thread takes the reply of the next request.
- The idle unload drops the models and keeps the process. A new process needs `import torch` again, which takes 1.2 to 2 s. Check that the worker is still idle after you get the lock, because a request can end while the idle task waits.
- A worker target that a test module defines makes the spawned child import the whole test module. Put each fake worker into `tests/fake_worker.py`.
