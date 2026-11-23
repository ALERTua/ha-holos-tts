---
name: loudness
description: Read this file before you change the level of the audio, `audio.loudness`, `audio.limit`, `audio.normalize`, `TARGET_LOUDNESS` or a path that gives audio to gradio.
metadata:
  type: project
---

- The decoder of HolosTTS gives about −25 LUFS with a true peak near −7 dBFS. The packages of the author do not change the level. `Synthesizer.synthesize` brings each chunk to `TARGET_LOUDNESS` with `audio.normalize`, so all three paths give the same audio.
- The old StyleTTS2 server was loud only because gradio converts float audio to 16-bit by a division by its peak. Give gradio int16 audio, as `web_ui.speak` does, or gradio changes the level again.
- `audio.loudness` must agree with `ffmpeg -af ebur128` within 0.1 LU. To measure a file, run `ffmpeg -hide_banner -nostats -i <file> -af ebur128=peak=true -f null -` and read `I` and `Peak` in the summary.
- `audio.limit` computes its gain in float64. A running sum of float32 values drifted past the ceiling.
- The limiter ceiling is −2 dBFS for each sample. A ceiling of −1.5 dBFS let the true peak reach −0.6 dBFS after the limiter.
- `TARGET_LOUDNESS` stops at −14 LUFS. The peaks of the model are about 19 dB above its loudness, so the limiter works hard above that. On 14 model files in MP3, −12 LUFS gave a true peak of −0.2 dBFS and reached only −14.0 to −12.2 LUFS, and −10 LUFS gave +0.6 dBFS. Before you raise the limit or the ceiling, measure the true peak of the MP3 again.
- `audio.MAX_GAIN_DB` caps the gain at 24 dB, so a test signal must be louder than about −40 LUFS to reach −16 LUFS.
- `tests.conftest.make_settings` sets `target_loudness=0`, because the fake worker returns marker samples, not speech. A test of the level uses the marker `SPEECH` of `tests/fake_worker.py`.
