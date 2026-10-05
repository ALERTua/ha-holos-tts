---
name: home-assistant
description: Read this file before you test the server with Home Assistant.
metadata:
  type: project
---

- For a streamed text, the Wyoming integration sends `synthesize-start`, then the text in chunks, then `synthesize-stop`. The server starts the warm-up on `synthesize-start`.
- The openai_tts integration sends the voice "alloy" in a probe. The server answers with the default voice.
- The TTS cache of Home Assistant gives the old audio for the same text. Call `tts.clear_cache` before you repeat a test.
- `soundfile` reports a wrong length for a VBR MP3 without a Xing header, which is the MP3 of the Home Assistant TTS. To measure the length, decode the file with ffmpeg and count the samples.
- Test only on a test Home Assistant. Do not change a production Home Assistant.
