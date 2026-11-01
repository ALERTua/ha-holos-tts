"""Facts about the model that the main process needs without loading the model."""

SAMPLE_RATE = 24000
# A chunk takes the next sentence while it is not longer than this, the same as the HolosTTS demo
CHUNK_CHARS = 100
LANGUAGE = "uk"
PROGRAM_NAME = "holos-tts"
# the name of the installed package, which differs from the Wyoming program name
DISTRIBUTION_NAME = "ha-holos-tts"
MODEL_URL = "https://huggingface.co/patriotyk/HolosTTS"
# the speed range of the HolosTTS demo; 1.0 is the normal speed, and the speech at 2.0 is unintelligible
MIN_SPEED = 0.7
MAX_SPEED = 1.3
# the models of the worker, in the order of a warm-up: the stress model takes the longest
MODEL_PARTS = ("stress", "engine", "verbalizer")
