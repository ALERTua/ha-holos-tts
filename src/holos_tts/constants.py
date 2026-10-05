"""Facts about the model that the main process needs without loading the model."""

SAMPLE_RATE = 24000
# A chunk takes the next sentence while it is not longer than this, the same as the HolosTTS demo
CHUNK_CHARS = 100
LANGUAGE = "uk"
PROGRAM_NAME = "holos-tts"
MODEL_URL = "https://huggingface.co/patriotyk/HolosTTS"
# speed range that the models speak well with; 1.0 is the normal speed
MIN_SPEED = 0.5
MAX_SPEED = 2.0
# the models of the worker, in the order of a warm-up: the stress model takes the longest
MODEL_PARTS = ("stress", "engine", "verbalizer")
