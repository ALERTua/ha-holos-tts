[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/banner-direct-single.svg)](https://stand-with-ukraine.pp.ua)
[![Made in Ukraine](https://img.shields.io/badge/made_in-Ukraine-ffd700.svg?labelColor=0057b7)](https://stand-with-ukraine.pp.ua)
[![Stand With Ukraine](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/StandWithUkraine.svg)](https://stand-with-ukraine.pp.ua)
[![Russian Warship Go Fuck Yourself](https://raw.githubusercontent.com/vshymanskyy/StandWithUkraine/main/badges/RussianWarship.svg)](https://stand-with-ukraine.pp.ua)

# ha-holos-tts

A Ukrainian text-to-speech server for Home Assistant. It runs the [HolosTTS](https://huggingface.co/patriotyk/HolosTTS) model of Serhiy Stetskovych ([patriotyk](https://github.com/patriotyk)) in one container. See "Credits" for all the work that this server uses.

The server gives two interfaces:

- The Wyoming protocol. The Wyoming integration of Home Assistant uses it, so no custom integration is necessary.
- An OpenAI-compatible speech API. The [openai_tts](https://github.com/sfortis/openai_tts) custom integration of Home Assistant and Music Assistant use it.

Before speech, the server prepares the text in three steps:

1. A verbalizer model writes numbers, dates, units and acronyms as words. For example, "о 15:30, -3°C" becomes "о пʼятнадцятій тридцять, мінус три градуси Цельсія". Only the sentences with digits, symbols, Latin letters or acronyms go to the verbalizer.
2. A stress model puts the stress marks.
3. A phonemizer converts the text to the phonemes of the model.

The model runs in ONNX Runtime: an int8 model on the CPU, or a float32 model on an NVIDIA GPU.

## Quick start

1. Copy [docker-compose.yml](docker-compose.yml) and [.env.example](.env.example) to a folder.
2. Rename `.env.example` to `.env` and change the values that you need.
3. Create the data folder: `mkdir data`. If Docker creates it, the folder belongs to root, and the container user cannot write the models into it.
4. Run `docker compose up -d`.

The first start downloads the models into `./data` and builds a cache from them. See "Data folder" for the sizes. The log shows "Models are loaded in advance" when the models are in memory. The server accepts requests before that line. Such a request waits for the model that loads at that moment, and then it loads the other models that its text needs.

For an NVIDIA GPU, use the image tag `latest-cuda` and uncomment the `deploy` block in `docker-compose.yml`.

Image tags on `ghcr.io/alertua/ha-holos-tts`:

| Tag | What it is |
|---|---|
| `latest`, `latest-cuda` | The newest release. |
| `1`, `1.2`, `1.2.3`, and the same with `-cuda` | A release line or one exact release. `1` gets each new `1.x.y` release. |
| `edge`, `edge-cuda` | The newest commit of `main`. It can break at any time. |

## Data folder

The container keeps all its files in `/data`. The file docker-compose.yml mounts `./data` there.

| Folder | What it holds |
|---|---|
| `/data/huggingface` | The models that the server downloads from Hugging Face. The verbalizer is 1.9 GB of them. |
| `/data/stanza` | The stress model that the server downloads. |
| `/data/cache` | Fast copies that the server makes from the downloaded models on the first start. |
| `/data/voices` | Your extra voices. See "Voices". |

The downloaded models take about 2.5 GB in `huggingface` and `stanza` together. The cache takes about 880 MB in the CPU image. The total is about 3.4 GB.

<details>
<summary>What the cache holds</summary>

The cache holds three parts:

- The optimized graph of the speech model, about 306 MB. Only the CPU image makes it.
- An int8 copy of the verbalizer, about 470 MB.
- A fast copy of the vectors of the stress model, about 100 MB.

You can delete the folder `/data/cache`. The server makes the copies again when it loads the models.

On the first start, the first request after the download takes 10 to 12 s, because the server builds the cache. These numbers come from a 32-core CPU. Later starts read the cache.

</details>

## Home Assistant

### Wyoming integration

1. In Home Assistant, go to Settings > Devices & services > Add integration > Wyoming Protocol.
2. Enter the host of the server and the port `10200`.
3. Home Assistant adds the entity `tts.holos_tts`.

```yaml
action: tts.speak
target:
  entity_id: tts.holos_tts
data:
  media_player_entity_id: media_player.kitchen
  message: Пральна машина закінчила роботу о 15:30.
  options:
    voice: Гаська Шиян
```

The Wyoming protocol has no speed field. Set the speed with `DEFAULT_SPEED`.

### OpenAI TTS integration

1. Install the [openai_tts](https://github.com/sfortis/openai_tts) custom integration.
2. Set the URL to `http://<host>:8000/v1/audio/speech`. Leave the API key empty.
3. Set any model name, a voice from the voice list, and the speed.

The integration sends the voice name "alloy" in some requests. The server does not know this name, so it uses `DEFAULT_VOICE`.

### Music Assistant

Add the OpenAI text-to-speech provider. Set the API endpoint to `http://<host>:8000/v1`, with the `/v1` path.

To load the models before a voice command, see `POST /v1/warmup` in the API section.

## Voices

The model has 27 voices: "Гаська Шиян" and the numbered voices "Speaker_0" to "Speaker_84". `GET /v1/audio/voices` gives the list. The default voice is `Speaker_67`.

The model card does not tell whose voices these are, and it gives no license for the voices. For the voices of his older model, [styletts2-ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian/discussions/11), the author wrote that they are the voices of well-known voice actors and that they should not be used. For public content, such as a video or a podcast, use your own cloned voice (see below).

You can add your own voices:

1. Open the [HolosTTS demo](https://huggingface.co/spaces/patriotyk/HolosTTS).
2. Upload a recording of the voice into "Аудіо промт (клонування голосу)".
3. Download the `.pt` file of the style.
4. Put the file into `./data/voices`. The file name is the voice name, for example `./data/voices/Мій голос.pt`.
5. Restart the container.

A `.npy` file with 256 numbers also works.

## Stress marks

The stress model can choose a wrong stress. To set a stress, put `+` after the stressed vowel, for example `Му+дрого`. The server also accepts `` ` `` before the vowel, for example ``Зустр`іч``, as the older styletts2 API did.

## Long sentences

The server splits a sentence longer than 150 characters at commas, then at spaces. The model cannot speak a longer piece in one pass.

## Web UI

Set `WEB_UI=1` to add a web page to the server. Open `http://<host>:8000/web/` in a browser, type a text, choose a voice and the speed, and press "Speak". The page uses the same speech pipeline as the API, so the verbalizer and the stress marks work the same way. Press "Verbalize" to replace the text with its verbalized form (numbers, dates, units and acronyms written as words), so you can check and edit it before "Speak".

The page has no password and uses the port of the API. Turn it on only in a network that you trust. The page adds about 80 MB to the RAM of the main process. When `WEB_UI` is off, the server does not load the web page library at all.

## Configuration

Set a variable in the file `.env`. If you run the image without Docker Compose, use `-e NAME=value` instead. When the server cannot use a value, it stops at start, and the log names the variable.

A switch takes `1` or `0`. It also takes `true` or `false`, `yes` or `no`, and `on` or `off`. An empty value of a switch means `0`.

The file [.env.example](.env.example) explains each setting in plain words.

<details>
<summary>All settings in one table</summary>

| Variable | Default | What it does | When to change it |
|---|---|---|---|
| *Ports* | | | |
| `HTTP_PORT` | `8000` | Port of the OpenAI-compatible API. `0` turns the API off when you run the image without docker-compose.yml. | Change it when another program uses port 8000 on the host. |
| `WYOMING_PORT` | `10200` | Port of the Wyoming server for Home Assistant. `0` turns the server off when you run the image without docker-compose.yml. | Change it when another program uses port 10200 on the host. |
| `HTTP_HOST` | `0.0.0.0` | Network address that the API listens on. `0.0.0.0` means all addresses. | In a container, keep it. Without Docker, set `127.0.0.1` to accept only requests from the same computer. |
| `WYOMING_HOST` | `0.0.0.0` | Network address that the Wyoming server listens on. | Same as `HTTP_HOST`. |
| `WEB_UI` | `0` | `1` adds a web page at `/web/` on the port of the API. See "Web UI". | Set `1` to try the voices and the speed in a browser. The page has no password. |
| *Voice and speed* | | | |
| `DEFAULT_VOICE` | `Speaker_67` | Voice for requests without a voice, and for unknown voice names. It must be a name from `GET /v1/audio/voices`. | Change it to use another voice in the Wyoming integration, or with clients that send a name such as "alloy". |
| `DEFAULT_SPEED` | `1.0` | Speed of the Wyoming speech, and of OpenAI requests without a speed. The range is `0.5` to `2.0`. A value outside the range stops the server. | Change it when the speech is too fast or too slow. The Wyoming protocol has no speed field. |
| `VERBALIZE` | `1` | `1` writes numbers, dates, units and acronyms as words before the speech. `0` turns the verbalizer off. `AUTO_USE_VERBALIZER` is the old name. The server reads it when `VERBALIZE` is not set. | Set `0` only when your texts have no digits or symbols. Then the server does not download the verbalizer (1.9 GB) and saves about 0.5 GB of memory. |
| *Models and memory* | | | |
| `DEVICE` | `cpu` | Where the speech model runs: `cpu` or `cuda`. The CPU image sets `cpu`. The CUDA image sets `cuda`. | Leave it. Choose the image tag that fits your hardware. The CPU image cannot use `cuda`. |
| `VERBALIZER_DEVICE` | the value of `DEVICE` | Where the verbalizer runs: `cpu` or `cuda`. | In the CUDA image, set `cpu` to keep the verbalizer off the GPU, for example when the GPU has little free memory. |
| `PRELOAD` | `1` | `1`: at start, the server loads the models one at a time in the background. A request during this load does not wait for the end of the load. It loads the models that its text needs. `0`: the models load at the first request. The model process and the voice list still start with the server. | Keep `1`, so the first request is fast and the downloads of the first start finish early. Set `0` to keep the models out of memory until the first request. |
| `UNLOAD_AFTER_SECONDS` | `0` | Seconds without requests, after which the server unloads the models. `0` never unloads. The model process stays with its libraries, about 0.44 GB. The next request takes about 3.5 s instead of about 1.3 s. | Set it, for example to `600`, when the host needs the memory for other work. See "Memory and speed". |
| `VERBALIZER_UNLOAD_AFTER_SECONDS` | `0` | Seconds without use, after which the server unloads only the verbalizer. `0` keeps it loaded once it is loaded. When the value is above 0, `PRELOAD` and the warm-up skip the verbalizer. | Set it to give back about 0.5 GB when most of your texts have no digits or symbols. Use a value below `UNLOAD_AFTER_SECONDS`. |
| *CPU* | | | |
| `THREADS` | `0` | CPU threads for each model. `0` uses the CPU limit of the container, rounded up. For example, `--cpus 4` gives 4 threads. Without a limit, it uses all cores. The server reads the limit on hosts with cgroup v2. | Set it when the host runs other heavy work, or when you want the server to use fewer cores. |
| *Paths* | | | |
| `DATA_DIR` | `/data` | Folder in the container for your voices (`voices`) and the cache (`cache`). The Dockerfile sets it. | Do not change it. Mount your host folder to `/data`. |
| `HF_HOME` | `/data/huggingface` | Folder for the models that the server downloads. The download library reads it. The Dockerfile sets it. | Do not change it. |
| `STANZA_RESOURCES_DIR` | `/data/stanza` | Folder for the stress model. The stress library reads it. The Dockerfile sets it. | Do not change it. |
| `HF_TOKEN` | empty | Hugging Face access token. The download library reads it. | Set it when Hugging Face limits the downloads of the first start. See "Hugging Face token". |
| `HF_TOKEN_PATH` | empty | Path in the container to a file with the Hugging Face token, for example a Docker secret. | Use it instead of `HF_TOKEN` to keep the token out of the variables. See "Hugging Face token". |
| `PUID`, `PGID` | `1000` | User and group of the container. Only docker-compose.yml reads them. | Change them when the folder `./data` belongs to another user. Make sure that this user can write into it. |
| *Packages* | | | |
| `UV_OVERRIDE` | empty | Path in the container to a file with package versions that replace the versions of the image. See "Overriding packages". | Set it when your GPU needs another build of a package, for example `/data/user_requirements.txt`. |
| `UV_EXTRA_INDEX_URL` | empty | An extra package index for the versions in the `UV_OVERRIDE` file. The installer uv reads it. | Set it when a section of the override file names an index. |
| `UV_CACHE_DIR` | `/data/uv_cache` | Folder for the downloads of uv. The Dockerfile sets it. | Do not change it. You can delete the folder when the container does not run. |
| *Logs* | | | |
| `LOG_LEVEL` | `INFO` | Amount of log text: `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL`. `INFO` logs the text of each request. `DEBUG` also logs the verbalized text and the phonemes. | Set `DEBUG` to find out why a word sounds wrong. Set `WARNING` to keep the request texts out of the log. |

</details>

With docker-compose.yml, `HTTP_PORT` and `WYOMING_PORT` in `.env` choose the ports on the host. Inside the container, the servers keep the ports 8000 and 10200. To turn a server off, remove its line from `ports` in docker-compose.yml.

## Hugging Face token

The server downloads public models, so it needs no token. Set a token only when Hugging Face limits the downloads of the first start. A token with the "Read" role is enough. Create it at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens).

The simple way is `HF_TOKEN=<token>` in `.env`. But Docker shows the value of each variable to anyone who can run `docker inspect`. To keep the token out of the variables, give it as a Docker secret.

<details>
<summary>Give the token as a Docker secret</summary>

1. Put only the token into a file next to docker-compose.yml, for example `hf_token.txt`. The container user (`PUID`) must be able to read it.
2. In docker-compose.yml, uncomment the two `secrets` blocks and the line `HF_TOKEN_PATH: /run/secrets/hf_token`.
3. Remove `HF_TOKEN` from `.env`, and run `docker compose up -d`.

The download library reads the token from the file that `HF_TOKEN_PATH` names.

</details>

## Overriding packages

The image has fixed versions of its Python packages. Some GPUs need other builds. For example, a GTX 10xx card needs the CUDA 12 build of ONNX Runtime. You can replace the versions without a new image.

<details>
<summary>How to override packages</summary>

1. Copy [user_requirements.txt.example](user_requirements.txt.example) to the data folder as `user_requirements.txt`.
2. Uncomment the section for your GPU in that file, or write your own versions.
3. Set `UV_OVERRIDE=/data/user_requirements.txt` in `.env`. The path is the path inside the container.
4. If the section names an index, set `UV_EXTRA_INDEX_URL` to that index.
5. Restart the container: `docker compose up -d`.

Without `UV_OVERRIDE`, the container starts the server at once with the packages of the image.

With `UV_OVERRIDE`, the container installs all packages of the image into `/data/venv` before the server starts. Each version in your file replaces the version of the image. The installer [uv](https://docs.astral.sh/uv/pip/compatibility/) does this work. It also reads `UV_EXTRA_INDEX_URL`.

- The first start installs about 4.3 GB for the CUDA image. It takes several minutes, and the health check can show "unhealthy" until the server starts. Measured: 7.5 minutes.
- A later start finds the packages installed and installs nothing. Measured: the check takes 1 s.
- After you change the file or update the image, the next start installs only the changed packages.
- If the file does not exist, the log shows a warning, and the server starts with the packages of the image.
- To remove a package of the image, write it with a condition that is never true, for example `nvidia-cudnn-cu13 ; sys_platform == "never"`.

To go back to the packages of the image, remove `UV_OVERRIDE` and restart the container. Then delete the folders `/data/venv` and `/data/uv_cache` to free the disk.

</details>

## Memory and speed

The models run in a separate process. After `UNLOAD_AFTER_SECONDS` without requests, the process unloads the models and gives their memory back to the system. The process stays alive and keeps its libraries, about 0.44 GB. The next request loads the models again.

Some x86 CPUs have no VNNI instructions, for example the Intel Core i5-10400. On such a CPU, the fast int8 matrix multiplication of ONNX Runtime overflows, and the speech becomes unintelligible. The server reads the CPU flags and turns on the exact int8 multiplication (`session.x64quantprecision`) on such a CPU. On the i5-10400, a phrase takes about 14 % more time with it. At the model load, the log tells when the server uses it.

<details>
<summary>Measured memory and speed</summary>

Measured in containers on an Intel Core i9-13900HX and an RTX 4090 Laptop GPU. The text is "Пральна машина закінчила роботу о 15:30. Температура на вулиці -3°C, вологість 87%.", about 9 seconds of speech with the voice `Speaker_67`.

| | CPU image | CUDA image |
|---|---|---|
| Image size | 1.4 GB | 4.9 GB |
| RAM of the model process with all models loaded | about 1.5 GB | |
| GPU memory | | |
| Time to make the speech with all models loaded | 1.3 s | |
| RAM after `UNLOAD_AFTER_SECONDS` | 0.44 GB | |
| First request after the unload | 3.5 s | |
| First request after the start of the container, with no warm-up | 5.7 s | |
| Request 6 s after the start of a warm-up | 1.4 s | |

RAM of each part in the CPU image, from separate measurements:

| Part | RAM |
|---|---|
| Libraries, voices and the model process, always loaded | 0.44 GB |
| Verbalizer | about 0.5 GB |
| HolosTTS int8 model | about 0.3 GB |
| Stress model (stanza) | about 0.2 GB |

`docker stats` of a new CPU container with all models loaded and no requests. The Intel Core i9-13900HX (`wslc stats`) and the Intel Core i5-10400 (`docker stats`) gave the same values within 1 %:

| `VERBALIZE` | `WEB_UI` | MEM USAGE | CPU % without requests |
|---|---|---|---|
| `1` | `0` | 1.38 GiB | below 1.5 % |
| `0` | `0` | 0.87 GiB | below 1.5 % |
| `1` | `1` | 1.45 GiB | 5 to 8 % |
| `0` | `1` | 0.93 GiB | about 5 % |

The model process grows while the server works. On the i5-10400, after two hours of work with Home Assistant, the model process used 2.08 GB instead of 1.52 GB, and `docker stats` showed 2.01 GiB.

</details>

### Warm-up

A warm-up loads the models in the background, one at a time. A request during a warm-up does not wait for the end of the warm-up. It waits for the model that loads at that moment, and then it loads the other models that its text needs. A text with digits needs all three models, so such a request right after the start of the container takes 6 to 7 s. Three things start a warm-up:

- The start of the server, when `PRELOAD=1`.
- The Wyoming event `synthesize-start`, which Home Assistant sends at the start of a streamed text.
- A call of `POST /v1/warmup`.

After an unload, a warm-up loads the models again before the next request. A call of `POST /v1/warmup` while a warm-up runs does nothing new.

<details>
<summary>The warm-up and the verbalizer</summary>

When `VERBALIZER_UNLOAD_AFTER_SECONDS` is above 0, the warm-up and `PRELOAD` skip the verbalizer. The verbalizer then loads only for a text with digits, symbols, Latin letters or acronyms. This request takes about 1 s more, and the verbalizer frees about 0.5 GB again after its time.

With the default `0`, the verbalizer stays loaded once it is loaded. With `PRELOAD=1`, it is loaded at start. With `PRELOAD=0`, it loads at the first text that needs it.

</details>

## API

### `POST /v1/audio/speech`

| Field | Description |
|---|---|
| `input` | The text. |
| `voice` | A name from `GET /v1/audio/voices`. An unknown name uses the default voice. |
| `speed` | `1.0` is the normal speed. The server accepts `0.25` to `4.0`. It raises a value below `0.5` to `0.5`, and it lowers a value above `2.0` to `2.0`. |
| `response_format` | `mp3` (default), `wav`, `flac`, `opus` or `pcm`. `pcm` is raw 16-bit mono audio at 24 kHz. |
| `model` | The server accepts any name. |

```bash
curl http://127.0.0.1:8000/v1/audio/speech -H "Content-Type: application/json" -d '{"input": "Привіт! Зараз 7 година ранку.", "voice": "Гаська Шиян"}' -o speech.mp3
```

### `GET /v1/audio/voices`

Returns `{"voices": ["Speaker_67", "Гаська Шиян", "Speaker_0", ...]}`. The default voice is first.

### `POST /v1/warmup`

Starts a warm-up in the background and returns the status 202 at once. The body is `{"status": "started"}`. When a warm-up already runs, the body is `{"status": "running"}`.

Call it shortly before the first request, so the models are in memory. This matters most when `UNLOAD_AFTER_SECONDS` is above 0. Measured: a request 6 s after the start of a warm-up takes 1.4 s. A request without a warm-up takes 3.4 to 5.7 s.

<details>
<summary>An example for Home Assistant</summary>

First, add a REST command to `configuration.yaml`:

```yaml
rest_command:
  holos_tts_warmup:
    url: "http://<host>:8000/v1/warmup"
    method: post
```

Then call it when a voice satellite starts to listen. Replace the entity with your satellite:

```yaml
automation:
  - alias: Warm up holos-tts
    triggers:
      - trigger: state
        entity_id: assist_satellite.kitchen
        to: listening
    actions:
      - action: rest_command.holos_tts_warmup
```

</details>

### `GET /health`

Returns `{"status": "ok", ...}` and the state of the model process under the key `worker`:

- `running`: the other keys tell which models are in memory (`stress`, `engine`, `verbalizer`) and how much RAM the process uses (`rss_mb`).
- `busy`: the process makes speech or loads a model now.
- `stopped`: no model process runs.

## Limits

- The verbalizer is a neural model. It can choose a wrong grammatical case for a number.
- The models of patriotyk use the Ukrainian language only.

## Development

```bash
uv sync --extra cpu
uv run pytest
uv run ruff check .
```

The [justfile](justfile) has recipes for these commands and for a local container in [wslc](https://github.com/MicrosoftDocs/WSL/blob/main/WSL/wsl-container.md). Run `just help` to see them. For example, `just build` builds the CPU image, and `just run` starts it with the file `.env`. The container keeps the models and the cache in the folder `data` next to the justfile. `just say "Привіт"` writes the speech into `data/speech.mp3` with the voice of `DEFAULT_VOICE`.

## Credits

This server only connects the work of other people to Home Assistant. Many thanks to all of them.

Most of all, thank you to **Serhiy Stetskovych** ([patriotyk](https://github.com/patriotyk), [Hugging Face](https://huggingface.co/patriotyk)). He trained and published HolosTTS, the voices and the demo, and the earlier [StyleTTS2 Ukrainian](https://huggingface.co/spaces/patriotyk/styletts2-ukrainian) models. He also keeps the forks of the stress and IPA tools that the models need.

| Part | Project | Authors | License |
|---|---|---|---|
| Speech model | [HolosTTS](https://huggingface.co/patriotyk/HolosTTS), [demo](https://huggingface.co/spaces/patriotyk/HolosTTS), [code](https://github.com/patriotyk/HolosTTS) | Serhiy Stetskovych | MIT (model) |
| Numbers as words | [m2m100-ukr-verbalization](https://huggingface.co/skypro1111/m2m100-ukr-verbalization) and its [CTranslate2 build](https://huggingface.co/skypro1111/m2m100-ukr-verbalization-ct2) | Serhii Kravchenko ([skypro1111](https://huggingface.co/skypro1111)) | MIT |
| Base of the verbalizer | [M2M100 418M](https://huggingface.co/facebook/m2m100_418M) | Meta AI | MIT |
| Training text of the verbalizer | [UberText 2.0](https://aclanthology.org/2023.unlp-1.1/) news, [verbalized](https://huggingface.co/datasets/skypro1111/ubertext-2-news-verbalized) | Dmytro Chaplynskyi; Serhii Kravchenko | CC BY 4.0 (dataset) |
| Stress marks | [ukrainian-word-stress](https://github.com/lang-uk/ukrainian-word-stress) ([fork](https://github.com/patriotyk/ukrainian-word-stress)) | Oleksiy Syvokon, lang-uk; Serhiy Stetskovych | MIT |
| Stress dictionary | [ukrainian-word-stress-dictionary](https://github.com/lang-uk/ukrainian-word-stress-dictionary), from [«Словники України»](https://lcorp.ulif.org.ua/dictua/) | lang-uk; ULIF of the NAS of Ukraine | not given |
| Stress of unknown words | [ukrainian-accentor](https://github.com/egorsmkv/ukrainian-accentor) | Yehor Smoliakov, Bohdan Mykhailenko | MIT |
| Phonemes | [ipa-uk](https://github.com/lang-uk/ipa-uk) ([fork](https://github.com/patriotyk/ipa-uk)), after the Wiktionary module [uk-pronunciation](https://en.wiktionary.org/wiki/Module:uk-pronunciation) | Dmitry Chaplinsky, lang-uk; Serhiy Stetskovych | MIT |
| Parts of speech for the stress | [stanza](https://github.com/stanfordnlp/stanza) and its [Ukrainian models](https://huggingface.co/stanfordnlp/stanza-uk), trained on [UD Ukrainian-IU](https://github.com/UniversalDependencies/UD_Ukrainian-IU) | Stanford NLP Group; Institute for Ukrainian | Apache-2.0; treebank CC BY-NC-SA 4.0 |
| Runtimes | [ONNX Runtime](https://github.com/microsoft/onnxruntime), [CTranslate2](https://github.com/OpenNMT/CTranslate2) | Microsoft; OpenNMT | MIT |
| Home Assistant protocol | [Wyoming](https://github.com/OHF-Voice/wyoming) | Michael Hansen, Open Home Foundation | MIT |

## Licenses

This project uses the GPL-3.0 license. The table in "Credits" gives the license of each part that the server downloads or installs. Two parts need care:

- The voices of the model have no stated license. See "Voices".
- The Ukrainian stanza models learned from the UD Ukrainian-IU treebank, which allows only non-commercial use (CC BY-NC-SA 4.0). Ask its authors before you use the server for a commercial product.
