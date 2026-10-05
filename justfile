# https://github.com/casey/just
set dotenv-load

# Set shell for non-Windows OSs:
set shell := ["powershell", "-c"]

# Set shell for Windows OSs:
set windows-shell := ["powershell.exe", "-NoLogo", "-Command"]

image := "holos-tts"
container := "holos-tts"
# the folder next to this file keeps the downloaded models and the cache between containers
data := justfile_directory() / "data"
# the same variables as in .env; the file .env passes the other variables into the container
http_port := env("HTTP_PORT", "8000")
wyoming_port := env("WYOMING_PORT", "10200")
default_voice := env("DEFAULT_VOICE", "Speaker_43")
puid := env("PUID", "1000")
pgid := env("PGID", "1000")
url := "http://127.0.0.1:" + http_port

# Install the CPU dependencies and the development tools into .venv
install:
    uv sync --extra cpu

# Update the dependencies in uv.lock to the newest allowed versions and install them
upgrade:
    uv sync --extra cpu --upgrade

# Format the code, fix the lint findings that ruff can fix, and check the types with ty
lint:
    uv run ruff format .
    uv run ruff check --fix
    uvx --with pre-commit-uv pre-commit run ty --all-files

# Run all pre-commit hooks on all files
pre:
    uvx --with pre-commit-uv pre-commit run --all-files

# Update the hook versions in .pre-commit-config.yaml
pre-update:
    uvx --with pre-commit-uv pre-commit autoupdate

# Run all tests
test:
    uv run pytest

# Run all tests with the coverage report. CI fails below 85 %
cov:
    uv run pytest --cov=holos_tts --cov-report=term-missing --cov-fail-under=85

# Build the CPU image with wslc
build:
    wslc build -t {{image}}:cpu .

# Build the NVIDIA GPU image with wslc
build-cuda:
    wslc build -t {{image}}:cuda --target cuda .

# Start the CPU image in the background. The file .env sets the server variables when it exists
run: _data (_run "cpu" "")

# Start the GPU image in the background
run-cuda: _data (_run "cuda" "--gpus all")

# The container user cannot create the folder on the host, so it must exist before the start
_data:
    New-Item -ItemType Directory -Force '{{data}}' | Out-Null

# The ports inside the container stay 8000 and 10200, as in docker-compose.yml
_run tag gpus:
    $envFile = if (Test-Path .env) { @('--env-file', '.env') } else { @() }; \
    wslc run -d --name {{container}} {{gpus}} @envFile -u {{puid}}:{{pgid}} \
        -e HTTP_PORT=8000 -e WYOMING_PORT=10200 \
        -p {{http_port}}:8000 -p {{wyoming_port}}:10200 \
        -v '{{data}}:/data' \
        {{image}}:{{tag}}

# Stop and remove the container. The folder data with the models stays
stop:
    wslc rm -f {{container}}

# Rebuild the CPU image and start it again
restart: build stop run

# Follow the log of the running container
logs:
    wslc logs -f {{container}}

# A shell inside the running container
shell:
    wslc exec -it {{container}} sh

# Memory of each process in the container (the page cache is not counted)
rss:
    wslc exec {{container}} sh -c "grep -H VmRSS /proc/[0-9]*/status"

# Show which models are in memory and the memory of the model process
health:
    Invoke-RestMethod {{url}}/health | ConvertTo-Json

# Start to load the models in the background
warmup:
    Invoke-RestMethod -Method Post {{url}}/v1/warmup | ConvertTo-Json

# Speak TEXT with VOICE into the folder data. VOICE defaults to DEFAULT_VOICE. Example: just say "Привіт" "Гаська Шиян"
say TEXT VOICE=default_voice FILE="speech.mp3":
    $body = [Text.Encoding]::UTF8.GetBytes((@{ input = '{{TEXT}}'; voice = '{{VOICE}}' } | ConvertTo-Json)); \
    Invoke-WebRequest -Method Post {{url}}/v1/audio/speech \
        -ContentType 'application/json' -Body $body -OutFile '{{data}}/{{FILE}}'; \
    Get-Item '{{data}}/{{FILE}}' | Select-Object FullName, Length

# Show the voice names. The default voice is first
voices:
    [Console]::OutputEncoding = [Text.Encoding]::UTF8; \
    $raw = (Invoke-WebRequest {{url}}/v1/audio/voices -UseBasicParsing).RawContentStream.ToArray(); \
    ([Text.Encoding]::UTF8.GetString($raw) | ConvertFrom-Json).voices

# Delete the cache in the folder data. The server makes it again from the downloaded models
clean-cache:
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue '{{data}}/cache'

# Show available commands
help:
    @just --list
