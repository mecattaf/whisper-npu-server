# whisper-npu-server

A high-performance speech transcription server leveraging Intel NPU acceleration. This fork extends the original project with enhanced organization, error handling, and user-specific model management.

## Hardware Requirements

Tested and optimized for the ASUS Zenbook DUO UX8406 with Intel® Core™ Ultra 9 185H processor. The server utilizes the integrated Intel NPU for efficient model inference.

## Features

The server provides real-time speech transcription using OpenVINO-optimized Whisper models. It automatically manages model storage in the user's home directory, making it easy to persist models across container rebuilds and system updates. The implementation focuses on simplicity and reliability, with comprehensive error handling and logging.

## Models

The server stores models in `~/.whisper/models/` in your home directory. This location is automatically created when the server starts. The server supports various Whisper models including:

whisper-tiny.en through whisper-large-v3 models are supported, with whisper-medium.en being the default choice. The complete model list includes both English-specific and multilingual variants:

For English-only use:
- whisper-tiny.en (fastest)
- whisper-base.en
- whisper-small.en
- whisper-medium.en (default, recommended)

For multilingual support:
- whisper-tiny
- whisper-base
- whisper-small
- whisper-medium
- whisper-large-v3

## Setup

The server uses a user-specific directory structure for model storage, making it particularly suitable for Fedora Silverblue's immutable nature:

```bash
# Create the whisper directory structure (server will also create this automatically)
mkdir -p ~/.whisper/models

# Download and extract pre-converted models
cd ~/.whisper/models
wget <models-url> -O models.tar.gz
tar xzf models.tar.gz
rm models.tar.gz
```

## Container Management

```bash
# Build the container
podman build -t whisper-server-image .

# Run in development mode
podman run -d \
    --name whisper-server \
    -v $PWD/server.py:/src/dictation/server.py:Z \
    -v $HOME/.whisper/models:/root/.whisper/models:Z \
    -p 8009:5000 \
    --security-opt seccomp=unconfined \
    --ipc=host \
    --group-add keep-groups \
    --device=/dev/dri \
    --device=/dev/accel/accel0 \
    whisper-server-image

# Simple transcription test
curl --data-binary @audio.wav -X POST http://localhost:8009/transcribe
```

## Systemd Integration

The server can be automatically started using systemd user services, making it readily available for desktop integration with tools like Sway:

```bash
# Generate and enable systemd service
podman generate systemd whisper-server > $HOME/.config/systemd/user/whisper-server.service
systemctl --user daemon-reload
systemctl --user enable whisper-server.service
```

## API Response Format

The server provides clean, straightforward JSON responses:

Success response:
```json
{
    "text": "transcribed text here"
}
```

Error response:
```json
{
    "error": "error description here"
}
```

## Acknowledgments

This project is based on the original work by [ellenhp](https://github.com/ellenhp) who created the initial implementation for the ThinkPad T14. Modified for the ASUS Zenbook DUO with reorganized file structure and enhanced error handling.
