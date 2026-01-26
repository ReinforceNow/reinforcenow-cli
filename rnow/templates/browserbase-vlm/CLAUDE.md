# Browserbase VLM Agent Template

This template trains a Vision-Language Model (VLM) to control a web browser using screenshots, inspired by OpenAI's Computer Use Agent (CUA).

## Overview

Unlike text-based browser agents that use accessibility snapshots, this template:
- Uses **screenshots** as the primary feedback mechanism
- Provides **coordinate-based actions** (click at x,y) instead of element selectors
- Trains models to understand visual layouts and UI elements

## Setup

### 1. Get Browserbase Credentials

1. Sign up at [browserbase.com](https://browserbase.com)
2. Create a project in the dashboard
3. Copy your API key and project ID

### 2. Get OpenAI API Key (for LLM judge)

1. Go to [platform.openai.com/api-keys](https://platform.openai.com/api-keys)
2. Create a new API key

### 3. Configure Environment

Copy `example.env` to `.env` and fill in your credentials:

```bash
cp example.env .env
# Edit .env with your keys
```

### 4. Build the Docker Image

```bash
docker build -f Dockerfile.browserbase -t local/browserbase .
```

### 5. Test Locally

```bash
uv run rnow test -n 1 --verbose
```

## How It Works

### Tools

The agent has access to these browser control tools:

| Tool | Description |
|------|-------------|
| `click(x, y)` | Click at screen coordinates (0-1023, 0-767) |
| `type_text(text)` | Type text into the focused element |
| `press_key(key)` | Press a keyboard key (Enter, Tab, Escape, etc.) |
| `scroll(direction, amount)` | Scroll up or down |
| `navigate(url)` | Go directly to a URL |
| `screenshot()` | Get current page state without acting |

Every tool returns a screenshot of the resulting page state.

### Screen Coordinates

The browser viewport is 1024x768 pixels:
- X: 0 (left) to 1023 (right)
- Y: 0 (top) to 767 (bottom)

The model must learn to identify UI elements visually and click on their coordinates.

### Rewards

1. **used_browser** (precondition): The agent must use at least one browser tool. This prevents the model from answering questions from its training data without actually browsing.

2. **accuracy** (LLM judge): Evaluates whether the agent's final answer matches the expected answer using semantic comparison.

## Training Data Format

Each entry in `train.jsonl` should have:

```json
{
  "messages": [
    {"role": "system", "content": "You are a computer use agent..."},
    {"role": "user", "content": "Find the population of Tokyo"}
  ],
  "rewards": ["used_browser", "accuracy"],
  "tools": ["click", "type_text", "press_key", "scroll", "navigate", "screenshot"],
  "docker": "local/browserbase",
  "metadata": {"expected_answer": "13.96 million"}
}
```

## Customization

### Adding More Training Examples

Edit `train.jsonl` to add more question-answer pairs. Good examples:
- Factual questions that require web search
- Multi-step tasks (search, click result, find specific info)
- Form filling exercises
- Navigation tasks

### Changing the Model

Edit `config.yml` to use a different VLM:

```yaml
model:
  path: Qwen/Qwen3-VL-235B-A22B-Instruct  # Larger model
```

### Adjusting Difficulty

- Increase `max_turns` for more complex tasks
- Add multi-step tasks that require navigation
- Include tasks with distractor elements

## Technical Details

### VLM Image Format

Tools return images using a special format that the training environment recognizes:

```python
{
    "__vlm_image__": {
        "data": "base64-encoded-image-data",
        "format": "png"  # or "jpeg"
    },
    "text": "Description of what happened"
}
```

This triggers multimodal message creation, embedding the image alongside text in the conversation.

### Pixel Normalization

Pixel normalization is handled automatically by Tinker's image processor:

1. Screenshots are captured as PNG (lossless)
2. Converted to JPEG for efficient transmission
3. Qwen3-VL's image processor normalizes pixel values
4. Image patches are calculated based on dimensions

You don't need to manually normalize pixels - the pipeline handles it.

### Image Token Count

The Qwen3-VL model uses dynamic image tokens based on image size. A 1024x768 screenshot typically uses ~1000-1500 tokens. The `max_context_window` in config.yml should account for this.

## Troubleshooting

### "Failed to create session" Error

- Check your Browserbase API key and project ID
- Ensure your Browserbase plan has available sessions

### Screenshots Not Displaying

- The VLM receives screenshots as base64-encoded images
- Ensure your model supports vision (Qwen2-VL, GPT-4V, etc.)

### Agent Not Using Browser

- The `used_browser` precondition reward ensures the agent must browse
- If constantly 0, check that tools are being loaded correctly
