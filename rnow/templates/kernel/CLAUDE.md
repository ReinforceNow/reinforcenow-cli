# Kernel VLM Browser Agent

Train a VLM to browse websites using screenshots and coordinate-based actions via [Kernel](https://kernel.sh) cloud browsers.

## Setup

```bash
cp example.env .env
# Add KERNEL_API_KEY and OPENAI_API_KEY
```

## Usage

```bash
source .env
source setup_pool.sh       # Create browser pool (100 browsers)
rnow test -n 3             # Test
rnow run                   # Train
./cleanup_pool.sh          # Delete pool when done
```

## Browser Tools

Coordinates use 0-1000 scale: `[0,0]` = top-left, `[1000,1000]` = bottom-right.

| Tool | Description |
|------|-------------|
| `click(coordinate)` | Click at `[x, y]` |
| `type_text(text)` | Type text |
| `press_key(key)` | Press key (Enter, Tab, etc.) |
| `scroll(coordinate, direction)` | Scroll up/down |
| `navigate(url)` | Go to URL |
| `screenshot()` | Get screenshot |

## How It Works

1. `setup_pool.sh` creates a pool of browsers on Kernel
2. Each rollout: Docker acquires browser from pool → tools execute → browser released
3. Idle browsers auto-destroy after 5min to save costs
4. `cleanup_pool.sh` deletes the pool

## Costs

Idle browsers: ~$2.88/month each. See [Kernel Pricing](https://www.kernel.sh/docs/info/pricing).
