# Food Extract Template

Train a VLM to extract food and drink items from images using [FoodExtract-1k](https://huggingface.co/datasets/mrdbourke/FoodExtract-1k-Vision).

## Setup

```bash
pip install datasets pillow
python setup.py --samples 100
```

First run downloads ~280MB (cached after). Generates `train.jsonl` + `images/`.

## Train

```bash
rnow run
```

## Files

- `setup.py` - Download dataset from HuggingFace
- `rewards.py` - Jaccard similarity on food/drink lists
- `config.yml` - Training config (Qwen3-VL-8B)
