# Posttrain Project

This is a ReinforceNow **posttrain** (continued pretraining) project.

## What is Posttrain?

Posttrain (also called midtrain or continued pretraining) trains the model on **all tokens** in raw text, not just assistant responses like SFT. This is useful for:

- Domain adaptation (e.g., training on medical, legal, or financial text)
- Teaching the model new knowledge or terminology
- Improving performance on domain-specific tasks

## Data Format

Each line in `train.jsonl` must have a `text` field:

```json
{"text": "Your raw text content here. The model learns to predict every token."}
{"text": "Another text chunk. Can be paragraphs, documents, or any continuous text."}
```

## Key Differences from SFT

| Aspect | SFT | Posttrain |
|--------|-----|-----------|
| Input format | `{"messages": [...]}` | `{"text": "..."}` |
| Loss target | Assistant tokens only | All tokens |
| Use case | Instruction tuning | Domain adaptation |

## Best Practices

1. **Use base models** - Instruct models have chat formatting baked in. Use base models like `Qwen/Qwen3-8B-Base` for better results.

2. **Clean your data** - Remove duplicates, boilerplate, and low-quality text.

3. **Chunk appropriately** - Each text entry should be meaningful. Too short = wasted compute, too long = may exceed context.

4. **Consider tokenization** - The model learns token-by-token. Technical content with rare tokens may need more data.

## Commands

```bash
# Test locally (validates data format)
uv run rnow test -n 1 --verbose

# Start training
uv run rnow run
```

## Files

- `config.yml` - Training configuration
- `train.jsonl` - Training data (one `{"text": "..."}` per line)
