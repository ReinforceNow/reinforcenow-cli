# SkyRL-SQL Training

RL training for text-to-SQL using execution-based rewards.

## How It Works

1. **Input**: Model receives database schema + natural language question
2. **Output**: Model generates SQL wrapped in `<solution>...</solution>`
3. **Reward**: Execute both generated and expected SQL, compare results
   - `1.0` = Results match
   - `0.0` = Results differ, error, or bad format

## Files

| File | Purpose |
|------|---------|
| `config.yml` | Training config |
| `train.jsonl` | Text-to-SQL tasks (schema + question + expected SQL) |
| `rewards.py` | Execution-based reward |
| `tools.py` | SQL tool for multi-turn exploration |
| `.env` | AWS credentials for S3 database access |

## Setup

1. Add to `.env`:
```
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
S3_BUCKET=your_bucket
```

2. Upload SQLite databases to S3:
```
s3://your-bucket/dbs/{db_id}.sqlite
```

3. Test:
```bash
uv pip install -r requirements.txt
uv run --active rnow test -n 3
```

## Supported Models

- `Qwen/Qwen3-4B-Instruct-2507`
- `Qwen/Qwen3-8B`
- `Qwen/Qwen3-30B-A3B`
- `Qwen/Qwen3-30B-A3B-Instruct-2507`
- `Qwen/Qwen3-32B`

## Data Sources

- [NovaSky-AI/SkyRL-SQL-653-data](https://huggingface.co/datasets/NovaSky-AI/SkyRL-SQL-653-data)
- [seeklhy/OmniSQL-datasets](https://huggingface.co/datasets/seeklhy/OmniSQL-datasets) (SQLite files)
