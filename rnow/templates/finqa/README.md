# FinQA - Financial Numerical Reasoning

Train LLMs to answer numerical questions over financial tables and text.

Based on the [FinQA dataset](https://github.com/czyssrs/FinQA) (EMNLP 2021).

## Task

Given financial context (tables + text from earnings reports), answer a numerical question requiring:
- Identifying relevant numbers
- Multi-step arithmetic reasoning
- Returning a precise numerical answer

## Dataset

- **6,251 entries** from the full FinQA training set
- Real financial data from SEC filings
- Ground truth answers with calculation programs

## Reward

Single reward: **execution accuracy** (same as FinQA paper)
- 1.0 if answer matches ground truth (within 1% tolerance)
- 0.0 otherwise

## Example

**Input:**
```
Context:
The following table shows revenue by segment:

| Segment | 2019 | 2018 |
|---------|------|------|
| Cloud | $4,500 | $3,800 |
| Enterprise | $2,100 | $2,300 |

Question: What was the percent change in Cloud revenue?
```

**Expected Output:**
```
Cloud revenue: 2019 = $4,500, 2018 = $3,800
Change = (4500 - 3800) / 3800 = 0.1842 = 18.42%

**Answer: 18.42**
```

## Usage

```bash
rnow init -t finqa -n "my-finqa"
rnow test --smoke-test
rnow run
```

## References

- [FinQA Paper](https://arxiv.org/abs/2109.00122)
- [Dataset](https://github.com/czyssrs/FinQA)
