# AdaptLLM Finance Tasks - ConvFinQA

This template trains models on **conversational financial question answering** using the [AdaptLLM/finance-tasks](https://huggingface.co/datasets/AdaptLLM/finance-tasks) dataset (ConvFinQA subset).

## Dataset

The dataset contains multi-turn financial reasoning questions based on:
- 10-K filings and earnings reports
- Financial tables and statements
- Cash flow analysis
- Revenue breakdowns

Each entry includes:
- **Financial context**: Tables, numbers, and narrative text
- **Conversation chain**: Sequential questions building on previous answers
- **Final question**: The question the model must answer

## Rewards

1. **numerical_accuracy**: Extracts numbers from the response and compares with expected answer (1% tolerance)
2. **exact_match**: Checks if the exact answer string appears in the response (fallback)

## Example Entry

**Input context:**
```
Cash flow from 2012-2014:
| Year | Operating | Investing |
|------|-----------|-----------|
| 2014 | 7,385     | -3,214    |
| 2013 | 6,823     | -2,987    |
| 2012 | 6,161     | -2,456    |

Q: What was the cash provided by operating activities in 2013?
A: 6823.0

Q: And in 2012?
A: 6161.0

Q: What was the difference between the years?
```

**Expected answer:** `662.0`

## Training

```bash
rnow init -t adaptllm-finance -n "my-finance-qa"
cd my-finance-qa
rnow run
```
