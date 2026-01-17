# ConvFinQA - Conversational Financial Question Answering

Train LLMs to answer conversational numerical questions over financial tables and text.

Based on the [ConvFinQA dataset](https://github.com/czyssrs/ConvFinQA) (EMNLP 2022).

## Task

Given financial context (tables + text from earnings reports) and a conversation history, answer numerical questions requiring:
- Understanding conversational context and coreferences
- Identifying relevant numbers from tables and text
- Multi-step arithmetic reasoning
- Building on previous answers in the conversation

## Dataset

- **3,037 entries** from the ConvFinQA training set
- Real financial data from SEC 10-K filings
- Conversational questions that build on previous Q&A pairs
- Ground truth answers with step-by-step calculation programs

## Reward

Uses **GPT-4.1 Nano** to evaluate numerical correctness:
- Compares model's answer against ground truth
- Handles different formats (e.g., "$6,823" vs "6823.0")
- Returns 1.0 if correct, 0.0 otherwise

**Requires OpenAI API key.** Copy `.env.example` to `.env` and add your key:
```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

## Example

**Input:**
```
Context:
The following table shows operating expenses:

| Category | 2019 | 2018 |
|----------|------|------|
| R&D | $450M | $380M |
| Sales | $320M | $290M |

Conversation:
Q: What was the R&D expense in 2019?
A: $450M

Current Question: What is the percent increase from 2018?
```

**Expected Output:**
```
R&D expense: 2019 = $450M, 2018 = $380M
Percent increase = (450 - 380) / 380 = 0.1842 = 18.4%

**Answer: 18.4**
```

## Usage

```bash
rnow init -t convfinqa -n "my-convfinqa"
rnow test --smoke-test
rnow run
```

## References

- [ConvFinQA Paper](https://arxiv.org/abs/2210.03849)
- [Dataset](https://github.com/czyssrs/ConvFinQA)
