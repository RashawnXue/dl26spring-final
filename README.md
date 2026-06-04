# HumanEval Code Generation Experiment

This folder contains the code for the course report experiment:

- Task: Python function-level code generation
- Dataset: HumanEval
- Model: `deepseek-ai/deepseek-coder-1.3b-base`
- Metrics: syntax pass rate, pass@1, failure types

## Colab Setup

```bash
pip install -U -r requirements.txt
```

If Colab has already imported `transformers` before installation, restart the
runtime once after installing the requirements. This avoids tokenizer version
conflicts with `deepseek-ai/deepseek-coder-1.3b-base`.

Run the main experiment methods on a small subset first:

```bash
python run_humaneval_experiment.py --method all --limit 20 --max-new-tokens 128 --output-dir results-sub
```

`--method all` runs the four methods used in the report:

- `baseline_none`
- `baseline_template`
- `deepseek_basic`
- `deepseek_enhanced`

Run the full experiment:

```bash
python run_humaneval_experiment.py --method all --max-new-tokens 128 --output-dir results-final
```

Run only the DeepSeek-Coder basic prompt setting:

```bash
python run_humaneval_experiment.py --method deepseek_basic --limit 20 --max-new-tokens 128 --output-dir results-deepseek-sub
```

Run only the CodeT5-small enhanced prompt setting:

```bash
python run_humaneval_experiment.py --method deepseek_enhanced --max-new-tokens 128 --output-dir results-deepseek_enhanced
```

## Outputs

The script writes:

- `results/results_<method>.csv`: per-task prediction and status
- `results/results_<method>.json`: per-task structured records
- `results/summary_<method>.json`: aggregate metrics

Copy the aggregate metrics and several representative failure cases into the report.

## Notes

The script executes generated Python code in a subprocess with a timeout. This is suitable for a controlled course experiment, but generated code should not be executed on a machine containing sensitive files or credentials.
