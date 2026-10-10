# External dependencies

| Component | Search terms | Required files or action |
| --- | --- | --- |
| SpreadsheetBench data | `SpreadsheetBench Verified 400` | Obtain `spreadsheetbench_verified_400.tar.gz` from the official release. |
| Official evaluator | `SpreadsheetBench official evaluation` | Obtain `evaluation/evaluation.py` at revision `49b73a94775fb489063f60ca1865e3a650079a79`. |
| Model | `Qwen3-14B official model` | Obtain the checkpoint, tokenizer, and model configuration; place them in `models/Qwen3-14B`. |
| Python dependencies | Package names in `requirements-model.txt` | Install the declared packages in the experiment environment. |
| Spreadsheet runtime | `LibreOffice Calc headless` | Build the supplied Dockerfile to obtain the configured tool environment. |

Pass the local archive and evaluator to `prepare_data.py` with `--archive` and `--scorer`. The expected evaluator SHA-256 is `4ae77cee8df01d1f34684fceab972810d696886533d33be2e89373de6b4d3de3`.

External data, evaluator code, and model weights are obtained separately. Their license and attribution notices remain with the upstream distributions and should be retained with downloaded copies. This package contains no bundled dataset, evaluator, model weights, or third-party baseline implementation.
