# Papers

Two companion papers on data leakage in machine learning.

## Reproduce

### Option 1: Docker (no local dependencies)

```bash
cd papers
docker build -t ml-papers-verify .
docker run ml-papers-verify
```

### Option 2: Local (Python 3.10+ and [Quarto](https://quarto.org/))

```bash
# 1. Landscape paper (run first — grammar paper depends on its claims.json)
cd landscape
pip install -r requirements.txt
python compile_claims.py    # data/*.jsonl → claims.json
python figures.py           # data/*.jsonl → figures/*.pdf
quarto render paper.qmd     # → paper.pdf

# 2. Grammar paper
cd ../grammar
pip install -r requirements.txt
python compile_claims.py    # reads ../landscape/claims.json
quarto render paper.qmd     # → paper.pdf
```

## Structure

```
landscape/          Leakage landscape study (28 experiments, 2,047 datasets)
  data/             Raw experiment results (JSONL)
  figures/          Generated publication figures
  compile_claims.py Data → claims.json (single source of truth for all numbers)
  figures.py        Data → publication figures
  claims.py         Quarto inline lookup (used by paper.qmd)
  paper.qmd         Paper source
  Dockerfile        Standalone verification

grammar/            ML workflow grammar specification
  compile_claims.py Extracts subset from landscape claims
  claims.py         Quarto inline lookup
  paper.qmd         Paper source

Dockerfile          Verify both papers end-to-end
```
