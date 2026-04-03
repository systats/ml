# Papers

Two companion papers on data leakage in machine learning.

| Paper | DOI | PDF |
|-------|-----|-----|
| A Grammar of Machine Learning Workflows | [10.5281/zenodo.19406355](https://doi.org/10.5281/zenodo.19406355) | [grammar/paper.pdf](grammar/paper.pdf) |
| Which Leakage Types Matter? | [10.5281/zenodo.19406148](https://doi.org/10.5281/zenodo.19406148) | [landscape/paper.pdf](landscape/paper.pdf) |

## Verify claims

```bash
cd landscape
pip install numpy
python verify_from_raw.py    # 10/10 claims checked against raw JSONL
```

Also available in R, Julia, and Bash — no trust in `compile_claims.py` needed.

## Structure

```
grammar/                ML workflow grammar (specification paper)
  paper.qmd             Paper source
  paper.pdf             Compiled PDF
  refs.bib              Bibliography
  claims.py             Quarto inline lookup
  compile_claims.py     Reads landscape claims

landscape/              Leakage landscape (29 experiments, 2,047 datasets)
  paper.qmd             Paper source
  paper.pdf             Compiled PDF
  refs.bib              Bibliography
  data/                 Raw experiment results (JSONL)
  experiments/          Experiment runners (re-run from scratch)
  Dockerfile            Re-run experiments in container
  compile_claims.py     data/*.jsonl → claims.json
  figures.py            data/*.jsonl → figures/*.pdf
  claims.py             Quarto inline lookup
  verify_from_raw.py    Independent claim verification (Python)
  verify_from_raw.R     Independent claim verification (R)
  verify_from_raw.jl    Independent claim verification (Julia)
  verify_from_raw.sh    Independent claim verification (Bash)
```

## Data provenance

Every number in the papers traces to raw experiment data:

```
experiments/*.py → data/*.jsonl → compile_claims.py → claims.json → paper.qmd → paper.pdf
```

| File | Rows | Content |
|------|------|---------|
| leakage_landscape_v1_final.jsonl | 2,288 | Core experiments (A–L) |
| leakage_landscape_v1_extended.jsonl | 2,288 | Extended experiments |
| leakage_landscape_v2.jsonl | 2,047 | Experiments AQ, AC, AI, BB |
| v3_an.jsonl | 866 | N-scaling (50–10K) |
| v3_ap.jsonl | 1,965 | Seed dose-response (K=5–100) |
| v3_ao_merged.jsonl | 2,047 | CV coverage gap |
| phase2_ao_v2.jsonl | 3,522 | 6-method CI comparison |

## Re-run experiments

```bash
cd landscape
docker build -t landscape-experiments .
docker run landscape-experiments
```

Or locally (~48–72h, 64GB+ RAM, downloads from OpenML):

```bash
cd landscape/experiments
pip install -r requirements.txt
python run_v3_experiments.py
```

## Compile papers

Requires [Quarto](https://quarto.org/).

```bash
# Landscape first (grammar depends on its claims.json)
cd landscape && pip install -r requirements.txt
python compile_claims.py && python figures.py && quarto render paper.qmd

# Grammar
cd ../grammar
python compile_claims.py && quarto render paper.qmd
```
