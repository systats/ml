FROM python:3.12-slim

WORKDIR /paper

# Install deps (experiments + verification)
COPY experiments/requirements.txt /tmp/requirements.txt
COPY requirements.txt /tmp/verify-requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt -r /tmp/verify-requirements.txt

# Copy everything (build context = papers/landscape/)
COPY . .

# Re-run all experiments, then verify results match paper claims
# ~48-72h on 8+ cores, downloads from OpenML
CMD ["sh", "-c", "\
  echo '=== Phase 1+2: Core experiments ===' && \
  cd experiments && python run_leakage_landscape.py && cd .. && \
  echo '' && \
  echo '=== Phase 3: V3 experiments ===' && \
  cd experiments && python run_v3_experiments.py && cd .. && \
  echo '' && \
  echo '=== Verify results match paper claims ===' && \
  python verify_from_raw.py && \
  echo '' && \
  echo 'All experiments complete. Results verified.' \
"]
