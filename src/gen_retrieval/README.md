Generative Retrieval

# 0. Prerequisite

## a. Data

Use HotpotQA data

## b. Library

```bash
uv sync
```

# 1. Data preparation

Step 1: Tokenize corpus

```bash
python -m src.gen_retrieval.1_1_tokenize_corpus -n <no_parallel_processes>
```

Step 2: Extract embedding for each document

```bash
python -m rc.gen_retrieval.1_2_extract_doc_embs -d cuda
```

Step 3: Extract semantic ID
Run pytest

```bash
pytest src/gen_retrieval/tests/test_output.py
```

Run module

```bash
python -m src.gen_retrieval.1_3_form_docid -c <no_centroids> -n <no_parallel_processes>
```
