# Quy trình phát triển và tái lập

Code triển khai nằm trong `src/agentic_rag/`; test bổ sung nằm riêng trong `tests/solution/`. Bốn test gốc được giữ nguyên tại thư mục `tests/` để người chấm nhận diện ngay. Các file đề bài ở root, `starter/`, test gốc và mẫu báo cáo không được chỉnh sửa. Báo cáo đã điền là `report/individual/2A202604019_LaiTrongDuc.md`.

## Chuẩn bị môi trường

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pip install -r requirements-runtime.txt
docker compose -f infra/docker-compose.yml up -d
```

Tạo `.env` cục bộ từ `.env.example`, điền `OPENROUTER_API_KEY` và `DATABASE_URL`. Không ghi secret vào config, trace, fixture hoặc commit.

## Quality gates

```bash
PYTHONPATH=src .venv/bin/python -m agentic_rag verify
```

Lệnh chạy Ruff format/lint, mypy strict, compileall và toàn bộ `unittest`. Kết quả chất lượng cuối là 54 test pass. Test offline chỉ xác minh hành vi code, không được dùng như chất lượng benchmark.

## Chuẩn bị dữ liệu thật

```bash
PYTHONPATH=src .venv/bin/python -m agentic_rag prepare-corpus \
  --pdf data/raw/1552-btttt-thh-de-an-06-v1.0.pdf \
  --config config/corpus.json --output data/corpus/chunks.json
PYTHONPATH=src .venv/bin/python -m agentic_rag embed-corpus \
  --chunks data/corpus/chunks.json --config config/embedding.json \
  --output data/index/embeddings.json
PYTHONPATH=src .venv/bin/python -m agentic_rag index-pgvector \
  --chunks data/corpus/chunks.json --embeddings data/index/embeddings.json \
  --dimensions 1024 --model baai/bge-m3
```

PDF phải đúng checksum đã ghim. Embedding phải là BGE-M3 thật, hữu hạn, 1.024 chiều và normalized; không dùng vector ngẫu nhiên.

## Benchmark và agent

```bash
PYTHONPATH=src .venv/bin/python -m agentic_rag verify-freeze \
  --manifest data/benchmark/freeze.json
PYTHONPATH=src .venv/bin/python -m agentic_rag evaluate-retrieval \
  --chunks data/corpus/chunks.json --questions data/benchmark/questions.json \
  --embedding-config config/embedding.json --split dev \
  --output results/dev/retrieval-new.json
PYTHONPATH=src .venv/bin/python -m agentic_rag run-agent-split \
  --chunks data/corpus/chunks.json --questions data/benchmark/questions.json \
  --agent-config config/agent.json --embedding-config config/embedding.json \
  --split dev --output results/dev/agent-new.json
```

Chỉ dùng dev để thay đổi cấu hình. Test đã được chạy single-pass sau freeze; không chạy lại để tuning. Gold chỉ đi vào validator/evaluator, không đi vào controller hoặc retriever runtime.

## Bằng chứng bàn giao

- Manifest: `data/index/*manifest.json`, `data/benchmark/freeze.json`, `config/*frozen.json`.
- Dev: `results/dev/retrieval_config_b.json`, `agent_config_c.json`, reviews và metrics C.
- Test: `results/test/retrieval_frozen.json`, `agent_frozen.json`, context reviews, answer reviews và metrics.
- Quyết định corpus: `docs/TOPIC_SPEC.md`; báo cáo: `report/individual/2A202604019_LaiTrongDuc.md`.
