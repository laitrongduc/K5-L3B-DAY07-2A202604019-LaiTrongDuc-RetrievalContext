# Báo cáo Day 7 — 2A202604019 Lai Trọng Đức

## 1. Dữ liệu, kiến trúc và benchmark

Corpus là bản tiếng Việt *Hướng dẫn về các thành phần công nghệ thông tin thiết yếu cần thiết và tiêu chuẩn, định mức phục vụ triển khai Đề án 06*, Công văn 1552/BTTTT-THH ngày 26/04/2022, phiên bản 1.0. PDF chính thức có 52 trang; pipeline chỉ lấy 48 trang nội dung (trang vật lý 3–50), loại hai trang trống và hai trang chứa đầu mối liên hệ. Nguồn: `https://onetouch.mic.gov.vn/tai-lieu/CHINH_PHU_SO/1552_BTTTT-THH.pdf`; SHA-256 PDF là `f925...2a97`. Đây là tài liệu hướng dẫn công khai, được giữ nguyên attribution; kết quả chỉ mô tả phiên bản 1.0, không dùng như tư vấn pháp lý hiện hành.

`pdftotext` trích 109.112 ký tự; chunk theo từng trang, cửa sổ 1.200 ký tự và overlap 180 ký tự, tạo 123 chunk. Mỗi chunk giữ URL, checksum PDF, trang vật lý, chương và khoảng ký tự. Embedding thật được tạo qua OpenRouter bằng `baai/bge-m3` (response alias `parasail-bge-m3`), 1.024 chiều, chuẩn hóa L2: 123 vector, 25.538 token, 17,629514 giây, chi phí ước tính 0,00025538 USD. Vector store là PostgreSQL 16.16 + pgvector 0.8.7, 123 hàng/123 ID; cosine probe sau khi mở kết nối mới đạt 1,0 và nạp trùng bị từ chối. Không dùng vector ngẫu nhiên.

Kiến trúc tách `ingestion → embedding → vector store → retriever → agent state machine → evaluation`. Agent dùng `openai/gpt-4o-mini` qua OpenRouter, temperature 0. Model chọn tool/query; runtime không nhận gold. Context tối đa 3 chunk/9.000 ký tự, tối đa 3 search. Citation phải ánh xạ duy nhất tới context; truy vấn tool+query lặp lại dẫn tới abstain; lỗi provider được ghi riêng, không tạo kết quả giả.

Benchmark đóng băng gồm 24 câu tiếng Việt: dev 15 có đáp án + 2 thiếu bằng chứng; test 5 + 2. Evidence là trích dẫn nguyên văn, đúng source/page. SHA-256 questions `5593...bb15`, chunks `f727...7954`. Cấu hình được chọn bằng dev và ghi vào `config/agent_frozen.json` trước khi mở test.

## 2. Retrieval và thử nghiệm

Tất cả phương pháp dùng cùng 123 chunk, candidate-k 20 và chấm top-3. BM25 dùng tokenizer Unicode tiếng Việt; dense dùng BGE-M3/cosine trên pgvector; hybrid dùng RRF với `k=60`. Reference-aware chỉ xử lý câu tham chiếu “Hình N”: nếu có caption ở thân bài thì ưu tiên nó hơn mục lục.

| Cấu hình | Phương pháp/thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---:|---:|---:|---:|
| Baseline | BM25 | dev | 10/15 (66,67%) | 14/15 (93,33%) | 14/15 (93,33%) | 0,010601 s; 0 USD |
| Phương pháp 2 | Dense BGE-M3 + pgvector | dev | 8/15 (53,33%) | 10/15 (66,67%) | 10/15 (66,67%) | 4,009323 s |
| Phương pháp 3 | RRF cân bằng | dev | 10/15 (66,67%) | 12/15 (80,00%) | 12/15 (80,00%) | 0,002111 s fusion |
| Cải thiện 1/chọn | BM25 reference-aware | dev | 11/15 (73,33%) | 15/15 (100%) | 15/15 (100%) | 0,000462 s hậu xử lý |
| Cải thiện 2 | RRF lexical 2× | dev | 10/15 (66,67%) | 13/15 (86,67%) | 13/15 (86,67%) | 0,001593 s fusion |
| Baseline | BM25 | test | 2/5 (40%) | 5/5 (100%) | 5/5 (100%) | 0,002534 s; 0 USD |
| Cấu hình chọn | BM25 reference-aware | test | 2/5 (40%) | 5/5 (100%) | 5/5 (100%) | 0,000143 s hậu xử lý |

Query embedding của toàn bộ dev dùng 344 token, ước tính 0,00000344 USD; test dùng 89 token, 0,00000089 USD. Thời gian dense đã gồm API + DB; thời gian BM25/reference/RRF chỉ đo phần local tương ứng nên không so sánh trực tiếp như cùng một latency end-to-end.

Hai lỗi dev được đọc và sửa:

1. `dev-09` hỏi “Hình 2” nhưng BM25 xếp caption đúng ở rank 4 do mục lục/đoạn QR lấn át. Reference-aware đưa caption thân bài lên rank 1, nâng Evidence Hit@3 từ 14/15 lên 15/15.
2. Dense bỏ lỡ top-3 ở `dev-03/04/05/07/09` do ưu tiên đoạn gần nghĩa thay vì điều kiện chính xác. RRF lexical 2× tăng 12/15 lên 13/15 nhưng vẫn kém reference-aware 15/15. Vì tiêu chí chọn trước là Evidence Hit@3/Context Evidence rồi mới latency/cost, cấu hình lexical reference-aware được đóng băng.

Audit context test trong `results/test/context_reviews.json`: cả BM25 baseline và cấu hình chọn đạt mức 2 ở 5/5 câu. Dense đạt Hit@1 5/5 trên test, nhưng kết quả này chỉ được báo cáo; không dùng để đổi cấu hình sau freeze.

## 3. Agent, trace và đánh giá

Agent có ba tool `search_lexical`, `search_dense`, `search_hybrid`; GPT-4o-mini tự chọn tool và query. Thực nghiệm thực tế chọn lexical cho mọi câu. Context deduplicate theo chunk ID, giữ page/source, giới hạn 9.000 ký tự. Terminal state là `answered`, `abstain` hoặc `error`; citation sai/thiếu, search trùng hoặc hết lượt đều fail-closed thành abstain.

| Cấu hình / split | Tool use | Model/search calls | Grounded answer (0/1/2; TB) | Citation hỗ trợ | Câu trả lời có citation | Abstention đúng | Từ chối nhầm |
|---|---:|---:|---:|---:|---:|---:|---:|
| A dev — chưa retry/checkpoint | 11/11 ca thành công | 25/11 | chưa chấm đầy đủ; 6 provider/tool errors | chưa chấm | 7 câu trả lời | — | — |
| B dev — retry + checkpoint | 16/16 | 34/16 | 1/0/14; 1,8667/2 | 15/18 (83,33%) | 14/14 (100%) | 1/1 | 1/15 |
| C dev — guardrail đã chọn | 17/17 | 34/17 | 0/1/14; 1,9333/2 | 17/20 (85%) | 15/15 (100%) | 2/2 (100%) | 0/15 |
| Frozen test | 7/7 | 14/7 | 1/0/4; 1,6/2 | 4/4 (100%) | 4/4 (100%) | 2/2 (100%) | 1/5 (20%) |

Cấu hình A dev gặp 5 lỗi mạng tạm thời và một action JSON sai. Cấu hình B thêm retry giới hạn, checkpoint atomic và renderer citation; còn `dev-14` từ chối nhầm vì model rút gọn chunk ID, `dev-u01` lỗi do lặp search rồi sinh tên tool sai. Cấu hình C chỉ mở rộng citation prefix khi khớp duy nhất một context ID và chặn tool+query trùng bằng abstain. Hai lỗi này có regression test; kết quả dev sau sửa không còn error/false abstention. Cấu hình C là artifact hợp nhất có kiểm tra từ run B và hai repair run, nên 264,803911 giây là thời gian cộng dồn của các source run, không phải một lượt wall-clock duy nhất; tổng 28.108 token, 0,0053349 USD.

### Audit từng câu test

| Query | Loại | Kết quả | Nhãn và lý do |
|---|---|---|---|
| test-01 | Có đáp án | Lexical 1 lượt; trả “dữ liệu dân cư”, 1 citation | 2; citation chứa đúng mệnh đề dữ liệu gốc |
| test-02 | Có đáp án | Lexical 1 lượt; Mô hình 03 + đặc thù Bộ Quốc phòng | 2; citation hỗ trợ đủ hai ý |
| test-03 | Có đáp án | Lexical 1 lượt; tối thiểu 30 km | 2; đúng số và cấp độ |
| test-04 | Có đáp án | Lexical 1 lượt; nội dung QCVN 109 | 2; citation hỗ trợ đầy đủ |
| test-05 | Có đáp án | Query viết lại “Văn bản hướng dẫn kiểm thử phần mềm nội bộ”; abstain | 0; từ chối nhầm, query gốc mới đưa Công văn 3787 vào top-3 |
| test-u01 | Thiếu bằng chứng | Lexical 1 lượt; model abstain | Đúng; corpus không có Kubernetes |
| test-u02 | Thiếu bằng chứng | Lexical 1 lượt; model abstain | Đúng; corpus không chỉ định thuật toán nhận diện khuôn mặt |

Test chạy đúng một lượt sau freeze: 217,082701 giây, 12.250 token, 0,0022443 USD, không provider error. Ví dụ trả lời có bằng chứng là `test-03`; ví dụ thiếu bằng chứng là `test-u01`; lỗi thật là `test-05`. Lỗi cuối cho thấy query rewriting có thể làm giảm recall dù retrieval query gốc đạt 5/5; hướng cải thiện tương lai là tìm song song query gốc và query model rồi fusion, nhưng không áp dụng sau khi xem test.

## 4. Tái lập và bằng chứng

```bash
docker compose -f infra/docker-compose.yml up -d
PYTHONPATH=src .venv/bin/python -m agentic_rag verify
PYTHONPATH=src .venv/bin/python -m agentic_rag verify-freeze --manifest data/benchmark/freeze.json
PYTHONPATH=src .venv/bin/python -m agentic_rag evaluate-retrieval \
  --chunks data/corpus/chunks.json --questions data/benchmark/questions.json \
  --embedding-config config/embedding.json --split test \
  --output results/test/retrieval_frozen.json
PYTHONPATH=src .venv/bin/python -m agentic_rag run-agent-split \
  --chunks data/corpus/chunks.json --questions data/benchmark/questions.json \
  --agent-config config/agent.json --embedding-config config/embedding.json \
  --split test --output results/test/agent_frozen.json
```

Cần đặt `OPENROUTER_API_KEY` và `DATABASE_URL` trong môi trường; không commit `.env`. Trace/nhãn/metric chính: `results/dev/agent_config_c.json`, `results/dev/reviews_config_c.json`, `results/dev/agent_metrics_config_c.json`, `results/test/agent_frozen.json`, `results/test/reviews_frozen.json`, `results/test/agent_metrics_frozen.json`. Retrieval nằm tại `results/dev/retrieval_config_b.json` và `results/test/retrieval_frozen.json`.

Quality gate cuối gồm Ruff format/lint, mypy strict, compileall và 54 unit/integration tests offline. Runtime contracts kiểm tra checksum/schema/dimension/norm, citation-context, budget/state, chống nạp trùng và provider failure. Giới hạn: PDF extraction chưa hiểu cấu trúc bảng như layout model; semantic support vẫn cần audit người; OpenRouter latency/alias có thể thay đổi; corpus 48 trang không đạt bonus đa phương thức ≥100 trang nên không tuyên bố phần nâng cao.
