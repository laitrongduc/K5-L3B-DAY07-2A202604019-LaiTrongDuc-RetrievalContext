# Phân tách test gốc và test bổ sung

Bốn file test gốc của đề bài được giữ nguyên ngay tại thư mục này:

- `test_agent_metrics.py`
- `test_agentic.py`
- `test_baseline.py`
- `test_fixture.py`

Toàn bộ test do bài làm bổ sung nằm trong `tests/solution/`. Không file test gốc nào bị sửa. Lệnh sau tự động chạy cả hai nhóm:

```bash
python3 -m unittest discover -s tests -v
```
