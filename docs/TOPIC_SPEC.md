# Topic Decision Spec — Hướng dẫn kỹ thuật Đề án 06

## Quyết định

Sử dụng tài liệu tiếng Việt **Hướng dẫn về các thành phần công nghệ thông tin
thiết yếu cần thiết và tiêu chuẩn, định mức phục vụ triển khai Đề án 06** làm
corpus duy nhất cho phần cơ bản của Day 7.

- Văn bản: Công văn 1552/BTTTT-THH ngày 26/04/2022, phiên bản 1.0.
- Cơ quan ban hành: Bộ Thông tin và Truyền thông.
- Nguồn chính thức: <https://onetouch.mic.gov.vn/tai-lieu/CHINH_PHU_SO/1552_BTTTT-THH.pdf>.
- Trang danh mục: <https://onetouch.mic.gov.vn/tai-lieu/TaiLieu_Chinh_Phu_So.html>.
- SHA-256: `f92551f03b06242b40d10f7c2c9c29a5f73b847acbfed5ae3c1699de50982a97`.
- Ngôn ngữ hỏi/đáp: tiếng Việt.

Tài liệu có 52 trang vật lý. Corpus lấy trang 3–50 (48 trang có nội dung) và
loại trang 1–2 trống cùng trang 51–52 chứa thông tin liên hệ cá nhân. File PDF
gốc được giữ cục bộ trong `data/raw/` và bị Git ignore.

## Lý do chọn

Tài liệu vượt ngưỡng 30 trang, có nguồn nhà nước rõ ràng và cấu trúc ổn định:
phạm vi/đối tượng, nguyên tắc, mô hình tổng thể, yêu cầu chức năng, định danh
điện tử, an toàn thông tin, hiệu năng và tiêu chuẩn. Các bảng yêu cầu và 13 hình
vẽ tạo đủ câu hỏi định nghĩa, liệt kê, so sánh và tổng hợp nhiều mục. Chủ đề hẹp
đủ để audit citation nhưng vẫn phù hợp retrieval tiếng Việt.

Đây là corpus đánh giá nội dung của **phiên bản 1.0**, không phải nguồn tư vấn
pháp lý hoặc mô tả quy định hiện hành. Mục 7 đã được sửa đổi bởi Công văn
708/BTTTT-CATTT năm 2024; benchmark phải diễn đạt câu hỏi theo tài liệu 1552
và không suy diễn thành yêu cầu pháp luật hiện tại.

## Hợp đồng ingest

1. Chỉ nhận PDF có checksum đã ghim; sai checksum phải fail-fast.
2. Trích xuất layout-aware bằng `pdftotext`, giữ nguyên tiếng Việt NFC.
3. Mỗi page/chunk phải truy ngược được URL, SHA-256, trang vật lý, mục và
   khoảng ký tự nguồn.
4. Chunk xác định, có overlap cố định; không được sinh hoặc sửa nội dung.
5. Gọi OpenRouter với model đa ngôn ngữ `baai/bge-m3` (1024 chiều), chuẩn hóa
   vector phía client và nạp pgvector riêng của dự án; cấm vector ngẫu nhiên.
6. Ghi phiên bản công cụ, số trang/chunk và checksum vào manifest.

## Retrieval và benchmark

So sánh ba phương pháp trên cùng corpus và gold đã đóng băng: lexical, dense
và hybrid RRF. Top-k khác nhau chỉ là cấu hình, không phải phương pháp mới.
Benchmark ban đầu có đúng 24 câu: dev 15 answerable + 2 unanswerable; test 5
answerable + 2 unanswerable. Evidence phải là đoạn nguyên văn và trang vật lý;
gold chỉ được dùng trong validation/evaluation, không được đưa cho agent.

## Tiêu chí chấp nhận

- 48 trang nội dung và mọi chunk vượt validation; checksum corpus được ghim.
- Embedding hữu hạn, normalized, đúng 1024 chiều và được tạo bởi model thật.
- pgvector kiểm tra dimension, row count, chống trùng và tìm kiếm sau restart.
- Dev báo Hit@1, Evidence Hit@3, Context Evidence Rate, latency và cost.
- Cấu hình được đóng băng trước test; file đề bài gốc không bị sửa.

Tài liệu 52 trang không đạt điều kiện ≥100 trang của phần nâng cao; phạm vi hiện
tại nhắm hoàn thành đầy đủ phần cơ bản 85 điểm.
