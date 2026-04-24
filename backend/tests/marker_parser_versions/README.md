# Marker parser incremental versions

> Ghi chú: số dòng bên dưới áp dụng cho đúng các file `.py` trong thư mục này ở thời điểm hiện tại. Nếu bạn sửa thêm code, line number sẽ dịch chuyển.

## 1) Tóm tắt nhanh theo version

| Version | Mục tiêu | Thay đổi chính | File |
|---|---|---|---|
| v0 | Baseline để so sánh | Reconstruct lại parser đầy đủ từ 2 file hiện tại | `v0_marker_document_parser_baseline_reconstructed.py` |
| v1 | Sửa env/cache | Chuẩn hóa cache env cho Hugging Face / Transformers | `v1_marker_document_parser_env_cache_fix.py` |
| v2 | Tăng chất lượng OCR tiếng Việt + tune runtime | Thêm `langs=["vi", "en"]`, thêm `batch_multiplier` | `v2_marker_document_parser_langs_batch.py` |
| v3 | Tự động route config theo loại tài liệu | Auto detect profile + converter cache theo profile + `force_ocr` cho scanned | `v3_marker_document_parser_auto_profile_force_ocr.py` |

---

## 2) v0 — Baseline reconstructed

### Các block quan trọng
- Env/cache block: dòng **22-34**
- `_get_converter()`: dòng **70-106**
- `_parse_with_marker()`: dòng **134-175**
- `_save_marker_images()`: dòng **180-223**
- `_replace_image_refs_in_markdown()`: dòng **233-250**
- `_extract_tables_from_markdown()`: dòng **255-304**
- `_chunk_markdown()`: dòng **313-358**
- `_parse_legacy()`: dòng **393-431**
- CLI/test runner: dòng **434-510**

### Ý nghĩa
Bản này là mốc gốc để bạn diff từng bước nhỏ. Không cố gắng tối ưu mạnh, chỉ giữ cấu trúc gần với code hiện tại nhất.

---

## 3) v1 — Chỉ sửa env/cache

### Thay đổi nằm ở đâu
- **Dòng 22-34**: thay toàn bộ block env/cache của v0
  - thêm `_MARKER_MODEL_BASE`
  - dùng `HF_HOME`
  - thêm `HF_HUB_CACHE`
  - đồng bộ `HUGGINGFACE_HUB_CACHE`
  - đổi `TRANSFORMERS_CACHE` để trỏ cùng hub cache
- **Dòng 36-43**: import `app.*` được đặt sau env block để tránh thư viện đọc cache path quá sớm

### Cụ thể đổi gì so với v0
#### v0
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["MARKER_HOME"] = "/home/user/Workspace/healer/rag/backend/models/marker_models"
os.environ["DATALAB_CACHE_DIR"] = os.path.join(os.environ["MARKER_HOME"], ".cache/datalab")
os.environ["HF_HOME"] = os.path.join(os.environ["MARKER_HOME"], ".hf-cache")
os.environ["HUGGINGFACE_HUB_CACHE"] = os.path.join(os.environ["HF_HOME"], "hub")
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ["HF_HOME"], "transformers")
```

#### v1
```python
_MARKER_MODEL_BASE = "/home/user/Workspace/healer/rag/backend/models/marker_models"

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["MARKER_HOME"] = _MARKER_MODEL_BASE
os.environ["DATALAB_CACHE_DIR"] = os.path.join(_MARKER_MODEL_BASE, ".cache", "datalab")

_HF_HOME = os.path.join(_MARKER_MODEL_BASE, ".hf-cache")
_HF_HUB_CACHE = os.path.join(_HF_HOME, "hub")

os.environ["HF_HOME"] = _HF_HOME
os.environ["HF_HUB_CACHE"] = _HF_HUB_CACHE
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_HUB_CACHE
os.environ["TRANSFORMERS_CACHE"] = _HF_HUB_CACHE
```

### Tác động
- Không đổi logic parse/chunk/image/table.
- Chỉ giảm rủi ro tải model/cache sai chỗ hoặc bị tách nhiều thư mục cache.

---

## 4) v2 — Chỉ thêm 2 config OCR/runtime

### Thay đổi nằm ở đâu
- **Dòng 83-89** trong `_get_converter()`

### Cụ thể đổi gì so với v1
#### v1
```python
config = {
    "output_format": "markdown",
    "paginate_output": True,
    "disable_image_extraction": not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION,
}
```

#### v2
```python
config = {
    "output_format": "markdown",
    "paginate_output": True,
    "disable_image_extraction": not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION,
    "langs": ["vi", "en"],
    "batch_multiplier": getattr(settings, "HEALERRAG_MARKER_BATCH_MULTIPLIER", 1),
}
```

### 2 config mới nghĩa là gì
- `langs=["vi", "en"]`
  - giúp OCR ưu tiên tiếng Việt nhưng vẫn giữ khả năng đọc từ tiếng Anh, ký hiệu, viết tắt.
- `batch_multiplier`
  - cho phép scale batch nội bộ của Marker/Surya theo VRAM.
  - `1` là an toàn. GPU khỏe hơn có thể tăng dần.

### Tác động
- Không đổi flow parser.
- Không thêm route logic.
- Phù hợp rollout ngay sau v1.

---

## 5) v3 — Auto profile + force_ocr cho scanned docs

### Thay đổi nằm ở đâu
- `__init__()` đổi từ 1 converter thành converter cache theo profile: **58-61**
- `_load_artifacts()`: **70-81**
- `_build_config(profile)`: **83-101**
- `_get_converter(profile)`: **103-128**
- `_detect_profile(file_path)`: **130-143**
- `_detect_pdf_profile(file_path)`: **145-181**
- `_parse_with_marker()` dùng profile trước khi tạo converter: **226-228**

### Cụ thể đổi gì
#### 5.1 `__init__` — đổi state
- v2: dùng `self._converter`
- v3: dùng `self._converter_cache: dict[str, object] = {}`

#### 5.2 `_build_config(profile)` — gom config builder riêng
- `general`: giữ config thường
- `scanned`: thêm
  - `force_ocr = True`
  - `strip_existing_ocr = False`
- `spreadsheet`: ép `disable_image_extraction = True`

#### 5.3 `_detect_profile()` — route theo extension
- `.docx/.pptx/.html/.epub` → `general`
- `.xlsx` → `spreadsheet`
- ảnh `.png/.jpg/...` → `scanned`
- `.pdf` → gọi `_detect_pdf_profile()`

#### 5.4 `_detect_pdf_profile()` — route PDF scan/digital bằng PyMuPDF
- sample tối đa 3 trang đầu
- tính `avg_chars/page`
- tính `image_ratio`
- rule hiện tại:
  - nếu `avg_chars < 80` và `image_ratio > 0.3` → `scanned`
  - ngược lại → `general`

#### 5.5 `_parse_with_marker()` — chèn route profile vào pipeline
Thay vì:
```python
converter = self._get_converter()
```
đổi thành:
```python
profile = self._detect_profile(file_path)
converter = self._get_converter(profile)
```

### Tác động
- Đây là bản đầu tiên bắt đầu thay đổi behavior thực sự theo loại tài liệu.
- Phù hợp khi bạn đã ổn với v1/v2 và muốn xử lý scan PDF/ảnh tốt hơn.

---

## 6) Rollout khuyến nghị

1. Chạy **v1** trước để ổn định cache/model path.
2. Sau đó chạy **v2** để xem tiếng Việt và tốc độ/VRAM ổn chưa.
3. Cuối cùng mới lên **v3** vì v3 bắt đầu đổi routing hành vi parser.

---

## 7) Logo extraction — Marker đang làm được gì và cần thêm gì

### Flow hiện tại trong code
Trong cả v0-v3, ảnh Marker extract đi theo flow này:
1. Marker render tài liệu → trả về `marker_images`
2. `_save_marker_images()` lưu PIL image ra disk
3. `_replace_image_refs_in_markdown()` thay ref ảnh trong markdown thành static URL
4. caption ảnh chỉ là bước bổ sung sau đó, **không phải** bước quyết định ảnh có được lưu hay không

### Những chỗ cần đọc trong code
- v0 `_parse_with_marker()` dòng **144-154**
- v0 `_save_marker_images()` dòng **180-223**
- v0 `_replace_image_refs_in_markdown()` dòng **233-250**
- v3 `_parse_with_marker()` dòng **226-237**
- v3 `_save_marker_images()` dòng **263-305**

### Điều quan trọng
Logo **không phải OCR output**. Logo chỉ được giữ lại nếu nó đi qua nhánh **image extraction**.

Nói cách khác:
- OCR = đọc chữ trong tài liệu hoặc trong ảnh scan
- caption = mô tả nội dung ảnh sau khi đã lưu ảnh
- logo retention = phải có bước giữ/sao chép asset ảnh ra disk ngay từ pipeline image extraction

### Khi nào logo dễ bị mất
- `disable_image_extraction=True`
- logo quá nhỏ và Marker không surface nó vào `marker_images`
- tài liệu PDF có logo là embedded image nhưng layout pipeline không render ra markdown image block
- bạn đạt ngưỡng `HEALERRAG_MAX_IMAGES_PER_DOC` quá sớm nên logo ở trang sau không được lưu

### Cách làm để giữ logo tốt hơn
#### Cách 1 — giữ nguyên pipeline hiện tại nhưng nới điều kiện
- đảm bảo `HEALERRAG_ENABLE_IMAGE_EXTRACTION=True`
- tăng `HEALERRAG_MAX_IMAGES_PER_DOC`
- không tắt image extraction ở các profile cần giữ logo

#### Cách 2 — thêm PDF raw image fallback cho logo/embedded asset
Đây là cách thực tế nhất nếu bạn muốn giữ logo chắc chắn hơn:
- sau khi có `marker_images`, nếu file là PDF thì dùng **PyMuPDF** đọc thêm `page.get_images(full=True)`
- extract raw embedded images trực tiếp từ PDF
- save ra disk
- dedupe với ảnh Marker đã lưu bằng hash hoặc `(width, height, bbox gần giống)`

Cách này rất hữu ích cho:
- logo công ty ở header/footer
- icon nhỏ
- con dấu dạng raster nhỏ
- ảnh embedded mà Marker không đẩy vào markdown

#### Cách 3 — gắn nhãn ảnh là `logo` thay vì chỉ `image`
Sau khi ảnh đã được lưu, bạn có thể thêm bước classify nhẹ:
- heuristic: ảnh nhỏ, nằm ở top-left/top-right, xuất hiện lặp lại nhiều trang, nền trong/sắc nét
- hoặc vision model để gán `image_type = logo | photo | stamp | chart | signature`

### Gợi ý patch ít xâm lấn
Nếu muốn thêm logo extraction mà vẫn incremental, patch nên chạm đúng 2 điểm:
1. thêm `_extract_raw_pdf_images()`
2. merge kết quả đó vào `_parse_with_marker()` ngay sau dòng save `marker_images`

Pseudo flow:
```python
images = self._save_marker_images(marker_images, document_id)
if file_path.suffix.lower() == ".pdf":
    raw_images = self._extract_raw_pdf_images(file_path, document_id)
    images = self._merge_and_dedupe_images(images, raw_images)
```

Patch như vậy tách biệt rõ:
- Marker layout images
- raw embedded PDF assets
- caption/classification ở bước sau

---

## 8) Kết luận ngắn
- Muốn rollout an toàn: **v1 → v2 → v3**.
- Muốn giữ logo tốt: pipeline phải coi logo là **image asset cần lưu**, không phải chỉ là đối tượng để OCR/caption.
- Nếu file PDF có nhiều logo/icon nhỏ, nên thêm **raw PDF image fallback** bằng PyMuPDF.
