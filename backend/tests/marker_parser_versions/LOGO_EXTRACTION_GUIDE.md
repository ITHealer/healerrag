# Logo extraction guide for Marker parser

## Mục tiêu
Giữ lại logo như một asset ảnh riêng, lưu ra disk, có metadata và có thể reference lại trong markdown / DB, thay vì chỉ OCR text hoặc mô tả ảnh.

## 1) Phân biệt 3 pipeline khác nhau

### OCR
- nhiệm vụ: đọc text
- đầu ra: markdown/text
- không đảm bảo giữ file ảnh logo

### Image captioning
- nhiệm vụ: mô tả ảnh đã được lưu
- đầu ra: caption
- không quyết định ảnh có được extract hay không

### Image retention / logo retention
- nhiệm vụ: lưu asset ảnh ra disk/DB và giữ metadata
- đầu ra: `ExtractedImage`
- đây mới là thứ quyết định bạn có giữ được logo hay không

## 2) Trong code hiện tại, logo đi qua đâu

### v0
- `_parse_with_marker()` dòng 144-154
- `_save_marker_images()` dòng 180-223
- `_replace_image_refs_in_markdown()` dòng 233-250

### v3
- `_parse_with_marker()` dòng 226-237
- `_save_marker_images()` dòng 263-305

Nghĩa là hiện tại nếu Marker surface logo vào `marker_images`, code của bạn **đã có khả năng lưu logo**.
Vấn đề thường nằm ở chỗ Marker có trả logo về `marker_images` hay không.

## 3) Nguyên nhân phổ biến khiến logo không được lưu
- image extraction đang bị tắt
- `HEALERRAG_MAX_IMAGES_PER_DOC` quá thấp
- logo quá nhỏ nên Marker không coi là ảnh cần render
- logo là embedded PDF image nhưng không xuất hiện thành markdown image block
- logo bị lặp nhiều trang và bạn muốn dedupe nhưng chưa có logic dedupe hợp lý

## 4) Cách nâng cấp thực tế nhất

### Bước A — giữ pipeline Marker hiện tại
Giữ nguyên:
- `text, _ext, marker_images = text_from_rendered(rendered)`
- `_save_marker_images(marker_images, document_id)`

### Bước B — thêm raw PDF image fallback
Với PDF, đọc thêm embedded image trực tiếp từ PyMuPDF:

```python
import fitz


def _extract_raw_pdf_images(self, file_path: Path, document_id: int) -> list[ExtractedImage]:
    doc = fitz.open(str(file_path))
    images_dir = self._get_served_images_dir()
    results: list[ExtractedImage] = []

    for page_index in range(len(doc)):
        page = doc[page_index]
        for img in page.get_images(full=True):
            xref = img[0]
            base = doc.extract_image(xref)
            image_bytes = base["image"]
            ext = base.get("ext", "png")

            image_id = str(uuid.uuid4())
            image_path = images_dir / f"{image_id}.{ext}"
            image_path.write_bytes(image_bytes)

            results.append(
                ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_index + 1,
                    file_path=str(image_path),
                    caption="",
                    width=0,
                    height=0,
                )
            )

    doc.close()
    return results
```

### Bước C — merge + dedupe
Sau khi có `images` từ Marker và `raw_images` từ PDF:
- dedupe bằng SHA256 nội dung file, hoặc
- dedupe bằng perceptual hash nếu bạn muốn tolerant hơn

Pseudo:
```python
images = self._save_marker_images(marker_images, document_id)
if file_path.suffix.lower() == ".pdf":
    raw_images = self._extract_raw_pdf_images(file_path, document_id)
    images = self._merge_and_dedupe_images(images, raw_images)
```

## 5) Nếu muốn biết ảnh nào là logo
Có 2 mức:

### Heuristic nhẹ
Gắn cờ `is_logo_candidate=True` nếu:
- nằm gần đầu trang
- kích thước nhỏ
- tỷ lệ khung ổn định
- xuất hiện lặp lại nhiều trang

### Vision classify
Cho model vision nhìn ảnh và trả:
- logo
- seal/stamp
- signature
- chart
- photo
- other

## 6) Thiết kế DB/metadata nên có
Nếu bạn muốn dùng lâu dài, `ExtractedImage` nên có thêm:
- `image_type`
- `source` = `marker_layout` | `pdf_raw`
- `sha256`
- `is_duplicate`
- `page_bbox` (nếu lấy được)

## 7) Khuyến nghị triển khai thực tế
- Bắt đầu bằng raw PDF fallback trước vì nó cho hiệu quả rõ nhất với logo.
- Chưa cần caption/classifier ngay.
- Khi pipeline giữ được asset ổn định rồi mới thêm bước gắn nhãn `logo`.
