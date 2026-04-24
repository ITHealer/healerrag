"""
docling_parser_v1_vietnamese_ocr.py
====================================
VERSION 1 — Fix duy nhất: Tiếng Việt + Tiếng Anh OCR

VẤN ĐỀ CỐ ĐỊNH:
  ✓ Mất dấu tiếng Việt (àáâãèéê... bị strip thành a, e...)
  ✗ Table cross-page vẫn chưa fix (xem v2)
  ✗ Watermark/stamp chưa fix (xem v3)

NGUYÊN NHÂN MẤT DẤU:
  Code gốc không set `ocr_options` → Docling dùng EasyOCR mặc định
  với lang=["en"] → English-only charset → strip hết dấu Latin mở rộng.

  Với PDF có text layer (programmatic PDF từ Word/Excel):
    → Docling đọc thẳng từ PDF text layer, KHÔNG qua OCR.
    → Nếu font encoding lỗi (CID/Type1 cũ): text layer đã corrupt → garbage.
    → Giải pháp: set force_full_page_ocr=True để bypass text layer.

  Với PDF scan (ảnh chụp):
    → Luôn qua OCR, nếu không có "vi" trong lang thì mất dấu.

THAY ĐỔI SO VỚI CODE GỐC:
  Dòng duy nhất thêm vào _get_converter():
    pipeline_options.ocr_options = EasyOcrOptions(
        lang=["vi", "en"],           # <-- THÊM: Vietnamese + English
        confidence_threshold=0.4,    # <-- THÊM: giảm từ 0.5
        use_gpu=None,                # <-- THÊM: auto GPU detect
        force_full_page_ocr=False,   # PDF programmatic: False. Scan: True
        bitmap_area_threshold=0.02,  # <-- THÊM: giảm từ 0.05
        download_enabled=True,
    )

CONFIG MỚI CẦN THÊM VÀO settings:
  HEALERRAG_OCR_LANGUAGES: list[str] = ["vi", "en"]
  HEALERRAG_OCR_FORCE_FULL_PAGE: bool = False
  HEALERRAG_OCR_CONFIDENCE: float = 0.4

TEST:
  python3 test_parser_cli.py --file your.pdf --version v1
  python3 test_parser_cli.py --file scan.pdf --version v1 --force-full-page-ocr
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}


class DoclingDocumentParser(BaseDocumentParser):
    """V1: Fix Vietnamese OCR. Minimal change từ code gốc."""

    parser_name = "docling"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._converter = None

    @staticmethod
    def supported_extensions() -> set[str]:
        return _DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Converter — CHỈ THÊM ocr_options, mọi thứ khác giữ nguyên
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is not None:
            return self._converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            # [V1-NEW] Import EasyOcrOptions
            EasyOcrOptions,
        )

        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION
        pipeline_options.images_scale = settings.HEALERRAG_DOCLING_IMAGES_SCALE
        pipeline_options.do_formula_enrichment = settings.HEALERRAG_ENABLE_FORMULA_ENRICHMENT

        # ════════════════════════════════════════════════════════════════
        # [V1] CHỈ THÊM ĐOẠN NÀY — OCR với tiếng Việt + tiếng Anh
        # ════════════════════════════════════════════════════════════════
        #
        # Tại sao cần set tường minh?
        #   Docling Discussion #792: khi không set ocr_options tường minh,
        #   EasyOCR default lang=["en"] → mất dấu tiếng Việt.
        #
        # Tại sao ocr_options phải set SAU CÙNG?
        #   EasyOcrOptions() constructor reset mọi field về default.
        #   Nếu set trước rồi set lại thì bị override hết.
        #
        # QUAN TRỌNG cho PDF có text layer (Word → PDF export):
        #   do_ocr mặc định là True nhưng chỉ OCR bitmap regions.
        #   Nếu PDF lỗi font encoding → set force_full_page_ocr=True.
        #
        # OCR languages:
        #   "vi" = Vietnamese (Latin + diacritics: ăâđêôơư + 5 tone marks)
        #   "en" = English
        #   Cả hai dùng Latin script → chạy chung 1 model, không tốn thêm RAM.
        #
        pipeline_options.ocr_options = EasyOcrOptions(
            # [V1-NEW] Vietnamese + English (thay vì default English-only)
            lang=getattr(settings, "HEALERRAG_OCR_LANGUAGES", ["vi", "en"]),

            # [V1-NEW] Auto-detect GPU (None = auto CUDA/CPU)
            # Set False để force CPU nếu GPU có vấn đề
            use_gpu=None,

            # [V1-NEW] Giảm confidence threshold: 0.5 → 0.4
            # 0.4: bắt được chữ mờ hơn, đánh đổi ít noise hơn
            # Với chữ in rõ: 0.5 vẫn tốt. Với scan/viết tay: 0.3-0.4.
            confidence_threshold=getattr(settings, "HEALERRAG_OCR_CONFIDENCE", 0.4),

            # [V1-NEW] False: chỉ OCR bitmap (nhanh, cho programmatic PDF)
            # True:  OCR toàn trang (chậm, cho scan/viết tay/font lỗi)
            # → Để True nếu tiếng Việt vẫn bị mất dấu sau khi set lang
            force_full_page_ocr=getattr(
                settings, "HEALERRAG_OCR_FORCE_FULL_PAGE", False
            ),

            # [V1-NEW] Giảm từ 0.05 → 0.02:
            # OCR cả bitmap chiếm >= 2% diện tích trang (stamp, chữ ký nhỏ)
            bitmap_area_threshold=0.02,

            # Tự download EasyOCR models nếu chưa có
            download_enabled=True,
        )
        # ════════════════════════════════════════════════════════════════
        # Hết phần thay đổi V1
        # ════════════════════════════════════════════════════════════════

        self._converter = DocumentConverter(
            format_options={
                "pdf": PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
        return self._converter

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    # ------------------------------------------------------------------
    # Từ đây xuống: GIỐNG HỆT code gốc, không thay đổi gì
    # ------------------------------------------------------------------

    def parse(self, file_path, document_id, original_filename):
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            result = self._parse_with_docling(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(f"Unsupported file type: {suffix}.")

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[docling/v1] Parsed {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    def _parse_with_docling(self, file_path, document_id, original_filename):
        converter = self._get_converter()
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        images, pic_url_list = self._extract_images_with_urls(doc, document_id)
        tables = self._extract_tables(doc, document_id)
        if settings.HEALERRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        markdown = self._export_markdown(doc)
        markdown = self._inject_image_references(markdown, pic_url_list)
        markdown = self._inject_table_captions(markdown, tables)

        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=markdown,
            page_count=page_count,
            chunks=chunks,
            images=images,
            tables=tables,
            tables_count=len(tables),
        )

    def _chunk_document(self, doc, document_id, original_filename, images=None, tables=None):
        from docling_core.transforms.chunker import HybridChunker
        chunker = HybridChunker(max_tokens=settings.HEALERRAG_CHUNK_MAX_TOKENS, merge_peers=True)

        page_images = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        page_tables = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        assigned_images = set()
        assigned_tables = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            page_no = 0
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "page"):
                    page_no = chunk.meta.page or 0
                elif hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        if hasattr(item, "prov") and item.prov:
                            for prov in item.prov:
                                if hasattr(prov, "page_no"):
                                    page_no = prov.page_no or 0
                                    break
                            if page_no > 0:
                                break

            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            has_table = has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower(): has_table = True
                        if "code" in label.lower(): has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            chunk_image_refs = []
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = [
                    f"[Image on page {img_by_id[id].page_no}]: {img_by_id[id].caption}"
                    for id in chunk_image_refs
                    if img_by_id.get(id) and img_by_id[id].caption
                ]
                if desc_parts:
                    enriched_text = chunk_text + "\n\n" + "\n".join(desc_parts)

            chunk_table_refs = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = [
                    f"[Table on page {tbl_by_id[id].page_no} "
                    f"({tbl_by_id[id].num_rows}x{tbl_by_id[id].num_cols})]: {tbl_by_id[id].caption}"
                    for id in chunk_table_refs
                    if tbl_by_id.get(id) and tbl_by_id[id].caption
                ]
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text, chunk_index=i,
                source_file=original_filename, document_id=document_id,
                page_no=page_no, heading_path=heading_path,
                image_refs=chunk_image_refs, table_refs=chunk_table_refs,
                has_table=has_table, has_code=has_code, contextualized=contextualized,
            ))
        return chunks

    def _export_markdown(self, doc):
        try:
            return doc.export_to_markdown(page_break_placeholder="\n\n---\n\n")
        except TypeError:
            return doc.export_to_markdown()

    def _extract_images_with_urls(self, doc, document_id):
        if not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION:
            return [], []
        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        images = []
        pic_to_image_idx = []
        picture_count = 0
        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []
        for pic in doc.pictures:
            if picture_count >= settings.HEALERRAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1); continue
            image_id = str(uuid.uuid4())
            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"): page_no = prov.page_no or 0; break
            try:
                image_path = images_dir / f"{image_id}.png"
                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1); continue
                else:
                    pic_to_image_idx.append(-1); continue
                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")
                images.append(ExtractedImage(
                    image_id=image_id, document_id=document_id,
                    page_no=page_no, file_path=str(image_path),
                    caption=caption, width=width, height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}")
                pic_to_image_idx.append(-1)
        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)
        pic_url_list = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))
        return images, pic_url_list

    def _inject_image_references(self, markdown, pic_url_list):
        if not pic_url_list:
            return markdown
        pic_iter = iter(pic_url_list)
        def replacer(match):
            try:
                caption, url = next(pic_iter)
                if url:
                    safe = " ".join(caption.replace("[","").replace("]","").split())
                    return f"\n![{safe}]({url})\n"
                return ""
            except StopIteration:
                return ""
        return re.sub(r'<!--\s*image\s*-->', replacer, markdown)

    def _extract_tables(self, doc, document_id):
        if not hasattr(doc, "tables") or not doc.tables:
            return []
        tables = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())
            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"): page_no = prov.page_no or 0; break
            try:
                content_md = table.export_to_markdown(doc)
            except Exception:
                content_md = ""
            if not content_md.strip():
                continue
            num_rows = num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0
            tables.append(ExtractedTable(
                table_id=table_id, document_id=document_id,
                page_no=page_no, content_markdown=content_md,
                num_rows=num_rows, num_cols=num_cols,
            ))
        return tables

    def _parse_legacy(self, file_path, document_id, original_filename):
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker
        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content, source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )
        chunks = [
            EnrichedChunk(content=tc.content, chunk_index=tc.chunk_index,
                          source_file=original_filename, document_id=document_id, page_no=0)
            for tc in text_chunks
        ]
        return ParsedDocument(
            document_id=document_id, original_filename=original_filename,
            markdown=loaded.content, page_count=loaded.page_count,
            chunks=chunks, images=[], tables_count=0,
        )
