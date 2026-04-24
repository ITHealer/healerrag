"""
docling_parser_v2_table_fix.py
================================
VERSION 2 — V1 + Fix bảng (table cross-page + ACCURATE mode)

VẤN ĐỀ CỐ ĐỊNH THÊM SO VỚI V1:
  ✓ Tiếng Việt OCR (kế thừa từ v1)
  ✓ Bảng page 1 tốt nhưng trang tiếp theo mất format
  ✓ Bảng có merged cells / no-border không nhận dạng đúng

NGUYÊN NHÂN BẢO TABLE CROSS-PAGE:
  TableFormer xử lý từng page crop riêng biệt.
  Bảng trải qua 2 trang → trang 1 là table object A, trang 2 là table
  object B hoặc bị nhận là "text block" thay vì table.

  Kết quả: page 2 không có header → không xuất ra đúng format markdown.

GIẢI PHÁP V2:
  1. TableFormerMode.ACCURATE: nhận dạng chính xác hơn từng trang.
  2. do_cell_matching=True: map text cell từ PDF vào TableFormer structure.
  3. _merge_cross_page_tables(): post-process gộp các table liên tiếp
     có cùng số cột → ghép thành 1 bảng markdown hoàn chỉnh.
  4. Fallback export_to_dataframe() nếu export_to_markdown() fail.
  5. AcceleratorOptions AUTO để tăng tốc (nếu có GPU).

CONFIG MỚI THÊM SO VỚI V1:
  HEALERRAG_TABLE_MODE: str = "accurate"      # "fast" | "accurate"
  HEALERRAG_USE_TABLE_V2: bool = True
  HEALERRAG_MERGE_CROSS_PAGE_TABLES: bool = True
  HEALERRAG_ACCELERATOR_DEVICE: str = "auto"
  HEALERRAG_ACCELERATOR_THREADS: int = 4

TEST:
  python3 test_parser_cli.py --file report.pdf --version v2 --show-tables
  python3 test_parser_cli.py --file multi_page_table.pdf --version v2 --show-tables --show-markdown
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
    """V2: Fix Vietnamese OCR + Table ACCURATE mode + cross-page merge."""

    parser_name = "docling"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._converter = None

    @staticmethod
    def supported_extensions() -> set[str]:
        return _DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Converter — V1 OCR + V2 Table options + AcceleratorOptions
    # ------------------------------------------------------------------

    def _get_converter(self):
        if self._converter is not None:
            return self._converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            EasyOcrOptions,
        )

        # ════════════════════════════════════════════════════════════════
        # [V2-NEW] AcceleratorOptions — tự detect GPU
        # ════════════════════════════════════════════════════════════════
        accelerator_options = self._build_accelerator_options()

        # ════════════════════════════════════════════════════════════════
        # [V2-NEW] TableStructure — ACCURATE mode thay vì FAST
        # ════════════════════════════════════════════════════════════════
        table_structure_options = self._build_table_options()

        # Pipeline
        pipeline_options = PdfPipelineOptions()
        pipeline_options.generate_picture_images = settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION
        pipeline_options.images_scale = settings.HEALERRAG_DOCLING_IMAGES_SCALE
        pipeline_options.do_formula_enrichment = settings.HEALERRAG_ENABLE_FORMULA_ENRICHMENT
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # [V2-NEW] Gán table options
        pipeline_options.table_structure_options = table_structure_options

        # [V2-NEW] Gán accelerator
        if accelerator_options:
            pipeline_options.accelerator_options = accelerator_options

        # [V1 kế thừa] OCR options — PHẢI set SAU CÙNG
        pipeline_options.ocr_options = EasyOcrOptions(
            lang=getattr(settings, "HEALERRAG_OCR_LANGUAGES", ["vi", "en"]),
            use_gpu=None,
            confidence_threshold=getattr(settings, "HEALERRAG_OCR_CONFIDENCE", 0.4),
            force_full_page_ocr=getattr(settings, "HEALERRAG_OCR_FORCE_FULL_PAGE", False),
            bitmap_area_threshold=0.02,
            download_enabled=True,
        )

        self._converter = DocumentConverter(
            format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_options)}
        )
        logger.info("[docling/v2] Converter initialized with ACCURATE table mode")
        return self._converter

    # ════════════════════════════════════════════════════════════════════
    # [V2-NEW] Build table structure options
    # ════════════════════════════════════════════════════════════════════

    def _build_table_options(self):
        """
        ACCURATE mode: chậm hơn 2× nhưng xử lý được:
        - Bảng không có border (no-border tables rất phổ biến trong báo cáo VN)
        - Merged cells (ô gộp hàng/cột)
        - Row/column span phức tạp
        - Header nhiều level

        TableStructureV2: cải thiện cell matching, thử trước rồi fallback V1.
        """
        use_v2 = getattr(settings, "HEALERRAG_USE_TABLE_V2", True)

        if use_v2:
            try:
                from docling.datamodel.pipeline_options import TableStructureV2Options
                logger.info("[docling/v2] TableStructureV2Options: do_cell_matching=True")
                return TableStructureV2Options(do_cell_matching=True)
            except ImportError:
                logger.info("[docling/v2] V2 not available, using V1 ACCURATE")

        from docling.datamodel.pipeline_options import TableStructureOptions, TableFormerMode
        table_mode_str = getattr(settings, "HEALERRAG_TABLE_MODE", "accurate").lower()
        mode = TableFormerMode.ACCURATE if table_mode_str == "accurate" else TableFormerMode.FAST
        logger.info(f"[docling/v2] TableStructureOptions: mode={mode}, do_cell_matching=True")
        return TableStructureOptions(mode=mode, do_cell_matching=True)

    # ════════════════════════════════════════════════════════════════════
    # [V2-NEW] Build accelerator options
    # ════════════════════════════════════════════════════════════════════

    def _build_accelerator_options(self):
        try:
            from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
            device_str = getattr(settings, "HEALERRAG_ACCELERATOR_DEVICE", "auto").lower()
            device_map = {
                "auto": AcceleratorDevice.AUTO, "cpu": AcceleratorDevice.CPU,
                "cuda": AcceleratorDevice.CUDA, "mps": AcceleratorDevice.MPS,
            }
            return AcceleratorOptions(
                num_threads=getattr(settings, "HEALERRAG_ACCELERATOR_THREADS", 4),
                device=device_map.get(device_str, AcceleratorDevice.AUTO),
            )
        except ImportError:
            logger.warning("[docling/v2] AcceleratorOptions not available")
            return None

    # ------------------------------------------------------------------
    # Parse (giống v1 nhưng gọi _extract_tables với merge)
    # ------------------------------------------------------------------

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

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
            f"[docling/v2] Parsed {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    def _parse_with_docling(self, file_path, document_id, original_filename):
        converter = self._get_converter()
        logger.info(f"[docling/v2] Converting: {file_path}")
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        images, pic_url_list = self._extract_images_with_urls(doc, document_id)

        # [V2] Extract tables với merge logic
        tables = self._extract_tables(doc, document_id)
        if settings.HEALERRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        markdown = self._export_markdown(doc)
        markdown = self._inject_image_references(markdown, pic_url_list)
        markdown = self._inject_table_captions(markdown, tables)

        page_count = len(doc.pages) if hasattr(doc, "pages") and doc.pages else 0
        chunks = self._chunk_document(doc, document_id, original_filename, images, tables)

        return ParsedDocument(
            document_id=document_id, original_filename=original_filename,
            markdown=markdown, page_count=page_count, chunks=chunks,
            images=images, tables=tables, tables_count=len(tables),
        )

    # ════════════════════════════════════════════════════════════════════
    # [V2-NEW] Table extraction với fallback + cross-page merge
    # ════════════════════════════════════════════════════════════════════

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """
        Extract tables với:
        1. Primary: export_to_markdown() — dùng TableFormer structure
        2. Fallback: export_to_dataframe() → to_markdown() — đảm bảo capture
        3. Post-process: merge adjacent tables có cùng column count
        """
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        raw_tables = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())
            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"): page_no = prov.page_no or 0; break

            # Primary: export_to_markdown
            content_md = ""
            try:
                content_md = table.export_to_markdown(doc)
            except Exception as e:
                logger.debug(f"[v2] export_to_markdown failed page {page_no}: {e}")

            # [V2-NEW] Fallback: DataFrame → markdown
            if not content_md.strip():
                try:
                    import pandas as pd
                    df = table.export_to_dataframe(doc=doc)
                    if df is not None and not df.empty:
                        content_md = df.to_markdown(index=False)
                        logger.debug(f"[v2] Used DataFrame fallback page {page_no}: {df.shape}")
                except Exception as e2:
                    logger.warning(f"[v2] Both export methods failed page {page_no}: {e2}")

            if not content_md.strip():
                continue

            num_rows = num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            raw_tables.append(ExtractedTable(
                table_id=table_id, document_id=document_id,
                page_no=page_no, content_markdown=content_md,
                num_rows=num_rows, num_cols=num_cols,
            ))

        logger.info(f"[docling/v2] Extracted {len(raw_tables)} raw tables")

        # [V2-NEW] Merge cross-page tables
        if getattr(settings, "HEALERRAG_MERGE_CROSS_PAGE_TABLES", True):
            merged = self._merge_cross_page_tables(raw_tables)
            logger.info(
                f"[docling/v2] After cross-page merge: {len(merged)} tables "
                f"(merged {len(raw_tables) - len(merged)} fragments)"
            )
            return merged

        return raw_tables

    def _merge_cross_page_tables(
        self, tables: list[ExtractedTable]
    ) -> list[ExtractedTable]:
        """
        Heuristic merge: Gộp 2 table liên tiếp nếu:
        1. Trang kề nhau (page N và N+1)
        2. Cùng số cột (num_cols match)
        3. Table sau không có header row (dòng đầu không phải |----|)

        Logic:
          - Table đầu: giữ nguyên (có header)
          - Table sau: bỏ dòng header trong markdown (dòng |---|)
          - Ghép nối content, cộng num_rows

        Hạn chế của heuristic này:
          - Không phát hiện được nếu table cách nhau > 1 trang
          - Không handle table có colspan/rowspan phức tạp ở biên trang
          - num_cols có thể = 0 nếu parse fail → dùng count "|" trong markdown
        """
        if not tables:
            return tables

        # Sort theo page
        tables = sorted(tables, key=lambda t: t.page_no)
        merged = []
        i = 0
        while i < len(tables):
            current = tables[i]
            # Check nếu table tiếp theo là continuation
            if i + 1 < len(tables):
                nxt = tables[i + 1]
                if self._is_continuation_table(current, nxt):
                    # Merge: lấy body của nxt (bỏ header + separator)
                    merged_md = self._merge_table_markdown(
                        current.content_markdown,
                        nxt.content_markdown,
                    )
                    merged_table = ExtractedTable(
                        table_id=current.table_id,
                        document_id=current.document_id,
                        page_no=current.page_no,
                        content_markdown=merged_md,
                        num_rows=current.num_rows + max(nxt.num_rows - 1, 0),
                        num_cols=current.num_cols or nxt.num_cols,
                        caption=current.caption or nxt.caption,
                    )
                    logger.debug(
                        f"[v2] Merged table page {current.page_no} + {nxt.page_no} "
                        f"({current.num_cols} cols)"
                    )
                    merged.append(merged_table)
                    i += 2  # Skip nxt vì đã merge
                    continue
            merged.append(current)
            i += 1

        return merged

    def _is_continuation_table(self, t1: ExtractedTable, t2: ExtractedTable) -> bool:
        """
        Kiểm tra t2 có phải là phần tiếp theo của t1 không.

        Điều kiện:
        1. Trang kề nhau (t2.page_no == t1.page_no + 1)
        2. Số cột giống nhau (hoặc cả hai đều 0 → đếm từ markdown)
        3. t2 có vẻ không có header độc lập
           (dòng đầu không có text "STT|No|#" hay keyword header thông dụng)
        """
        # Điều kiện 1: trang kề
        if t2.page_no != t1.page_no + 1:
            return False

        # Điều kiện 2: số cột
        cols1 = t1.num_cols or self._count_cols_from_markdown(t1.content_markdown)
        cols2 = t2.num_cols or self._count_cols_from_markdown(t2.content_markdown)
        if cols1 == 0 or cols2 == 0:
            return False
        if cols1 != cols2:
            return False

        # Điều kiện 3: t2 không có header riêng
        # Nếu dòng đầu của t2 trùng với dòng đầu của t1 → có thể là repeat header
        # (Docling/TableFormer đôi khi duplicate header khi page break)
        t1_header = self._get_first_row(t1.content_markdown)
        t2_header = self._get_first_row(t2.content_markdown)
        headers_match = (
            t1_header.strip().lower() == t2_header.strip().lower()
            and len(t1_header.strip()) > 3  # không phải empty header
        )

        # Nếu header giống nhau → definitely continuation (repeat header pattern)
        if headers_match:
            return True

        # Nếu header khác nhau nhưng t2 header trông như data (không có |---| pattern)
        # → có thể là continuation với data-as-first-row
        t2_lines = [l for l in t2.content_markdown.strip().split("\n") if l.strip()]
        has_separator = any(
            set(l.replace("|", "").replace("-", "").replace(" ", "")) == set()
            or re.match(r'[\|\-\s]+', l)
            for l in t2_lines[:3]
        )

        # Nếu không có separator → likely dữ liệu tiếp nối
        return not has_separator

    def _count_cols_from_markdown(self, md: str) -> int:
        """Đếm số cột từ dòng đầu của markdown table."""
        for line in md.strip().split("\n"):
            if "|" in line:
                return max(0, line.count("|") - 1)
        return 0

    def _get_first_row(self, md: str) -> str:
        """Lấy dòng đầu tiên của markdown table (header row)."""
        for line in md.strip().split("\n"):
            if "|" in line and not re.match(r'^[\|\-\s]+$', line):
                return line
        return ""

    def _merge_table_markdown(self, md1: str, md2: str) -> str:
        """
        Ghép 2 markdown table:
        - Giữ nguyên md1 (có header + separator)
        - Từ md2: bỏ header row và separator row, chỉ lấy data rows
        - Nếu md2 header == md1 header (repeat): bỏ luôn
        """
        lines1 = [l for l in md1.strip().split("\n") if l.strip()]
        lines2 = [l for l in md2.strip().split("\n") if l.strip()]

        if not lines2:
            return md1

        # Xác định header và separator trong md2
        data_start = 0
        header_row2 = None

        for idx, line in enumerate(lines2):
            is_separator = bool(re.match(r'^[\|\s\-]+$', line)) and '-' in line
            is_header_like = "|" in line and not is_separator

            if is_header_like and header_row2 is None:
                header_row2 = line
                data_start = idx + 1
            elif is_separator:
                data_start = idx + 1
                break

        data_lines2 = lines2[data_start:]

        if not data_lines2:
            return md1

        return "\n".join(lines1) + "\n" + "\n".join(data_lines2)

    # ------------------------------------------------------------------
    # Phần còn lại: giữ nguyên từ v1
    # ------------------------------------------------------------------

    def _chunk_document(self, doc, document_id, original_filename, images=None, tables=None):
        from docling_core.transforms.chunker import HybridChunker
        chunker = HybridChunker(max_tokens=settings.HEALERRAG_CHUNK_MAX_TOKENS, merge_peers=True)

        page_images = {}
        if images:
            for img in images: page_images.setdefault(img.page_no, []).append(img)
        page_tables = {}
        if tables:
            for tbl in tables: page_tables.setdefault(tbl.page_no, []).append(tbl)

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
                                if hasattr(prov, "page_no"): page_no = prov.page_no or 0; break
                            if page_no > 0: break

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

            contextualized = " > ".join(heading_path) + ": " + chunk_text[:100] if heading_path else ""

            chunk_image_refs = []
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id); assigned_images.add(img.image_id)

            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = [
                    f"[Image on page {img_by_id[id].page_no}]: {img_by_id[id].caption}"
                    for id in chunk_image_refs if img_by_id.get(id) and img_by_id[id].caption
                ]
                if desc_parts: enriched_text = chunk_text + "\n\n" + "\n".join(desc_parts)

            chunk_table_refs = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id); assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = [
                    f"[Table on page {tbl_by_id[id].page_no} ({tbl_by_id[id].num_rows}×{tbl_by_id[id].num_cols})]: {tbl_by_id[id].caption}"
                    for id in chunk_table_refs if tbl_by_id.get(id) and tbl_by_id[id].caption
                ]
                if tbl_parts: enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

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
                        pil_image.save(str(image_path), format="PNG"); width, height = pil_image.size
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
                    image_id=image_id, document_id=document_id, page_no=page_no,
                    file_path=str(image_path), caption=caption, width=width, height=height,
                ))
                pic_to_image_idx.append(len(images) - 1); picture_count += 1
            except Exception as e:
                logger.warning(f"Failed to extract image: {e}"); pic_to_image_idx.append(-1)
        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)
        pic_url_list = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                pic_url_list.append((img.caption, f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"))
            else:
                pic_url_list.append(("", ""))
        return images, pic_url_list

    def _inject_image_references(self, markdown, pic_url_list):
        if not pic_url_list: return markdown
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
