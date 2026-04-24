"""
Docling Document Parser
=======================

Wraps the existing Docling-based parsing pipeline (formerly in
``deep_document_parser.py``) behind the ``BaseDocumentParser`` interface.

=== CHANGELOG / CÁC THAY ĐỔI CHÍNH ===

1. [OCR ENGINE - EasyOCR multilingual]
   - Thêm lang=["vi", "en"] để hỗ trợ tiếng Việt lẫn tiếng Anh cùng lúc.
     Vietnamese dùng bộ chữ Latin mở rộng, EasyOCR hỗ trợ ngôn ngữ "vi".
   - Thêm `confidence_threshold=0.4` (giảm từ default 0.5) để bắt được chữ
     viết tay / chữ mờ tốt hơn mà không quá nhiều noise.
   - Thêm `force_full_page_ocr` (configurable) — khi True, toàn bộ trang sẽ
     đi qua OCR thay vì chỉ các bitmap. Quan trọng cho scan / chữ viết tay.
   - `use_gpu=None` → tự detect GPU (CUDA/MPS). Set False để force CPU.
   - `bitmap_area_threshold=0.02` (giảm từ 0.05) → OCR ngay cả bitmap nhỏ
     hơn, tránh bỏ sót chữ trong figure/stamp nhỏ.

2. [TABLE EXTRACTION - TableFormer ACCURATE mode + do_cell_matching]
   - Thay mode FAST → ACCURATE: chậm hơn ~2× nhưng cấu trúc bảng chính xác
     hơn nhiều, đặc biệt bảng merged-cell, no-border, row/col spans.
   - `do_cell_matching=True`: khớp cell text từ PDF với TableFormer predictions
     → tránh re-OCR bảng, giữ được font/encoding gốc.
   - Thử dùng TableStructureV2Options nếu có (V2 có cải tiến cell matching),
     fallback về V1 nếu chưa upgrade docling.

3. [ACCELERATOR - AcceleratorOptions AUTO]
   - Dùng `AcceleratorDevice.AUTO` để tự chọn CUDA/MPS/CPU.
   - `num_threads` configurable qua settings.
   - Khi có CUDA: layout model nhanh 14×, OCR nhanh 8×, table nhanh 4×
     so với CPU (theo Docling benchmark).

4. [PDF BACKEND]
   - `DLPARSE_V2` (default) cho parsing programmatic PDF chính xác.
   - Có thể switch sang `PYPDFIUM2` qua settings nếu cần tốc độ hơn accuracy.
   - `images_scale=2.0` → 216 DPI, giúp OCR bắt được chữ nhỏ.

5. [NEW SETTINGS]
   Thêm vào app/core/config.py (xem phần cuối file):
   - HEALERRAG_OCR_LANGUAGES          : list[str]   = ["vi", "en"]
   - HEALERRAG_OCR_FORCE_FULL_PAGE    : bool         = False
   - HEALERRAG_OCR_CONFIDENCE         : float        = 0.4
   - HEALERRAG_TABLE_MODE             : str          = "accurate"  # "fast" | "accurate"
   - HEALERRAG_USE_TABLE_V2           : bool         = True
   - HEALERRAG_ACCELERATOR_DEVICE     : str          = "auto"      # "auto"|"cpu"|"cuda"|"mps"
   - HEALERRAG_ACCELERATOR_THREADS    : int          = 4
   - HEALERRAG_PDF_BACKEND            : str          = "dlparse_v2"# "dlparse_v2"|"pypdfium2"
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

# File extensions handled by Docling vs legacy
_DOCLING_EXTENSIONS = {".pdf", ".docx", ".pptx", ".html"}
_LEGACY_EXTENSIONS = {".txt", ".md"}


class DoclingDocumentParser(BaseDocumentParser):
    """
    Document parser powered by Docling.

    - Converts PDF/DOCX/PPTX/HTML via Docling DocumentConverter
    - Chunks using HybridChunker (semantic + structural)
    - Extracts images and optionally captions them via LLM Vision
    - Falls back to legacy text extraction for TXT/MD
    - [NEW] EasyOCR với tiếng Việt + Anh, force_full_page_ocr cho handwriting
    - [NEW] TableFormer ACCURATE mode + TableStructureV2 cho bảng phức tạp
    - [NEW] AcceleratorOptions AUTO (CUDA/MPS/CPU auto-detect)
    """

    parser_name = "docling"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._converter = None

    @staticmethod
    def supported_extensions() -> set[str]:
        return _DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # [CHANGED] Converter — toàn bộ phần _get_converter được viết lại
    # ------------------------------------------------------------------

    def _get_converter(self):
        """
        Lazy-init Docling DocumentConverter với đầy đủ OCR + Table options.

        Thứ tự quan trọng (theo docling Discussion #792):
        Phải set ocr_options SAU khi set các flag khác vì EasyOcrOptions()
        constructor reset về default values.
        """
        if self._converter is not None:
            return self._converter

        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions,
            TableStructureOptions,
        )

        # ── [CHANGED] Accelerator: AUTO chọn CUDA/MPS/CPU tự động ──────────
        # Khi có GPU NVIDIA: layout 14×, OCR 8×, table 4× nhanh hơn CPU
        accelerator_options = self._build_accelerator_options()

        # ── [CHANGED] Table structure: dùng ACCURATE thay vì FAST ───────────
        # ACCURATE xử lý được: merged cells, no-border tables, row/col spans
        table_structure_options = self._build_table_structure_options()

        # ── Pipeline base options ────────────────────────────────────────────
        pipeline_options = PdfPipelineOptions()

        # Image extraction (giữ nguyên từ code gốc)
        pipeline_options.generate_picture_images = (
            settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION
        )
        # [CHANGED] scale=2.0 → 216 DPI, giúp OCR bắt chữ nhỏ, watermark, stamp
        # Code gốc dùng settings.HEALERRAG_DOCLING_IMAGES_SCALE, giữ nguyên
        pipeline_options.images_scale = settings.HEALERRAG_DOCLING_IMAGES_SCALE

        # Formula enrichment (giữ nguyên)
        pipeline_options.do_formula_enrichment = (
            settings.HEALERRAG_ENABLE_FORMULA_ENRICHMENT
        )

        # [CHANGED] Bật OCR và table structure tường minh
        pipeline_options.do_ocr = True
        pipeline_options.do_table_structure = True

        # [CHANGED] Gán table options (ACCURATE mode + cell matching)
        pipeline_options.table_structure_options = table_structure_options

        # [CHANGED] Gán accelerator
        pipeline_options.accelerator_options = accelerator_options

        # ── [CHANGED] OCR options — PHẢI set CUỐI CÙNG ──────────────────────
        # Lý do: EasyOcrOptions() sẽ override bất kỳ flag nào set trước đó
        # trên pipeline_options.ocr_options. Luôn khởi tạo object hoàn chỉnh.
        pipeline_options.ocr_options = self._build_ocr_options()

        # ── PDF backend selection ────────────────────────────────────────────
        # [CHANGED] Hỗ trợ switch backend qua settings
        format_options = self._build_format_options(pipeline_options)

        self._converter = DocumentConverter(format_options=format_options)
        logger.info(
            f"[docling] Converter initialized — "
            f"OCR langs={self._get_ocr_languages()}, "
            f"table_mode={getattr(settings, 'HEALERRAG_TABLE_MODE', 'accurate')}, "
            f"accelerator={getattr(settings, 'HEALERRAG_ACCELERATOR_DEVICE', 'auto')}, "
            f"force_full_page_ocr={getattr(settings, 'HEALERRAG_OCR_FORCE_FULL_PAGE', False)}"
        )
        return self._converter

    # ------------------------------------------------------------------
    # [NEW] Helper: build OCR options với multilingual + handwriting support
    # ------------------------------------------------------------------

    def _get_ocr_languages(self) -> list[str]:
        """
        Lấy danh sách ngôn ngữ OCR từ settings.
        Default: ["vi", "en"] — tiếng Việt + tiếng Anh.

        EasyOCR language codes:
          - "en"  : English
          - "vi"  : Vietnamese (Latin-based với diacritics)
          - Có thể thêm "fr", "de", "es"... nếu cần

        Lưu ý: Languages dùng chung script (Latin) có thể kết hợp tốt.
        Vietnamese và English đều Latin → model chạy chung, ít overhead.
        """
        return getattr(settings, "HEALERRAG_OCR_LANGUAGES", ["vi", "en"])

    def _build_ocr_options(self):
        """
        Xây dựng EasyOcrOptions với các tham số tối ưu cho:
        - Tiếng Việt + Anh (Latin script)
        - Chữ viết tay (qua force_full_page_ocr + lower confidence threshold)
        - GPU acceleration (use_gpu=None → auto-detect)

        Tại sao chọn EasyOCR thay vì Tesseract hay RapidOCR?
        - EasyOCR: Tốt nhất cho multilingual mixed-script docs, hỗ trợ 80+
          ngôn ngữ, có GPU acceleration, balance giữa accuracy và speed.
        - Tesseract: Nhanh hơn nhưng kém hơn ở chữ nghiêng/xấu/viết tay.
        - RapidOCR (PP-OCRv5): Nhanh, nhưng model mặc định tập trung Chinese;
          cần custom models cho Vietnamese (xem _build_rapidocr_options).

        Alternative: nếu muốn dùng RapidOCR với PP-OCRv5, uncomment
        _build_rapidocr_options() bên dưới và thay thế ở đây.
        """
        from docling.datamodel.pipeline_options import EasyOcrOptions

        ocr_langs = self._get_ocr_languages()

        # [CHANGED] force_full_page_ocr từ settings (default False)
        # Khi True: mọi trang đều qua OCR hoàn toàn — tốt cho tài liệu scan
        # và chữ viết tay. Khi False: chỉ OCR các vùng bitmap (nhanh hơn).
        force_full_page = getattr(settings, "HEALERRAG_OCR_FORCE_FULL_PAGE", False)

        # [CHANGED] confidence_threshold=0.4 (giảm từ default 0.5)
        # Lower threshold → bắt được chữ mờ/viết tay hơn, đánh đổi một chút noise
        # Với chữ in rõ: 0.5 vẫn tốt. Với scan xấu/viết tay: 0.3-0.4 tốt hơn.
        confidence = getattr(settings, "HEALERRAG_OCR_CONFIDENCE", 0.4)

        return EasyOcrOptions(
            # [CHANGED] Multilingual: Vietnamese + English (Latin script shared)
            lang=ocr_langs,
            # [CHANGED] Auto GPU detection (None = auto). Set False để force CPU.
            use_gpu=None,
            # [CHANGED] Lower confidence cho handwriting / scan chất lượng thấp
            confidence_threshold=confidence,
            # [CHANGED] Force full page OCR — set True nếu tài liệu là scan/handwritten
            force_full_page_ocr=force_full_page,
            # [CHANGED] Giảm bitmap threshold: OCR cả bitmap nhỏ hơn (stamp, chữ ký)
            # Default = 0.05 (5% diện tích trang). Giảm xuống 0.02 để bắt hết.
            bitmap_area_threshold=0.02,
            # download_enabled=True: tự download model nếu chưa có
            download_enabled=True,
        )

    def _build_rapidocr_options(self):
        """
        [ALTERNATIVE] RapidOCR với PP-OCRv5 server models — tốc độ cao hơn,
        phù hợp nếu có tài nguyên và muốn dùng ONNX runtime.

        Cách dùng: thay thế _build_ocr_options() bằng hàm này.

        Yêu cầu: pip install rapidocr_onnxruntime modelscope
        Download models từ HuggingFace: RapidAI/RapidOCR

        Lưu ý: PP-OCRv5 multilingual hỗ trợ Vietnamese qua Latin script.
        Model ch_PP-OCRv5_server bao gồm Latin characters.
        """
        from docling.datamodel.pipeline_options import RapidOcrOptions

        # [COMMENTED] Uncommment và set đúng paths nếu dùng custom models
        # import os
        # from modelscope import snapshot_download
        # download_path = snapshot_download(repo_id="RapidAI/RapidOCR")
        # det_model_path = os.path.join(download_path, "onnx", "PP-OCRv5",
        #                               "det", "ch_PP-OCRv5_server_det.onnx")
        # rec_model_path = os.path.join(download_path, "onnx", "PP-OCRv5",
        #                               "rec", "ch_PP-OCRv5_rec_server_infer.onnx")
        # cls_model_path = os.path.join(download_path, "onnx", "PP-OCRv4",
        #                               "cls", "ch_ppocr_mobile_v2.0_cls_infer.onnx")

        return RapidOcrOptions(
            # backend: "onnxruntime" (default), "openvino" (Intel GPU), "paddle"
            # openvino chạy nhanh hơn EasyOCR+GPU trên Intel iGPU/dGPU
            backend="onnxruntime",
            force_full_page_ocr=getattr(
                settings, "HEALERRAG_OCR_FORCE_FULL_PAGE", False
            ),
            # det_model_path=det_model_path,   # uncomment nếu dùng custom models
            # rec_model_path=rec_model_path,
            # cls_model_path=cls_model_path,
        )

    # ------------------------------------------------------------------
    # [NEW] Helper: build Table structure options (ACCURATE + V2 fallback)
    # ------------------------------------------------------------------

    def _build_table_structure_options(self):
        """
        Xây dựng TableStructureOptions với chế độ ACCURATE.

        TableFormerMode.FAST vs ACCURATE:
        - FAST:     ~400ms/table (GPU L4). Tốt cho bảng đơn giản có border.
        - ACCURATE: ~800ms/table (GPU L4). Tốt hơn cho:
                    * Bảng không có border (no-border tables)
                    * Merged cells (ô gộp hàng/cột)
                    * Row/column spans phức tạp
                    * Bảng với indentation không đều
                    * Nested headers

        do_cell_matching=True: Khớp text cell từ PDF parser với TableFormer
        predictions — tránh phải OCR lại bảng, giữ được text chính xác hơn
        (đặc biệt quan trọng với số liệu và ký tự đặc biệt).

        Thử TableStructureV2Options (improved cell matching) nếu có.
        """
        # [CHANGED] Thử dùng TableStructureV2Options trước (newer, better)
        use_v2 = getattr(settings, "HEALERRAG_USE_TABLE_V2", True)

        if use_v2:
            try:
                from docling.datamodel.pipeline_options import TableStructureV2Options
                logger.info("[docling] Using TableStructureV2Options (improved cell matching)")
                # V2 tự động dùng ACCURATE và có enhanced cell matching logic
                return TableStructureV2Options(do_cell_matching=True)
            except ImportError:
                logger.info(
                    "[docling] TableStructureV2Options not available, "
                    "falling back to V1 ACCURATE mode"
                )

        # [CHANGED] Fallback: TableFormer V1 với ACCURATE mode
        from docling.datamodel.pipeline_options import (
            TableStructureOptions,
            TableFormerMode,
        )

        table_mode_str = getattr(settings, "HEALERRAG_TABLE_MODE", "accurate").lower()
        # Map string → enum
        mode = (
            TableFormerMode.ACCURATE
            if table_mode_str == "accurate"
            else TableFormerMode.FAST
        )

        return TableStructureOptions(
            mode=mode,
            # do_cell_matching: match PDF text cells với table predictions
            # Tắt nếu PDF là scan hoàn toàn (không có text layer)
            do_cell_matching=True,
        )

    # ------------------------------------------------------------------
    # [NEW] Helper: build AcceleratorOptions
    # ------------------------------------------------------------------

    def _build_accelerator_options(self):
        """
        Xây dựng AcceleratorOptions để tận dụng GPU nếu có.

        AcceleratorDevice options:
        - AUTO: Tự chọn tốt nhất (CUDA → MPS → CPU). Khuyến nghị.
        - CUDA: Force NVIDIA GPU. Cần torch+cuda và onnxruntime-gpu.
        - MPS:  Apple Silicon GPU. Note: TableFormer disable MPS tự động.
        - CPU:  Force CPU. Dùng khi debug hoặc không có GPU.

        num_threads: Số CPU threads cho inference. Nên = số physical cores.
        Default 4, có thể tăng lên 8-16 trên server.
        """
        try:
            from docling.datamodel.accelerator_options import (
                AcceleratorDevice,
                AcceleratorOptions,
            )

            device_str = getattr(
                settings, "HEALERRAG_ACCELERATOR_DEVICE", "auto"
            ).lower()
            num_threads = getattr(settings, "HEALERRAG_ACCELERATOR_THREADS", 4)

            # Map string → AcceleratorDevice enum
            device_map = {
                "auto": AcceleratorDevice.AUTO,
                "cpu": AcceleratorDevice.CPU,
                "cuda": AcceleratorDevice.CUDA,
                "mps": AcceleratorDevice.MPS,
            }
            device = device_map.get(device_str, AcceleratorDevice.AUTO)

            return AcceleratorOptions(num_threads=num_threads, device=device)

        except ImportError:
            # Docling version cũ chưa có AcceleratorOptions
            logger.warning(
                "[docling] AcceleratorOptions not available in this docling version. "
                "Upgrade: pip install docling --upgrade"
            )
            return None

    # ------------------------------------------------------------------
    # [NEW] Helper: build format_options với PDF backend selection
    # ------------------------------------------------------------------

    def _build_format_options(self, pipeline_options) -> dict:
        """
        Xây dựng format_options dict cho DocumentConverter.

        PDF Backends:
        - DLPARSE_V2 (default): Docling native parser, tốt nhất cho
          programmatic PDFs, giữ được reading order, formula, layout.
        - PYPDFIUM2: Nhanh hơn, stable hơn với large PDFs, nhưng kém hơn
          về layout analysis. Dùng khi speed > accuracy.
        - DLPARSE_V1: Legacy, không nên dùng.
        """
        from docling.datamodel.base_models import InputFormat
        from docling.document_converter import PdfFormatOption

        backend_str = getattr(
            settings, "HEALERRAG_PDF_BACKEND", "dlparse_v2"
        ).lower()

        # [CHANGED] Support configurable PDF backend
        if backend_str == "pypdfium2":
            try:
                from docling.backend.pypdfium2_backend import PyPdfiumDocumentBackend
                logger.info("[docling] Using PyPdfium2 PDF backend (speed-optimized)")
                return {
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        backend=PyPdfiumDocumentBackend,
                    )
                }
            except ImportError:
                logger.warning(
                    "[docling] PyPdfiumDocumentBackend not available, "
                    "falling back to default DLPARSE_V2"
                )

        # Default: dùng string key "pdf" như code gốc (DLPARSE_V2 là default)
        # Giữ backward compatible với code gốc
        logger.info("[docling] Using DLPARSE_V2 PDF backend (accuracy-optimized)")
        return {
            "pdf": PdfFormatOption(pipeline_options=pipeline_options),
        }

    # ------------------------------------------------------------------
    # Remaining code: giữ nguyên cấu trúc từ code gốc
    # ------------------------------------------------------------------

    @staticmethod
    def is_docling_supported(file_path: str | Path) -> bool:
        """Check if the file format is supported by Docling (not legacy)."""
        return Path(file_path).suffix.lower() in _DOCLING_EXTENSIONS

    # ------------------------------------------------------------------
    # Main parse entry
    # ------------------------------------------------------------------

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _DOCLING_EXTENSIONS:
            result = self._parse_with_docling(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_DOCLING_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            f"[docling] Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    # ------------------------------------------------------------------
    # Docling pipeline (giữ nguyên logic, chỉ thêm table export DataFrame)
    # ------------------------------------------------------------------

    def _parse_with_docling(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse with Docling for rich structural extraction."""
        converter = self._get_converter()

        logger.info(f"Docling converting: {file_path}")
        conv_result = converter.convert(str(file_path))
        doc = conv_result.document

        # Extract images and build URL mapping for markdown references
        images, pic_url_list = self._extract_images_with_urls(doc, document_id)

        # Extract tables
        tables = self._extract_tables(doc, document_id)
        if settings.HEALERRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)

        # Export to markdown
        markdown = self._export_markdown(doc)

        # Post-process: replace image placeholders with real markdown images
        markdown = self._inject_image_references(markdown, pic_url_list)

        # Post-process: inject table captions into markdown
        markdown = self._inject_table_captions(markdown, tables)

        # Get page count
        page_count = 0
        if hasattr(doc, "pages") and doc.pages:
            page_count = len(doc.pages)

        # Chunk with HybridChunker
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

    # ------------------------------------------------------------------
    # Chunking (Docling HybridChunker) — giữ nguyên từ code gốc
    # ------------------------------------------------------------------

    def _chunk_document(
        self,
        doc,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk document using Docling's HybridChunker with image/table enrichment."""
        from docling_core.transforms.chunker import HybridChunker

        chunker = HybridChunker(
            max_tokens=settings.HEALERRAG_CHUNK_MAX_TOKENS,
            merge_peers=True,
        )

        # Build page→images lookup
        page_images: dict[int, list[ExtractedImage]] = {}
        if images:
            for img in images:
                page_images.setdefault(img.page_no, []).append(img)

        # Build page→tables lookup
        page_tables: dict[int, list[ExtractedTable]] = {}
        if tables:
            for tbl in tables:
                page_tables.setdefault(tbl.page_no, []).append(tbl)

        chunks = []
        assigned_images: set[str] = set()
        assigned_tables: set[str] = set()

        for i, chunk in enumerate(chunker.chunk(doc)):
            # Extract page number
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

            # Extract heading path
            heading_path = []
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "headings") and chunk.meta.headings:
                    heading_path = list(chunk.meta.headings)

            # Detect content types
            chunk_text = chunk.text if hasattr(chunk, "text") else str(chunk)
            has_table = False
            has_code = False
            if hasattr(chunk, "meta") and chunk.meta:
                if hasattr(chunk.meta, "doc_items") and chunk.meta.doc_items:
                    for item in chunk.meta.doc_items:
                        label = getattr(item, "label", "") or ""
                        if "table" in label.lower():
                            has_table = True
                        if "code" in label.lower():
                            has_code = True

            contextualized = ""
            if heading_path:
                contextualized = " > ".join(heading_path) + ": " + chunk_text[:100]

            # ── Image-aware enrichment ──
            chunk_image_refs: list[str] = []
            if page_no > 0 and page_no in page_images:
                for img in page_images[page_no]:
                    if img.image_id not in assigned_images:
                        chunk_image_refs.append(img.image_id)
                        assigned_images.add(img.image_id)

            enriched_text = chunk_text
            if chunk_image_refs and images:
                img_by_id = {im.image_id: im for im in images}
                desc_parts = []
                for img_id in chunk_image_refs:
                    img = img_by_id.get(img_id)
                    if img and img.caption:
                        desc_parts.append(
                            f"[Image on page {img.page_no}]: {img.caption}"
                        )
                if desc_parts:
                    enriched_text = chunk_text + "\n\n" + "\n".join(desc_parts)

            # ── Table-aware enrichment ──
            chunk_table_refs: list[str] = []
            if page_no > 0 and page_no in page_tables:
                for tbl in page_tables[page_no]:
                    if tbl.table_id not in assigned_tables:
                        chunk_table_refs.append(tbl.table_id)
                        assigned_tables.add(tbl.table_id)

            if chunk_table_refs and tables:
                tbl_by_id = {t.table_id: t for t in tables}
                tbl_parts = []
                for tbl_id in chunk_table_refs:
                    tbl = tbl_by_id.get(tbl_id)
                    if tbl and tbl.caption:
                        tbl_parts.append(
                            f"[Table on page {tbl.page_no} ({tbl.num_rows}x{tbl.num_cols})]: {tbl.caption}"
                        )
                if tbl_parts:
                    enriched_text = enriched_text + "\n\n" + "\n".join(tbl_parts)

            chunks.append(EnrichedChunk(
                content=enriched_text,
                chunk_index=i,
                source_file=original_filename,
                document_id=document_id,
                page_no=page_no,
                heading_path=heading_path,
                image_refs=chunk_image_refs,
                table_refs=chunk_table_refs,
                has_table=has_table,
                has_code=has_code,
                contextualized=contextualized,
            ))

        if images:
            logger.info(
                f"Image-aware chunking: {len(assigned_images)}/{len(images)} images "
                f"assigned to {len(chunks)} chunks"
            )
        if tables:
            logger.info(
                f"Table-aware chunking: {len(assigned_tables)}/{len(tables)} tables "
                f"assigned to {len(chunks)} chunks"
            )

        return chunks

    # ------------------------------------------------------------------
    # Markdown export (giữ nguyên)
    # ------------------------------------------------------------------

    def _export_markdown(self, doc) -> str:
        """Export document to markdown with page break markers if supported."""
        try:
            return doc.export_to_markdown(
                page_break_placeholder="\n\n---\n\n",
            )
        except TypeError:
            return doc.export_to_markdown()

    # ------------------------------------------------------------------
    # Image extraction (giữ nguyên từ code gốc)
    # ------------------------------------------------------------------

    def _extract_images_with_urls(
        self,
        doc,
        document_id: int,
    ) -> tuple[list[ExtractedImage], list[tuple[str, str]]]:
        """Extract images and build URL mapping for markdown placeholders."""
        if not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION:
            return [], []

        images_dir = self.output_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        images: list[ExtractedImage] = []
        pic_to_image_idx: list[int] = []
        picture_count = 0

        if not hasattr(doc, "pictures") or not doc.pictures:
            return [], []

        for pic in doc.pictures:
            if picture_count >= settings.HEALERRAG_MAX_IMAGES_PER_DOC:
                pic_to_image_idx.append(-1)
                continue

            image_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(pic, "prov") and pic.prov:
                for prov in pic.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            try:
                image_path = images_dir / f"{image_id}.png"

                if hasattr(pic, "image") and pic.image:
                    pil_image = pic.image.pil_image
                    if pil_image:
                        pil_image.save(str(image_path), format="PNG")
                        width, height = pil_image.size
                    else:
                        pic_to_image_idx.append(-1)
                        continue
                else:
                    pic_to_image_idx.append(-1)
                    continue

                caption = ""
                if hasattr(pic, "caption_text"):
                    caption = pic.caption_text(doc) if callable(pic.caption_text) else str(pic.caption_text or "")
                elif hasattr(pic, "text"):
                    caption = str(pic.text or "")

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption=caption,
                    width=width,
                    height=height,
                ))
                pic_to_image_idx.append(len(images) - 1)
                picture_count += 1

            except Exception as e:
                logger.warning(f"Failed to extract image from document {document_id}: {e}")
                pic_to_image_idx.append(-1)
                continue

        logger.info(f"Extracted {len(images)} images from document {document_id}")

        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        pic_url_list: list[tuple[str, str]] = []
        for idx in pic_to_image_idx:
            if idx >= 0:
                img = images[idx]
                url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                pic_url_list.append((img.caption, url))
            else:
                pic_url_list.append(("", ""))

        return images, pic_url_list

    def _inject_image_references(
        self, markdown: str, pic_url_list: list[tuple[str, str]]
    ) -> str:
        """Replace <!-- image --> placeholders with ![caption](url) markdown."""
        placeholder_count = len(re.findall(r"<!--\s*image\s*-->", markdown))

        if not pic_url_list:
            if placeholder_count > 0:
                logger.warning(
                    f"Markdown has {placeholder_count} image placeholders but "
                    f"pic_url_list is empty — images will NOT be injected"
                )
            return markdown

        logger.info(
            f"Injecting {len(pic_url_list)} image URLs into "
            f"{placeholder_count} placeholders"
        )

        injected = 0
        pic_iter = iter(pic_url_list)

        def replacer(match):
            nonlocal injected
            try:
                caption, url = next(pic_iter)
                if url:
                    safe_caption = caption.replace("[", "").replace("]", "")
                    safe_caption = " ".join(safe_caption.split())
                    injected += 1
                    return f"\n![{safe_caption}]({url})\n"
                return ""
            except StopIteration:
                return ""

        result = re.sub(r'<!--\s*image\s*-->', replacer, markdown)
        logger.info(f"Injected {injected}/{placeholder_count} image references")
        return result

    # ------------------------------------------------------------------
    # [CHANGED] Table extraction — thêm DataFrame export backup
    # ------------------------------------------------------------------

    def _extract_tables(self, doc, document_id: int) -> list[ExtractedTable]:
        """
        Extract tables from Docling document.

        [CHANGED] Thêm fallback export_to_dataframe() nếu export_to_markdown()
        thất bại. DataFrame → markdown đảm bảo bảng phức tạp vẫn được capture.
        """
        if not hasattr(doc, "tables") or not doc.tables:
            return []

        tables: list[ExtractedTable] = []
        for table in doc.tables:
            table_id = str(uuid.uuid4())

            page_no = 0
            if hasattr(table, "prov") and table.prov:
                for prov in table.prov:
                    if hasattr(prov, "page_no"):
                        page_no = prov.page_no or 0
                        break

            # [CHANGED] Primary: export_to_markdown (dùng TableFormer structure)
            content_md = ""
            try:
                content_md = table.export_to_markdown(doc)
            except Exception as e:
                logger.warning(
                    f"export_to_markdown failed for table on page {page_no}: {e}"
                )

            # [CHANGED] Fallback: export qua DataFrame nếu markdown rỗng
            if not content_md.strip():
                try:
                    import pandas as pd
                    df = table.export_to_dataframe(doc=doc)
                    if df is not None and not df.empty:
                        content_md = df.to_markdown(index=False)
                        logger.debug(
                            f"Used DataFrame fallback for table on page {page_no} "
                            f"({df.shape[0]}r × {df.shape[1]}c)"
                        )
                except Exception as e2:
                    logger.warning(
                        f"DataFrame fallback also failed for table on page {page_no}: {e2}"
                    )

            if not content_md.strip():
                continue

            num_rows = 0
            num_cols = 0
            if hasattr(table, "data") and table.data:
                num_rows = getattr(table.data, "num_rows", 0) or 0
                num_cols = getattr(table.data, "num_cols", 0) or 0

            tables.append(ExtractedTable(
                table_id=table_id,
                document_id=document_id,
                page_no=page_no,
                content_markdown=content_md,
                num_rows=num_rows,
                num_cols=num_cols,
            ))

        logger.info(f"Extracted {len(tables)} tables from document {document_id}")
        return tables

    # ------------------------------------------------------------------
    # Legacy fallback (TXT/MD) — giữ nguyên
    # ------------------------------------------------------------------

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD with legacy loader."""
        from app.services.document_loader import load_document
        from app.services.chunker import DocumentChunker

        loaded = load_document(str(file_path))
        chunker = DocumentChunker(chunk_size=500, chunk_overlap=50)
        text_chunks = chunker.split_text(
            text=loaded.content,
            source=original_filename,
            extra_metadata={"document_id": document_id, "file_type": loaded.file_type},
        )

        chunks = [
            EnrichedChunk(
                content=tc.content,
                chunk_index=tc.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for tc in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables_count=0,
        )


# ===========================================================================
# CONFIG SNIPPET — Thêm vào app/core/config.py
# ===========================================================================
#
# class Settings(BaseSettings):
#     ...
#     # ── Docling OCR ──────────────────────────────────────────────────────
#     # Danh sách ngôn ngữ cho EasyOCR. Vietnamese + English là mặc định.
#     # Xem language codes: https://www.jaided.ai/easyocr/
#     HEALERRAG_OCR_LANGUAGES: list[str] = Field(
#         default=["vi", "en"],
#         description="EasyOCR language codes. vi=Vietnamese, en=English."
#     )
#
#     # Khi True: toàn bộ trang đi qua OCR (tốt cho scan/handwriting).
#     # Khi False: chỉ OCR các vùng bitmap (nhanh hơn, cho programmatic PDFs).
#     HEALERRAG_OCR_FORCE_FULL_PAGE: bool = Field(
#         default=False,
#         description=(
#             "Force full-page OCR on every page. "
#             "Set True for scanned docs or handwritten content."
#         )
#     )
#
#     # Confidence threshold cho OCR text. Range 0.0–1.0.
#     # 0.4 = bắt được chữ mờ/viết tay. 0.5 = mặc định EasyOCR.
#     HEALERRAG_OCR_CONFIDENCE: float = Field(
#         default=0.4,
#         description="EasyOCR confidence threshold. Lower = more text but more noise."
#     )
#
#     # ── Docling Table ────────────────────────────────────────────────────
#     # "accurate" = TableFormer ACCURATE mode (khuyến nghị cho bảng phức tạp)
#     # "fast"     = FAST mode (nhanh hơn ~2x nhưng kém chính xác hơn)
#     HEALERRAG_TABLE_MODE: str = Field(
#         default="accurate",
#         description='TableFormer mode: "accurate" or "fast".'
#     )
#
#     # Dùng TableStructureV2Options nếu có (improved cell matching).
#     HEALERRAG_USE_TABLE_V2: bool = Field(
#         default=True,
#         description="Use TableStructureV2Options if available."
#     )
#
#     # ── Docling Accelerator ──────────────────────────────────────────────
#     # "auto" = tự chọn CUDA/MPS/CPU (khuyến nghị)
#     # "cuda" = force NVIDIA GPU (cần onnxruntime-gpu)
#     # "cpu"  = force CPU
#     HEALERRAG_ACCELERATOR_DEVICE: str = Field(
#         default="auto",
#         description='Accelerator device: "auto", "cuda", "mps", "cpu".'
#     )
#
#     # Số CPU threads cho inference. Nên = số physical cores.
#     HEALERRAG_ACCELERATOR_THREADS: int = Field(
#         default=4,
#         description="Number of CPU threads for Docling inference."
#     )
#
#     # ── Docling PDF Backend ──────────────────────────────────────────────
#     # "dlparse_v2" = Docling native (accuracy-first, default)
#     # "pypdfium2"  = PyMuPDF-based (speed-first, cho large PDFs)
#     HEALERRAG_PDF_BACKEND: str = Field(
#         default="dlparse_v2",
#         description='PDF backend: "dlparse_v2" (accurate) or "pypdfium2" (fast).'
#     )
#
# ===========================================================================
# REQUIREMENTS — Thêm vào requirements.txt / pyproject.toml
# ===========================================================================
#
# # Core
# docling>=2.0.0                   # Upgrade nếu đang dùng version cũ
# docling-core>=2.0.0
#
# # OCR backends
# easyocr>=1.7.0                   # Vietnamese + English OCR (đã có trong code gốc)
#
# # Optional: RapidOCR alternative (nhanh hơn, cần custom models cho Vietnamese)
# # rapidocr_onnxruntime>=1.4.0
# # modelscope>=1.9.0              # Để download PP-OCRv5 models
#
# # Optional: GPU acceleration (NVIDIA)
# # onnxruntime-gpu>=1.17.0        # Thay onnxruntime thường
#
# # Table export fallback
# pandas>=2.0.0
# tabulate>=0.9.0                  # Cần cho df.to_markdown()
#
# ===========================================================================
# ENVIRONMENT VARIABLES (alternative config via .env)
# ===========================================================================
#
# HEALERRAG_OCR_LANGUAGES=["vi","en"]
# HEALERRAG_OCR_FORCE_FULL_PAGE=false
# HEALERRAG_OCR_CONFIDENCE=0.4
# HEALERRAG_TABLE_MODE=accurate
# HEALERRAG_USE_TABLE_V2=true
# HEALERRAG_ACCELERATOR_DEVICE=auto
# HEALERRAG_ACCELERATOR_THREADS=4
# HEALERRAG_PDF_BACKEND=dlparse_v2
#
# ===========================================================================
# QUICK REFERENCE — Khi nào dùng setting nào?
# ===========================================================================
#
# Tài liệu PDF programmatic (Word → PDF export):
#   HEALERRAG_OCR_FORCE_FULL_PAGE=false  ← nhanh hơn, dùng text layer
#   HEALERRAG_PDF_BACKEND=dlparse_v2     ← chính xác hơn
#
# Tài liệu scan (chụp ảnh, fax, photocopy):
#   HEALERRAG_OCR_FORCE_FULL_PAGE=true   ← OCR toàn trang
#   HEALERRAG_OCR_CONFIDENCE=0.35        ← bắt thêm text mờ
#   HEALERRAG_DOCLING_IMAGES_SCALE=2.0   ← 216 DPI cho chữ nhỏ
#
# Tài liệu có chữ viết tay:
#   HEALERRAG_OCR_FORCE_FULL_PAGE=true
#   HEALERRAG_OCR_CONFIDENCE=0.3         ← thấp hơn nữa
#   HEALERRAG_OCR_LANGUAGES=["vi","en"]  ← giữ nguyên
#
# Bảng phức tạp (merged cells, no-border):
#   HEALERRAG_TABLE_MODE=accurate         ← ACCURATE mode
#   HEALERRAG_USE_TABLE_V2=true           ← V2 nếu có
#
# Môi trường có NVIDIA GPU:
#   HEALERRAG_ACCELERATOR_DEVICE=cuda
#   HEALERRAG_ACCELERATOR_THREADS=8
#   # pip install onnxruntime-gpu (thay onnxruntime)
#
# ===========================================================================