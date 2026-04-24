"""
Marker Document Parser
======================

Alternative document parser using Marker (marker-pdf) for high-quality
math/formula extraction (LaTeX via Surya), lighter GPU footprint (~2-4GB VRAM),
and broad format support (PDF, DOCX, PPTX, XLSX, EPUB, HTML, images).

Install: ``pip install marker-pdf[full]``

────────────────────────────────────────────────────────────────────────────────
CONFIG ORIGIN LEGEND (dùng trong comment xuyên suốt file):
  [ORIGINAL]  — có trong code gốc, giữ nguyên
  [FIXED]     — có trong code gốc nhưng sai/thiếu, đã sửa
  [NEW]       — config / logic hoàn toàn mới
────────────────────────────────────────────────────────────────────────────────

LLM trong code gốc — tóm tắt:
  Model   : gemini-2.0-flash  (marker.services.gemini.GoogleGeminiService)
  Khi nào : chỉ khi settings.HEALERRAG_MARKER_USE_LLM = True
  Env var  : GOOGLE_API_KEY bắt buộc
  Tác dụng : merge bảng qua nhiều trang, sửa inline math, format bảng phức tạp,
             extract form values, sửa lỗi OCR layout khó
  Code gốc không set custom prompt, không set langs, không detect doc type.
"""
from __future__ import annotations

# ==============================================================================
# STEP 0 — Model cache env vars
# [FIXED] PHẢI đặt trước MỌI import khác.
#
# Lý do code gốc bị sai:
#   1. os.environ được set SAU "from app.* import ..." — huggingface_hub đọc
#      cache path ngay khi import, nên path đã bị freeze về ~/.cache/huggingface
#      trước khi env vars của chúng ta có hiệu lực.
#   2. Thiếu HF_HUB_CACHE (tên mới, ưu tiên cao hơn trong hf >= 0.14).
#   3. TRANSFORMERS_CACHE trỏ subfolder "transformers" riêng → model bị
#      download 2 lần vào 2 nơi khác nhau.
#
# [FIXED] Priority đọc cache (cao → thấp):
#   HF_HUB_CACHE > HUGGINGFACE_HUB_CACHE > HF_HOME/hub > XDG_CACHE_HOME/huggingface/hub
# ==============================================================================
import os

_MARKER_MODEL_BASE = "/home/user/Workspace/healer/rag/backend/models/marker_models"

# [ORIGINAL] GPU selection — giữ nguyên
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

# [FIXED] HF_HOME: biến "master" — khi set, hub cache tự động = <HF_HOME>/hub
_HF_HOME = os.path.join(_MARKER_MODEL_BASE, "hf_cache")
os.environ["HF_HOME"] = _HF_HOME

# [FIXED] Set CẢ HAI tên trỏ cùng một thư mục (mới + cũ backward compat)
_HF_HUB_CACHE = os.path.join(_HF_HOME, "hub")
os.environ["HF_HUB_CACHE"] = _HF_HUB_CACHE           # [NEW] tên mới, ưu tiên cao
os.environ["HUGGINGFACE_HUB_CACHE"] = _HF_HUB_CACHE  # [ORIGINAL] giữ cho backward compat

# [FIXED] Phải trỏ cùng hub dir, KHÔNG phải subfolder "transformers" riêng
os.environ["TRANSFORMERS_CACHE"] = _HF_HUB_CACHE

# [NEW] Datasets cache tách biệt khỏi model cache
os.environ["HF_DATASETS_CACHE"] = os.path.join(_HF_HOME, "datasets")

# [ORIGINAL] DATALAB_CACHE_DIR — giữ nguyên, đây là cache riêng của Marker/Surya
os.environ["DATALAB_CACHE_DIR"] = os.path.join(_MARKER_MODEL_BASE, "datalab_cache")

# [NEW] PyTorch hub cache (timm, torch.hub) — code gốc không set
os.environ["TORCH_HOME"] = os.path.join(_MARKER_MODEL_BASE, "torch_cache")

# [NEW] XDG fallback — last resort cho một số C/Rust lib trong wheel
os.environ["XDG_CACHE_HOME"] = os.path.join(_MARKER_MODEL_BASE, "xdg_cache")

# [NEW] Surya OCR detector thresholds — code gốc không set, dùng Surya default
# BLANK_THRESHOLD: xác suất dưới ngưỡng → khoảng trống giữa dòng (range 0.0–1.0)
# TEXT_THRESHOLD:  xác suất trên ngưỡng → là text, PHẢI > BLANK_THRESHOLD
# Default (0.1 / 0.3) ổn cho PDF sạch. Scan mờ: thử BLANK=0.05, TEXT=0.2
os.environ.setdefault("DETECTOR_BLANK_THRESHOLD", "0.1")
os.environ.setdefault("DETECTOR_TEXT_THRESHOLD", "0.3")

# [ORIGINAL] TORCH_DEVICE — giữ nguyên (Marker auto-detect nếu không set)
os.environ.setdefault("TORCH_DEVICE", "cuda")

# ==============================================================================
# Imports — an toàn sau block env vars
# ==============================================================================
import argparse
import json
import logging
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

# [ORIGINAL] Extension sets — giữ nguyên
_MARKER_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".epub"}
_LEGACY_EXTENSIONS = {".txt", ".md"}

# [ORIGINAL] Page separator — giữ nguyên
_PAGE_SEPARATOR = "-" * 48

# [NEW] Image file extensions → luôn là scanned, không có text layer
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}


class MarkerDocumentParser(BaseDocumentParser):
    """
    Document parser powered by Marker (marker-pdf).

    [ORIGINAL] Features giữ nguyên:
    - Math/formula → LaTeX via Surya
    - GPU footprint ~2-4GB
    - Image extraction, table → markdown, code blocks
    - LLM mode (Gemini gemini-2.0-flash) khi HEALERRAG_MARKER_USE_LLM=True

    [NEW] Additions:
    - Vietnamese language hint (langs=["vi","en"]) cho Surya OCR
    - Auto-detect document profile theo extension + PDF content analysis
    - Converter pool: mỗi profile có converter riêng, dùng chung artifact dict
    - Vietnamese-aware block_correction_prompt khi LLM enabled
    - batch_multiplier tunable từ settings
    """

    parser_name = "marker"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        # [ORIGINAL] artifact dict — giữ nguyên (load một lần, ~2GB)
        self._artifact_dict = None
        # [ORIGINAL → CHANGED] _converter (single) → [NEW] _converter_cache (dict theo profile)
        # Lý do: profile "scanned" cần force_ocr=True, "general" thì không
        # → phải là 2 converter khác nhau nhưng dùng chung artifact dict
        self._converter_cache: dict[str, object] = {}

    @staticmethod
    def supported_extensions() -> set[str]:
        # [ORIGINAL] giữ nguyên
        return _MARKER_EXTENSIONS | _LEGACY_EXTENSIONS

    # ──────────────────────────────────────────────────────────────────────────
    # [NEW] Document Profile Detection
    #
    # Vấn đề: user upload nhiều loại file (pdf/docx/xlsx/csv/txt/md/ảnh...)
    # và không tự chọn loại → hệ thống phải tự xác định config phù hợp.
    #
    # Giải pháp: routing 2 tầng
    #   Tầng 1 — Extension routing (không đọc file, instant):
    #     .docx / .pptx / .html / .epub → "general"   (Marker native, có text)
    #     .xlsx                          → "spreadsheet" (bản chất toàn bảng)
    #     .png / .jpg / .tiff / ...      → "scanned"   (ảnh = luôn OCR)
    #     .txt / .md                     → "general"   (đi qua _parse_legacy)
    #     .pdf                           → sang Tầng 2
    #
    #   Tầng 2 — PDF content analysis (đọc nhẹ qua PyMuPDF, không dùng GPU):
    #     avg_chars/page < 80 VÀ ảnh chiếm > 30% diện tích → "scanned"
    #     ngược lại                                          → "general"
    #
    # Profile map:
    #   "general"     — digital PDF / DOCX / PPTX: text layer có sẵn, Marker extract
    #   "scanned"     — scan/ảnh: không có text layer → force_ocr=True
    #   "spreadsheet" — XLSX: bản chất toàn bảng, tắt image extraction
    #
    # NOTE: "handwriting" KHÔNG auto-detect vì:
    #   - Cần LLM vision để xác nhận → tốn cost
    #   - Surya có thể nhận diện được một số handwriting (cho kết quả trước)
    #   - Opt-in qua settings.HEALERRAG_MARKER_USE_LLM = True (LLM sẽ sửa lại)
    # ──────────────────────────────────────────────────────────────────────────

    def _detect_profile(self, file_path: Path) -> str:
        """
        [NEW] Tự động xác định document profile dựa trên extension + content.

        Không yêu cầu user chọn, không load ML model, chỉ dùng PyMuPDF cho PDF.
        """
        suffix = file_path.suffix.lower()

        # ── Tầng 1: Extension routing ──────────────────────────────────────────
        # Office formats: Marker xử lý native, không cần OCR config đặc biệt
        if suffix in {".docx", ".pptx", ".html", ".epub"}:
            return "general"

        # Excel: bản chất toàn bảng → profile riêng tắt image extraction
        if suffix == ".xlsx":
            return "spreadsheet"

        # File ảnh: không có text layer → luôn OCR
        if suffix in _IMAGE_EXTENSIONS:
            return "scanned"

        # TXT/MD: đi qua _parse_legacy, profile không được dùng
        if suffix in _LEGACY_EXTENSIONS:
            return "general"

        # ── Tầng 2: PDF — phân tích nội dung nhẹ ─────────────────────────────
        if suffix == ".pdf":
            return self._detect_pdf_profile(file_path)

        return "general"  # safe fallback

    def _detect_pdf_profile(self, file_path: Path) -> str:
        """
        [NEW] Phân tích nhẹ PDF (không dùng GPU) để phân biệt digital vs scanned.

        Sample tối đa 3 trang đầu để nhanh. Điều kiện "scanned":
          - avg_chars/page < 80  (text layer rỗng hoặc rất sparse)
          - image_ratio > 0.3    (ảnh chiếm hơn 30% diện tích trang)
        """
        try:
            import fitz  # PyMuPDF — đã là dependency của Marker

            doc = fitz.open(str(file_path))
            sample_count = min(3, len(doc))
            total_chars = 0
            total_image_area = 0.0
            total_page_area = 0.0

            for idx in range(sample_count):
                page = doc[idx]
                total_chars += len(page.get_text().strip())

                rect = page.rect
                total_page_area += rect.width * rect.height

                for img in page.get_images(full=True):
                    try:
                        for r in page.get_image_rects(img[0]):
                            total_image_area += r.width * r.height
                    except Exception:
                        pass  # embedded image không expose rect → skip

            doc.close()

            avg_chars = total_chars / max(sample_count, 1)
            image_ratio = total_image_area / max(total_page_area, 1)

            if avg_chars < 80 and image_ratio > 0.3:
                logger.info(
                    "[marker] PDF profile=scanned "
                    "(avg_chars/page=%.0f, image_ratio=%.2f)",
                    avg_chars, image_ratio,
                )
                return "scanned"

        except Exception as exc:
            logger.warning(
                "[marker] PDF profile detection failed: %s — fallback to 'general'", exc
            )

        return "general"

    # ──────────────────────────────────────────────────────────────────────────
    # [NEW] Config builder — thay thế inline dict của code gốc
    #
    # Code gốc (inline trong _get_converter):
    #   config = {
    #       "output_format": "markdown",           # [ORIGINAL]
    #       "paginate_output": True,               # [ORIGINAL]
    #       "disable_image_extraction": ...,       # [ORIGINAL]
    #   }
    #   if settings.HEALERRAG_MARKER_USE_LLM:
    #       config["use_llm"] = True               # [ORIGINAL]
    #
    # Config MỚI thêm vào:
    #   "langs"                  [NEW] — Vietnamese hint cho Surya OCR
    #   "batch_multiplier"       [NEW] — scale batch size theo VRAM
    #   "force_ocr"              [NEW] — chỉ cho profile "scanned"
    #   "strip_existing_ocr"     [NEW] — cho scanned PDFs có OCR layer cũ lỗi
    #   "block_correction_prompt"[NEW] — Vietnamese prompt khi LLM enabled
    # ──────────────────────────────────────────────────────────────────────────

    def _build_config(self, profile: str) -> dict:
        """[NEW] Build Marker ConfigParser dict theo document profile."""

        # ── Base config — [ORIGINAL] 3 keys giữ nguyên ────────────────────────
        base: dict = {
            "output_format": "markdown",                                          # [ORIGINAL]
            "paginate_output": True,                                              # [ORIGINAL]
            "disable_image_extraction": not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION,  # [ORIGINAL]
        }

        # [NEW] Language hint cho Surya OCR
        # "vi" = kích hoạt Vietnamese diacritics weights trong Surya recognition model
        # "en" = cover từ viết tắt tiếng Anh phổ biến trong tài liệu VN (EBITDA, GDP...)
        base["langs"] = ["vi", "en"]

        # [NEW] Batch multiplier — scale Surya internal batch sizes theo VRAM
        # Default 1 (~3GB VRAM). HEALERRAG_MARKER_BATCH_MULTIPLIER=2 nếu GPU >= 6GB
        base["batch_multiplier"] = getattr(
            settings, "HEALERRAG_MARKER_BATCH_MULTIPLIER", 1
        )

        # ── Profile-specific overrides ─────────────────────────────────────────
        if profile == "scanned":
            # [NEW] force_ocr: bỏ qua text layer PDF hoàn toàn, OCR từ ảnh pixel
            base["force_ocr"] = True
            # strip_existing_ocr=False: nếu có digital text thì giữ, chỉ OCR phần ảnh
            # Set True nếu biết PDF có OCR layer cũ bị lỗi (garbled text)
            base["strip_existing_ocr"] = False

        elif profile == "spreadsheet":
            # [NEW] Excel không có ảnh quan trọng → tắt image extraction để nhanh hơn
            base["disable_image_extraction"] = True

        # ── LLM config — [ORIGINAL] logic giữ nguyên, thêm Vietnamese prompt ──
        if settings.HEALERRAG_MARKER_USE_LLM:
            base["use_llm"] = True  # [ORIGINAL]

            # [NEW] Vietnamese correction prompt — code gốc không có prompt nào
            # LLM dùng prompt này khi sửa lỗi OCR / format lại output
            # Gemini model: gemini-2.0-flash (default của Marker)
            base["block_correction_prompt"] = (
                "Tài liệu này bằng tiếng Việt, có thể lẫn tiếng Anh. "
                "Khi sửa lỗi OCR, hãy giữ nguyên dấu thanh điệu (sắc, huyền, hỏi, ngã, nặng) "
                "và ký tự đặc biệt: ă, â, ê, ô, ơ, ư, đ. "
                "Số liệu tài chính (VNĐ, %, tỷ đồng) phải chính xác tuyệt đối."
            )

        return base

    # ──────────────────────────────────────────────────────────────────────────
    # Lazy initialization
    # ──────────────────────────────────────────────────────────────────────────

    def _load_artifacts(self) -> None:
        """[ORIGINAL] Load shared ML artifact dict — giữ nguyên logic, thêm log."""
        if self._artifact_dict is not None:
            return
        from marker.models import create_model_dict
        logger.info(
            "[marker] Loading ML models | HF_HUB_CACHE=%s | DATALAB_CACHE_DIR=%s",
            os.environ.get("HF_HUB_CACHE", "default"),
            os.environ.get("DATALAB_CACHE_DIR", "default"),
        )
        self._artifact_dict = create_model_dict()
        logger.info("[marker] ML models loaded.")

    def _get_converter(self, profile: str = "general"):
        """
        [ORIGINAL → CHANGED] Code gốc: một self._converter duy nhất.
        [NEW] Converter pool: cache theo profile, dùng chung artifact dict (~2GB).

        Vì sao cần pool:
          - "scanned" cần force_ocr=True → Marker init khác pipeline
          - "general" không cần force_ocr
          → Hai converter khác nhau nhưng artifact dict (model weights) vẫn share
        """
        if profile in self._converter_cache:
            return self._converter_cache[profile]

        from marker.converters.pdf import PdfConverter
        from marker.config.parser import ConfigParser

        self._load_artifacts()

        config = self._build_config(profile)
        logger.info(
            "[marker] Init converter | profile=%r | langs=%s | force_ocr=%s | use_llm=%s",
            profile,
            config.get("langs"),
            config.get("force_ocr", False),
            config.get("use_llm", False),
        )

        config_parser = ConfigParser(config)
        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self._artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        # [ORIGINAL] LLM service attachment — giữ nguyên logic
        # Model mặc định: gemini-2.0-flash (marker.services.gemini.GoogleGeminiService)
        # Cần env var GOOGLE_API_KEY
        if config.get("use_llm"):
            try:
                converter.llm_service = config_parser.get_llm_service()
                logger.info(
                    "[marker] LLM service (Gemini gemini-2.0-flash) attached | profile=%r",
                    profile,
                )
            except Exception as exc:
                logger.warning(
                    "[marker] Failed to attach LLM service | profile=%r | error=%s",
                    profile, exc,
                )

        self._converter_cache[profile] = converter
        return converter

    # ──────────────────────────────────────────────────────────────────────────
    # Main parse entry
    # ──────────────────────────────────────────────────────────────────────────

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """
        [ORIGINAL] Public entry point — giữ nguyên signature để không break callers.
        [NEW] Auto-detect profile bên trong, không expose ra ngoài API.
        """
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _MARKER_EXTENSIONS:
            result = self._parse_with_marker(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_MARKER_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "[marker] Parsed %s (%s) in %dms — pages=%d chunks=%d images=%d tables=%d",
            document_id, original_filename, elapsed_ms,
            result.page_count, len(result.chunks),
            len(result.images), result.tables_count,
        )
        return result

    # ──────────────────────────────────────────────────────────────────────────
    # Marker pipeline
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_with_marker(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """
        [ORIGINAL] Core Marker pipeline — giữ nguyên cấu trúc.
        [NEW] Thêm _detect_profile() + _get_converter(profile) thay cho _get_converter().
        """
        from marker.output import text_from_rendered

        # [NEW] Auto-detect profile — không cần user chọn
        profile = self._detect_profile(file_path)

        converter = self._get_converter(profile)

        logger.info("[marker] Converting %s | profile=%s", file_path.name, profile)
        rendered = converter(str(file_path))
        text, _ext, marker_images = text_from_rendered(rendered)

        # [ORIGINAL] Image pipeline — giữ nguyên
        images = self._save_marker_images(marker_images, document_id)
        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        # [ORIGINAL] Markdown cleanup — giữ nguyên
        markdown = re.sub(r"\n\{(\d+)\}", "", text)
        markdown = self._replace_image_refs_in_markdown(markdown, marker_images, images)

        # [ORIGINAL] Table pipeline — giữ nguyên
        tables = self._extract_tables_from_markdown(markdown, document_id)
        if settings.HEALERRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)
        markdown = self._inject_table_captions(markdown, tables)

        # [ORIGINAL] Chunking — giữ nguyên
        page_count = self._count_pages(markdown)
        chunks = self._chunk_markdown(
            markdown, document_id, original_filename, images, tables
        )

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

    # ──────────────────────────────────────────────────────────────────────────
    # Image handling — [ORIGINAL] giữ nguyên toàn bộ
    # ──────────────────────────────────────────────────────────────────────────

    def _save_marker_images(
        self,
        marker_images: dict,
        document_id: int,
    ) -> list[ExtractedImage]:
        """[ORIGINAL] Save Marker-extracted PIL images to disk."""
        if not marker_images or not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION:
            return []

        images_dir = self._get_served_images_dir()
        images: list[ExtractedImage] = []
        count = 0

        for filename, pil_image in marker_images.items():
            if count >= settings.HEALERRAG_MAX_IMAGES_PER_DOC:
                break
            try:
                image_id = str(uuid.uuid4())
                image_path = images_dir / f"{image_id}.png"

                if pil_image.mode in ("RGBA", "P", "LA"):
                    pil_image = pil_image.convert("RGB")

                pil_image.save(str(image_path), format="PNG")
                width, height = pil_image.size
                page_no = self._extract_page_from_filename(filename)

                images.append(ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=page_no,
                    file_path=str(image_path),
                    caption="",
                    width=width,
                    height=height,
                ))
                count += 1
            except Exception as exc:
                logger.warning("Failed to save Marker image %s: %s", filename, exc)

        logger.info("Saved %d images from document %s", len(images), document_id)
        return images

    @staticmethod
    def _extract_page_from_filename(filename: str) -> int:
        """[ORIGINAL] Extract page number from Marker image filenames."""
        match = re.search(r"page[_-]?(\d+)", filename, re.IGNORECASE)
        return int(match.group(1)) if match else 0

    def _replace_image_refs_in_markdown(
        self,
        markdown: str,
        marker_images: dict,
        images: list[ExtractedImage],
    ) -> str:
        """[ORIGINAL] Replace Marker relative image paths with served static URLs."""
        if not marker_images or not images:
            return markdown

        filenames = list(marker_images.keys())
        for i, img in enumerate(images):
            if i >= len(filenames):
                break
            original_name = filenames[i]
            served_url = (
                f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
            )
            markdown = markdown.replace(f"]({original_name})", f"]({served_url})")

        return markdown

    # ──────────────────────────────────────────────────────────────────────────
    # Table extraction — [ORIGINAL] giữ nguyên toàn bộ
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_tables_from_markdown(
        markdown: str, document_id: int
    ) -> list[ExtractedTable]:
        """[ORIGINAL] Extract Markdown table blocks from paginated output."""
        tables: list[ExtractedTable] = []
        lines = markdown.split("\n")
        current_page = 1
        i = 0

        while i < len(lines):
            line = lines[i]

            if line.strip() == _PAGE_SEPARATOR:
                current_page += 1
                i += 1
                continue

            if line.strip().startswith("|"):
                table_lines = [line]
                while i + 1 < len(lines) and lines[i + 1].strip().startswith("|"):
                    i += 1
                    table_lines.append(lines[i])

                content_md = "\n".join(table_lines)
                data_rows = [
                    ln for ln in table_lines
                    if ln.strip().startswith("|") and "---" not in ln
                ]
                num_rows = max(0, len(data_rows) - 1)
                num_cols = (
                    len([c for c in data_rows[0].split("|") if c.strip()])
                    if data_rows else 0
                )

                if num_rows > 0 or num_cols > 0:
                    tables.append(ExtractedTable(
                        table_id=str(uuid.uuid4()),
                        document_id=document_id,
                        page_no=current_page,
                        content_markdown=content_md,
                        num_rows=num_rows,
                        num_cols=num_cols,
                    ))

            i += 1

        if tables:
            logger.info("Extracted %d tables from Marker markdown", len(tables))
        return tables

    # ──────────────────────────────────────────────────────────────────────────
    # Page counting — [ORIGINAL] giữ nguyên
    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _count_pages(markdown: str) -> int:
        """[ORIGINAL] Count pages from paginated markdown (pages = separators + 1)."""
        if not markdown:
            return 0
        return markdown.count(_PAGE_SEPARATOR) + 1

    # ──────────────────────────────────────────────────────────────────────────
    # Chunking — [ORIGINAL] giữ nguyên toàn bộ
    # ──────────────────────────────────────────────────────────────────────────

    def _chunk_markdown(
        self,
        markdown: str,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """[ORIGINAL] Chunk: page → heading section → token-bounded sub-chunk."""
        pages = markdown.split(_PAGE_SEPARATOR)
        chunks: list[EnrichedChunk] = []
        chunk_index = 0

        for page_idx, page_text in enumerate(pages):
            page_no = page_idx + 1
            page_text = page_text.strip()
            if not page_text:
                continue

            page_text = re.sub(r"^\{(\d+)\}\s*", "", page_text)

            for heading_path, section_text in self._split_by_headings(page_text):
                if not section_text.strip():
                    continue

                for sub_text in self._split_text_by_tokens(
                    section_text,
                    max_tokens=settings.HEALERRAG_CHUNK_MAX_TOKENS,
                ):
                    if not sub_text.strip():
                        continue

                    contextualized = ""
                    if heading_path:
                        contextualized = " > ".join(heading_path) + ": " + sub_text[:100]

                    chunks.append(EnrichedChunk(
                        content=sub_text,
                        chunk_index=chunk_index,
                        source_file=original_filename,
                        document_id=document_id,
                        page_no=page_no,
                        heading_path=heading_path,
                        has_table="|" in sub_text and "---" in sub_text,
                        has_code="```" in sub_text,
                        contextualized=contextualized,
                    ))
                    chunk_index += 1

        chunks = self._enrich_chunks_with_refs(chunks, images, tables)
        return chunks

    @staticmethod
    def _split_by_headings(text: str) -> list[tuple[list[str], str]]:
        """[ORIGINAL] Split page text into (heading_path, section_text) pairs."""
        heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
        matches = list(heading_pattern.finditer(text))

        if not matches:
            return [([], text)]

        sections: list[tuple[list[str], str]] = []
        heading_stack: list[tuple[int, str]] = []

        if matches[0].start() > 0:
            pre_text = text[: matches[0].start()].strip()
            if pre_text:
                sections.append(([], pre_text))

        for i, match in enumerate(matches):
            level = len(match.group(1))
            title = match.group(2).strip()

            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            heading_path = [h[1] for h in heading_stack]

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                sections.append((heading_path, section_text))

        return sections

    @staticmethod
    def _split_text_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
        """[ORIGINAL] Split text by approximate token limit (1 token ≈ 4 chars)."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return [text]

        paragraphs = re.split(r"\n\s*\n", text)
        chunks: list[str] = []
        current = ""

        for para in paragraphs:
            if len(current) + len(para) + 2 > max_chars:
                if current:
                    chunks.append(current.strip())
                if len(para) > max_chars:
                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    current = ""
                    for sent in sentences:
                        if len(current) + len(sent) + 1 > max_chars:
                            if current:
                                chunks.append(current.strip())
                            current = sent
                        else:
                            current = current + " " + sent if current else sent
                else:
                    current = para
            else:
                current = current + "\n\n" + para if current else para

        if current.strip():
            chunks.append(current.strip())

        return chunks if chunks else [text]

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy fallback — [ORIGINAL] giữ nguyên
    # ──────────────────────────────────────────────────────────────────────────

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """[ORIGINAL] Fallback for TXT/MD via legacy loader."""
        from app.services.loader.document_loader import load_document
        from app.services.chunking.chunker import DocumentChunker

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


# ==============================================================================
# [ORIGINAL → REFACTORED] CLI runner
#
# Thay đổi: tách _run_cli() nhận tham số trực tiếp (không parse sys.argv)
# để if __name__ == "__main__" có thể pass config tường minh và dễ đọc.
# ==============================================================================

def _run_cli(
    input_path: Path,
    output_dir: Path,
    workspace_id: int,
    document_id: int,
) -> int:
    """[ORIGINAL → REFACTORED] Execute parser và ghi artifacts ra disk."""
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    run_dir = output_dir / f"{input_path.stem}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Input:              %s", input_path)
    logger.info("Output:             %s", run_dir)
    logger.info("HF_HUB_CACHE:       %s", os.environ.get("HF_HUB_CACHE", "default"))
    logger.info("DATALAB_CACHE_DIR:  %s", os.environ.get("DATALAB_CACHE_DIR", "default"))

    parser_instance = MarkerDocumentParser(
        workspace_id=workspace_id, output_dir=run_dir
    )

    try:
        result = parser_instance.parse(
            file_path=input_path,
            document_id=document_id,
            original_filename=input_path.name,
        )
    except Exception as exc:
        logger.exception("Parse failed: %s", exc)
        return 1

    # Ghi artifacts
    run_dir.joinpath("parsed_markdown.md").write_text(result.markdown, encoding="utf-8")
    run_dir.joinpath("chunks.json").write_text(
        json.dumps([asdict(c) for c in result.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_dir.joinpath("images.json").write_text(
        json.dumps([asdict(i) for i in result.images], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_dir.joinpath("tables.json").write_text(
        json.dumps([asdict(t) for t in result.tables], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    run_dir.joinpath("summary.json").write_text(
        json.dumps(
            {
                "input": str(input_path),
                "output_dir": str(run_dir),
                "page_count": result.page_count,
                "chunk_count": len(result.chunks),
                "image_count": len(result.images),
                "table_count": len(result.tables),
                "env_snapshot": {
                    "HF_HOME": os.environ.get("HF_HOME"),
                    "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE"),
                    "DATALAB_CACHE_DIR": os.environ.get("DATALAB_CACHE_DIR"),
                    "TORCH_HOME": os.environ.get("TORCH_HOME"),
                    "TORCH_DEVICE": os.environ.get("TORCH_DEVICE"),
                    "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    logger.info(
        "Done — pages=%d chunks=%d images=%d tables=%d | dir=%s",
        result.page_count, len(result.chunks),
        len(result.images), len(result.tables), run_dir,
    )
    return 0


# ==============================================================================
#  TEST SECTION
#  Chạy: python marker_document_parser.py
#  Không cần CLI arguments — mọi thứ config tại đây.
#
#  Các dòng cần chỉnh:  tìm comment "← CHỈNH"
#  Các dòng bật/tắt:   bỏ/thêm dấu # ở đầu dòng theo hướng dẫn
# ==============================================================================
if __name__ == "__main__":

    # ── Logging ───────────────────────────────────────────────────────────────
    # Đổi INFO → DEBUG để xem layout bbox, OCR confidence từng bước
    logging.basicConfig(
        level=logging.INFO,                                    # ← CHỈNH nếu cần DEBUG
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )

    # =========================================================================
    # INPUT / OUTPUT
    # =========================================================================
    _INPUT  = Path("/path/to/your/test_document.pdf")         # ← CHỈNH đường dẫn file
    _OUTDIR = Path("/tmp/marker_test_output")                  # ← CHỈNH thư mục output
    _WORKSPACE_ID = 0
    _DOCUMENT_ID  = 1

    # =========================================================================
    # GPU / DEVICE
    # =========================================================================
    # Mặc định đã set CUDA_VISIBLE_DEVICES=1 ở đầu file.
    # Bỏ comment dòng nào cần:

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # ← dùng GPU 0 thay vì GPU 1
    # os.environ["TORCH_DEVICE"] = "cpu"           # ← chạy CPU (không cần GPU, chậm hơn)

    # =========================================================================
    # SURYA OCR THRESHOLDS  [NEW config]
    # Bỏ comment để override giá trị mặc định (0.1 / 0.3)
    # =========================================================================
    # os.environ["DETECTOR_BLANK_THRESHOLD"] = "0.05"  # scan mờ: nhạy hơn với khoảng trắng
    # os.environ["DETECTOR_TEXT_THRESHOLD"]  = "0.2"   # scan mờ: pick up text mờ hơn

    # =========================================================================
    # DOCUMENT PROFILE OVERRIDE  [NEW config]
    # Mặc định: auto-detect (None) — khuyến nghị production
    # Bỏ comment 1 dòng để force profile cụ thể khi test
    # =========================================================================
    _FORCE_PROFILE: Optional[str] = None           # auto-detect
    # _FORCE_PROFILE = "general"                   # PDF digital / DOCX / PPTX
    # _FORCE_PROFILE = "scanned"                   # PDF scan / ảnh chụp
    # _FORCE_PROFILE = "spreadsheet"               # XLSX

    # =========================================================================
    # LLM ENHANCEMENT (Gemini gemini-2.0-flash)  [ORIGINAL config]
    # Mặc định lấy từ settings.HEALERRAG_MARKER_USE_LLM
    # Override tạm cho test — bỏ comment 1 trong 2 dòng
    # =========================================================================
    # Cần: export GOOGLE_API_KEY="your-key" trước khi chạy
    # settings.HEALERRAG_MARKER_USE_LLM = True    # BẬT LLM cho test này
    # settings.HEALERRAG_MARKER_USE_LLM = False   # TẮT LLM (nhanh hơn, không cần API key)

    # =========================================================================
    # IMAGE EXTRACTION  [ORIGINAL config]
    # =========================================================================
    # settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION = True    # BẬT trích ảnh (default)
    # settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION = False   # TẮT — test nhanh hơn

    # =========================================================================
    # BATCH MULTIPLIER  [NEW config]
    # Scale Surya batch size. 1=default(~3GB VRAM), 2=(~5GB), 4=(~9GB)
    # =========================================================================
    # settings.HEALERRAG_MARKER_BATCH_MULTIPLIER = 2

    # =========================================================================
    # MARKER DEBUG MODE  [NEW config]
    # Lưu ảnh layout detection (bbox từng block) vào thư mục debug/
    # Rất hữu ích khi bảng/ảnh bị nhận sai vùng
    # =========================================================================
    # os.environ["DEBUG"] = "true"

    # ── Apply force profile nếu được set ─────────────────────────────────────
    if _FORCE_PROFILE is not None:
        logger.info("Profile override: %r (auto-detect disabled)", _FORCE_PROFILE)
        MarkerDocumentParser._detect_profile = lambda self, fp: _FORCE_PROFILE  # type: ignore[method-assign]

    # ── Run ───────────────────────────────────────────────────────────────────
    raise SystemExit(
        _run_cli(
            input_path=_INPUT,
            output_dir=_OUTDIR,
            workspace_id=_WORKSPACE_ID,
            document_id=_DOCUMENT_ID,
        )
    )






##v1