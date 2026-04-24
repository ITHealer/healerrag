"""
Marker Document Parser
======================
Alternative document parser using Marker (marker-pdf) for high-quality
math/formula extraction (LaTeX via Surya), lighter GPU footprint (~2-4GB VRAM),
and broad format support (PDF, DOCX, PPTX, XLSX, EPUB, HTML, images).
Install: pip install marker-pdf[full]
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Optional

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

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    EnrichedChunk,
    ExtractedImage,
    ExtractedTable,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_MARKER_EXTENSIONS = {".pdf", ".docx", ".pptx", ".xlsx", ".html", ".epub"}
_LEGACY_EXTENSIONS = {".txt", ".md"}
_PAGE_SEPARATOR = "-" * 48

class MarkerDocumentParser(BaseDocumentParser):
    """
    Document parser powered by Marker (marker-pdf).
    """

    parser_name = "marker"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self._artifact_dict = None
        self._converter_cache: dict[str, object] = {}

    @staticmethod
    def supported_extensions() -> set[str]:
        return _MARKER_EXTENSIONS | _LEGACY_EXTENSIONS

    # ------------------------------------------------------------------
    # Lazy initialization
    # ------------------------------------------------------------------
    def _load_artifacts(self) -> None:
        if self._artifact_dict is not None:
            return

        from marker.models import create_model_dict

        logger.info(
            "Loading Marker ML models... HF_HUB_CACHE=%s DATALAB_CACHE_DIR=%s",
            os.environ.get("HF_HUB_CACHE"),
            os.environ.get("DATALAB_CACHE_DIR"),
        )
        self._artifact_dict = create_model_dict()

    def _build_config(self, profile: str) -> dict:
        config = {
            "output_format": "markdown",
            "paginate_output": True,
            "disable_image_extraction": not settings.HEALERRAG_ENABLE_IMAGE_EXTRACTION,
            "langs": ["vi", "en"],
            "batch_multiplier": getattr(settings, "HEALERRAG_MARKER_BATCH_MULTIPLIER", 1),
        }

        if profile == "scanned":
            config["force_ocr"] = True
            config["strip_existing_ocr"] = False
        elif profile == "spreadsheet":
            config["disable_image_extraction"] = True

        if settings.HEALERRAG_MARKER_USE_LLM:
            config["use_llm"] = True

        return config

    def _get_converter(self, profile: str = "general"):
        if profile in self._converter_cache:
            return self._converter_cache[profile]

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter

        self._load_artifacts()
        config = self._build_config(profile)
        config_parser = ConfigParser(config)

        converter = PdfConverter(
            config=config_parser.generate_config_dict(),
            artifact_dict=self._artifact_dict,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
        )

        if config.get("use_llm"):
            try:
                converter.llm_service = config_parser.get_llm_service()
            except Exception as exc:
                logger.warning("Failed to init Marker LLM service for profile %s: %s", profile, exc)

        self._converter_cache[profile] = converter
        return converter

    def _detect_profile(self, file_path: Path) -> str:
        suffix = file_path.suffix.lower()

        if suffix in {".docx", ".pptx", ".html", ".epub"}:
            return "general"
        if suffix == ".xlsx":
            return "spreadsheet"
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}:
            return "scanned"
        if suffix in _LEGACY_EXTENSIONS:
            return "general"
        if suffix == ".pdf":
            return self._detect_pdf_profile(file_path)
        return "general"

    def _detect_pdf_profile(self, file_path: Path) -> str:
        try:
            import fitz

            doc = fitz.open(str(file_path))
            sample_count = min(3, len(doc))
            total_chars = 0
            total_image_area = 0.0
            total_page_area = 0.0

            for idx in range(sample_count):
                page = doc[idx]
                total_chars += len(page.get_text().strip())
                total_page_area += page.rect.width * page.rect.height

                for img in page.get_images(full=True):
                    try:
                        for rect in page.get_image_rects(img[0]):
                            total_image_area += rect.width * rect.height
                    except Exception:
                        continue

            doc.close()

            avg_chars = total_chars / max(sample_count, 1)
            image_ratio = total_image_area / max(total_page_area, 1)
            if avg_chars < 80 and image_ratio > 0.3:
                logger.info(
                    "PDF profile detected as scanned (avg_chars/page=%.0f, image_ratio=%.2f)",
                    avg_chars,
                    image_ratio,
                )
                return "scanned"
        except Exception as exc:
            logger.warning("Failed to detect PDF profile for %s: %s", file_path, exc)

        return "general"

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
            f"[marker] Parsed document {document_id} ({original_filename}) in {elapsed_ms}ms: "
            f"{result.page_count} pages, {len(result.chunks)} chunks, "
            f"{len(result.images)} images, {result.tables_count} tables"
        )
        return result

    # ------------------------------------------------------------------
    # Marker pipeline
    # ------------------------------------------------------------------
    def _parse_with_marker(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Parse with Marker for rich document extraction."""
        from marker.output import text_from_rendered

        profile = self._detect_profile(file_path)
        converter = self._get_converter(profile)
        logger.info("Marker converting: %s | profile=%s", file_path, profile)
        rendered = converter(str(file_path))
        text, _ext, marker_images = text_from_rendered(rendered)

        images = self._save_marker_images(marker_images, document_id)
        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        markdown = re.sub(r"\n\{(\d+)\}", "", text)
        markdown = self._replace_image_refs_in_markdown(markdown, marker_images, images)

        tables = self._extract_tables_from_markdown(markdown, document_id)
        if settings.HEALERRAG_ENABLE_TABLE_CAPTIONING and tables:
            self._caption_tables(tables)
        markdown = self._inject_table_captions(markdown, tables)

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

    # ------------------------------------------------------------------
    # Image handling
    # ------------------------------------------------------------------
    def _save_marker_images(
        self,
        marker_images: dict,
        document_id: int,
    ) -> list[ExtractedImage]:
        """Save Marker-extracted images (PIL) to disk and create ExtractedImage list."""
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
                logger.warning(f"Failed to save Marker image {filename}: {exc}")
                continue

        logger.info(f"Saved {len(images)} Marker images from document {document_id}")
        return images

    @staticmethod
    def _extract_page_from_filename(filename: str) -> int:
        """Try to extract page number from Marker image filenames."""
        match = re.search(r"page[_-]?(\d+)", filename, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return 0

    def _replace_image_refs_in_markdown(
        self,
        markdown: str,
        marker_images: dict,
        images: list[ExtractedImage],
    ) -> str:
        """Replace Marker image filenames in markdown with served URLs."""
        if not marker_images or not images:
            return markdown

        filenames = list(marker_images.keys())
        for i, img in enumerate(images):
            if i < len(filenames):
                original_name = filenames[i]
                served_url = f"/static/doc-images/kb_{self.workspace_id}/images/{img.image_id}.png"
                markdown = markdown.replace(f"]({original_name})", f"]({served_url})")

        return markdown

    # ------------------------------------------------------------------
    # Table extraction from markdown
    # ------------------------------------------------------------------
    @staticmethod
    def _extract_tables_from_markdown(
        markdown: str,
        document_id: int,
    ) -> list[ExtractedTable]:
        """Extract table blocks from markdown output."""
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
                    row for row in table_lines
                    if row.strip().startswith("|") and "---" not in row
                ]
                num_rows = max(0, len(data_rows) - 1)
                num_cols = 0
                if data_rows:
                    num_cols = len([cell for cell in data_rows[0].split("|") if cell.strip()])

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
            logger.info(f"Extracted {len(tables)} tables from Marker markdown")
        return tables

    @staticmethod
    def _count_pages(markdown: str) -> int:
        """Count pages from paginated markdown output."""
        if not markdown:
            return 0
        return markdown.count(_PAGE_SEPARATOR) + 1

    def _chunk_markdown(
        self,
        markdown: str,
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage] | None = None,
        tables: list[ExtractedTable] | None = None,
    ) -> list[EnrichedChunk]:
        """Chunk markdown by page, heading, and token-bounded text blocks."""
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
        """Split page text into sections by markdown headings."""
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
            heading_path = [item[1] for item in heading_stack]

            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            section_text = text[start:end].strip()

            if section_text:
                sections.append((heading_path, section_text))

        return sections

    @staticmethod
    def _split_text_by_tokens(text: str, max_tokens: int = 512) -> list[str]:
        """Split text into chunks respecting approximate token limit."""
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

    # ------------------------------------------------------------------
    # Legacy fallback (TXT/MD)
    # ------------------------------------------------------------------
    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback: parse TXT/MD with legacy loader."""
        from app.services.chunking.chunker import DocumentChunker
        from app.services.loader.document_loader import load_document

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


def _build_cli_parser() -> argparse.ArgumentParser:
    """Build CLI parser for local Marker parsing tests."""
    parser = argparse.ArgumentParser(
        description=(
            "Run MarkerDocumentParser on a local document and export parse artifacts "
            "to an output folder for inspection."
        )
    )
    parser.add_argument(
        "input_path",
        help="Path to input document (pdf, docx, pptx, xlsx, html, epub, txt, md).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination folder to write parser output artifacts.",
    )
    parser.add_argument(
        "--workspace-id",
        type=int,
        default=0,
        help="Workspace ID used to initialize parser context (default: 0).",
    )
    parser.add_argument(
        "--document-id",
        type=int,
        default=1,
        help="Document ID used in output metadata (default: 1).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level for this parser test run.",
    )
    return parser


def _run_cli() -> int:
    """Run parser from CLI and export inspectable artifacts."""
    args = _build_cli_parser().parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    input_path = Path(args.input_path).expanduser().resolve()
    if not input_path.exists():
        logger.error("Input file not found: %s", input_path)
        return 1

    output_root = Path(args.output_dir).expanduser().resolve()
    run_dir = output_root / f"{input_path.stem}_{int(time.time())}"
    run_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running Marker parser on: %s", input_path)
    logger.info("Writing artifacts to: %s", run_dir)

    parser = MarkerDocumentParser(
        workspace_id=args.workspace_id,
        output_dir=run_dir,
    )

    try:
        result = parser.parse(
            file_path=input_path,
            document_id=args.document_id,
            original_filename=input_path.name,
        )
    except Exception as exc:
        logger.exception("Failed to parse document: %s", exc)
        return 1

    markdown_path = run_dir / "parsed_markdown.md"
    chunks_path = run_dir / "chunks.json"
    images_path = run_dir / "images.json"
    tables_path = run_dir / "tables.json"
    summary_path = run_dir / "summary.json"

    markdown_path.write_text(result.markdown, encoding="utf-8")
    chunks_path.write_text(
        json.dumps([asdict(chunk) for chunk in result.chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    images_path.write_text(
        json.dumps([asdict(image) for image in result.images], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    tables_path.write_text(
        json.dumps([asdict(table) for table in result.tables], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    summary = {
        "input_path": str(input_path),
        "output_dir": str(run_dir),
        "parser_name": parser.parser_name,
        "workspace_id": args.workspace_id,
        "document_id": args.document_id,
        "page_count": result.page_count,
        "chunk_count": len(result.chunks),
        "image_count": len(result.images),
        "table_count": len(result.tables),
        "files": {
            "markdown": str(markdown_path),
            "chunks": str(chunks_path),
            "images": str(images_path),
            "tables": str(tables_path),
        },
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(
        "Parse completed: pages=%s chunks=%s images=%s tables=%s",
        result.page_count,
        len(result.chunks),
        len(result.images),
        len(result.tables),
    )
    logger.info("Summary file: %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli())
