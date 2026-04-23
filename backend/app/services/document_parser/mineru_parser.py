"""
MinerU Document Parser
======================

Document parser backed by the official MinerU CLI/tooling.

Implementation notes:
- Uses the official `mineru` command to parse full documents.
- Keeps MinerU config, Hugging Face cache, and local model snapshots inside
  the backend project tree for easier operational control.
- Defaults to the official MinerU2.5-Pro-2604-1.2B VLM model when using the
  VLM backend.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from app.core.config import settings
from app.services.document_parser.base import BaseDocumentParser
from app.services.models.parsed_document import (
    ExtractedImage,
    ExtractedTable,
    EnrichedChunk,
    ParsedDocument,
)

logger = logging.getLogger(__name__)

_MINERU_EXTENSIONS = {
    ".pdf",
    ".docx",
    ".pptx",
    ".xlsx",
    ".png",
    ".jpeg",
    ".jpg",
    ".bmp",
    ".tiff",
    ".tif",
    ".gif",
    ".webp",
}
_LEGACY_EXTENSIONS = {".txt", ".md"}
_MINERU_CONFIG_TEMPLATE = {
    "bucket_info": {},
    "latex-delimiter-config": {
        "display": {"left": "$$", "right": "$$"},
        "inline": {"left": "$", "right": "$"},
    },
    "llm-aided-config": {
        "title_aided": {
            "api_key": "",
            "base_url": "",
            "model": "",
            "enable_thinking": False,
            "enable": False,
        }
    },
}
_MINERU_BACKENDS = {
    "pipeline",
    "hybrid-auto-engine",
    "hybrid-http-client",
    "vlm-auto-engine",
    "vlm-http-client",
}


class MineruExecutionError(RuntimeError):
    """Raised when the MinerU CLI exits unsuccessfully."""

    def __init__(self, return_code: int, stderr: str):
        super().__init__(
            f"MinerU command failed with return code {return_code}: {stderr.strip()}"
        )
        self.return_code = return_code
        self.stderr = stderr


class MineruDocumentParser(BaseDocumentParser):
    """
    Document parser powered by the official MinerU CLI.

    Default strategy:
    - Use MinerU VLM backend for the primary parse flow.
    - Keep model snapshot, HF cache, and MinerU config under backend/models.
    - Fall back to legacy TXT/MD parsing for plain text inputs.
    """

    parser_name = "mineru"

    def __init__(self, workspace_id: int, output_dir: Optional[Path] = None):
        super().__init__(workspace_id, output_dir)
        self.models_root = settings.HEALERRAG_MINERU_MODELS_DIR
        self.hf_cache_dir = self.models_root / ".hf-cache"
        self.config_path = self.models_root / "mineru.json"
        self.vlm_model_dir = self.models_root / "MinerU2.5-Pro-2604-1.2B"

    @staticmethod
    def supported_extensions() -> set[str]:
        return _MINERU_EXTENSIONS | _LEGACY_EXTENSIONS

    def parse(
        self,
        file_path: str | Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        path = Path(file_path)
        suffix = path.suffix.lower()
        start_time = time.time()

        if suffix in _MINERU_EXTENSIONS:
            result = self._parse_with_mineru(path, document_id, original_filename)
        elif suffix in _LEGACY_EXTENSIONS:
            result = self._parse_legacy(path, document_id, original_filename)
        else:
            raise ValueError(
                f"Unsupported file type: {suffix}. "
                f"Supported: {_MINERU_EXTENSIONS | _LEGACY_EXTENSIONS}"
            )

        elapsed_ms = int((time.time() - start_time) * 1000)
        logger.info(
            "[mineru] Parsed document %s (%s) in %sms: %s pages, %s chunks, %s images, %s tables",
            document_id,
            original_filename,
            elapsed_ms,
            result.page_count,
            len(result.chunks),
            len(result.images),
            result.tables_count,
        )
        return result

    def ensure_local_vlm_model(self) -> Path:
        """Ensure the official MinerU2.5-Pro model exists inside the project."""
        if self._is_local_model_ready(self.vlm_model_dir):
            return self.vlm_model_dir

        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                "huggingface_hub is required to download MinerU models locally. "
                "Install backend dependencies first."
            ) from exc

        self.models_root.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            "Downloading MinerU VLM model %s to %s",
            settings.HEALERRAG_MINERU_MODEL_ID,
            self.vlm_model_dir,
        )
        snapshot_download(
            repo_id=settings.HEALERRAG_MINERU_MODEL_ID,
            local_dir=str(self.vlm_model_dir),
            cache_dir=str(self.hf_cache_dir),
        )
        return self.vlm_model_dir

    def _parse_with_mineru(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
        *,
        backend: str | None = None,
        source: str | None = None,
        lang: str | None = None,
        device: str | None = None,
        backend_url: str | None = None,
        formula: bool = True,
        table: bool = True,
    ) -> ParsedDocument:
        """Parse supported documents with MinerU and map outputs to ParsedDocument."""
        backend_name = (backend or settings.HEALERRAG_MINERU_BACKEND).strip().lower()
        source_name = (source or settings.HEALERRAG_MINERU_SOURCE).strip().lower()

        if backend_name not in _MINERU_BACKENDS:
            raise ValueError(
                f"Unsupported MinerU backend: {backend_name}. "
                f"Supported: {sorted(_MINERU_BACKENDS)}"
            )

        run_dir = self.output_dir / "runs" / f"doc_{document_id}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)

        env = self._build_mineru_env(source_name, backend_name)
        self._run_mineru_command(
            input_path=file_path,
            output_dir=run_dir,
            backend=backend_name,
            lang=lang,
            device=device,
            source=source_name,
            backend_url=backend_url,
            formula=formula,
            table=table,
            env=env,
        )

        markdown, content_list, content_list_path = self._read_output_files(
            run_dir, file_path.stem
        )
        images = self._extract_images_from_content_list(
            content_list,
            base_dir=content_list_path.parent if content_list_path else run_dir,
            document_id=document_id,
        )
        tables = self._extract_tables_from_content_list(content_list, document_id)

        if settings.HEALERRAG_ENABLE_IMAGE_CAPTIONING and images:
            self._caption_images(images)

        page_count = self._count_pages_from_content_list(content_list, markdown)
        chunks = self._build_chunks_from_content_list(
            content_list=content_list,
            document_id=document_id,
            original_filename=original_filename,
            images=images,
            tables=tables,
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

    def _build_mineru_env(self, source: str, backend_name: str) -> dict[str, str]:
        """Build a controlled MinerU environment rooted in the backend project."""
        env = os.environ.copy()
        self.models_root.mkdir(parents=True, exist_ok=True)
        self.hf_cache_dir.mkdir(parents=True, exist_ok=True)

        config = dict(_MINERU_CONFIG_TEMPLATE)
        config["models-dir"] = {
            "pipeline": settings.HEALERRAG_MINERU_PIPELINE_MODEL_DIR,
            "vlm": str(self.vlm_model_dir),
        }
        config["config_version"] = settings.HEALERRAG_MINERU_CONFIG_VERSION
        self.config_path.write_text(
            json.dumps(config, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        env["MINERU_TOOLS_CONFIG_JSON"] = str(self.config_path)
        env["MINERU_MODEL_SOURCE"] = source
        env["HF_HOME"] = str(self.hf_cache_dir)
        env["HF_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        env["HUGGINGFACE_HUB_CACHE"] = str(self.hf_cache_dir / "hub")
        env["TRANSFORMERS_CACHE"] = str(self.hf_cache_dir / "transformers")

        if source == "local":
            self.ensure_local_vlm_model()
            if (
                backend_name in {"pipeline", "hybrid-auto-engine", "hybrid-http-client"}
                and not settings.HEALERRAG_MINERU_PIPELINE_MODEL_DIR
            ):
                raise RuntimeError(
                    "MinerU local source for pipeline/hybrid backends requires "
                    "HEALERRAG_MINERU_PIPELINE_MODEL_DIR to be configured. "
                    "For VLM-only parsing, use backend='vlm-auto-engine' or 'vlm-http-client'."
                )

        return env

    @staticmethod
    def _run_mineru_command(
        *,
        input_path: Path,
        output_dir: Path,
        backend: str,
        lang: str | None,
        device: str | None,
        source: str,
        backend_url: str | None,
        formula: bool,
        table: bool,
        env: dict[str, str],
    ) -> None:
        """Run the official MinerU CLI with controlled config and cache paths."""
        cmd = [
            "mineru",
            "-p",
            str(input_path),
            "-o",
            str(output_dir),
            "-b",
            backend,
        ]
        if backend in {"pipeline", "hybrid-auto-engine", "hybrid-http-client"}:
            cmd.extend(["-m", "auto"])
        if lang:
            cmd.extend(["-l", lang])
        if device:
            cmd.extend(["-d", device])
        if backend_url:
            cmd.extend(["-u", backend_url])
        if not formula:
            cmd.extend(["-f", "false"])
        if not table:
            cmd.extend(["-t", "false"])
        if "MINERU_MODEL_SOURCE" not in env:
            cmd.extend(["--source", source])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                env=env,
                check=False,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "The `mineru` CLI is not installed. Install backend dependencies "
                "that include MinerU first."
            ) from exc

        if result.returncode != 0:
            stderr = result.stderr or result.stdout or "unknown MinerU error"
            raise MineruExecutionError(result.returncode, stderr)

        if result.stdout.strip():
            logger.info("MinerU stdout: %s", result.stdout.strip())

    @staticmethod
    def _read_output_files(
        output_dir: Path,
        file_stem: str,
    ) -> tuple[str, list[dict[str, Any]], Path | None]:
        """Read MinerU markdown and content_list outputs from the run directory."""
        markdown_path = MineruDocumentParser._find_first_output_file(
            output_dir, [f"{file_stem}.md"]
        )
        content_list_path = MineruDocumentParser._find_first_output_file(
            output_dir,
            [f"{file_stem}_content_list.json", "content_list.json"],
        )

        markdown = ""
        if markdown_path:
            markdown = markdown_path.read_text(encoding="utf-8")

        content_list: list[dict[str, Any]] = []
        if content_list_path:
            raw_data = json.loads(content_list_path.read_text(encoding="utf-8"))
            if isinstance(raw_data, list):
                content_list = [item for item in raw_data if isinstance(item, dict)]

        return markdown, content_list, content_list_path

    @staticmethod
    def _find_first_output_file(output_dir: Path, names: list[str]) -> Path | None:
        """Search recursively for the first matching output file."""
        candidates: list[Path] = []
        for name in names:
            candidates.extend(output_dir.rglob(name))
        if not candidates:
            return None
        return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0]

    @staticmethod
    def _count_pages_from_content_list(content_list: list[dict[str, Any]], markdown: str) -> int:
        """Count pages from structured output, with a markdown fallback."""
        page_indices = [
            int(item.get("page_idx", 0))
            for item in content_list
            if isinstance(item.get("page_idx"), int)
        ]
        if page_indices:
            return max(page_indices) + 1
        if markdown.strip():
            return 1
        return 0

    @staticmethod
    def _extract_images_from_content_list(
        self,
        content_list: list[dict[str, Any]],
        *,
        base_dir: Path,
        document_id: int,
    ) -> list[ExtractedImage]:
        """Convert MinerU image/chart blocks to ExtractedImage objects."""
        images: list[ExtractedImage] = []
        images_dir = self._get_served_images_dir()
        image_sequence = 1
        for item in content_list:
            if item.get("type") not in {"image", "chart"}:
                continue

            raw_img_path = str(item.get("img_path", "")).strip()
            if not raw_img_path:
                continue

            img_path = Path(raw_img_path)
            if not img_path.is_absolute():
                img_path = (base_dir / raw_img_path).resolve()
            if not img_path.exists():
                logger.debug("Skipping missing MinerU image asset: %s", img_path)
                continue

            width = 0
            height = 0
            image_id = f"doc_{document_id}_image_{image_sequence}"
            served_image_path = images_dir / f"{image_id}.png"
            try:
                from PIL import Image

                with Image.open(img_path) as image:
                    if image.mode in ("RGBA", "P", "LA"):
                        image = image.convert("RGB")
                    image.save(str(served_image_path), format="PNG")
                    width, height = image.size
            except Exception as exc:
                if img_path.suffix.lower() == ".png":
                    shutil.copy2(img_path, served_image_path)
                else:
                    logger.warning(
                        "Failed to normalize MinerU image asset %s: %s",
                        img_path,
                        exc,
                    )
                    continue

            caption_parts: list[str] = []
            for field_name in ("image_caption", "chart_caption", "image_footnote", "chart_footnote"):
                value = item.get(field_name)
                if isinstance(value, list):
                    caption_parts.extend(str(part).strip() for part in value if str(part).strip())
                elif isinstance(value, str) and value.strip():
                    caption_parts.append(value.strip())

            images.append(
                ExtractedImage(
                    image_id=image_id,
                    document_id=document_id,
                    page_no=int(item.get("page_idx", 0) or 0) + 1,
                    file_path=str(served_image_path),
                    caption=" ".join(caption_parts).strip(),
                    width=width,
                    height=height,
                )
            )
            image_sequence += 1
        return images

    @staticmethod
    def _extract_tables_from_content_list(
        content_list: list[dict[str, Any]],
        document_id: int,
    ) -> list[ExtractedTable]:
        """Convert MinerU table blocks to ExtractedTable objects."""
        tables: list[ExtractedTable] = []
        table_sequence = 1
        for item in content_list:
            if item.get("type") != "table":
                continue

            table_body = item.get("table_body", "")
            if isinstance(table_body, list):
                table_content = "\n".join(str(part) for part in table_body)
            elif isinstance(table_body, dict):
                table_content = json.dumps(table_body, ensure_ascii=False, indent=2)
            else:
                table_content = str(table_body or "")

            caption_parts: list[str] = []
            for field_name in ("table_caption", "table_footnote"):
                value = item.get(field_name)
                if isinstance(value, list):
                    caption_parts.extend(str(part).strip() for part in value if str(part).strip())
                elif isinstance(value, str) and value.strip():
                    caption_parts.append(value.strip())

            num_rows = 0
            num_cols = 0
            if isinstance(table_body, list) and table_body and all(isinstance(row, list) for row in table_body):
                num_rows = len(table_body)
                num_cols = max((len(row) for row in table_body if isinstance(row, list)), default=0)

            tables.append(
                ExtractedTable(
                    table_id=f"doc_{document_id}_table_{table_sequence}",
                    document_id=document_id,
                    page_no=int(item.get("page_idx", 0) or 0) + 1,
                    content_markdown=table_content,
                    caption=" ".join(caption_parts).strip(),
                    num_rows=num_rows,
                    num_cols=num_cols,
                )
            )
            table_sequence += 1
        return tables

    @staticmethod
    def _extract_block_text(item: dict[str, Any]) -> str:
        """Extract text-bearing content from a MinerU content_list block."""
        block_type = str(item.get("type", "") or "")
        if block_type in {"text", "equation"}:
            return str(item.get("text", "") or "").strip()
        if block_type == "list":
            list_items = item.get("list_items", [])
            if isinstance(list_items, list):
                return "\n".join(str(part).strip() for part in list_items if str(part).strip())
            return str(list_items or "").strip()
        if block_type == "table":
            parts: list[str] = []
            for field_name in ("table_caption", "table_footnote"):
                value = item.get(field_name)
                if isinstance(value, list):
                    parts.extend(str(part).strip() for part in value if str(part).strip())
                elif isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            body = item.get("table_body", "")
            if isinstance(body, list):
                parts.extend(str(part).strip() for part in body if str(part).strip())
            elif isinstance(body, str) and body.strip():
                parts.append(body.strip())
            return "\n".join(parts).strip()
        if block_type in {"code", "algorithm"}:
            parts = []
            for key in ("code_body", "code_caption", "code_footnote"):
                value = item.get(key)
                if isinstance(value, list):
                    parts.extend(str(part).strip() for part in value if str(part).strip())
                elif isinstance(value, str) and value.strip():
                    parts.append(value.strip())
            return "\n".join(parts).strip()
        return ""

    def _build_chunks_from_content_list(
        self,
        *,
        content_list: list[dict[str, Any]],
        document_id: int,
        original_filename: str,
        images: list[ExtractedImage],
        tables: list[ExtractedTable],
    ) -> list[EnrichedChunk]:
        """Build EnrichedChunk objects from MinerU's flattened reading-order JSON."""
        max_chars = settings.HEALERRAG_CHUNK_MAX_TOKENS * 4
        chunks: list[EnrichedChunk] = []
        heading_stack: list[str] = []

        buffer_parts: list[str] = []
        buffer_page_no = 0
        buffer_heading_path: list[str] = []
        chunk_index = 0

        def flush_buffer() -> None:
            nonlocal buffer_parts, buffer_page_no, buffer_heading_path, chunk_index
            text = "\n\n".join(part for part in buffer_parts if part.strip()).strip()
            if not text:
                buffer_parts = []
                return
            contextualized = ""
            if buffer_heading_path:
                contextualized = " > ".join(buffer_heading_path) + ": " + text[:100]
            chunks.append(
                EnrichedChunk(
                    content=text,
                    chunk_index=chunk_index,
                    source_file=original_filename,
                    document_id=document_id,
                    page_no=buffer_page_no,
                    heading_path=list(buffer_heading_path),
                    has_table="|" in text and "---" in text,
                    has_code="```" in text,
                    contextualized=contextualized,
                )
            )
            chunk_index += 1
            buffer_parts = []

        for item in content_list:
            block_text = self._extract_block_text(item)
            if not block_text:
                continue

            page_no = int(item.get("page_idx", 0) or 0) + 1
            text_level = int(item.get("text_level", 0) or 0)
            is_heading = item.get("type") == "text" and text_level > 0

            if is_heading:
                flush_buffer()
                while len(heading_stack) >= text_level:
                    heading_stack.pop()
                heading_stack.append(block_text)
                buffer_page_no = page_no
                buffer_heading_path = list(heading_stack)
                buffer_parts = [block_text]
                continue

            heading_path = list(heading_stack)
            candidate_parts = buffer_parts + [block_text]
            candidate_text = "\n\n".join(candidate_parts)
            needs_new_chunk = (
                not buffer_parts
                or page_no != buffer_page_no
                or heading_path != buffer_heading_path
                or len(candidate_text) > max_chars
            )

            if needs_new_chunk:
                flush_buffer()
                buffer_page_no = page_no
                buffer_heading_path = heading_path
                buffer_parts = [block_text]
            else:
                buffer_parts.append(block_text)

        flush_buffer()
        return self._enrich_chunks_with_refs(chunks, images, tables)

    @staticmethod
    def _is_local_model_ready(model_dir: Path) -> bool:
        """Check whether a local Hugging Face snapshot looks usable."""
        if not model_dir.exists():
            return False
        return any(model_dir.glob("*.json")) and any(model_dir.rglob("*.safetensors"))

    def _parse_legacy(
        self,
        file_path: Path,
        document_id: int,
        original_filename: str,
    ) -> ParsedDocument:
        """Fallback parser for TXT/MD using the existing lightweight loader."""
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
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                source_file=original_filename,
                document_id=document_id,
                page_no=0,
            )
            for chunk in text_chunks
        ]

        return ParsedDocument(
            document_id=document_id,
            original_filename=original_filename,
            markdown=loaded.content,
            page_count=loaded.page_count,
            chunks=chunks,
            images=[],
            tables=[],
            tables_count=0,
        )

if __name__ == "__main__":
    import argparse
    from dataclasses import asdict

    def build_cli_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=(
                "Run MineruDocumentParser on a local document and export parse artifacts "
                "to an output folder for inspection."
            )
        )
        parser.add_argument(
            "input_path",
            nargs="?",
            help="Path to input document (pdf, docx, pptx, xlsx, image, txt, md).",
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
            "--backend",
            default=settings.HEALERRAG_MINERU_BACKEND,
            choices=sorted(_MINERU_BACKENDS),
            help="Official MinerU backend to use.",
        )
        parser.add_argument(
            "--source",
            default=settings.HEALERRAG_MINERU_SOURCE,
            choices=["local", "huggingface", "modelscope"],
            help="MinerU model source. `local` keeps model files inside backend/models/mineru_models.",
        )
        parser.add_argument(
            "--lang",
            default=None,
            help="OCR language hint for MinerU when applicable.",
        )
        parser.add_argument(
            "--device",
            default=None,
            help="Device hint forwarded to MinerU, e.g. cpu, cuda, cuda:0, mps.",
        )
        parser.add_argument(
            "--backend-url",
            default=None,
            help="OpenAI-compatible URL for vlm-http-client or hybrid-http-client.",
        )
        parser.add_argument(
            "--download-model-only",
            action="store_true",
            help="Download the official MinerU2.5-Pro VLM model locally, then exit.",
        )
        parser.add_argument(
            "--no-formula",
            action="store_true",
            help="Disable formula parsing.",
        )
        parser.add_argument(
            "--no-table",
            action="store_true",
            help="Disable table parsing.",
        )
        parser.add_argument(
            "--log-level",
            default="INFO",
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            help="Logging level for this parser test run.",
        )
        return parser

    def run_cli() -> int:
        args = build_cli_parser().parse_args()
        logging.basicConfig(
            level=getattr(logging, args.log_level),
            format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        )

        output_root = Path(args.output_dir).expanduser().resolve()
        output_root.mkdir(parents=True, exist_ok=True)

        parser = MineruDocumentParser(
            workspace_id=args.workspace_id,
            output_dir=output_root,
        )

        if args.download_model_only:
            parser.ensure_local_vlm_model()
            logger.info("MinerU VLM model is ready at: %s", parser.vlm_model_dir)
            return 0

        if not args.input_path:
            logger.error("input_path is required unless --download-model-only is used.")
            return 1

        input_path = Path(args.input_path).expanduser().resolve()
        if not input_path.exists():
            logger.error("Input file not found: %s", input_path)
            return 1

        run_dir = output_root / f"{input_path.stem}_{int(time.time())}"
        run_dir.mkdir(parents=True, exist_ok=True)

        parser = MineruDocumentParser(
            workspace_id=args.workspace_id,
            output_dir=run_dir,
        )

        try:
            if input_path.suffix.lower() in _MINERU_EXTENSIONS:
                result = parser._parse_with_mineru(
                    file_path=input_path,
                    document_id=args.document_id,
                    original_filename=input_path.name,
                    backend=args.backend,
                    source=args.source,
                    lang=args.lang,
                    device=args.device,
                    backend_url=args.backend_url,
                    formula=not args.no_formula,
                    table=not args.no_table,
                )
            else:
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
            json.dumps(
                [asdict(chunk) for chunk in result.chunks],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        images_path.write_text(
            json.dumps(
                [asdict(image) for image in result.images],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        tables_path.write_text(
            json.dumps(
                [asdict(table) for table in result.tables],
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        summary = {
            "input_path": str(input_path),
            "output_dir": str(run_dir),
            "parser_name": parser.parser_name,
            "workspace_id": args.workspace_id,
            "document_id": args.document_id,
            "backend": args.backend,
            "source": args.source,
            "page_count": result.page_count,
            "chunk_count": len(result.chunks),
            "image_count": len(result.images),
            "table_count": len(result.tables),
            "mineru_config_path": str(parser.config_path),
            "vlm_model_dir": str(parser.vlm_model_dir),
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

    raise SystemExit(run_cli())
