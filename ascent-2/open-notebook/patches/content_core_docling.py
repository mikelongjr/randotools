"""
Docling-based document extraction processor.

Patched for ascent-2:
- When DOCLING_VISION_API_URL is set and docling_vision is enabled, picture
  descriptions go to an OpenAI-compatible VLM (Scout) instead of Docling's
  built-in local SmolVLM/Granite models.
- Pin RapidOCR to onnxruntime + English to avoid CUDA-torch CPU segfaults on
  aarch64 (GB10) when onnxruntime would otherwise be missing and fall back.
"""

import os

from content_core.config import ContentCoreConfig
from content_core.common.state import ExtractionOutput

DOCLING_AVAILABLE = False
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        PdfPipelineOptions,
        PictureDescriptionApiOptions,
        RapidOcrOptions,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption

    DOCLING_AVAILABLE = True
except ImportError:
    InputFormat = None  # type: ignore
    PdfPipelineOptions = None  # type: ignore
    PdfFormatOption = None  # type: ignore
    PictureDescriptionApiOptions = None  # type: ignore
    RapidOcrOptions = None  # type: ignore

    class DocumentConverter:  # type: ignore[no-redef]
        """Stub when docling is not installed."""

        def __init__(self, **kwargs):
            raise ImportError(
                "Docling not installed. Install with: pip install content-core[docling] "
                "or use CCORE_DOCUMENT_ENGINE=simple to skip docling."
            )

        def convert(self, source: str):
            raise ImportError(
                "Docling not installed. Install with: pip install content-core[docling] "
                "or use CCORE_DOCUMENT_ENGINE=simple to skip docling."
            )

# Supported MIME types for Docling extraction
DOCLING_SUPPORTED = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "text/markdown",
    # "text/plain", #docling currently not supporting txt
    "text/x-markdown",
    "text/csv",
    "text/html",
    "image/png",
    "image/jpeg",
    "image/tiff",
    "image/bmp",
}


def _scout_picture_options():
    """Build PictureDescriptionApiOptions from env, or None to use Docling defaults."""
    url = (os.environ.get("DOCLING_VISION_API_URL") or "").strip()
    if not url or PictureDescriptionApiOptions is None:
        return None

    model = os.environ.get("DOCLING_VISION_MODEL", "llama4-scout").strip()
    api_key = (os.environ.get("DOCLING_VISION_API_KEY") or "sk-local").strip()
    timeout = float(os.environ.get("DOCLING_VISION_TIMEOUT", "120"))
    max_tokens = int(os.environ.get("DOCLING_VISION_MAX_TOKENS", "256"))
    prompt = os.environ.get(
        "DOCLING_VISION_PROMPT",
        "Describe this image in a few concise sentences. Be accurate and specific.",
    )

    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    return PictureDescriptionApiOptions(
        url=url,
        headers=headers,
        params={
            "model": model,
            "max_tokens": max_tokens,
        },
        prompt=prompt,
        timeout=timeout,
        concurrency=1,
    )


def _rapid_ocr_options():
    """ARM-safe RapidOCR: onnxruntime backend, English."""
    if RapidOcrOptions is None:
        return None
    backend = (os.environ.get("DOCLING_OCR_BACKEND") or "onnxruntime").strip()
    lang_raw = (os.environ.get("DOCLING_OCR_LANG") or "en").strip()
    lang = [x.strip() for x in lang_raw.split(",") if x.strip()] or ["en"]
    return RapidOcrOptions(backend=backend, lang=lang)


async def extract_docling(source: str, config: ContentCoreConfig) -> ExtractionOutput:
    """Extract content using Docling."""
    if DOCLING_AVAILABLE and PdfPipelineOptions is not None:
        use_vision = bool(config.docling_vision)
        api_opts = _scout_picture_options() if use_vision else None

        pipeline_kwargs = dict(
            do_ocr=config.docling_ocr,
            do_formula_enrichment=config.docling_formulas,
            do_picture_description=use_vision,
            # Skip local Granite chart path when using remote Scout captions.
            do_chart_extraction=use_vision and api_opts is None,
        )
        if config.docling_ocr:
            ocr_opts = _rapid_ocr_options()
            if ocr_opts is not None:
                pipeline_kwargs["ocr_options"] = ocr_opts
        if api_opts is not None:
            pipeline_kwargs["enable_remote_services"] = True
            pipeline_kwargs["picture_description_options"] = api_opts

        pipeline_options = PdfPipelineOptions(**pipeline_kwargs)
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
            }
        )
    else:
        converter = DocumentConverter()

    if not source:
        raise ValueError("No input provided for Docling extraction.")

    result = converter.convert(source)
    doc = result.document

    fmt = config.docling_output_format
    if fmt == "html":
        output = doc.export_to_html()
    elif fmt == "json":
        output = doc.export_to_json()
    else:
        output = doc.export_to_markdown()

    return ExtractionOutput(
        content=output,
        source_type="file",
        identified_type="",
        metadata={"docling_format": fmt},
    )
