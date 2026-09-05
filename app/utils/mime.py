"""MIME type detection and attachment classification.

Pure-python sniffing of magic bytes with extension-based fallback — no
system-level libraries (e.g. libmagic) required.
"""

from __future__ import annotations

import mimetypes

from app.models.attachments import AttachmentType

# ── extension → MIME type ─────────────────────────────────────────────────────

EXTENSION_MIME: dict[str, str] = {
    # images
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "webp": "image/webp",
    "gif": "image/gif",
    "svg": "image/svg+xml",
    # documents / text
    "txt": "text/plain",
    "md": "text/markdown",
    "markdown": "text/markdown",
    "csv": "text/csv",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "doc": "application/msword",
    "pdf": "application/pdf",
    # spreadsheets
    "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # structured data
    "json": "application/json",
    "xml": "application/xml",
    "yaml": "application/yaml",
    "yml": "application/yaml",
    "toml": "application/toml",
    # source code
    "py": "text/x-python",
    "js": "text/javascript",
    "mjs": "text/javascript",
    "ts": "text/x-typescript",
    "tsx": "text/x-typescript",
    "jsx": "text/javascript",
    "java": "text/x-java",
    "c": "text/x-c",
    "cpp": "text/x-c++",
    "h": "text/x-c",
    "hpp": "text/x-c++",
    "go": "text/x-go",
    "rs": "text/x-rust",
    "rb": "text/x-ruby",
    "php": "text/x-php",
    "sh": "text/x-shellscript",
    "bash": "text/x-shellscript",
    "html": "text/html",
    "css": "text/css",
    "scss": "text/x-scss",
    "sql": "text/x-sql",
    "ini": "text/plain",
    "conf": "text/plain",
    "env": "text/plain",
    "log": "text/plain",
    # audio
    "mp3": "audio/mpeg",
    "wav": "audio/wav",
    "m4a": "audio/mp4",
    "ogg": "audio/ogg",
    "weba": "audio/webm",
    # video
    "mp4": "video/mp4",
    "mov": "video/quicktime",
    "webm": "video/webm",
}

# Extension patterns that also come in uppercase or with leading dots.
_TEXTUAL_TYPES = {
    AttachmentType.text,
    AttachmentType.document,
    AttachmentType.json,
    AttachmentType.spreadsheet,
}


def mime_from_filename(filename: str) -> str | None:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if not ext:
        return None
    return EXTENSION_MIME.get(ext) or mimetypes.guess_type(filename)[0]


# ── magic-byte sniffing ───────────────────────────────────────────────────────


def sniff_mime(data: bytes) -> str | None:
    """Detect common binary formats from their magic bytes."""
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    if data.startswith(b"%PDF"):
        return "application/pdf"
    if data.startswith(b"OggS"):
        return "audio/ogg"
    if data.startswith(b"ID3") or (len(data) > 2 and data.startswith(b"\xff\xfb")):
        return "audio/mpeg"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "audio/wav"
    if data.startswith(b"\x1a\x45\xdf\xa3"):
        return "video/webm"
    if len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"isom", b"mp42", b"avc1", b"m4a "):
            return "audio/mp4" if brand == b"m4a " else "video/mp4"
    if data.startswith(b"PK\x03\x04"):
        return "application/zip"
    return None


def detect_mime(filename: str, declared: str | None = None, data: bytes = b"") -> str:
    """Best-effort MIME detection: magic bytes, then filename, then header."""
    if data:
        sniffed = sniff_mime(data)
        if sniffed:
            return sniffed
    from_ext = mime_from_filename(filename)
    if from_ext:
        return from_ext
    if declared and "/" in declared:
        return declared.split(";")[0].strip()
    return "application/octet-stream"


def classify_attachment(filename: str, mime: str) -> AttachmentType:
    """Map a filename + MIME pair to an :class:`AttachmentType`."""
    mime = (mime or "").split(";")[0].strip().lower()
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

    if mime.startswith("image/") or ext in {"jpg", "jpeg", "png", "webp", "gif", "svg"}:
        return AttachmentType.image
    if mime == "application/pdf" or ext == "pdf":
        return AttachmentType.pdf
    if mime.startswith("audio/") or ext in {"mp3", "wav", "m4a", "ogg", "weba"}:
        return AttachmentType.audio
    if mime.startswith("video/") or ext in {"mp4", "mov", "webm"}:
        return AttachmentType.video
    if mime == "application/json" or ext == "json":
        return AttachmentType.json
    if "spreadsheet" in mime or ext == "xlsx":
        return AttachmentType.spreadsheet
    if mime == "text/plain" or ext in {"txt", "log", "ini", "conf", "env"}:
        return AttachmentType.text
    return AttachmentType.document


def is_textual(attachment_type: AttachmentType) -> bool:
    return attachment_type in _TEXTUAL_TYPES
