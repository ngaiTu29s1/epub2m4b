from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Chapter:
    index: int
    title: str
    markdown: str
    enabled: bool = True


@dataclass
class BookMetadata:
    title: str
    author: str
    cover_path: Path | None
    chapters: list[Chapter] = field(default_factory=list)


@dataclass
class AudioSegmentSpec:
    text: str
    pause_after_ms: int
    is_chapter_title: bool = False
