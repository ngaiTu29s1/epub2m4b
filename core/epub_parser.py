import re
from pathlib import Path
import ebooklib
from ebooklib import epub
from core.models import BookMetadata, Chapter
from core.text_normalizer import clean_html_to_markdown

_BOILERPLATE_TITLES = (
    'mục lục',
    'table of contents',
    'bản quyền',
    'copyright',
    'cover',
    'bìa sách',
    'thông tin xuất bản',
)


def _is_boilerplate_chapter(title: str, markdown: str) -> bool:
    t_low = title.strip().lower()
    if any(k in t_low for k in _BOILERPLATE_TITLES):
        return True
    body_lines = [line.strip() for line in markdown.splitlines() if not line.startswith('#')]
    body_text = ' '.join(body_lines).strip()
    return len(body_text) < 40


def _extract_cover(book: epub.EpubBook, workspace_dir: Path) -> Path | None:
    workspace_dir.mkdir(parents=True, exist_ok=True)
    for item in book.get_items():
        if item.get_type() in (ebooklib.ITEM_COVER, ebooklib.ITEM_IMAGE):
            name = (item.get_name() or '').lower()
            if item.get_type() == ebooklib.ITEM_COVER or 'cover' in name:
                ext = Path(name).suffix or '.jpg'
                cover_path = workspace_dir / f'cover{ext}'
                cover_path.write_bytes(item.get_content())
                return cover_path
    return None


def parse_epub(epub_path: Path, workspace_dir: Path) -> BookMetadata:
    workspace_dir = Path(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)
    book = epub.read_epub(str(epub_path))

    titles = book.get_metadata('DC', 'title')
    title = titles[0][0] if titles else epub_path.stem

    creators = book.get_metadata('DC', 'creator')
    author = creators[0][0] if creators else 'Không rõ tác giả'

    cover_path = _extract_cover(book, workspace_dir)

    chapters: list[Chapter] = []
    chap_idx = 1

    spine_ids = [item_id for item_id, _ in book.spine]
    seen_ids = set()

    for item_id in spine_ids:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        if isinstance(item, epub.EpubNav):
            continue
        seen_ids.add(item_id)
        raw_html = item.get_content().decode('utf-8', errors='ignore')
        chap_title, md = clean_html_to_markdown(raw_html, default_title=f'Chương {chap_idx}')
        body_check = re.sub(r'^#.*$', '', md, flags=re.MULTILINE).strip()
        if not body_check:
            continue
        enabled = not _is_boilerplate_chapter(chap_title, md)
        chapters.append(
            Chapter(index=chap_idx, title=chap_title, markdown=md, enabled=enabled)
        )
        chap_idx += 1

    return BookMetadata(
        title=title,
        author=author,
        cover_path=cover_path,
        chapters=chapters,
    )


def export_book_markdown(book: BookMetadata, out_md_path: Path) -> Path:
    out_md_path = Path(out_md_path)
    out_md_path.parent.mkdir(parents=True, exist_ok=True)
    enabled_md = [c.markdown.strip() for c in book.chapters if c.enabled and c.markdown.strip()]
    out_md_path.write_text('\n\n'.join(enabled_md) + '\n', encoding='utf-8')
    return out_md_path


def parse_book_markdown(md_text: str) -> list[Chapter]:
    chunks = re.split(r'(?=^# )', md_text.strip(), flags=re.MULTILINE)
    chapters: list[Chapter] = []
    idx = 1
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk:
            continue
        lines = chunk.splitlines()
        first = lines[0].strip()
        if first.startswith('# '):
            title = first[2:].strip()
        else:
            title = f'Chương {idx}'
            chunk = f'# {title}\n\n{chunk}'
        chapters.append(Chapter(index=idx, title=title, markdown=chunk, enabled=True))
        idx += 1
    return chapters
