from pathlib import Path
from ebooklib import epub
from core.epub_parser import parse_epub, export_book_markdown, parse_book_markdown


def _make_sample_epub(path: Path) -> Path:
    book = epub.EpubBook()
    book.set_identifier('id123')
    book.set_title('Mắt Biếc Thử Nghiệm')
    book.set_language('vi')
    book.add_author('Nguyễn Nhật Ánh')
    c1 = epub.EpubHtml(title='Chương 1', file_name='chap_01.xhtml', lang='vi')
    c1.content = '<h1>Chương 1: Làng Đo Đo</h1><p>Tuổi thơ tôi gắn liền với làng Đo Đo và những chiều mưa trên đồi sim tím.</p>'
    c2 = epub.EpubHtml(title='Mục lục', file_name='toc.xhtml', lang='vi')
    c2.content = '<h1>Mục lục</h1><p>Trang 1</p>'
    book.add_item(c1)
    book.add_item(c2)
    book.spine = ['nav', c1, c2]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub.write_epub(str(path), book)
    return path


def test_parse_epub_and_roundtrip_markdown(tmp_path: Path):
    epub_file = _make_sample_epub(tmp_path / 'sample.epub')
    meta = parse_epub(epub_file, tmp_path / 'work')
    assert meta.title == 'Mắt Biếc Thử Nghiệm'
    assert meta.author == 'Nguyễn Nhật Ánh'
    assert len(meta.chapters) == 2
    assert meta.chapters[0].enabled is True
    assert meta.chapters[1].enabled is False  # Auto-unselect short TOC/boilerplate

    md_path = export_book_markdown(meta, tmp_path / 'work' / 'book.md')
    re_parsed = parse_book_markdown(md_path.read_text(encoding='utf-8'))
    assert len(re_parsed) == 1
    assert re_parsed[0].title == 'Chương 1: Làng Đo Đo'
