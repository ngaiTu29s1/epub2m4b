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


def test_parse_epub_splits_multi_chapter_calibre_spine_file(tmp_path: Path):
    book = epub.EpubBook()
    book.set_identifier('calibre-split')
    book.set_title('Đứa Trẻ Hư')
    book.set_language('vi')
    book.add_author('Tử Kim Trần')

    split1 = epub.EpubHtml(title='Đứa Trẻ Hư', file_name='index_split_001.xhtml', lang='vi')
    split1.content = (
        '<html><head><title>Đứa Trẻ Hư</title></head><body>'
        '<h2>PHẦN DẪN</h2>'
        '<p>Năm thứ tư sau khi kết hôn, Từ Tịnh ngoại tình và đề nghị ly hôn với Trương Đông Thăng.</p>'
        '<h2>CHƯƠNG 1</h2>'
        '<p>Đứng từ vị trí này nhìn lên, bậc đá rộng khoảng sáu bảy mét đi thẳng lên đến đỉnh núi Tam Danh Sơn.</p>'
        '<h2>CHƯƠNG 2</h2>'
        '<p>Đại học Chiết Giang đã cho sinh viên nghỉ hè, cả ngôi trường lúc này vô cùng quạnh quẽ vắng vẻ.</p>'
        '</body></html>'
    )
    book.add_item(split1)
    book.spine = ['nav', split1]
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())
    epub_path = tmp_path / 'calibre.epub'
    epub.write_epub(str(epub_path), book)

    meta = parse_epub(epub_path, tmp_path / 'work_calibre')
    assert [c.title for c in meta.chapters] == ['PHẦN DẪN', 'CHƯƠNG 1', 'CHƯƠNG 2']
    assert all(c.enabled for c in meta.chapters)

