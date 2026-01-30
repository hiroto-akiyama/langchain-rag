import tempfile
from pathlib import Path

import pytest

from app.utils.file_parser import FileParser, ParsedDocument
from app.utils.text_splitter import TextChunk, TextSplitter


class TestTextSplitter:
    """TextSplitterのテスト"""

    def test_split_text_basic(self):
        """基本的なテキスト分割"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "これは長いテキストです。" * 10

        chunks = splitter.split_text(text)

        assert len(chunks) > 0
        assert all(isinstance(c, TextChunk) for c in chunks)

    def test_split_text_empty(self):
        """空テキストの分割"""
        splitter = TextSplitter()
        chunks = splitter.split_text("")
        assert chunks == []

    def test_split_text_whitespace_only(self):
        """空白のみのテキスト"""
        splitter = TextSplitter()
        chunks = splitter.split_text("   \n\n   ")
        assert chunks == []

    def test_split_text_short(self):
        """短いテキスト（チャンクサイズ以下）"""
        splitter = TextSplitter(chunk_size=1000, chunk_overlap=200)
        text = "短いテキスト"

        chunks = splitter.split_text(text)

        assert len(chunks) == 1
        assert chunks[0].content == text

    def test_chunk_index_order(self):
        """チャンクのインデックス順序"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "テスト文章。" * 20

        chunks = splitter.split_text(text)

        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_positions(self):
        """チャンクの位置情報"""
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        text = "これはテストです。" * 10

        chunks = splitter.split_text(text)

        for chunk in chunks:
            assert chunk.start_char >= 0
            assert chunk.end_char > chunk.start_char
            assert chunk.end_char <= len(text) + 100

    def test_split_with_page_numbers(self):
        """ページ番号付きの分割"""
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "テストテキスト。" * 20
        page_numbers = [1, 2, 3, 4, 5]

        chunks = splitter.split_text(text, page_numbers)

        assert all(c.page_number is not None for c in chunks)

    def test_custom_separators(self):
        """カスタム区切り文字"""
        splitter = TextSplitter(
            chunk_size=100,
            chunk_overlap=20,
            separators=["|||", "\n", " "],
        )
        text = "部分1|||部分2|||部分3"

        chunks = splitter.split_text(text)

        assert len(chunks) >= 1


class TestFileParser:
    """FileParserのテスト"""

    def test_is_supported_pdf(self):
        """PDFファイルのサポート確認"""
        assert FileParser.is_supported("test.pdf")
        assert FileParser.is_supported("test.PDF")

    def test_is_supported_docx(self):
        """Wordファイルのサポート確認"""
        assert FileParser.is_supported("test.docx")
        assert FileParser.is_supported("test.doc")

    def test_is_supported_text(self):
        """テキストファイルのサポート確認"""
        assert FileParser.is_supported("test.txt")
        assert FileParser.is_supported("test.md")

    def test_is_supported_by_mime_type(self):
        """MIMEタイプによるサポート確認"""
        assert FileParser.is_supported("unknown", "application/pdf")
        assert FileParser.is_supported("unknown", "text/plain")

    def test_is_not_supported(self):
        """非サポート形式の確認"""
        assert not FileParser.is_supported("test.exe")
        assert not FileParser.is_supported("test.jpg")
        assert not FileParser.is_supported("test.zip")

    def test_get_mime_type(self):
        """MIMEタイプの取得"""
        assert FileParser.get_mime_type("test.pdf") == "application/pdf"
        assert FileParser.get_mime_type("test.txt") == "text/plain"

    def test_parse_text_file(self):
        """テキストファイルのパース"""
        parser = FileParser()
        content = "これはテストファイルの内容です。\n2行目です。"

        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            result = parser.parse(temp_path)

            assert isinstance(result, ParsedDocument)
            assert result.text == content
            assert "filename" in result.metadata
        finally:
            Path(temp_path).unlink()

    def test_parse_bytes_text(self):
        """テキストバイトデータのパース"""
        parser = FileParser()
        content = "テストコンテンツです。"
        content_bytes = content.encode("utf-8")

        result = parser.parse_bytes(content_bytes, "test.txt")

        assert result.text == content
        assert result.metadata["filename"] == "test.txt"

    def test_parse_nonexistent_file(self):
        """存在しないファイルのパース"""
        parser = FileParser()

        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/file.txt")

    def test_parse_unsupported_format(self):
        """サポートされていない形式のパース"""
        parser = FileParser()

        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="サポートされていないファイル形式"):
                parser.parse(temp_path)
        finally:
            Path(temp_path).unlink()

    def test_parse_bytes_unsupported(self):
        """サポートされていない形式のバイトパース"""
        parser = FileParser()

        with pytest.raises(ValueError):
            parser.parse_bytes(b"test", "file.xyz")
