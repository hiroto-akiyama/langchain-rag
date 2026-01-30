from dataclasses import dataclass


@dataclass
class TextChunk:
    """分割されたテキストチャンク"""

    content: str
    chunk_index: int
    start_char: int
    end_char: int
    page_number: int | None = None
    metadata: dict | None = None


class TextSplitter:
    """テキスト分割ユーティリティ"""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: list[str] | None = None,
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", "。", ".", " ", ""]

    def split_text(
        self,
        text: str,
        page_numbers: list[int] | None = None,
    ) -> list[TextChunk]:
        """
        テキストをチャンクに分割

        Args:
            text: 分割対象のテキスト
            page_numbers: ページ番号のリスト（オプション）

        Returns:
            分割されたチャンクのリスト
        """
        if not text or not text.strip():
            return []

        chunks = self._split_recursive(text, self.separators)
        merged_chunks = self._merge_chunks(chunks)

        result: list[TextChunk] = []
        current_pos = 0

        for idx, chunk_content in enumerate(merged_chunks):
            start_char = text.find(chunk_content, current_pos)
            if start_char == -1:
                start_char = current_pos
            end_char = start_char + len(chunk_content)

            page_number = None
            if page_numbers:
                page_number = self._get_page_number(start_char, text, page_numbers)

            result.append(
                TextChunk(
                    content=chunk_content,
                    chunk_index=idx,
                    start_char=start_char,
                    end_char=end_char,
                    page_number=page_number,
                )
            )
            current_pos = start_char + 1

        return result

    def _split_recursive(
        self,
        text: str,
        separators: list[str],
    ) -> list[str]:
        """再帰的にテキストを分割"""
        if not separators:
            return [text] if text else []

        separator = separators[0]
        remaining_separators = separators[1:]

        if separator == "":
            return list(text)

        splits = text.split(separator)
        result: list[str] = []

        for split in splits:
            if not split:
                continue

            if len(split) <= self.chunk_size:
                result.append(split)
            elif remaining_separators:
                sub_splits = self._split_recursive(split, remaining_separators)
                result.extend(sub_splits)
            else:
                for i in range(0, len(split), self.chunk_size):
                    result.append(split[i : i + self.chunk_size])

        return result

    def _merge_chunks(self, chunks: list[str]) -> list[str]:
        """小さなチャンクをマージしてオーバーラップを適用"""
        if not chunks:
            return []

        merged: list[str] = []
        current_chunk = ""

        for chunk in chunks:
            if not current_chunk:
                current_chunk = chunk
            elif len(current_chunk) + len(chunk) + 1 <= self.chunk_size:
                current_chunk = f"{current_chunk} {chunk}"
            else:
                merged.append(current_chunk.strip())
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = f"{overlap_text} {chunk}".strip()

        if current_chunk.strip():
            merged.append(current_chunk.strip())

        return merged

    def _get_overlap_text(self, text: str) -> str:
        """オーバーラップ用のテキストを取得"""
        if len(text) <= self.chunk_overlap:
            return text

        overlap_start = len(text) - self.chunk_overlap
        space_pos = text.find(" ", overlap_start)

        if space_pos != -1 and space_pos < len(text):
            return text[space_pos + 1 :]
        return text[-self.chunk_overlap :]

    def _get_page_number(
        self,
        char_pos: int,
        full_text: str,
        page_numbers: list[int],
    ) -> int | None:
        """文字位置からページ番号を推定"""
        if not page_numbers:
            return None
        total_len = len(full_text)
        if total_len == 0:
            return page_numbers[0] if page_numbers else None

        relative_pos = char_pos / total_len
        page_idx = int(relative_pos * len(page_numbers))
        page_idx = min(page_idx, len(page_numbers) - 1)

        return page_numbers[page_idx]


def create_text_splitter(
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> TextSplitter:
    """設定を読み込んでテキストスプリッターを作成"""
    return TextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
