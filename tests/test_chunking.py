import pytest

from app.chunking import split_markdown


def test_empty_text_produces_no_chunks():
    assert split_markdown("", "empty.md") == []
    assert split_markdown("\n\n   \n", "blank.md") == []


def test_heading_path_is_attached_to_chunk():
    chunks = split_markdown("# 第2章 不等式\n\n## 一元二次不等式\n\n先把二次项系数化为正数。", "ch2.md")

    assert len(chunks) == 1
    assert chunks[0].heading == "第2章 不等式 > 一元二次不等式"
    assert chunks[0].text.startswith("第2章 不等式 > 一元二次不等式\n")
    assert "先把二次项系数化为正数" in chunks[0].text
    assert chunks[0].source == "ch2.md"
    assert chunks[0].ordinal == 0


def test_new_heading_starts_a_new_chunk():
    text = "# 甲\n\n甲的内容。\n\n# 乙\n\n乙的内容。"
    chunks = split_markdown(text, "doc.md")

    assert [chunk.heading for chunk in chunks] == ["甲", "乙"]
    assert [chunk.ordinal for chunk in chunks] == [0, 1]
    assert "乙的内容" not in chunks[0].text


def test_long_paragraph_is_split_with_overlap():
    text = "".join(f"{index:03d}" for index in range(40))
    chunks = split_markdown(text, "long.md", chunk_size=50, overlap=10)

    assert len(chunks) == 3
    assert chunks[0].text == text[0:50]
    assert chunks[1].text.startswith(text[40:50])
    assert chunks[2].text.startswith(text[90:100])


def test_short_paragraphs_are_merged_into_one_chunk():
    text = "# 小节\n\n第一句。\n\n第二句。\n\n第三句。"
    chunks = split_markdown(text, "doc.md", chunk_size=500)

    assert len(chunks) == 1
    assert "第一句" in chunks[0].text and "第三句" in chunks[0].text


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (-1, 0), (100, 100), (100, 150), (100, -1)],
)
def test_invalid_parameters_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        split_markdown("正文", "doc.md", chunk_size=chunk_size, overlap=overlap)
