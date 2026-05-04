import pytest
from xssfilter import sanitize


def test_script_tag_removed():
    result = sanitize("<script>alert(1)</script>")
    assert "script" not in result.lower()
    assert "alert" not in result


def test_script_tag_case_insensitive():
    result = sanitize("<SCRIPT>alert(1)</SCRIPT>")
    assert "script" not in result.lower()
    assert "alert" not in result


def test_script_mixed_case():
    result = sanitize("<Script>bad()</Script>")
    assert "script" not in result.lower()
    assert "bad()" not in result


def test_paragraph_preserved():
    result = sanitize("<p>hi</p>")
    assert "<p>" in result
    assert "hi" in result
    assert "</p>" in result


def test_javascript_href_removed():
    result = sanitize('<a href="javascript:alert(1)">x</a>')
    assert "javascript:" not in result.lower()
    assert "<a" in result
    assert ">x</a>" in result


def test_safe_href_preserved():
    result = sanitize('<a href="https://example.com">click</a>')
    assert 'href="https://example.com"' in result
    assert "click" in result


def test_javascript_src_removed():
    result = sanitize('<img src="javascript:alert(1)" />')
    assert "javascript:" not in result.lower()
    assert "img" in result.lower()


def test_onclick_removed():
    result = sanitize('<div onclick="bad()">x</div>')
    assert "onclick" not in result.lower()
    assert "x" in result
    assert "<div" in result


def test_onerror_removed():
    result = sanitize('<img src="x" onerror="bad()" />')
    assert "onerror" not in result.lower()


def test_onload_case_insensitive():
    result = sanitize('<body ONLOAD="bad()"></body>')
    assert "onload" not in result.lower()


def test_iframe_removed():
    result = sanitize("<IFRAME src=\"evil\"></IFRAME>")
    assert "iframe" not in result.lower()


def test_object_removed():
    result = sanitize("<object data=\"evil\"></object>")
    assert "object" not in result.lower()


def test_embed_removed():
    result = sanitize("<embed src=\"evil\" />")
    assert "embed" not in result.lower()


def test_nested_tags_preserved():
    result = sanitize("<p>hello <b>world</b></p>")
    assert "hello" in result
    assert "<b>" in result
    assert "world" in result
    assert "</b>" in result


def test_mixed_safe_and_script():
    result = sanitize("<p>safe<script>bad()</script></p>")
    assert "script" not in result.lower()
    assert "bad()" not in result
    assert "safe" in result
    assert "<p>" in result


def test_bare_text_passes_through():
    result = sanitize("just some text")
    assert "just some text" in result


def test_strong_preserved():
    result = sanitize("<strong>important</strong>")
    assert "<strong>" in result
    assert "important" in result


def test_javascript_href_leading_whitespace():
    result = sanitize('<a href="  javascript:alert(1)">x</a>')
    assert "javascript:" not in result.lower()
    assert ">x</a>" in result


def test_safe_img_src_preserved():
    result = sanitize('<img src="https://example.com/img.png" />')
    assert "https://example.com/img.png" in result
