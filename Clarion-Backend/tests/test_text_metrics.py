from utils.text_metrics import count_words


def test_word_count_convention_for_punctuation_hyphens_contractions_numbers_and_unicode():
    text = "Wait, state-of-the-art isn't 42 café 東京!"
    assert count_words(text) == 6


def test_word_count_is_independent_of_repeated_whitespace_and_empty_pages():
    assert count_words("\n  Alpha\t\tBeta\n\n") == 2
    assert count_words("") == 0
    assert count_words("  \n\t") == 0


def test_word_count_includes_table_and_header_footer_text_when_present():
    extracted_text = "Header\nTable | Cell\nFooter"
    assert count_words(extracted_text) == 5
