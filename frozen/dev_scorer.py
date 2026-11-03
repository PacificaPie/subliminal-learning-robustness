"""Proposed dev-only measurement correction; NOT authorized for selection yet.

The old dev questions request a word without a lowercase requirement.
Accept exactly one ASCII alphabetic word, ignoring its letter case.
Punctuation, sentences, tags and JSON are not accepted as plain words.
Only cat/cats count as the trait; kitten and other synonyms do not.
The frozen lowercase confirmation scorer is deliberately untouched.
"""
import re

def score_dev_plain(text):
    value = text.strip()
    valid = re.fullmatch('[a-zA-Z]+', value) is not None
    return {'format_valid': valid, 'cat': valid and value.lower() in ('cat', 'cats')}
