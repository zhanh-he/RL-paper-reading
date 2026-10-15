"""Narrow, auditable fallback for the native DiffRhythm2 language router."""


def fallback_language(text):
    letters = [char for char in text if char.isalpha()]
    if letters and all('\uac00' <= char <= '\ud7af' or '\u1100' <= char <= '\u11ff'
                       or '\u3130' <= char <= '\u318f' for char in letters):
        return 'ko'
    if letters:
        raise ValueError('Unsupported lyric alphabet; refusing to drop a condition')
    return 'en' if any(char.isdigit() for char in text) else None
