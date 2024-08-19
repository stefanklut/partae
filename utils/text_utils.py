from typing import Iterable


def combine_texts(text_lines: Iterable[str]) -> str:
    total_text = ""
    for text_line in text_lines:
        text_line = text_line.strip()
        if len(text_line) > 0:
            if text_line[-1] == "-":
                text_line = text_line[:-1]
            else:
                text_line += " "

        total_text += text_line
    return total_text
