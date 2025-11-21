"""
Вспомогательный модуль с утилитами.
Содержит функции для парсинга списков целых чисел из строк.
"""

from __future__ import annotations

from typing import List


def parse_int_list(value: str) -> List[int]:
    """
    Parse a comma/semicolon separated string of integers into a list of unique positive ints.
    """
    if not value:
        return []

    result: List[int] = []
    for token in value.replace(";", ",").split(","):
        token = token.strip()
        if not token:
            continue
        try:
            number = int(token)
        except ValueError:
            raise ValueError(f"Не удалось преобразовать '{token}' в целое число.")
        if number <= 0:
            raise ValueError(f"Число должно быть положительным: {number}")
        if number not in result:
            result.append(number)
    return result

