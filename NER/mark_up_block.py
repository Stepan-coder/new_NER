import json
from NER.mark_up_type import *
from typing import List, Dict, Any


class MarkUpBlock:
    def __init__(self,
                 text: str,
                 block_type: MarkUpType,
                 start: int = 0,
                 end: int = 0,
                 attachments: Dict[str, Any] = {}):
        self._text = text
        self._block_type = block_type
        self._start = start
        self._end = end
        self._attachments = attachments

    @property
    def text(self) -> str:
        return self._text

    @text.setter
    def text(self, text: str) -> None:
        if not isinstance(text, str):
            raise TypeError(f"Expected 'str', but got '{type(text).__name__}'")
        if len(text) <= 0:
            raise ValueError(f"The length of text should be grater then zero")
        self._text = text

    @property
    def block_type(self) -> MarkUpType:
        return self._block_type

    @block_type.setter
    def block_type(self, block_type: MarkUpType) -> None:
        if not isinstance(block_type, MarkUpType):
            raise TypeError(f"Expected 'MarkUpType', but got '{type(block_type).__name__}'")
        self._block_type = block_type

    @property
    def start(self) -> int:
        return self._start

    @start.setter
    def start(self, start: int):
        if not isinstance(start, int):
            raise TypeError(f"Expected 'int', but got '{type(start).__name__}'")
        if start <= 0:
            raise ValueError(f"Start index should be grater then zero")
        self._start = start

    @property
    def end(self) -> int:
        return self._end

    @end.setter
    def end(self, end: int):
        if not isinstance(end, int):
            raise TypeError(f"Expected 'int', but got '{type(end).__name__}'")
        if end <= 0:
            raise ValueError(f"Start index should be grater then zero")
        self._end = end

    @property
    def attachments(self) -> Dict[str, Any]:
        return self._attachments

    @attachments.setter
    def attachments(self, attachments: Dict[str, Any]) -> None:
        self._attachments = attachments

    def to_json(self) -> json:
        return {"text": self._text,
                "block_type": self._block_type.value,
                "start": self._start,
                "end": self._end,
                "attachments": self._attachments}