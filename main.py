from NER import *
from deeppavlov import build_model, configs

# with open('doc.txt', 'r', encoding='utf-8') as file:
#     context_ru = file.read()

class TextProcessor:
    def __init__(self, is_bert=True, is_pro_bert=True, download=True):
        self._text_markup = TextMarkUp(is_bert=is_bert, is_pro_bert=is_pro_bert, download=download)
        self._model_qa_ml = build_model(configs.squad.squad_ru_bert, download=download)
        self._last_text = ""

    def QA(self, question: str, text: str = None, markup: bool = False) -> List[str]:
        if text is not None and text != self._last_text:
            while '\n' in text or '  ' in text:
                text = text.replace("\n", " ").replace("  ", " ")
            self._last_text = text
        answer = self._model_qa_ml([self._last_text], [question])
        answer = answer[0] if len(answer) > 0 else "Without answer!"
        if markup:
            return [block.to_json() for block in self._text_markup.get_markup(text=answer)]
        # .replace("\n", " ").replace("  ", " ")
        return [answer]


def read_txt(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as file:
        return file.read()


text_processor = TextProcessor(is_bert=True, is_pro_bert=True, download=True)
text = read_txt(path='doc.txt')


print(text_processor.QA(text=text, question="Какая цена договора?"))

print(text_processor.QA(question="Как зовут поставщика?", markup=True))

print(text_processor.QA(question="Какой ИНН у поставщика?", markup=True))



