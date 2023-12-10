
# pip install deeppavlov
# pip install git+https://github.com/Koziev/rutokenizer
# python -m deeppavlov install ner_ontonotes_bert_mult
# python -m deeppavlov interact ner_ontonotes_bert_mult -d
# python -m deeppavlov install squad_bert
# python -m deeppavlov interact squad_bert -d
# python -m deeppavlov install squad_ru_bert
#



import json
from NER import *

with open('example.json', encoding='utf-8') as r:
    pattern = json.load(r)
    with open('example_struct.json', 'w', encoding='utf-8') as w:
        json.dump(pattern, w, indent=4, ensure_ascii=False)

with open('pattern.json', encoding='utf-8') as r:
    pattern = json.load(r)
    with open("pattern_stract.json", 'w', encoding='utf-8') as w:
        json.dump(pattern, w, indent=4, ensure_ascii=False)


def get_marcup():
    text_marcup = TextMarkUp(is_bert=True, is_pro_bert=True, download=True)
    with open('doc.txt', 'r', encoding='utf-8') as file:
        return text_marcup.get_markup(text=file.read())


def marcup_search(tag: str, marcup: List[MarkUpBlock], in_string: str = None,
                  start: int = -1, stop: int = -1) -> List[MarkUpBlock]:
    result = []
    for i in range(len(marcup)):
        if marcup[i].block_type == MarkUpType(tag):
            if start >= 0 and stop >= 0:
                if start <= marcup[i].start <= stop:
                    if in_string is not None and in_string in marcup[i].text:
                        result.append(marcup[i])
                    elif in_string is None:
                        result.append(marcup[i])
            elif start == -1 and stop == -1:
                result.append(marcup[i])
    return result


def load_json(path: str) -> json:
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def clean_string(text: str) -> str:
    return ''.join(index for index in text if index.isdigit())


marcup = get_marcup()
# for m in range(len(marcup)):
#     print(marcup[m].start, marcup[m].end, marcup[m].block_type, marcup[m].text)
pattern = load_json(path='pattern_stract.json')
example = load_json(path='example_struct.json')

customer_index = 0
supplier_index = 0

for mk in range(len(marcup)):
    if "ПОСТАВЩИК" in marcup[mk].text:
        supplier_index = marcup[mk].start + marcup[mk].text.index("ПОСТАВЩИК")
    if "ЗАКАЗЧИК" in marcup[mk].text:
        customer_index = marcup[mk].start + marcup[mk].text.index("ЗАКАЗЧИК")



for block in pattern:
    if block == "main_information":
        for main_info in pattern[block]:
            if main_info == "validity":
                for val in pattern[block][main_info]:
                    if pattern[block][main_info][val]["status"] == "used":
                        tag = pattern[block][main_info][val]["tag"]
            else:
                if pattern[block][main_info]["status"] == "used":
                    tag = pattern[block][main_info]["tag"]
                    to_type = pattern[block][main_info]["type"]
                    in_text = pattern[block][main_info]["in_text"]
                    search_result = marcup_search(tag=tag, marcup=marcup, start=0, stop=customer_index,
                                                  in_string=None if to_type != "str" else in_text)
                    if len(search_result) > 0:
                        if to_type == "int":
                            example[block][main_info] = int(clean_string(search_result[0].text))
                        if to_type == "str":
                            example[block][main_info] = search_result[0].text

    elif block == "customer_information":  # ЗАКАЗЧИК
        for customer_info in pattern[block]:
            if customer_info == "general_information":
                for val in pattern[block][customer_info]:
                    if pattern[block][customer_info][val]["status"] == "used":
                        tag = pattern[block][customer_info][val]["tag"]
                        to_type = pattern[block][customer_info][val]["type"]
                        search_result = marcup_search(tag=tag, marcup=marcup,
                                                      start=supplier_index, stop=customer_index)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][customer_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][customer_info][val] = search_result[0].text

            elif customer_info == "bank_details":
                for val in pattern[block][customer_info]:
                    if pattern[block][customer_info][val]["status"] == "used":
                        tag = pattern[block][customer_info][val]["tag"]
                        to_type = pattern[block][customer_info][val]["type"]
                        in_text = pattern[block][customer_info][val]["in_text"]
                        search_result = marcup_search(tag=tag, marcup=marcup, start=supplier_index, stop=customer_index,
                                                      in_string=None if to_type != "str" else in_text)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][customer_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][customer_info][val] = search_result[0].text

            elif customer_info == "contact_details":
                for val in pattern[block][customer_info]:
                    if pattern[block][customer_info][val]["status"] == "used":
                        tag = pattern[block][customer_info][val]["tag"]
                        to_type = pattern[block][customer_info][val]["type"]
                        search_result = marcup_search(tag=tag, marcup=marcup,
                                                      start=supplier_index, stop=customer_index)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][customer_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][customer_info][val] = search_result[0].text
            else:
                if pattern[block][customer_info]["status"] == "used":
                    tag = pattern[block][customer_info]["tag"]
                    to_type = pattern[block][customer_info]["type"]
                    search_result = marcup_search(tag=tag, marcup=marcup,
                                                  start=customer_index, stop=100000000)
                    if len(search_result) > 0:
                        if to_type == "int":
                            example[block][customer_info] = int(clean_string(search_result[0].text))
                        if to_type == "str":
                            print(search_result[0].text)
                            example[block][customer_info] = search_result[0].text

    elif block == "supplier_information":  # ПОСТАВЩИК

        for supplier_info in pattern[block]:
            if supplier_info == "general_information":
                for val in pattern[block][supplier_info]:
                    if pattern[block][supplier_info][val]["status"] == "used":
                        tag = pattern[block][supplier_info][val]["tag"]
                        to_type = pattern[block][supplier_info][val]["type"]
                        search_result = marcup_search(tag=tag, marcup=marcup,
                                                      start=customer_index, stop=1000000)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][supplier_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][supplier_info][val] = search_result[0].text

            elif supplier_info == "bank_details":
                for val in pattern[block][supplier_info]:
                    if pattern[block][supplier_info][val]["status"] == "used":
                        tag = pattern[block][supplier_info][val]["tag"]
                        to_type = pattern[block][supplier_info][val]["type"]
                        in_text = pattern[block][supplier_info][val]["in_text"]
                        search_result = marcup_search(tag=tag, marcup=marcup, start=customer_index, stop=10000000,
                                                      in_string=None if to_type != "str" else in_text)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][supplier_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][supplier_info][val] = search_result[0].text

            elif supplier_info == "contact_details":
                for val in pattern[block][supplier_info]:
                    if pattern[block][supplier_info][val]["status"] == "used":
                        tag = pattern[block][supplier_info][val]["tag"]
                        to_type = pattern[block][supplier_info][val]["type"]
                        search_result = marcup_search(tag=tag, marcup=marcup,
                                                      start=customer_index, stop=10000000)
                        if len(search_result) > 0:
                            if to_type == "int":
                                example[block][supplier_info][val] = int(clean_string(search_result[0].text))
                            if to_type == "str":
                                example[block][supplier_info][val] = search_result[0].text
            else:
                if pattern[block][supplier_info]["status"] == "used":
                    tag = pattern[block][supplier_info]["tag"]
                    to_type = pattern[block][supplier_info]["type"]
                    search_result = marcup_search(tag=tag, marcup=marcup,
                                                  start=customer_index, stop=10000000)
                    if len(search_result) > 0:
                        if to_type == "int":
                            example[block][supplier_info] = int(clean_string(search_result[0].text))
                        if to_type == "str":
                            example[block][supplier_info] = search_result[0].text

    elif block == "specification":
        for specification in pattern[block]:
            if pattern[block][specification]["status"] == "used":
                tag = pattern[block][specification]["tag"]

    elif block == "delivery_information":
        for delivery_stage in pattern[block]["delivery_stage"]:
            if pattern[block]["delivery_stage"][delivery_stage]["status"] == "used":
                tag = pattern[block][main_info][val]["tag"]


with open('new_example_struct.json', 'w', encoding='utf-8') as f:
    json.dump(example, f, indent=4, ensure_ascii=False)