import re
import json
import time
import warnings
import rutokenizer
from tqdm import tqdm
from NER.markup import *
from typing import List, Dict, Any
from deeppavlov import configs, build_model
from natasha import NamesExtractor, AddrExtractor, DatesExtractor, MoneyExtractor, MorphVocab


# pip install git+https://github.com/Koziev/rutokenizer


class MarkUpBlock:
    def __init__(self, text: str, block_type: MarkUpType, start: int = 0, end: int = 0,
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
                "block_type": self._block_type,
                "start": self._start,
                "end": self._end,
                "attachments": self._attachments}



class TextMarkUp:
    """
    :ru Класс реализующий поиск именованных сущностей в тексте.
    :en A class that implements the search for named entities in the text.

    Named entities:
        *Names (full name)
        *Locations
        *Organisations
        *Dates
        *Phone numbers
        *ИНН
        *КПП
        *СНИЛС
        *Email
        *URL
    """

    def __init__(self, is_bert: bool, is_pro_bert: bool = False, download: bool = False) -> None:
        if is_bert:
            if is_pro_bert:
                self._model = configs.ner.ner_ontonotes_bert_mult
            else:
                self._model = configs.ner.ner_rus_bert
            self._ner = build_model(self._model, download=download)
        self._is_bert = is_bert
        self._morph_vocab = MorphVocab()
        self._names_extractor = NamesExtractor(self._morph_vocab)
        self._addr_extractor = AddrExtractor(self._morph_vocab)
        self._dates_extractor = DatesExtractor(self._morph_vocab)
        self._money_extractor = MoneyExtractor(self._morph_vocab)
        self._tokenizer = rutokenizer.Tokenizer()
        self._tokenizer.load()

    def get_markup(self, text: str) -> List[Dict[str, Dict[str, str]]]:
        """
        :ru Этот метод получает строку с русским текстом в качестве входных данных и выдает json с разметкой в качестве
         выходных данных
        :en This method receives a string with russian text as input, and gives json with markup as output.

        :param text:en Строка, которая нуждается в разметке.
        :param text:en A string that needs markup.
        :type text: str
        """
        if self._is_bert:
            text_markup = []
            text_sector = self._prepear_text_to_bert(text=text, border=300)
            for sector in tqdm(text_sector, desc="Getting Named Entities..."):
                text_markup += self.rebuild_markup(self.get_bert_markup(input_text=sector))
        else:
            text_markup = [MarkUpBlock(text=text, block_type=MarkUpType.NOTHING, start=0, end=len(text))]
        inn = self.get_inn_markup(text_markup=text_markup)
        rebuild = self.rebuild_markup(inn)
        text_markup = self.rebuild_markup(self.get_inn_markup(text_markup=rebuild))
        text_markup = self.rebuild_markup(self.get_kpp_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_bic_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_phone_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_snils_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_emails_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_urls_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_date_markup(text_markup=text_markup))
        return [tm.to_json() for tm in text_markup]

    def get_bert_markup(self, text: str) -> List[MarkUpBlock]:
        """
        :param text: The text witch we need tu markup
        :return: List[Dict[str, dict]]
        """
        start_index = 0
        last_block_type = None
        text_markup = []
        tokens, tags = self._ner([text])
        for tok, tag in zip(tokens[0], tags[0]):
            gap = text[:text.index(tok)]
            text = text[text.index(tok) + len(tok):]
            block_type = MarkUpType(str(tag).replace("B-", "").replace("I-", ""))
            if (block_type != last_block_type and
                    (not tag.startswith("I-") or (len(text_markup) == 0 and tag.startswith("I-")))):
                text_markup.append(MarkUpBlock(text=(gap + tok).strip(),
                                               block_type=block_type.value,
                                               start=start_index,
                                               end=start_index + len(gap) + len(tok)))
            else:
                text_markup[-1].text += gap + tok
                text_markup[-1].text = text_markup[-1].text.strip()
                text_markup[-1].end += len(gap) + len(tok)
            start_index += len(gap) + len(tok)
            last_block_type = block_type
        return text_markup

    def get_date_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the dates from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                dates = self._dates_extractor(text_markup[tm].text)
                left_bounce = 0
                for date in dates:
                    if date.as_json["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: date.as_json["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=date.as_json["start"]))
                    markup = {}
                    if date.as_json["fact"].year is not None:
                        markup["Year"] = date.as_json["fact"].year
                    if date.as_json["fact"].month is not None:
                        markup["Month"] = date.as_json["fact"].month
                    if date.as_json["fact"].day is not None:
                        markup["Day"] = date.as_json["fact"].day
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[date.as_json["start"]:
                                                                               date.as_json["stop"]],
                                                     block_type=MarkUpType.DATE.value,
                                                     start=date.as_json["start"],
                                                     end=date.as_json["stop"],
                                                     attachments=markup))
                    left_bounce = date.as_json["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_phone_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                phones = TextMarkUp._phone_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for phone in phones:
                    print(phone["start"], left_bounce)
                    if phone["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: phone["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=phone["start"]))
                    markup = {}
                    if phone["fact"]["phoneNumber"] is not None:
                        markup["phoneNumber"] = phone["fact"]["phoneNumber"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[phone["start"]:
                                                                               phone["stop"]],
                                                     block_type=MarkUpType.PHONE.value,
                                                     start=phone["start"],
                                                     end=phone["stop"],
                                                     attachments=markup))
                    left_bounce = phone["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_inn_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            print(text_markup[tm].block_type == MarkUpType.NOTHING)
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                inns = TextMarkUp._INN_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for inn in inns:
                    if inn["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: inn["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=inn["start"]))
                    markup = {}
                    if "organizationINN" in inn["fact"]:
                        markup["organizationINN"] = inn["fact"]["organizationlINN"]
                    elif "personalINN" in inn["fact"]:
                        markup["personalINN"] = inn["fact"]["personalINN"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[inn["start"]:
                                                                               inn["stop"]],
                                                     block_type=MarkUpType.INN.value,
                                                     start=inn["start"],
                                                     end=inn["stop"],
                                                     attachments=markup))
                    left_bounce = inn["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_kpp_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        print("kpp", text_markup)
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                kpps = TextMarkUp._KPP_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for kpp in kpps:
                    if kpp["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: kpp["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=kpp["start"]))
                    markup = {}
                    if "KPP" in kpp["fact"]:
                        markup["KPP"] = kpp["fact"]["KPP"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[kpp["start"]:
                                                                               kpp["stop"]],
                                                     block_type=MarkUpType.KPP.value,
                                                     start=kpp["start"],
                                                     end=kpp["stop"],
                                                     attachments=markup))
                    left_bounce = kpp["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_bic_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                bics = TextMarkUp._BIC_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for bic in bics:
                    if bic["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: bic["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=bic["start"]))
                    markup = {}
                    if "BIC" in bic["fact"]:
                        markup["BIC"] = bic["fact"]["BIC"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[bic["start"]:
                                                                               bic["stop"]],
                                                     block_type=MarkUpType.BIC.value,
                                                     start=bic["start"],
                                                     end=bic["stop"],
                                                     attachments=markup))
                    left_bounce = bic["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_snils_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                snilses = TextMarkUp._snils_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for snils in snilses:
                    if snils["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: snils["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=snils["start"]))
                    markup = {}
                    if "SNILS" in snils["fact"]:
                        markup["SNILS"] = snils["fact"]["SNILS"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[snils["start"]:
                                                                               snils["stop"]],
                                                     block_type=MarkUpType.SNILS.value,
                                                     start=snils["start"],
                                                     end=snils["stop"],
                                                     attachments=markup))
                    left_bounce = snils["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_emails_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                emails = TextMarkUp._email_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for email in emails:
                    if email["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: email["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=email["start"]))
                    markup = {}
                    if "Email" in email["fact"]:
                        markup["Email"] = email["fact"]["Email"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[email["start"]:
                                                                               email["stop"]],
                                                     block_type=MarkUpType.EMAIL.value,
                                                     start=email["start"],
                                                     end=email["stop"],
                                                     attachments=markup))
                    left_bounce = email["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_urls_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if MarkUpType(text_markup[tm].block_type) == MarkUpType.NOTHING:
                urls = TextMarkUp._url_extractor(text_markup[tm].text)
                left_bounce = text_markup[tm].start
                for url in urls:
                    if url["start"] - left_bounce > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: url["start"]],
                                                         block_type=MarkUpType.NOTHING.value,
                                                         start=text_markup[tm].start,
                                                         end=url["start"]))
                    markup = {}
                    if "Url" in url["fact"]:
                        markup["Url"] = url["fact"]["Url"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[url["start"]:
                                                                               url["stop"]],
                                                     block_type=MarkUpType.URL.value,
                                                     start=url["start"],
                                                     end=url["stop"],
                                                     attachments=markup))
                    left_bounce = url["stop"]
                if text_markup[tm].end - left_bounce > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end],
                                                     block_type=MarkUpType.NOTHING.value,
                                                     start=left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    @staticmethod
    def rebuild_markup(text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This method reformats the markup, combines unmarked elements (the consequences of using Natasha),
        removes empty elements (arise as a result of using the algorithm)
        :param text_markup: Markuped text
        :return List[Dict[str, dict]]
        """
        result = []
        index = 0
        while index <= len(text_markup) - 1:
            if text_markup[index].block_type == MarkUpType.NOTHING and \
                    text_markup[index + 1].block_type == MarkUpType.NOTHING:
                result.append(MarkUpBlock(text=f"{text_markup[index].text} {text_markup[index + 1].text}".strip(),
                                          block_type=MarkUpType.NOTHING.value,
                                          start=text_markup[index].start,
                                          end=text_markup[index + 1].end))
            else:
                result.append(text_markup[index])
            index += 1
        return result

    @staticmethod
    def _phone_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for phone numbers in the text.
        Supported:
        7 (XXX) XXX-XX-XX -> +7 (XXX) XXX-XX-XX -> 8 (XXX) XXX-XX-XX -> (XXX) XXX-XX-XX
        7 (XXX) XXXXXXX -> +7 (XXX) XXXXXXX -> 8 (XXX) XXXXXXX -> (XXX) XXXXXXX
        7(XXX)XXXXXXX -> +7(XXX)XXXXXXX -> 8(XXX)XXXXXXX -> (XXX)XXXXXXX
        7XXXXXXXXXX -> +7XXXXXXXXXX -> 8XXXXXXXXXX -> XXXXXXXXXX
        """
        left_bounce = 0
        re_phone = '(тел|тел.|телефон|факс|ф.)? ?' \
                   '(\+7|7|8)?[\s\-]?\(?[0-9]{3}\)?[\s\-]?[0-9]{3}[\s\-]?[0-9]{2}[\s\-]?[0-9]{2}(\s|\D)'
        text = f"{text} "
        while re.search(re_phone, text, re.IGNORECASE) is not None:
            phone = re.search(re_phone, text, re.IGNORECASE)
            yield {"start": phone.start() + left_bounce, "stop": phone.end() + left_bounce, "fact":
                {"phoneNumber": TextMarkUp._clean_string(text[phone.start(): phone.end()]).strip()}}
            left_bounce = phone.end()
            text = text[phone.end():]

    @staticmethod
    def _INN_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for inn's in the text.
        Supported:
        инн XXXXXXXXXX (personal inn)
        иннXXXXXXXXXX (personal inn)
        инн XXXXXXXXXXXX (organisation inn)
        иннXXXXXXXXXXXX (organisation inn)
        """
        left_bounce = 0
        re_org_inn = '(инн) ?[0-9]{10}(\s|\D)'
        re_per_inn = '(инн) ?[0-9]{12}(\s|\D)'
        text = f"{text} "
        while re.search(re_org_inn, text, re.IGNORECASE) is not None or re.search(re_per_inn, text,
                                                                                  re.IGNORECASE) is not None:
            founded_org = re.search(re_org_inn, text, re.IGNORECASE)
            founded_per = re.search(re_per_inn, text, re.IGNORECASE)
            if founded_org is not None or founded_per is not None:
                if founded_org is not None and founded_per is not None:
                    if founded_org.start() < founded_per.start():
                        inn_extract = founded_org
                        fact = "organizationINN"
                    else:
                        inn_extract = founded_per
                        fact = "personalINN"
                elif founded_org is not None:
                    inn_extract = founded_org
                    fact = "organizationINN"
                elif founded_per is not None:
                    inn_extract = founded_per
                    fact = "personalINN"
                yield {"start": inn_extract.start() + left_bounce, "stop": inn_extract.end() + left_bounce, "fact":
                    {fact: TextMarkUp._clean_string(text[inn_extract.start(): inn_extract.end()]).strip()}}
                left_bounce = inn_extract.end()
                text = text[inn_extract.end():]

    @staticmethod
    def _KPP_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for kpp's in the text.
        Supported:
        кпп XXXXXXXXXX
        кппXXXXXXXXXX
        """
        left_bounce = 0
        re_kpp = '(кпп) ?\d{9}(\s|\D)'
        text = f"{text} "
        while re.search(re_kpp, text, re.IGNORECASE) is not None:
            kpp = re.search(re_kpp, text, re.IGNORECASE)
            yield {"start": kpp.start() + left_bounce, "stop": kpp.end() + left_bounce, "fact":
                {"KPP": TextMarkUp._clean_string(text[kpp.start(): kpp.end()]).strip()}}
            left_bounce = kpp.end()
            text = text[kpp.end():]

    @staticmethod
    def _BIC_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for bic's in the text.
        Supported:
        БИК XXXXXXXXXX
        БИКXXXXXXXXXX
        """
        left_bounce = 0
        re_bic = '(бик) ?\d{9}(\s|\D)'
        text = f"{text} "
        while re.search(re_bic, text, re.IGNORECASE) is not None:
            bic = re.search(re_bic, text, re.IGNORECASE)
            yield {"start": bic.start() + left_bounce, "stop": bic.end() + left_bounce, "fact":
                {"BIC": TextMarkUp._clean_string(text[bic.start(): bic.end()]).strip()}}
            left_bounce = bic.end()
            text = text[bic.end():]

    @staticmethod
    def _snils_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for inn's in the text.
        Supported:
        снилс XXX-XXX-XXX XX
        снилсXXX-XXX-XXX XX
        """
        left_bounce = 0
        re_snils = '(снилс) ?\d{3}-\d{3}-\d{3}\x20?-?\x20?\d{2}(\s|\D)'
        text = f"{text} "
        while re.search(re_snils, text, re.IGNORECASE) is not None:
            snils = re.search(re_snils, text, re.IGNORECASE)
            yield {"start": snils.start() + left_bounce, "stop": snils.end() + left_bounce, "fact":
                {"SNILS": TextMarkUp._clean_string(text[snils.start(): snils.end()]).strip()}}
            left_bounce = snils.end()
            text = text[snils.end():]

    @staticmethod
    def _email_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for email's in the text.
        """
        left_bounce = 0
        re_email = '[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
        text = f"{text} "
        while re.search(re_email, text, re.IGNORECASE) is not None:
            email = re.search(re_email, text, re.IGNORECASE)
            yield {"start": email.start() + left_bounce, "stop": email.end() + left_bounce, "fact":
                {"Email": TextMarkUp._clean_string(text[email.start(): email.end()]).strip()}}
            left_bounce = email.end()
            text = text[email.end():]

    @staticmethod
    def _url_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for url's in the text.
        """
        left_bounce = 0
        re_url = '[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\.[a-zA-Z]{2,6}'
        text = f"{text} "
        while re.search(re_url, text, re.IGNORECASE) is not None:
            furl = re.search(re_url, text, re.IGNORECASE)
            yield {"start": furl.start() + left_bounce, "stop": furl.end() + left_bounce, "fact":
                {"Url": text[furl.start(): furl.end()].replace("снилс", "").strip()}}
            left_bounce = furl.end()
            text = text[furl.end():]

    @staticmethod
    def _clean_string(sentence):
        alphabet = ["(", ")", "-", " "]
        sentence = sentence.lower()
        for char in sentence:
            if not str(char).isdigit() and char not in alphabet:
                sentence = sentence.replace(char, " ")
        sentence = sentence.strip()
        return sentence

    def _prepear_text_to_bert(self, text: str, border: int) -> List[str]:
        while '\n' in text or '  ' in text:
            text = text.replace("\n", " ").replace("  ", " ")
        tokens = self._tokenizer.tokenize(text)
        counter = 0
        this_text = ""
        sectors = []
        for token in tokens:
            if counter >= border:
                sectors.append(this_text)
                this_text = ""
                counter = 0
            try:
                this_text += text[:text.index(token) + len(token)]
                text = text[text.index(token) + len(token):]
                counter += 1
            except:
                pass
        sectors.append(this_text)
        return sectors
