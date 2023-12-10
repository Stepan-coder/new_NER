import re
import json
import time
import torch
import warnings
import rutokenizer
from tqdm import tqdm
from NER.markup import *
from NER.mark_up_block import *
from typing import List, Dict, Any
from transformers import AutoTokenizer
from deeppavlov import configs, build_model
from natasha import NamesExtractor, AddrExtractor, DatesExtractor, MoneyExtractor, MorphVocab


# pip install git+https://github.com/Koziev/rutokenizer


class TextMarkUp:
    """
    :ru Класс реализующий поиск именованных сущностей в тексте.
    :en A class that implements the search for named entities in the text.

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

    def get_markup(self, text: str) -> List[MarkUpBlock]:
        """
        :ru Этот метод получает строку с русским текстом в качестве входных данных и выдает json с разметкой в качестве
         выходных данных
        :en This method receives a string with russian text as input, and gives json with markup as output.

        :param text:en Строка, которая нуждается в разметке.
        :param text:en A string that needs markup.
        :type text: str
        """
        if self._is_bert:
            start_index = 0
            text_markup = []
            text_sector = self._prepear_text_to_bert(text=text, border=200)
            for sector in tqdm(text_sector, desc="Getting Named Entities..."):
                markup = self.get_bert_markup(text=sector, start_index=start_index)
                text_markup += markup
                start_index = markup[-1].end
            text_markup = self.rebuild_markup(text_markup=text_markup, delete_empty=True, join_similar=True)
        else:
            text_markup = [MarkUpBlock(text=text, block_type=MarkUpType.NOTHING, start=0, end=len(text))]
        text_markup = self.rebuild_markup(self.get_ikz_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_inn_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_kpp_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_ogrn_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_okpo_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_oktmo_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_okato_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_bic_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_phone_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_snils_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_emails_markup(text_markup=text_markup))
        text_markup = self.rebuild_markup(self.get_urls_markup(text_markup=text_markup))
        # text_markup = self.rebuild_markup(self.get_date_markup(text_markup=text_markup))
        return self.rebuild_markup(text_markup=text_markup, delete_empty=True, join_similar=True)

    def get_bert_markup(self, text: str, start_index: int = 0) -> List[MarkUpBlock]:
        """
        :param start_index:
        :param text: The text witch we need tu markup
        :return: List[Dict[str, dict]]
        """
        last_block_type = None
        text_markup = []
        tokens, tags = self._ner([text])
        for tok, tag in zip(tokens[0], tags[0]):
            gap = text[:text.index(tok)]
            text = text[text.index(tok) + len(tok):]
            block_type = MarkUpType(str(tag).replace("B-", "").replace("I-", ""))
            if tag.startswith("O") or (block_type != last_block_type and
                                       (not tag.startswith("I-") or (len(text_markup) == 0 and tag.startswith("I-")))):
                text_markup.append(MarkUpBlock(text=(gap + tok).strip(),
                                               block_type=block_type,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                dates = self._dates_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for date in dates:
                    if date["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: date["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + date["start"]))
                    markup = {}
                    if date.as_json["fact"].year is not None:
                        markup["Year"] = date.as_json["fact"].year
                    if date.as_json["fact"].month is not None:
                        markup["Month"] = date.as_json["fact"].month
                    if date.as_json["fact"].day is not None:
                        markup["Day"] = date.as_json["fact"].day
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[date.as_json["start"]:
                                                                               date.as_json["stop"]],
                                                     block_type=MarkUpType.DATE,
                                                     start=increment + left_bounce + date.as_json["start"],
                                                     end=increment + left_bounce + date.as_json["stop"],
                                                     attachments=markup))
                    left_bounce = date.as_json["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                phones = TextMarkUp._phone_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for phone in phones:
                    if phone["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: phone["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + phone["start"]))
                    markup = {}
                    if phone["fact"]["phoneNumber"] is not None:
                        markup["phoneNumber"] = phone["fact"]["phoneNumber"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[phone["start"]:
                                                                               phone["stop"]],
                                                     block_type=MarkUpType.PHONE,
                                                     start=increment + left_bounce + phone["start"],
                                                     end=increment + left_bounce + phone["stop"],
                                                     attachments=markup))
                    left_bounce = phone["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                inns = TextMarkUp._INN_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for inn in inns:
                    if inn["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: inn["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + inn["start"]))
                    markup = {}
                    if "organizationINN" in inn["fact"]:
                        markup["organizationINN"] = inn["fact"]["organizationINN"]
                    elif "personalINN" in inn["fact"]:
                        markup["personalINN"] = inn["fact"]["personalINN"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[inn["start"]:
                                                                               inn["stop"]],
                                                     block_type=MarkUpType.INN,
                                                     start=increment + left_bounce + inn["start"],
                                                     end=increment + left_bounce + inn["stop"],
                                                     attachments=markup))
                    left_bounce = inn["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                kpps = TextMarkUp._KPP_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for kpp in kpps:
                    if kpp["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: kpp["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + kpp["start"]))
                    markup = {}
                    if "KPP" in kpp["fact"]:
                        markup["KPP"] = kpp["fact"]["KPP"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[kpp["start"]:
                                                                               kpp["stop"]],
                                                     block_type=MarkUpType.KPP,
                                                     start=increment + left_bounce + kpp["start"],
                                                     end=increment + left_bounce + kpp["stop"],
                                                     attachments=markup))
                    left_bounce = kpp["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_ikz_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                ikzs = TextMarkUp._IKZ_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for ikz in ikzs:
                    if ikz["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: ikz["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + ikz["start"]))
                    markup = {}
                    if "IKZ" in ikz["fact"]:
                        markup["IKZ"] = ikz["fact"]["IKZ"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[ikz["start"]:
                                                                               ikz["stop"]],
                                                     block_type=MarkUpType.IKZ,
                                                     start=increment + left_bounce + ikz["start"],
                                                     end=increment + left_bounce + ikz["stop"],
                                                     attachments=markup))
                    left_bounce = ikz["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_ogrn_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                ogrns = TextMarkUp._OGRN_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for ogrn in ogrns:
                    if ogrn["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: ogrn["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + ogrn["start"]))
                    markup = {}
                    if "OGRN" in ogrn["fact"]:
                        markup["OGRN"] = ogrn["fact"]["OGRN"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[ogrn["start"]:
                                                                               ogrn["stop"]],
                                                     block_type=MarkUpType.OGRN,
                                                     start=increment + left_bounce + ogrn["start"],
                                                     end=increment + left_bounce + ogrn["stop"],
                                                     attachments=markup))
                    left_bounce = ogrn["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_okpo_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                okpos = TextMarkUp._OKPO_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for okpo in okpos:
                    if okpo["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: okpo["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + okpo["start"]))
                    markup = {}
                    if "OKPO" in okpo["fact"]:
                        markup["OKPO"] = okpo["fact"]["OKPO"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[okpo["start"]:
                                                                               okpo["stop"]],
                                                     block_type=MarkUpType.OKPO,
                                                     start=increment + left_bounce + okpo["start"],
                                                     end=increment + left_bounce + okpo["stop"],
                                                     attachments=markup))
                    left_bounce = okpo["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    def get_oktmo_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                oktmos = TextMarkUp._OKTMO_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for oktmo in oktmos:
                    if oktmo["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: oktmo["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + oktmo["start"]))
                    markup = {}
                    if "OKTMO" in oktmo["fact"]:
                        markup["OKTMO"] = oktmo["fact"]["OKTMO"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[oktmo["start"]:
                                                                               oktmo["stop"]],
                                                     block_type=MarkUpType.OKTMO,
                                                     start=increment + left_bounce + oktmo["start"],
                                                     end=increment + left_bounce + oktmo["stop"],
                                                     attachments=markup))
                    left_bounce = oktmo["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup


    def get_okato_markup(self, text_markup: List[MarkUpBlock]) -> List[MarkUpBlock]:
        """
        This class receives the pre-marked text as input and places the phones from the pieces
         that have not yet been marked up.
        :param text_markup: Pre-marked text
        :return: List[Dict[str, dict]]
        """
        result_markup = []
        for tm in range(len(text_markup)):
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                okatos = TextMarkUp._OKATO_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for okato in okatos:
                    if okato["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: okato["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + okato["start"]))
                    markup = {}
                    if "OKATO" in okato["fact"]:
                        markup["OKATO"] = okato["fact"]["OKATO"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[okato["start"]:
                                                                               okato["stop"]],
                                                     block_type=MarkUpType.OKATO,
                                                     start=increment + left_bounce + okato["start"],
                                                     end=increment + left_bounce + okato["stop"],
                                                     attachments=markup))
                    left_bounce = okato["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                bics = TextMarkUp._BIC_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for bic in bics:
                    if bic["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: bic["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + bic["start"]))
                    markup = {}
                    if "BIC" in bic["fact"]:
                        markup["BIC"] = bic["fact"]["BIC"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[bic["start"]:
                                                                               bic["stop"]],
                                                     block_type=MarkUpType.BIC,
                                                     start=increment + left_bounce + bic["start"],
                                                     end=increment + left_bounce + bic["stop"],
                                                     attachments=markup))
                    left_bounce = bic["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                snilses = TextMarkUp._snils_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for snils in snilses:
                    if snils["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: snils["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + snils["start"]))
                    markup = {}
                    if "SNILS" in snils["fact"]:
                        markup["SNILS"] = snils["fact"]["SNILS"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[snils["start"]:
                                                                               snils["stop"]],
                                                     block_type=MarkUpType.SNILS,
                                                     start=increment + left_bounce + snils["start"],
                                                     end=increment + left_bounce + snils["stop"],
                                                     attachments=markup))
                    left_bounce = snils["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
                increment = text_markup[tm].start
                left_bounce = 0
                for email in emails:
                    if email["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: email["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + email["start"]))
                    markup = {}
                    if "Email" in email["fact"]:
                        markup["Email"] = email["fact"]["Email"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[email["start"]:
                                                                               email["stop"]],
                                                     block_type=MarkUpType.EMAIL,
                                                     start=increment + left_bounce + email["start"],
                                                     end=increment + left_bounce + email["stop"],
                                                     attachments=markup))
                    left_bounce = email["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
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
            if text_markup[tm].block_type == MarkUpType.NOTHING:
                urls = TextMarkUp._url_extractor(text_markup[tm].text)
                increment = text_markup[tm].start
                left_bounce = 0
                for url in urls:
                    if url["start"] > 0:
                        result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce: url["start"]],
                                                         block_type=MarkUpType.NOTHING,
                                                         start=increment + left_bounce,
                                                         end=increment + left_bounce + url["start"]))
                    markup = {}
                    if "Url" in url["fact"]:
                        markup["Url"] = url["fact"]["Url"]
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[url["start"]:
                                                                               url["stop"]],
                                                     block_type=MarkUpType.URL,
                                                     start=increment + left_bounce + url["start"],
                                                     end=increment + left_bounce + url["stop"],
                                                     attachments=markup))
                    left_bounce = url["stop"]
                if text_markup[tm].end - (increment + left_bounce) > 0:
                    result_markup.append(MarkUpBlock(text=text_markup[tm].text[left_bounce:
                                                                               text_markup[tm].end - increment],
                                                     block_type=MarkUpType.NOTHING,
                                                     start=increment + left_bounce,
                                                     end=text_markup[tm].end))
            else:
                result_markup.append(text_markup[tm])
        return result_markup

    @staticmethod
    def rebuild_markup(text_markup: List[MarkUpBlock],
                       delete_empty: bool = False,
                       join_similar: bool = False) -> List[MarkUpBlock]:
        """
        This method reformats the markup, combines unmarked elements (the consequences of using Natasha),
        removes empty elements (arise as a result of using the algorithm)
        :param join_similar:
        :param delete_empty:
        :param text_markup: Markuped text
        :return List[Dict[str, dict]]
        """
        text_markup = sorted(text_markup, key=lambda x: x.start)
        # for index in range(len(text_markup) - 1):
        #     if text_markup[index].block_type == MarkUpType.NOTHING and \
        #             text_markup[index + 1].block_type == MarkUpType.NOTHING:
        #         result.append(MarkUpBlock(text=f"{text_markup[index].text} {text_markup[index + 1].text}".strip(),
        #                                   block_type=MarkUpType.NOTHING.value,
        #                                   start=text_markup[index].start,
        #                                   end=text_markup[index + 1].end))
        #     else:
        #         result.append(text_markup[index])
        if delete_empty:
            text_markup = [text_markup[index] for index in range(len(text_markup)) if len(text_markup[index].text) > 0]

        if join_similar:
            index = 0
            result = []
            last_tag = None
            while index < len(text_markup):

                if text_markup[index].block_type != last_tag:
                    result.append(MarkUpBlock(text=text_markup[index].text,
                                              block_type=text_markup[index].block_type,
                                              start=text_markup[index].start,
                                              end=text_markup[index].end))
                else:

                    result[-1].text = f"{result[-1].text} {text_markup[index].text}".strip()
                    result[-1].text = result[-1].text.strip()
                    result[-1].end = result[-1].start + len(result[-1].text)
                last_tag = text_markup[index].block_type
                index += 1
            return sorted(result, key=lambda x: x.start)
        return sorted(text_markup, key=lambda x: x.start)

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
        re_phone = '(тел|тел.|телефон|факс|ф.).?.?' \
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
        re_org_inn = '(инн).?.?[0-9]{10}(\s|\D)'
        re_per_inn = '(инн).?.?[0-9]{12}(\s|\D)'
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
        re_kpp = '(кпп).?.?\d{9}(\s|\D)'
        text = f"{text} "
        while re.search(re_kpp, text, re.IGNORECASE) is not None:
            kpp = re.search(re_kpp, text, re.IGNORECASE)
            yield {"start": kpp.start() + left_bounce, "stop": kpp.end() + left_bounce, "fact":
                {"KPP": TextMarkUp._clean_string(text[kpp.start(): kpp.end()]).strip()}}
            left_bounce = kpp.end()
            text = text[kpp.end():]

    @staticmethod
    def _IKZ_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for ikz's in the text.
        Supported:
        икз XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        икзXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        """
        left_bounce = 0
        re_ikz = '(икз).?.?\d{36}(\s|\D)?'
        text = f"{text} "
        while re.search(re_ikz, text, re.IGNORECASE) is not None:
            ikz = re.search(re_ikz, text, re.IGNORECASE)
            yield {"start": ikz.start() + left_bounce, "stop": ikz.end() + left_bounce, "fact":
                {"IKZ": TextMarkUp._clean_string(text[ikz.start(): ikz.end()]).strip()}}
            left_bounce = ikz.end()
            text = text[ikz.end():]

    @staticmethod
    def _OGRN_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for ikz's in the text.
        Supported:
        огрн XXXXXXXXXXXXX
        огрнXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        """
        left_bounce = 0
        re_ogrn = '(огрн).?.?\d{13}(\s|\D)'
        text = f"{text} "
        while re.search(re_ogrn, text, re.IGNORECASE) is not None:
            ogrn = re.search(re_ogrn, text, re.IGNORECASE)
            yield {"start": ogrn.start() + left_bounce, "stop": ogrn.end() + left_bounce, "fact":
                {"OGRN": TextMarkUp._clean_string(text[ogrn.start(): ogrn.end()]).strip()}}
            left_bounce = ogrn.end()
            text = text[ogrn.end():]

    @staticmethod
    def _OKPO_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for ikz's in the text.
        Supported:
        окпо XXXXXXXX
        окпоXXXXXXXX
        """
        left_bounce = 0
        re_okpo = '(окпо).?.?\d{8}(\s|\D)'
        text = f"{text} "
        while re.search(re_okpo, text, re.IGNORECASE) is not None:
            okpo = re.search(re_okpo, text, re.IGNORECASE)
            yield {"start": okpo.start() + left_bounce, "stop": okpo.end() + left_bounce, "fact":
                {"OKPO": TextMarkUp._clean_string(text[okpo.start(): okpo.end()]).strip()}}
            left_bounce = okpo.end()
            text = text[okpo.end():]

    @staticmethod
    def _OKTMO_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for ikz's in the text.
        Supported:
        октмо XXXXXXXXX
        октмоXXXXXXXXX
        """
        left_bounce = 0
        re_oktmo = '(октмо).?.?\d{9}(\s|\D)'
        text = f"{text} "
        while re.search(re_oktmo, text, re.IGNORECASE) is not None:
            oktmo = re.search(re_oktmo, text, re.IGNORECASE)
            yield {"start": oktmo.start() + left_bounce, "stop": oktmo.end() + left_bounce, "fact":
                {"OKTMO": TextMarkUp._clean_string(text[oktmo.start(): oktmo.end()]).strip()}}
            left_bounce = oktmo.end()
            text = text[oktmo.end():]

    @staticmethod
    def _OKATO_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for ikz's in the text.
        Supported:
        окато XXX XXX XXX XX
        окатоXXXXXXXXXXX
        """
        left_bounce = 0
        re_okato = '(окато).?.?\d{11}(\s|\D)'
        text = f"{text} "
        while re.search(re_okato, text, re.IGNORECASE) is not None:
            okato = re.search(re_okato, text, re.IGNORECASE)
            yield {"start": okato.start() + left_bounce, "stop": okato.end() + left_bounce, "fact":
                {"OKATO": TextMarkUp._clean_string(text[okato.start(): okato.end()]).strip()}}
            left_bounce = okato.end()
            text = text[okato.end():]

    @staticmethod
    def _BIC_extractor(text: str) -> List[Dict[str, Any]] or None:
        """
        This is a generator that searches for bic's in the text.
        Supported:
        БИК XXXXXXXXXX
        БИКXXXXXXXXXX
        """
        left_bounce = 0
        re_bic = '(БИК).?.?\d{9}(\s|\D)?'
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
        re_snils = '(снилс).?.?\d{3}-\d{3}-\d{3}\x20?-?\x20?\d{2}(\s|\D)'
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
        re_email = 'e?-?mail?.?.?[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}'
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
        # while '\n' in text or '  ' in text:
        #     text = text.replace("\n", " ").replace("  ", " ")
        tokens = self._tokenizer.tokenize(text)
        counter = 0
        this_text = ""
        sectors = []
        for token in tokens:
            this_text += text[:text.index(token) + len(token)]
            text = text[text.index(token) + len(token):]
            counter += 1
            if counter >= border:
                sectors.append(this_text)
                this_text = ""
                counter = 0
        sectors.append(this_text)
        return sectors
