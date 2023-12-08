from enum import Enum
from typing import List, Dict, Any


class MarkUpType(Enum):
    PERSON = "PERSON"  # People including fictional
    NORP = "NORP"  # Nationalities or religious or political groups
    FACILITY = "FACILITY"  # Buildings, airports, highways, bridges, etc.
    ORGANIZATION = "ORGANIZATION"  # Companies, agencies, institutions, etc.
    GPE = "GPE"  # Countries, cities, states
    LOCATION = "LOCATION"  # Non-GPE locations, mountain ranges, bodies of water
    PRODUCT = "PRODUCT"  # Vehicles, weapons, foods, etc. (Not services)
    EVENT = "EVENT"  # Named hurricanes, battles, wars, sports events, etc.
    WORK_OF_ART = "WORK_OF_ART"  # Titles of books, songs, etc.
    LAW = "LAW"  # Named documents made into laws
    LANGUAGE = "LANGUAGE"  # Any named language
    DATE = "DATE"  # Absolute or relative dates or periods
    TIME = "TIME"  # Times smaller than a day
    PERCENT = "PERCENT"  # Percentage (including “%”)
    MONEY = "MONEY"  # Monetary values, including unit
    QUANTITY = "QUANTITY"  # Measurements such as weight or distance
    ORDINAL = "ORDINAL"  # “first”, “second”, etc.
    CARDINAL = "CARDINAL"  # Numerals that do not fall under another type
    NOTHING = "O"


# class MarkUpType(Enum):
#     PERSON = "PERSON"
#     LOCALITY = "LOCALITY"
#     ORGANISATION = "ORGANISATION"
#     DATE = "DATE"
#     MONEY = "MONEY"
#     PHONE = "PHONE"
#     INN = "INN"
#     KPP = "KPP"
#     BIC = "BIC"
#     SNILS = "SNILS"
#     EMAIL = "EMAIL"
#     URL = "URL"
#     NONE = None


class MarkUp:
    def __init__(self, item: Dict[str, Dict[str, str]]):
        self.__item = item
        self.__text = list(item.keys())[0]
        self.__is_empty = len(item[list(item.keys())[0]]) == 0
        self.__type = self.__get_type()
        self.__value = self.__item[list(self.__item.keys())[0]]

    @property
    def is_empty(self) -> bool:
        ""
        return self.__is_empty

    @property
    def type(self) -> MarkUpType:
        return self.__type

    @property
    def text(self) -> str:
        return self.__text

    def get_param_value(self, param: str) -> str:
        value = self.__item[list(self.__item.keys())[0]]
        if self.__type is MarkUpType.PERSON and param in ["Person"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.LOCALITY and param in ["Locality"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.ORGANISATION and param in ["Organisation"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.DATE and param in ["Year", "Month", "Day"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.MONEY and param in ["Amount", "Currency"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.PHONE and param in ["phoneNumber"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.INN and param in ["organizationINN", "personalINN"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.KPP and param in ["KPP"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.BIC and param in ["BIC"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.SNILS and param in ["SNILS"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.EMAIL and param in ["Email"]:
            return value[param] if param in value else None
        elif self.__type is MarkUpType.URL and param in ["Url"]:
            return value[param] if param in value else None
        else:
            raise Exception(f"Param {param} does not exist in type {self.__type}")

    def get_json(self) -> Dict[str, Dict[str, str]]:
        return self.__item

    def __get_type(self) -> MarkUpType:
        value = self.__item[list(self.__item.keys())[0]]
        if "Person" in value:
            return MarkUpType.PERSON
        elif "Locality" in value:
            return MarkUpType.LOCALITY
        elif "Organisation" in value:
            return MarkUpType.ORGANISATION
        elif "Year" in value or "Month" in value or "Day" in value:
            return MarkUpType.DATE
        elif "Amount" in value or "Currency" in value:
            return MarkUpType.MONEY
        elif "phoneNumber" in value:
            return MarkUpType.PHONE
        elif "organizationINN" in value or "personalINN" in value:
            return MarkUpType.INN
        elif "KPP" in value:
            return MarkUpType.KPP
        elif "BIC" in value:
            return MarkUpType.BIC
        elif "SNILS" in value:
            return MarkUpType.SNILS
        elif "Email" in value:
            return MarkUpType.EMAIL
        elif "Url" in value:
            return MarkUpType.URL
        else:
            return MarkUpType.NONE
