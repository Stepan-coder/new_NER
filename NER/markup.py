from enum import Enum
from typing import List, Dict, Any


class MarkUpType(Enum):
    PERSON = "PERSON"  # People including fictional
    NORP = "NORP"  # Nationalities or religious or political groups
    FACILITY = "FACILITY"  # Buildings, airports, highways, bridges, etc.
    ORGANIZATION = "ORG"  # Companies, agencies, institutions, etc.
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
    PHONE = "PHONE"
    INN = "INN"
    KPP = "KPP"
    BIC = "BIC"
    SNILS = "SNILS"
    EMAIL = "EMAIL"
    URL = "URL"
    NOTHING = "O"
    NONE = None



