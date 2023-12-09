# pip install deeppavlov
# pip install git+https://github.com/Koziev/rutokenizer
# python -m deeppavlov install ner_ontonotes_bert_mult
# python -m deeppavlov interact ner_ontonotes_bert_mult -d
# python -m deeppavlov install squad_bert
# python -m deeppavlov interact squad_bert -d
# python -m deeppavlov install squad_ru_bert
#
# import NER
#
# from deeppavlov import configs, build_model
#
#
# ner_model_ml = build_model(configs.ner.ner_ontonotes_bert_mult, download=True)
#
#
# print(ner_model_ml(["Меня зовут Бородин Степан и я работаю в компани И я первый, кто полетел в космос. Я прочитал книгу Льва Толстого Война и мир. Я вылетел из аэрпорта Домодедово в аэропорт внукво, там мне пришлось говорить на английском"]))
#


# NER.MarkUp()
# from deeppavlov import build_model, configs
# model_qa_ml = build_model(configs.squad.squad_ru_bert, download=True)
#
# context_ru = "Меня зовут Бородин Степан, я студент 2-го курса магистратуры, Уральского Федерального Университета. Живу я в городе Екатеринбург, мои адреса электронной почты test@test.com и mymail@mail.com. В ближайшее время, я еду на хакатон в Пермь."
#
# print(model_qa_ml([context_ru],
#       ["я обучаюсь?"]))

import NER

text = "Мы заключили договор 25 января 2005 года тел 89801856564 inn 00000000000"
marcup = NER.TextMarkUp(is_bert=False, is_pro_bert=True, download=True)
# print(marcup.get_bert_markup(text=text))
print(marcup.get_markup(text=text))