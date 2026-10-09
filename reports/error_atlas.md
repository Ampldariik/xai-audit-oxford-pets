# Атлас ошибок основной модели (test)

Всего тестовых кадров: 1050, ошибок: 110 (10.5%).

| Категория | Ошибок | Доля | Правило |
|---|---|---|---|
| Зависимость от фона | 25 | 22.7% | FG-energy<0.35, или без фона верно, или без объекта тот же ответ |
| Низкое качество кадра | 18 | 16.4% | размытие / мелкий объект / пере- или недосвет / малое разрешение |
| Сходство похожих пород | 55 | 50.0% | тот же вид, истинный класс в топ-3 |
| Прочее / нетипичный кадр | 12 | 10.9% | остальное |

## Самые частые пары «истина → предсказание»

| Истина | Предсказание | n |
|---|---|---|
| american_pit_bull_terrier | staffordshire_bull_terrier | 7 |
| Egyptian_Mau | Bengal | 5 |
| english_setter | english_cocker_spaniel | 5 |
| Maine_Coon | Ragdoll | 4 |
| Ragdoll | Birman | 4 |
| British_Shorthair | Russian_Blue | 3 |
| american_pit_bull_terrier | american_bulldog | 3 |
| english_cocker_spaniel | newfoundland | 3 |
| beagle | basset_hound | 3 |
| american_bulldog | boxer | 3 |

Верных предсказаний «по неверной причине» (Clever Hans): 7 (0.7% от верных).