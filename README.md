# Цель
Рассматривается торговая компания, занимающаяся поставками промышленной трубопроводной арматуры (задвижки, вентили, шаровые краны, и .т.д.). Имеется заявка от клиента на покупку продукции – перечень оборудования с техническими характеристиками.    В реальной жизни сроки могут быть очень сжатыми и нет времени запрашивать производителей и ждать от них цен.
Решено создать модель машинного обучения и на ее основе написать скрипт, который считывает техническую информацию из таблицы клиента и заносит в нее предсказанные цены.
В нашем распоряжении таблица с данными по предыдущим продажам с известными ценами.
# Результат
Лучшая модель -  осредненная модель на основе нескольких моделей LGBM, подобранных с помощью RandomizedSearchCV.
Данные можно улучшить за счет увеличения числа моделей и количества итераций, а также за счет увеличения данных для обучения
# Стек
pandas, numpy, seaborn, sklearn, Lightgbm, RandomizedSearchCV
# Стутус
Планируются улучшения
