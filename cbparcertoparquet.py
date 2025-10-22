import requests
import pandas as pd
import re
from pathlib import Path
BASE_URL = "https://cbr.ru/dataservice"
YEAR_FROM = 2010
YEAR_TO = 2025


def get_publications():
    url = f"{BASE_URL}/publications"
    response = requests.get(url)
    response.raise_for_status()
    return [p for p in response.json() if p.get("NoActive") != 1]  # отфильтровываем неактивные

def get_datasets(publication_id):
    url = f"{BASE_URL}/datasets"
    response = requests.get(url, params={"publicationId": publication_id})
    response.raise_for_status()
    return response.json()

def get_measures(dataset_id):
    url = f"{BASE_URL}/measures"
    response = requests.get(url, params={"datasetId": dataset_id})
    response.raise_for_status()
    return response.json().get("measure", [])

def get_data(y1, y2, dataset_id, publication_id, measure_id=None):
    url = f"{BASE_URL}/data"
    params = {
        "y1": y1,
        "y2": y2,
        "datasetId": dataset_id,
        "publicationId": publication_id
    }
    if measure_id is not None:
        params["measureId"] = measure_id
    response = requests.get(url, params=params)
    response.raise_for_status()
    return response.json()

def abbr(text):
    return ''.join([w[:3] for w in re.findall(r'\w+', text)])[:15]

# ТОЛЬКО нужные публикации и их показатели
required_structure = {
    5: [  # Денежные агрегаты (структура денежной массы)
        5,  # Денежный агрегат М0
        6,  # Денежный агрегат М1
        7,  # Денежный агрегат М2
        8,  # Широкая денежная масса
    ],
    8: [  # Платежный баланс — сводные сальдо счетов
        9,  # Сальдо счета текущих операций
        10, # Сальдо счета операций с капиталом
        11, # Сальдо финансового счета
        12, # Чистые ошибки и пропуски
    ],
    9: [  # Платежный баланс — компоненты счета текущих операций
        13, # Товары
        14, # Услуги
        15, # Первичные доходы
        16, # Вторичные доходы
    ],
    10: [  # Счет операций с капиталом
        17, # Приобретение/выбытие непроизведенных нефинансовых активов
        18, # Капитальные трансферты
    ],
    11: [  # Финансовый счет — структура инвестиций
        19, # Прямые инвестиции
        20, # Портфельные инвестиции
        21, # Производные финансовые инструменты и опционы
        22, # Прочие инвестиции
        23, # Резервные активы
    ],
    12: [  # Балансирующая статья платежного баланса
        24, # Чистые ошибки и пропуски
    ],
    14: [  # Средние ставки по кредитам (всего, без валютного деления)
        25, # Ставки по кредитам нефинансовым организациям
        26, # Ставки по кредитам нефинансовым организациям — субъектам МСП
        27, # Ставки по кредитам физическим лицам
        28, # Ставки по автокредитам
        29, # Ставки по ипотечным жилищным кредитам
    ],
    15: [  # Ставки по кредитам в рублях
        30, # Ставки по кредитам нефинансовым организациям в рублях
        31, # Ставки по кредитам нефинансовым организациям — субъектам МСП в рублях
        32, # Ставки по кредитам физическим лицам в рублях
        33, # Ставки по автокредитам в рублях
        34, # Ставки по ипотечным жилищным кредитам
    ],
    16: [  # Ставки по кредитам нефинансовым организациям (укороченный набор)
        35, # Ставки по кредитам нефинансовым организациям
        36, # Ставки по кредитам нефинансовым организациям — субъектам МСП
    ],
    18: [  # Ставки по депозитам (вклады физических и юридических лиц)
        37, # Ставки по вкладам физических лиц
        38, # Ставки по вкладам нефинансовых организаций
    ],
    19: [  # Ставки по депозитам в рублях
        39, # Ставки по вкладам физических лиц в рублях
        40, # Ставки по вкладам нефинансовых организаций в рублях
    ],
    20: [  # Объемы и задолженность по кредитам (агрегированные)
        41, # Объем кредитов
        42, # Задолженность по кредитам
        43, # Просроченная задолженность по кредитам
    ],
    21: [  # Кредитная активность — детализированные показатели
        44, # Количество кредитов
        45, # Объем кредитов
        46, # Задолженность по кредитам
        55, # Сезонно скорректированная задолженность по кредитам
        56, # Сезонно скорректированная задолженность с учетом прав требования
        47, # Просроченная задолженность по кредитам
        48, # Средневзвешенный срок по кредитам
    ],
    22: [  # Кредитная активность — нефинансовые организации
        49, # Объем кредитов
        50, # Задолженность по кредитам
        51, # Просроченная задолженность по кредитам
    ],
    23: [  # Кредитная активность — физические лица
        52, # Объем кредитов
        53, # Задолженность по кредитам
        54, # Просроченная задолженность по кредитам
    ],
    25: [  # Диффузные индикаторы по секторам и ожиданиям (опрос предприятий)
        58,  # Экономика всего
        59,  # Промышленность
        60,  # Добыча
        61,  # Обработка
        62,  # Обработка (потребительские товары)
        63,  # Обработка (инвестиционные товары)
        64,  # Обработка (промежуточные товары)
        65,  # ЭЭГП
        66,  # Водоснабжение
        67,  # Сельское хозяйство
        68,  # Строительство
        69,  # Торговля всего
        70,  # Автотранспорт
        71,  # Оптовая торговля
        72,  # Розничная торговля
        73,  # Транспортировка и хранение
        74,  # Услуги
        75,  # Загрузка производственных мощностей
        76,  # Инвестиционная активность
        77,  # Ожидания изменения инвестиционной активности
        78,  # Обеспеченность персоналом
        79,  # Ожидания изменения численности персонала
        140, # Ожидания изменения расходов на оплату труда
    ],
    26: [  # Участники опроса Банка России
        57,  # Число участников опроса
    ],
    28: [  # Диффузные индикаторы по секторам — панель 1
        80,  # Экономика всего
        83,  # Промышленность
        86,  # Добыча
        89,  # Обработка
        92,  # Сельское хозяйство
        95,  # Строительство
        98,  # Торговля всего
        101, # Транспортировка и хранение
        104, # Услуги
        107, # Загрузка производственных мощностей
        110, # Инвестиционная активность
        113, # Ожидания изменения инвестиционной активности
        116, # Обеспеченность персоналом
        119, # Ожидания изменения численности персонала
    ],
    29: [  # Диффузные индикаторы по секторам — панель 2
        81,  # Экономика всего
        84,  # Промышленность
        87,  # Добыча
        90,  # Обработка
        93,  # Сельское хозяйство
        96,  # Строительство
        99,  # Торговля всего
        102, # Транспортировка и хранение
        105, # Услуги
        108, # Загрузка производственных мощностей
        111, # Инвестиционная активность
        114, # Ожидания изменения инвестиционной активности
        117, # Обеспеченность персоналом
        120, # Ожидания изменения численности персонала
    ],
    30: [  # Диффузные индикаторы по секторам — панель 3
        82,  # Экономика всего
        85,  # Промышленность
        88,  # Добыча
        91,  # Обработка
        94,  # Сельское хозяйство
        97,  # Строительство
        100, # Торговля всего
        103, # Транспортировка и хранение
        106, # Услуги
        109, # Загрузка производственных мощностей
        112, # Инвестиционная активность
        115, # Ожидания изменения инвестиционной активности
        118, # Обеспеченность персоналом
        121, # Ожидания изменения численности персонала
    ],
    33: [  # Курсы валют — уровни (type=2)
        127, # Номинальный курс
        128, # Средний номинальный курс за период
        139, # Средний номинальный курс за период с начала года
    ],
    34: [  # Индексы валютного курса — номинальный/реальный, эффективные (type=1)
        129, # Индекс номинального курса
        130, # Индекс номинального эффективного курса рубля
        131, # Индекс реального курса
        132, # Индекс реального эффективного курса рубля
    ],
    35: [  # Курсы валют — уровни (type=2)
        133, # Номинальный курс
        134, # Средний номинальный курс за период
        141, # Средний номинальный курс за период с начала года
    ],
    36: [  # Индексы валютного курса — номинальный/реальный, эффективные (type=1)
        135, # Индекс номинального курса
        136, # Индекс номинального эффективного курса рубля
        137, # Индекс реального курса
        138, # Индекс реального эффективного курса рубля
    ],
}



# Получаем все публикации, но фильтруем нужные
all_publications = get_publications()
publications = [p for p in all_publications if p["id"] in required_structure]

print(f"🔍 Выгружаем только {len(publications)} заданных публикаций")

dataset_map = {
    "Ставки по кредитам нефинансовым организациям": "СтаКредНеФин",
    "Ставки по кредитам нефинансовым организациям-субъектам МСП": "СтаКредМСП",
    "Ставки по кредитам физическим лицам": "СтаКредФЛ",
    "Ставки по автокредитам": "СтаКредАвто",
    "Ставки по ипотечным жилищным кредитам": "СтаИпот",
    "Ставки по вкладам (депозитам) физических лиц":"СтаВклФЛ",
    "Ставки по вкладам (депозитам) нефинансовых организаций":"СтаВклЮЛ",
    "Объем кредитов": "ОбъКред",
    "Ставки по вкладам (депозитам) физических лиц в рублях": "СтаВклФЛРуб",
    "Ставки по вкладам (депозитам) нефинансовых организаций в рублях":"СтаВклНеФинРуб",
    "Задолженность по кредитам": "ДолгКред",
    "Просроченная задолженность по кредитам": "Просрочка",
    "Количество кредитов": "КолвоКред",
    "Сезонно скорректированная задолженность по кредитам": "СезонКред",
    "Сезонно скорректированная задолженность по кредитам с учетом приобретенных кредитными организациями прав требования": "СезонКредТреб",
    "Средневзвешенный срок по кредитам": "ВзвСрокКред",
    "Денежный агрегат М0": "М0",
    "Денежный агрегат М1": "М1",
    "Денежный агрегат М2": "М2",
    "Широкая денежная масса": "Масса",
    "Сальдо счета текущих операций": "СТО",
    "Сальдо счета операций с капиталом": "СОК",
    "Сальдо финансового счета": "СФС",
    "Чистые ошибки и пропуски": "ЧОП",
    "Товары": "Товары",
    "Услуги": "Услуги",
    "Первичные доходы": "ПервДох",
    "Вторичные доходы": "ВторДох",
    "Приобретение/выбытие непроизведенных нефинансовых активов": "НефинАкт",
    "Капитальные трансферты": "КапТранс",
    "Прямые инвестиции": "ПрямыеИн",
    "Портфельные инвестиции": "ПортфИн",
    "Производные финансовые инструменты (кроме резервов) и опционы на акции для работников": "Опционы",
    "Прочие инвестиции": "ПрочИн",
    "Резервные активы": "Резерв",
    "Номинальный курс": "НК",
    "Средний номинальный курс за период": "СрНК",
    "Средний номинальный курс за период с начала года": "СрНК_YtD",
    "Индекс номинального курса": "ИнНК",
    "Индекс номинального эффективного курса рубля к иностранным валютам": "ИнНЭфКРубкИн",
    "Индекс реального курса": "ИнРК",
    "Индекс реального эффективного курса рубля к иностранным валютам": "ИнРЭфКРубИн",
    "Экономика всего": "Эк_всего",
    "Промышленность": "Пром",
    "Добыча": "Доб",
    "Обработка": "Обр",
    "Обработка (потребительские товары)": "Обр_потреб",
    "Обработка (инвестиционные товары)": "Обр_инв",
    "Обработка (промежуточные товары)": "Обр_промеж",
    "ЭЭГП": "ЭЭГП",
    "Водоснабжение":"Вод",
    "Сельское хозяйство": "СХ",
    "Строительство": "Строй",
    "Торговля всего": "Торг_всего",
    "Автотранспорт": "АвтТран",
    "Оптовая": "Опт",
    "Розничная": "Розн",
    "Транспортировка и хранение":"ТранХран",
    "Загрузка производственных мощностей":"ПроизвМощ",
    "Инвестиционная активность": "ИнвАкт",
    "Ожидания изменения инвестиционной активности": "ОжДелИнвАкт",
    "Ожидания изменения численности": "ОжДелЧисл",
    "Ожидания изменения расходов на оплату труда": "ОжДелЗП",
    "Число участников опроса Банка России": "ЧУОпрЦБ"

}

output_dir = Path("cbr_parquet")
output_dir.mkdir(parents=True, exist_ok=True)

for existing_file in output_dir.glob("*.parquet"):
    existing_file.unlink()

metadata_rows = []
saved_files = set()

for pub in publications:
    pub_id = pub["id"]
    pub_name = pub["category_name"]
    print(f"\n📘 Публикация ID={pub_id}: {pub_name}")

    try:
        datasets = get_datasets(pub_id)
    except Exception as e:
        print(f"❌ Ошибка при получении показателей: {e}")
        continue

    for dataset in datasets:
        if dataset["id"] not in required_structure[pub_id]:
            continue  # пропускаем лишние показатели

        dataset_id = dataset["id"]
        dataset_name = dataset["name"] or f"dataset_{dataset_id}"
        dataset_type = dataset.get("type", 0)

        print(f"📡  → Показатель ID={dataset_id}: {dataset_name} (type={dataset_type})")

        # measureId обязателен только для type == 1
        measure_ids = []
        measure_map = {}
        if dataset_type == 1:
            try:
                measures = get_measures(dataset_id)
                measure_ids = [m["id"] for m in measures if m["id"] is not None]
                measure_map = {m["id"]: m["name"] for m in measures if m["id"] is not None}
                if not measure_ids:
                    print("⚠ Нет разрезов, пропускаем")
                    continue
            except:
                continue
        else:
            measure_ids = [None]  # хотя бы один запуск

        # фильтрация по dataset_id и нужным measureId
        if dataset_id in [30, 31, 32, 33]:
            measure_ids = [23, 42, 55, 64, 72, 87, 95, 106]
        elif dataset_id in [34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]:
            measure_ids = [22]
        # остальные оставляем как есть

        pub_map = {
            "В целом по Российской Федерации": "Рус",
            "В территориальном разрезе": "Раз",
            "В разрезе по видам экономической деятельности": "Отр",
            "По кредитам физическим лицам": "КредФЛ",
            "По ипотечным жилищным кредитам": "Ипот",
            "По кредитам юридическим лицам и индивидуальным предпринимателям (в т.ч. МСП)": "КредЮЛиИП",
            "По кредитам субъектам МСП": "КредМСП",
            "Структура денежной массы": "ДенМас",
            "Ключевые агрегаты": "КлАгр",
            "Счет текущих операций": "СТО",
            "Счет операций с капиталом": "СОК",
            "Финансовый счет": "ФС",
            "Чистые ошибки и пропуски": "ЧОП",
            "Номинальные курсы иностранных валют к рублю (рублей за единицу иностранной валюты) (ежемесячные данные)": "НКИкР_мес",
            "Индексы обменного курса рубля (ежемесячные данные)": "ИнОКРуб_мес",
            "Номинальные курсы иностранных валют к рублю (рублей за единицу иностранной валюты) (ежеквартальные данные)": "НКИкР_ежк",
            "Индексы обменного курса рубля (ежеквартальные данные)": "ИнОКРуб_ежк",
            "Данные мониторинга всего": "ДМ",
            "Участники опроса": "УчОпр",
            "Крупные": "Круп",
            "Средние": "Сред",
            "Малые и микро": "МалМик"
        }

        meas_map = {
            "Центральный федеральный округ": "ЦентрОкр",
            "Северо-Западный федеральный округ": "СевЗапОкр",
            "Южный федеральный округ": "ЮжнОкруг",
            "Северо-Кавказский федеральный округ": "СевКавОкр",
            "Приволжский федеральный округ": "ПривОкр",
            "Уральский федеральный округ": "УралОкр",
            "Сибирский федеральный округ": "СибОкр",
            "Дальневосточный федеральный округ": "ДВОкр",
            "Российская Федерация": "РФ",
            "Прочее": "Проч",
            "В рублях": "руб",
            "В долларах США": "дол",
            "В евро": "евр",
            "A. Сельское, лесное хозяйство, охота, рыболовство и рыбоводство": "AСелл",
            "B. Добыча полезных ископаемых": "Добыч",
            "C. Обрабатывающие производства": "Обраб",
            "D. Обеспечение электрической энергией, газом и паром; кондиционирование воздуха": "Энерг",
            "E. Водоснабжение; водоотведение, организация сбора и утилизации отходов, деятельность по ликвидации загрязнений": "Водв",
            "F. Строительство": "Строй",
            "G. Торговля оптовая и розничная; ремонт автотранспортных средств и мотоциклов": "Торг",
            "H. Транспортировка и хранение": "Тран",
            "I. Деятельность гостиниц и предприятий общественного питания": "Гот",
            "J. Деятельность в области информации и связи": "Инф",
            "L. Деятельность по операциям с недвижимым имуществом": "Недв",
            "M. Деятельность профессиональная, научная и техническая": "Проф",
            "P. Образование": "Обр",
            "R. Деятельность в области культуры, спорта, организации досуга и развлечений": "Культ",
            "В целом по Российской Федерации": "Рус",
            "В % прироста к декабрю предыдущего года": "%дек-Y",
            "В % прироста к предыдущему периоду": "%-Пер",
            "В % прироста к соответствующему периоду предыдущего года": "%-гг",
            "Исходные данные": "Исх",
            "Сезонно скорректированные данные": "СезСкор",
            "Экономика": "Экон",
            "Промышленное производство": "Пром",
            "Добыча полезных ископаемых": "Добыч",
            "Обрабатывающие производства": "Обраб",
            "Сельское, лесное хозяйство, охота, рыболовство и рыбоводство": "СХ",
            "Строительство": "Строй",
            "Торговля оптовая и розничная, ремонт автотранспортных средств и мотоциклов": "Торг",
            "Транспортировка и хранение": "ТранХран",
            "Услуги": "Услуги"
        }

        for measure_id in measure_ids:
            meas_abbr_raw = "" if measure_id is None else measure_map.get(measure_id, "")
            meas_abbr = meas_map.get(meas_abbr_raw, meas_abbr_raw)  # используем meas_map, если можем
            meas_short = meas_abbr if meas_abbr else ""

            pub_short = pub_map.get(pub_name, abbr(pub_name))
            dataset_short = dataset_map.get(dataset_name, abbr(dataset_name))

            name_parts = [pub_short, dataset_short]
            if meas_short:
                name_parts.append(meas_short)

            # Ограничим длину каждого компонента до 12 символов, чтобы избежать повторов
            name_parts_trimmed = [part[:12] for part in name_parts]
            table_alias = "_".join(name_parts_trimmed)

            safe_suffix = "all" if measure_id is None else f"m{measure_id}"
            table_name = f"pub{pub_id}_ds{dataset_id}_{safe_suffix}"

            try:
                data_json = get_data(YEAR_FROM, YEAR_TO, dataset_id, pub_id, measure_id)
                raw = data_json.get("RawData", [])
                headers = data_json.get("headerData", [])
                units = data_json.get("units", [])
                df = pd.DataFrame(raw)

                if df.empty:
                    print(f"⚠ Нет данных (measureId={measure_id})")
                    continue

                # Расшифровка
                element_map = {h["id"]: h.get("elname", "") for h in headers}
                unit_map = {u["id"]: u.get("val", "") for u in units}

                df["indicator_name"] = headers[0]["elname"] if headers else dataset_name
                if "element_id" in df.columns:
                    df["element_name"] = df["element_id"].map(element_map)
                if "measure_id" in df.columns and measure_map:
                    df["measure_name"] = df["measure_id"].map(measure_map)
                if "unit_id" in df.columns:
                    df["unit_name"] = df["unit_id"].map(unit_map)

                # Добавляем метаданные, чтобы сохранить контекст выгрузки
                df["publication_id"] = pub_id
                df["publication_name"] = pub_name
                df["dataset_id"] = dataset_id
                df["dataset_name"] = dataset_name
                df["measure_id_requested"] = measure_id
                df["table_alias"] = table_alias

                if "obs_val" in df.columns:
                    df = df[[col for col in df.columns if col != "obs_val"] + ["obs_val"]]

                file_name = f"{table_name}.parquet"
                file_path = output_dir / file_name

                if table_name in saved_files and file_path.exists():
                    existing_df = pd.read_parquet(file_path)
                    df = pd.concat([existing_df, df], ignore_index=True)
                else:
                    saved_files.add(table_name)
                    metadata_rows.append({
                        "table_name": table_name,
                        "file_name": file_name,
                        "table_alias": table_alias,
                        "publication_id": pub_id,
                        "publication_name": pub_name,
                        "dataset_id": dataset_id,
                        "dataset_name": dataset_name,
                        "measure_id": measure_id,
                        "measure_name": meas_abbr_raw
                    })

                df.to_parquet(file_path, index=False)

                print(f"✅ Сохранено: {file_name}")

            except Exception as e:
                print(f"❌ Ошибка при обработке ID={dataset_id}, measureId={measure_id}: {e}")

if metadata_rows:
    metadata_df = pd.DataFrame(metadata_rows)
    metadata_df.to_parquet(output_dir / "metadata.parquet", index=False)

print(f"\n📁 Готово! Данные сохранены в каталоге {output_dir.resolve()}")



import pandas as pd
from pathlib import Path

parquet_dir = Path("cbr_parquet")
all_frames = [pd.read_parquet(p) for p in parquet_dir.glob("*.parquet") if p.name != "metadata.parquet"]
combined = pd.concat(all_frames, ignore_index=True)
combined.to_parquet("cbr_parquet/all_data.parquet", index=False)



import pandas as pd
import pyodbc
from datetime import datetime
from pathlib import Path

PARQUET_FILE = Path("cbr_parquet/all_data.parquet")
TARGET_TABLE = "u_m26pu.cbr_all_data"
DSN_NAME = "impala51"          # замените на свой DSN
BATCH_SIZE = 10_000            # можно менять под объём

# если хотим ограничиться определёнными колонками — перечислите их здесь
required_columns = [
    "publication_id",
    "publication_name",
    "dataset_id",
    "dataset_name",
    "measure_id_requested",
    "obs_val",
    "time",
    "indicator_name",
    "element_name",
    "measure_name",
    "unit_name",
]

def load_data_to_hadoop():
    try:
        print(f"[{datetime.now()}] Начало чтения Parquet: {PARQUET_FILE}")
        df = pd.read_parquet(PARQUET_FILE)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        df = df.fillna("")  # при необходимости замените на свои правила заполнения

        print(f"[{datetime.now()}] Подключение к Impala (DSN={DSN_NAME})")
        conn = pyodbc.connect(
            f"DSN={DSN_NAME};",
            autocommit=True,
            ansi=True,
            timeout=0,
        )
        cursor = conn.cursor()

        # создаём таблицу (все поля как STRING — подстройте типы при необходимости)
        column_defs = ",\n    ".join(f"`{col}` STRING" for col in required_columns)
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
                {column_defs}
            )
            STORED AS PARQUET
            TBLPROPERTIES ('transactional'='false', 'parquet.compression'='SNAPPY')
        """)
        print(f"[{datetime.now()}] Таблица проверена/создана")

        total_rows = len(df)
        inserted_rows = 0
        print(f"[{datetime.now()}] Старт загрузки: {total_rows} строк...")

        for start in range(0, total_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_rows)
            batch = df.loc[start:end - 1, required_columns]

            placeholders = ", ".join(["?"] * len(required_columns))
            data = [tuple(row) for row in batch.itertuples(index=False, name=None)]

            try:
                cursor.fast_executemany = True
                cursor.executemany(
                    f"INSERT INTO {TARGET_TABLE} VALUES ({placeholders})",
                    data,
                )
                inserted_rows += len(data)
                print(f"[{datetime.now()}] Загружено: {inserted_rows}/{total_rows}")
            except pyodbc.Error as e:
                print(f"[{datetime.now()}] Ошибка пакета {start}-{end}: {e}")
                with open("error_batch.txt", "a", encoding="utf-8") as err:
                    err.write(f"{datetime.now()} — пакет {start}-{end}: {e}\n")
                    batch.to_csv(err, sep="\t", index=False)
                continue

        print(f"[{datetime.now()}] Успешно загружено {inserted_rows} строк")

    except Exception as e:
        print(f"[{datetime.now()}] Критическая ошибка: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
        print(f"[{datetime.now()}] Соединение закрыто")

if __name__ == "__main__":
    load_data_to_hadoop()


import pandas as pd
import pyodbc
from datetime import datetime
from pathlib import Path

PARQUET_FILE = Path("cbr_parquet/all_data.parquet")
TARGET_TABLE = "u_m26pu.cbr_all_data"   # поменяйте на свою БД.таблицу
DSN_NAME = "impala51"                   # имя ODBC-DSN для Impala
BATCH_SIZE = 10_000                     # размер пачки для executemany

# перечень колонок и их типы в Impala
column_types = {
    "publication_id": "STRING",
    "publication_name": "STRING",
    "dataset_id": "STRING",
    "dataset_name": "STRING",
    "measure_id_requested": "STRING",
    "obs_val": "DOUBLE",                # числовой тип для значения показателя
    "time": "STRING",
    "indicator_name": "STRING",
    "element_name": "STRING",
    "measure_name": "STRING",
    "unit_name": "STRING",
}
required_columns = list(column_types)


def load_data_to_hadoop():
    try:
        print(f"[{datetime.now()}] Чтение Parquet: {PARQUET_FILE}")
        df = pd.read_parquet(PARQUET_FILE)

        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Отсутствуют колонки: {missing}")

        # obs_val → numeric (несчитываемые значения станут NaN)
        df["obs_val"] = pd.to_numeric(df["obs_val"], errors="coerce")

        # строковые поля → строки без NaN
        string_cols = [col for col, typ in column_types.items() if typ == "STRING"]
        df[string_cols] = df[string_cols].apply(lambda s: s.fillna("").astype(str))

        print(f"[{datetime.now()}] Подключение к Impala (DSN={DSN_NAME})")
        conn = pyodbc.connect(
            f"DSN={DSN_NAME};",
            autocommit=True,
            ansi=True,
            timeout=0,
        )
        cursor = conn.cursor()

        column_defs = ",\n    ".join(
            f"`{col}` {column_types[col]}" for col in required_columns
        )
        cursor.execute(f"""
            CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
                {column_defs}
            )
            STORED AS PARQUET
            TBLPROPERTIES ('transactional'='false', 'parquet.compression'='SNAPPY')
        """)
        print(f"[{datetime.now()}] Таблица готова")

        total_rows = len(df)
        inserted_rows = 0
        print(f"[{datetime.now()}] Старт загрузки: {total_rows} строк")

        for start in range(0, total_rows, BATCH_SIZE):
            end = min(start + BATCH_SIZE, total_rows)
            batch = df.iloc[start:end][required_columns]

            placeholders = ", ".join(["?"] * len(required_columns))
            data = [tuple(row) for row in batch.itertuples(index=False, name=None)]

            try:
                cursor.fast_executemany = True
                cursor.executemany(
                    f"INSERT INTO {TARGET_TABLE} VALUES ({placeholders})",
                    data,
                )
                inserted_rows += len(data)
                print(f"[{datetime.now()}] Загружено {inserted_rows}/{total_rows}")
            except pyodbc.Error as e:
                print(f"[{datetime.now()}] Ошибка пакета {start}-{end}: {e}")
                with open("error_batch.txt", "a", encoding="utf-8") as err:
                    err.write(f"{datetime.now()} — пакет {start}-{end}: {e}\n")
                    batch.to_csv(err, sep="\t", index=False)
                continue

        print(f"[{datetime.now()}] Успешно загружено {inserted_rows} строк")

    except Exception as e:
        print(f"[{datetime.now()}] Критическая ошибка: {e}")
    finally:
        if 'cursor' in locals():
            cursor.close()
        if 'conn' in locals():
            conn.close()
        print(f"[{datetime.now()}] Соединение закрыто")


if __name__ == "__main__":
    load_data_to_hadoop()
