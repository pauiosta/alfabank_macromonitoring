import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set, Tuple

import pandas as pd
import pyodbc
import requests
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


TABLE_PREFIX = os.getenv("CBR_TABLE_PREFIX", "cbr")
HADOOP_DSN = os.getenv("CBR_HADOOP_DSN", "impala51")
HADOOP_DATABASE = os.getenv("CBR_HADOOP_DATABASE", "u_cbr")
PARQUET_COMPRESSION = os.getenv("CBR_PARQUET_COMPRESSION", "SNAPPY")
RECREATE_TABLES = os.getenv("CBR_RECREATE_TABLES", "true").lower() == "true"
BATCH_SIZE = int(os.getenv("CBR_BATCH_SIZE", "1000"))
DICTIONARY_PATH = Path(os.getenv("CBR_DICTIONARY_PATH", "cbr_name_dictionary.csv"))
METADATA_TABLE_NAME = os.getenv("CBR_METADATA_TABLE", f"{TABLE_PREFIX}_metadata")
IDENTIFIER_MAX_LENGTH = 96
COLUMN_IDENTIFIER_MAX_LENGTH = 64


CYRILLIC_TO_LATIN: Dict[str, str] = {
    "а": "a",
    "б": "b",
    "в": "v",
    "г": "g",
    "д": "d",
    "е": "e",
    "ё": "e",
    "ж": "zh",
    "з": "z",
    "и": "i",
    "й": "y",
    "к": "k",
    "л": "l",
    "м": "m",
    "н": "n",
    "о": "o",
    "п": "p",
    "р": "r",
    "с": "s",
    "т": "t",
    "у": "u",
    "ф": "f",
    "х": "h",
    "ц": "ts",
    "ч": "ch",
    "ш": "sh",
    "щ": "sch",
    "ъ": "",
    "ы": "y",
    "ь": "",
    "э": "e",
    "ю": "yu",
    "я": "ya",
}


def transliterate(text: Optional[str]) -> str:
    if not text:
        return ""

    result = []
    for char in text.lower():
        if char in CYRILLIC_TO_LATIN:
            result.append(CYRILLIC_TO_LATIN[char])
        elif char.isascii():
            result.append(char)
        elif char.isdigit():
            result.append(char)
        else:
            # fallback: replace with underscore for unsupported characters
            result.append("_")
    return "".join(result)


def normalize_identifier(raw_value: Optional[str], max_length: int = IDENTIFIER_MAX_LENGTH) -> str:
    value = transliterate(raw_value)
    value = re.sub(r"[^0-9a-z_]+", "_", value)
    value = re.sub(r"_+", "_", value)
    value = value.strip("_")
    if not value:
        value = "value"
    if value[0].isdigit():
        value = f"_{value}"
    if len(value) > max_length:
        value = value[:max_length]
        value = value.rstrip("_") or value
    return value


def quote_identifier(identifier: str) -> str:
    return f"`{identifier}`"


def ensure_database(cursor: pyodbc.Cursor, schema: str) -> None:
    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {quote_identifier(schema)}")


def map_dtype(dtype: pd.api.extensions.ExtensionDtype) -> str:
    if pd.api.types.is_bool_dtype(dtype):
        return "BOOLEAN"
    if pd.api.types.is_integer_dtype(dtype):
        return "BIGINT"
    if pd.api.types.is_float_dtype(dtype):
        return "DOUBLE"
    if pd.api.types.is_datetime64_any_dtype(dtype):
        return "TIMESTAMP"
    if pd.api.types.is_string_dtype(dtype):
        return "STRING"
    return "STRING"


def fully_qualified_name(schema: str, table: str) -> str:
    return f"{quote_identifier(schema)}.{quote_identifier(table)}"


def create_impala_table(cursor: pyodbc.Cursor, schema: str, table: str, df: pd.DataFrame) -> None:
    qualified = fully_qualified_name(schema, table)
    if RECREATE_TABLES:
        cursor.execute(f"DROP TABLE IF EXISTS {qualified}")

    columns_ddl = []
    for column, dtype in df.dtypes.items():
        sql_type = map_dtype(dtype)
        columns_ddl.append(f"{quote_identifier(column)} {sql_type}")

    columns_sql = ",\n        ".join(columns_ddl)
    ddl = (
        f"CREATE TABLE IF NOT EXISTS {qualified} (\n"
        f"        {columns_sql}\n"
        ")\n"
        "STORED AS PARQUET\n"
        f"TBLPROPERTIES ('transactional'='false', 'parquet.compression'='{PARQUET_COMPRESSION}')"
    )
    cursor.execute(ddl)


def insert_dataframe(
    cursor: pyodbc.Cursor,
    schema: str,
    table: str,
    df: pd.DataFrame,
    batch_size: int = BATCH_SIZE,
) -> None:
    if df.empty:
        return

    qualified = fully_qualified_name(schema, table)
    placeholders = ", ".join(["?"] * len(df.columns))
    column_clause = ", ".join(quote_identifier(col) for col in df.columns)
    insert_sql = f"INSERT INTO {qualified} ({column_clause}) VALUES ({placeholders})"

    cursor.fast_executemany = True

    total_rows = len(df)
    for start in range(0, total_rows, batch_size):
        end = min(start + batch_size, total_rows)
        batch = df.iloc[start:end]
        data = []
        for row in batch.itertuples(index=False, name=None):
            prepared = []
            for value in row:
                missing = False
                if value is None:
                    missing = True
                else:
                    try:
                        missing = pd.isna(value)
                    except TypeError:
                        missing = False

                if missing:
                    prepared.append(None)
                    continue

                if isinstance(value, pd.Timestamp):
                    prepared.append(value.to_pydatetime())
                else:
                    prepared.append(value)
            data.append(tuple(prepared))

        if data:
            cursor.executemany(insert_sql, data)


COLUMN_OVERRIDES: Dict[str, str] = {
    "obs_val": "observation_value",
    "indicator_name": "indicator_name",
    "element_id": "element_id",
    "element_name": "element_name",
    "measure_id": "measure_id",
    "measure_name": "measure_name",
    "measure_id_requested": "requested_measure_id",
    "publication_id": "publication_id",
    "publication_name": "publication_name",
    "dataset_id": "dataset_id",
    "dataset_name": "dataset_name",
    "unit_id": "unit_id",
    "unit_name": "unit_name",
    "table_alias": "table_alias",
}


def rename_columns_with_translation(
    df: pd.DataFrame,
    table_name: str,
    dictionary_records: List[Dict[str, object]],
    context: Dict[str, object],
    dictionary_seen: Set[Tuple[str, str]],
) -> pd.DataFrame:
    assigned: Dict[str, int] = {}
    rename_map: Dict[str, str] = {}

    for original in df.columns:
        candidate = COLUMN_OVERRIDES.get(original)
        if candidate is None:
            candidate = normalize_identifier(original, COLUMN_IDENTIFIER_MAX_LENGTH)

        base = candidate
        suffix = 1
        while candidate in assigned:
            suffix += 1
            candidate = f"{base}_{suffix}"

        assigned[candidate] = 1
        rename_map[original] = candidate

        key = (table_name, candidate)
        if key not in dictionary_seen:
            dictionary_records.append(
                {
                    **context,
                    "entity_type": "column",
                    "table_name": table_name,
                    "column_name": candidate,
                    "english_name": candidate,
                    "russian_name": original,
                }
            )
            dictionary_seen.add(key)

    return df.rename(columns=rename_map)


def build_table_name(
    dataset_name: str,
    measure_label: Optional[str],
    dataset_id: int,
    measure_id: Optional[int],
    used_names: Set[str],
) -> str:
    parts = [TABLE_PREFIX, dataset_name]
    if measure_label:
        parts.append(measure_label)
    suffix = "all" if measure_id is None else f"m{measure_id}"
    parts.append(suffix)

    base = normalize_identifier("_".join(filter(None, parts)))
    if not base:
        base = f"{TABLE_PREFIX}_dataset_{dataset_id}_{suffix}"

    candidate = base
    counter = 1
    while candidate in used_names:
        counter += 1
        candidate = normalize_identifier(f"{base}_{counter}")

    if len(candidate) > IDENTIFIER_MAX_LENGTH:
        candidate = candidate[:IDENTIFIER_MAX_LENGTH]

    used_names.add(candidate)
    return candidate


def connect_impala() -> pyodbc.Connection:
    return pyodbc.connect(
        f"DSN={HADOOP_DSN};",
        autocommit=True,
        ansi=True,
        timeout=0,
    )

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

def load_data_to_hadoop() -> None:
    dictionary_records: List[Dict[str, object]] = []
    dictionary_seen: Set[Tuple[str, str]] = set()
    metadata_rows: List[Dict[str, object]] = []
    created_tables: Set[str] = set()
    documented_tables: Set[str] = set()
    used_table_names: Set[str] = set()

    conn: Optional[pyodbc.Connection] = None
    cursor: Optional[pyodbc.Cursor] = None

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
        "Малые и микро": "МалМик",
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
        "Услуги": "Услуги",
    }

    try:
        print(f"[{datetime.now()}] Подключение к Impala через DSN='{HADOOP_DSN}'...")
        conn = connect_impala()
        cursor = conn.cursor()
        ensure_database(cursor, HADOOP_DATABASE)
        print(f"[{datetime.now()}] Подключено. Целевая база: {HADOOP_DATABASE}")

        for pub in publications:
            pub_id = pub["id"]
            pub_name = pub["category_name"]
            print(f"\n📘 Публикация ID={pub_id}: {pub_name}")

            try:
                datasets = get_datasets(pub_id)
            except Exception as error:
                print(f"❌ Ошибка при получении показателей: {error}")
                continue

            for dataset in datasets:
                if dataset["id"] not in required_structure[pub_id]:
                    continue

                dataset_id = dataset["id"]
                dataset_name = dataset["name"] or f"dataset_{dataset_id}"
                dataset_type = dataset.get("type", 0)

                print(f"📡  → Показатель ID={dataset_id}: {dataset_name} (type={dataset_type})")

                measure_ids: List[Optional[int]] = []
                measure_map: Dict[Optional[int], str] = {}
                if dataset_type == 1:
                    try:
                        measures = get_measures(dataset_id)
                        measure_ids = [m["id"] for m in measures if m["id"] is not None]
                        measure_map = {m["id"]: m["name"] for m in measures if m["id"] is not None}
                        if not measure_ids:
                            print("⚠ Нет разрезов, пропускаем")
                            continue
                    except Exception as error:
                        print(f"⚠ Не удалось получить measure для dataset {dataset_id}: {error}")
                        continue
                else:
                    measure_ids = [None]

                if dataset_id in [30, 31, 32, 33]:
                    measure_ids = [23, 42, 55, 64, 72, 87, 95, 106]
                elif dataset_id in [34, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54]:
                    measure_ids = [22]

                for measure_id in measure_ids:
                    meas_abbr_raw = "" if measure_id is None else measure_map.get(measure_id, "")
                    meas_abbr = meas_map.get(meas_abbr_raw, meas_abbr_raw)
                    meas_short = meas_abbr if meas_abbr else ""

                    pub_short = pub_map.get(pub_name, abbr(pub_name))
                    dataset_short = dataset_map.get(dataset_name, abbr(dataset_name))

                    name_parts = [pub_short, dataset_short]
                    if meas_short:
                        name_parts.append(meas_short)

                    name_parts_trimmed = [part[:12] for part in name_parts]
                    table_alias_ru = "_".join(name_parts_trimmed)
                    table_alias_en = normalize_identifier(table_alias_ru, COLUMN_IDENTIFIER_MAX_LENGTH)

                    try:
                        data_json = get_data(YEAR_FROM, YEAR_TO, dataset_id, pub_id, measure_id)
                        raw = data_json.get("RawData", [])
                        headers = data_json.get("headerData", [])
                        units = data_json.get("units", [])
                        df = pd.DataFrame(raw)

                        if df.empty:
                            print(f"⚠ Нет данных (measureId={measure_id})")
                            continue

                        element_map = {h["id"]: h.get("elname", "") for h in headers}
                        unit_map = {u["id"]: u.get("val", "") for u in units}

                        df["indicator_name"] = headers[0]["elname"] if headers else dataset_name
                        if "element_id" in df.columns:
                            df["element_name"] = df["element_id"].map(element_map)
                        if "measure_id" in df.columns and measure_map:
                            df["measure_name"] = df["measure_id"].map(measure_map)
                        if "unit_id" in df.columns:
                            df["unit_name"] = df["unit_id"].map(unit_map)

                        df["publication_id"] = pub_id
                        df["publication_name"] = pub_name
                        df["dataset_id"] = dataset_id
                        df["dataset_name"] = dataset_name
                        df["measure_id_requested"] = measure_id
                        df["table_alias"] = table_alias_en

                        if "obs_val" in df.columns:
                            df = df[[col for col in df.columns if col != "obs_val"] + ["obs_val"]]

                        df = df.convert_dtypes()

                        context_info = {
                            "publication_id": pub_id,
                            "publication_name": pub_name,
                            "dataset_id": dataset_id,
                            "dataset_name": dataset_name,
                            "measure_id": measure_id,
                            "measure_name": meas_abbr_raw or "",
                        }

                        english_table_name = build_table_name(
                            dataset_name=dataset_name,
                            measure_label=meas_abbr_raw,
                            dataset_id=dataset_id,
                            measure_id=measure_id,
                            used_names=used_table_names,
                        )

                        if english_table_name not in documented_tables:
                            russian_caption_parts = [pub_name, dataset_name]
                            if meas_abbr_raw:
                                russian_caption_parts.append(meas_abbr_raw)
                            russian_caption = " | ".join(russian_caption_parts)
                            dictionary_records.append(
                                {
                                    **context_info,
                                    "entity_type": "table",
                                    "table_name": english_table_name,
                                    "column_name": "",
                                    "english_name": english_table_name,
                                    "russian_name": russian_caption,
                                }
                            )
                            documented_tables.add(english_table_name)

                        df = rename_columns_with_translation(
                            df,
                            english_table_name,
                            dictionary_records,
                            context_info,
                            dictionary_seen,
                        )

                        if english_table_name not in created_tables:
                            create_impala_table(cursor, HADOOP_DATABASE, english_table_name, df)
                            created_tables.add(english_table_name)

                        insert_dataframe(cursor, HADOOP_DATABASE, english_table_name, df)

                        metadata_rows.append(
                            {
                                "table_name": english_table_name,
                                "table_alias_en": table_alias_en,
                                "table_alias_ru": table_alias_ru,
                                "publication_id": pub_id,
                                "publication_name": pub_name,
                                "dataset_id": dataset_id,
                                "dataset_name": dataset_name,
                                "measure_id": measure_id,
                                "measure_name": meas_abbr_raw or "",
                            }
                        )

                        print(f"✅ Загружено в {HADOOP_DATABASE}.{english_table_name}: {len(df)} строк")

                    except Exception as error:
                        print(
                            f"❌ Ошибка при обработке dataset={dataset_id}, measureId={measure_id}: {error}"
                        )
                        continue

        if metadata_rows and cursor is not None:
            metadata_df = pd.DataFrame(metadata_rows).convert_dtypes()
            metadata_context = {
                "publication_id": None,
                "publication_name": "metadata",
                "dataset_id": None,
                "dataset_name": "metadata",
                "measure_id": None,
                "measure_name": "metadata",
            }

            if METADATA_TABLE_NAME not in documented_tables:
                dictionary_records.append(
                    {
                        **metadata_context,
                        "entity_type": "table",
                        "table_name": METADATA_TABLE_NAME,
                        "column_name": "",
                        "english_name": METADATA_TABLE_NAME,
                        "russian_name": "Metadata for loaded tables",
                    }
                )
                documented_tables.add(METADATA_TABLE_NAME)

            metadata_df = rename_columns_with_translation(
                metadata_df,
                METADATA_TABLE_NAME,
                dictionary_records,
                metadata_context,
                dictionary_seen,
            )

            create_impala_table(cursor, HADOOP_DATABASE, METADATA_TABLE_NAME, metadata_df)
            insert_dataframe(cursor, HADOOP_DATABASE, METADATA_TABLE_NAME, metadata_df)
            print(f"ℹ Метаданные обновлены в {HADOOP_DATABASE}.{METADATA_TABLE_NAME}")

        if dictionary_records:
            dictionary_df = pd.DataFrame(dictionary_records)
            dictionary_df.to_csv(DICTIONARY_PATH, index=False, encoding="utf-8")
            print(f"ℹ Справочник сохранен: {DICTIONARY_PATH}")

        print("\n📁 Готово! Данные загружены в Hadoop (Impala)")

    except Exception as error:
        print(f"[{datetime.now()}] Критическая ошибка: {error}")
        raise
    finally:
        if cursor is not None:
            cursor.close()
        if conn is not None:
            conn.close()
        print(f"[{datetime.now()}] Соединение закрыто")


if __name__ == "__main__":
    load_data_to_hadoop()
