import clickhouse_connect
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import requests
import json

# connection to server 
client = clickhouse_connect.get_client(
    host='clickhouse.moscow',
    port=8123,  # 8123 для HTTP, 9000 для Native (TCP)
    username='GrushkoIV',
    password='jNbrvzd1IcF0Yx5I',

)


# 2. Для поиска по строковым значениям  используй
#    функцию lower() или ILIKE, чтобы не было ошибок из-за регистра.
#    Например: WHERE lower(`Region MMH`) = 'москва'.

# --- Схема таблиц ---
TABLES_INFO = """
У тебя есть таблицы в ClickHouse.

Важные правила:
1. Если имя колонки содержит пробел, заключай его в обратные кавычки (`).
   Например: SELECT `Region MMH` FROM MDLP.SHOW_Disposal_reports.


используй фильтр, если указан один из препаратов (sku):
Тенотен детский, Эргоферон, ПРОСПЕКТА, Тенотен, Ренгалин, Климаксан, Анаферон детский, Анаферон, Афалаза, Колофорт, Ренгалин, РАФАМИН.

используй фильтр, если указан один из регионов (`Region MMH`):
Волга, Москва, Центр, Сибирь, Урал, Северо-запад.
"""

TABLES_INFO += """
У тебя есть следующие таблицы в ClickHouse:
1. MDLP.SHOW_Disposal_reports — данные о выбытии препаратов:
   - exit_date (Дата, Date)
   - Year (Год, Int)
   - Month (Месяц, Int)
   - Type (тип контрагента, String)
   - contragent (название контрагента, String)
   - `Region MMH` (Регион, String)
   - sku (Препарат, String)
   - exit_type (тип выбытия, например, Продажа, String)
   - cnt (Продажи количество, Int)

2. MDLP.SHOW_Remaining_reports — данные об остатках:
   - date (Дата, Date)
   - Year (Год, Int)
   - Month (Месяц, Int)
   - Type (тип контрагента, String)
   - contragent (название контрагента, String)
   - `Region MMH`  (Регион, String)
   - sku (Препарат, String)
   - remains_full (Остаток, Int)
"""

# --- Функция обращения к Ollama ---
def ask_ollama(prompt, model="mistral"):
    r = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt}
    )
    out = ""
    for line in r.iter_lines():
        if line:
            out += json.loads(line)["response"]
    return out.strip()

# --- Основной бот ---
def ask_bot(question: str):
    sql_prompt = f"""
    Пользователь задал вопрос: "{question}".
    Используй эти таблицы:

    {TABLES_INFO}

    Составь SQL для ClickHouse (без пояснений и комментариев).
    """

    sql_query = ask_ollama(sql_prompt)
    print(f"📝 SQL:\n{sql_query}\n")

    try:
        result = client.query(sql_query).result_rows
        print("📊 Результат:", result, "\n")

        # Красивый ответ
        answer_prompt = f"Вопрос: {question}\nSQL результат: {result}\nСделай краткий понятный ответ."
        return ask_ollama(answer_prompt)
    except Exception as e:
        return f"⚠️ Ошибка SQL: {e}"

if __name__ == "__main__":
    while True:
        q = input("❓ Вопрос: ")
        if q.lower() in ["exit", "quit", "выход"]:
            break
        print("✅ Ответ:", ask_bot(q), "\n")

# # --- Настройки Ollama ---
# MODEL_NAME = "mistral"   # можно заменить на llama2, phi3 и т.д.

# # --- Модель (локально через Ollama) ---
# llm = ChatOllama(model=MODEL_NAME, temperature=0)

# # --- Шаблон: переводим вопрос → SQL ---
# template = """
# Ты аналитик. У тебя есть база данных с таблицами по продажам лекарств (ClickHouse).
# Нужно составить SQL запрос для ответа на вопрос пользователя.

# Пример формата: SELECT ... FROM sales WHERE ...

# Вопрос: {question}
# SQL:
# """

# prompt = PromptTemplate(template=template, input_variables=["question"])
# sql_chain = LLMChain(llm=llm, prompt=prompt)


# def ask_bot(question: str):
#     # 1. LLM генерирует SQL-запрос
#     sql_query = sql_chain.run(question).strip()

#     print(f"📝 Сгенерированный SQL:\n{sql_query}\n")

#     # 2. Отправляем SQL в ClickHouse
#     try:
#         result = client.query(sql_query).result_rows
#         print("📊 Результат запроса:", result, "\n")

#         # 3. Попросим LLM красиво оформить ответ
#         answer_prompt = f"На вопрос: '{question}' результат запроса: {result}. Сформулируй понятный ответ."
#         answer = llm.invoke(answer_prompt)
#         return str(answer)
#     except Exception as e:
#         return f"⚠️ Ошибка выполнения SQL: {e}"


# def main():
#     print("💊 Фарм-бот готов. Пиши вопросы (или 'exit' для выхода).\n")

#     while True:
#         q = input("❓ Вопрос: ")
#         if q.lower() in ["exit", "quit", "выход"]:
#             print("👋 До встречи!")
#             break
#         print("✅ Ответ:", ask_bot(q), "\n")


# if __name__ == "__main__":
#     main()