import requests
import pandas as pd

def check_site(url: str):
    try:
        response = requests.get(url, timeout=10)

        if response.status_code != 200:
            return f"Не открылся (код {response.status_code})"

        content = response.text.strip()

        if len(content) < 50:
            return "Открылся, но пустая страница"

        return "Открыт, контент есть"

    except Exception as e:
        return f"Ошибка: {e}"


# ==== НАСТРОЙКА ====
input_file = "sites.xlsx"       # входной Excel
output_file = "sites_result.xlsx"  # выходной Excel
column_url = "Ссылка"   # колонка со ссылками
column_status = "Статус"  # куда писать статус
# ===================


# Загружаем Excel
df = pd.read_excel(input_file)

# Проверяем, что колонка существует
if column_url not in df.columns:
    raise ValueError(f"Колонки '{column_url}' нет в файле!")

# Создаём колонку, если её ещё нет
if column_status not in df.columns:
    df[column_status] = ""

# Проходим по всем ссылкам
for i, url in df[column_url].items():
    if pd.isna(url):
        df.loc[i, column_status] = "Пустая ячейка"
        continue

    print(f"Проверяю: {url}")
    status = check_site(url)
    df.loc[i, column_status] = status


# Сохраняем файл
df.to_excel(output_file, index=False)

print("\nГотово! Результаты записаны в", output_file)
