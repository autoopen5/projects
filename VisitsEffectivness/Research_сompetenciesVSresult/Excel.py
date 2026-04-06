import pandas as pd
from scipy.stats import spearmanr
import plotly.express as px

file_path = "1.xlsx"

sheets = [
    'РАФАМИН_регионы',
    # 'РАФАМИН_брики',
    # 'ПРОСПЕКТА_регионы',
    # 'ПРОСПЕКТА_брики'
]

score_cols = [
    'Оценка общая (рук.), %',
    'Оценка эффективности коммуникации (рук.), %',
    'Оценка навыки продаж (рук.), %'
]

kpi_cols = [
    'Динамика региона',
    'Выполнение плана'
]


for sheet in sheets:
    print(f'\n===== {sheet} =====')

    df = pd.read_excel(file_path, sheet_name=sheet)

    # очистка названий колонок (часто пробелы мешают)
    df.columns = df.columns.str.strip()

    for score in score_cols:
        for kpi in kpi_cols:

            tmp = df[[score, kpi, 'Сотрудник', 'Область']].copy()

            # убрать пропуски
            tmp = tmp.dropna()

            # иногда проценты как строки → приводим
            tmp[score] = (
                tmp[score].astype(str)
                .str.replace(',', '.')
                .astype(float)
            )

            tmp[kpi] = (
                tmp[kpi].astype(str)
                .str.replace(',', '.')
                .astype(float)
            )

            if len(tmp) < 3:
                continue

            corr = spearmanr(tmp[score], tmp[kpi])
            n = len(tmp)

            print(
                f'{score} vs {kpi}: '
                f'r={corr.statistic:.3f}, p={corr.pvalue:.4f}, n={n}'
            )

            # график
            fig = px.scatter(
                tmp,
                x=score,
                y=kpi,
                color='Область',
                hover_data=['Сотрудник'],
                title=(
                    f'{sheet}<br>'
                    f'{score} vs {kpi}<br>'
                    f'n={n} | r={corr.statistic:.3f}, p={corr.pvalue:.4f}'
                )
            )

            fig.add_hline(y=1, line_dash='dash', opacity=0.4)
            fig.show()