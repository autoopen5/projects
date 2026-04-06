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
    # 'Оценка эффективности коммуникации (рук.), %',
    # 'Оценка навыки продаж (рук.), %'
]

kpi_cols = [
    'Динамика региона',
    'Выполнение плана'
]


for sheet in sheets:
    print(f'\n===== {sheet} =====')

    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()

    # если есть эшелон — укажи колонку тут:
    # echelon_col = 'Эшелон' if 'Эшелон' in df.columns else None

    for score in score_cols:
        for kpi in kpi_cols:

            cols = [score, kpi, 'Область']
            # if echelon_col:
            #     cols.append(echelon_col)

            tmp = df[cols].dropna().copy()

            # перевод процентов
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

            # ===== АГРЕГАЦИЯ =====
            group_cols = ['Область']
            # if echelon_col:
                # group_cols.append(echelon_col)

            agg_df = (
                tmp
                .groupby(group_cols)
                .agg(
                    score_mean=(score, 'mean'),
                    kpi_val=(kpi, 'mean')  # обычно одинаковый
                )
                .reset_index()
            )

            if len(agg_df) < 3:
                continue
            
            # print((agg_df[:5]))
            corr = spearmanr(
                agg_df['score_mean'],
                agg_df['kpi_val']
            )

            print(
                f'{score} vs {kpi} | aggregated: '
                f'r={corr.statistic:.3f}, '
                f'p={corr.pvalue:.4f}, '
                f'n={len(agg_df)}'
            )

            fig = px.scatter(
                agg_df,
                x='score_mean',
                y='kpi_val',
                color='Область',
                title=(
                    f'{sheet}<br>'
                    f'АГРЕГИРОВАНО: {score} vs {kpi}<br>'
                    f'n={len(agg_df)} | '
                    f'r={corr.statistic:.3f}, '
                    f'p={corr.pvalue:.4f}'
                )
            )

            fig.show()
