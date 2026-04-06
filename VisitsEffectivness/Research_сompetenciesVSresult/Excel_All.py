from openpyxl import Workbook
import pandas as pd
import plotly.express as px
from scipy.stats import spearmanr

file_path = "1.xlsx"

sheets = [
    # 'РАФАМИН_регионы',
    'РАФАМИН_брики',
    'РАФАМИН_брикирегионы',
    # 'ПРОСПЕКТА_регионы',
    'ПРОСПЕКТА_брики'
    # 'ПРОСПЕКТА_брикирегионы'
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

region_col = "Область"
echelon_col = "Эшелон"
rep_col = "Сотрудник"
 
results = []
counter = 1


def clean_numeric(df, col):
    return (
        df[col].astype(str)
        .str.replace(',', '.', regex=False)
        .astype(float)
    )


def make_plot(df, x, y, title, color=None, hover=None):

    fig = px.scatter(
        df,
        x=x,
        y=y,
        color=color,
        hover_name=hover,   # главное изменение
        title=title
    )

    fig.add_hline(y=1, line_dash="dash", opacity=0.4)
    fig.show()

for sheet in sheets:

    df = pd.read_excel(file_path, sheet_name=sheet)
    df.columns = df.columns.str.strip()

    brand, group = sheet.split('_')
    has_echelon = echelon_col in df.columns

    for score in score_cols:
        for kpi in kpi_cols:

            cols = [score, kpi, region_col]
            if has_echelon:
                cols.append(echelon_col)
            if rep_col in df.columns:
                cols.append(rep_col)

            cols = [c for c in cols if c in df.columns]
            if len(cols) < 3:
                continue

            tmp = df[cols].dropna().copy()
            tmp[score] = clean_numeric(tmp, score)
            tmp[kpi] = clean_numeric(tmp, kpi)

            # =====================================================
            # 1️⃣ Индивидуально
            # =====================================================
            if len(tmp) >= 3:

                corr = spearmanr(tmp[score], tmp[kpi])
                n = len(tmp)

                make_plot(
                    tmp,
                    score,
                    kpi,
                    f"{sheet} | Индивидуально | n={n}<br>"
                    f"{score} vs {kpi}<br>"
                    f"r={corr.statistic:.3f}, p={corr.pvalue:.4f}",
                    color=region_col,
                    hover=rep_col if rep_col in tmp.columns else None
                )

                results.append([
                    counter, brand, group, "Индивидуально",
                    kpi, score, n,
                    round(corr.statistic, 3),
                    round(corr.pvalue, 4),
                    "Да" if corr.pvalue < 0.05 else "Нет",
                    ""
                ])
                counter += 1

            # =====================================================
            # 2️⃣ По региону
            # =====================================================
            agg = (
                tmp.groupby(region_col)
                .agg(score_mean=(score, 'mean'),
                     kpi_val=(kpi, 'mean'))
                .reset_index()
            )

            if len(agg) >= 3:

                corr = spearmanr(agg.score_mean, agg.kpi_val)
                n = len(agg)

                make_plot(
                    agg, "score_mean", "kpi_val",
                    f"{sheet} | По региону | n={n}<br>"
                    f"{score} vs {kpi}<br>"
                    f"r={corr.statistic:.3f}, p={corr.pvalue:.4f}",
                    color=region_col
                )

                results.append([
                    counter, brand, group, "По региону",
                    kpi, score, n,
                    round(corr.statistic, 3),
                    round(corr.pvalue, 4),
                    "Да" if corr.pvalue < 0.05 else "Нет",
                    ""
                ])
                counter += 1

            # =====================================================
            # 3️⃣ Эшелоны
            # =====================================================
            # if has_echelon:

            #     for ech in tmp[echelon_col].unique():

            #         df_ech = tmp[tmp[echelon_col] == ech]

            #         # --- индивидуально внутри эшелона
            #         if len(df_ech) >= 3:

            #             corr = spearmanr(
            #                 df_ech[score],
            #                 df_ech[kpi]
            #             )
            #             n = len(df_ech)

            #             make_plot(
            #                 df_ech, score, kpi,
            #                 f"{sheet} | Эшелон {ech} "
            #                 f"(индивидуально) | n={n}<br>"
            #                 f"r={corr.statistic:.3f}, "
            #                 f"p={corr.pvalue:.4f}",
            #                 color=region_col
            #             )

            #             results.append([
            #                 counter, brand, group,
            #                 f"Эшелон {ech} индивидуально",
            #                 kpi, score, n,
            #                 round(corr.statistic, 3),
            #                 round(corr.pvalue, 4),
            #                 "Да" if corr.pvalue < 0.05 else "Нет",
            #                 ""
            #             ])
            #             counter += 1

            #         # --- среднее по региону внутри эшелона
            #         agg_ech = (
            #             df_ech.groupby(region_col)
            #             .agg(score_mean=(score, 'mean'),
            #                  kpi_val=(kpi, 'mean'))
            #             .reset_index()
            #         )

            #         if len(agg_ech) >= 3:

            #             corr = spearmanr(
            #                 agg_ech.score_mean,
            #                 agg_ech.kpi_val
            #             )
            #             n = len(agg_ech)

            #             make_plot(
            #                 agg_ech,
            #                 "score_mean",
            #                 "kpi_val",
            #                 f"{sheet} | Эшелон {ech} "
            #                 f"(по региону) | n={n}<br>"
            #                 f"r={corr.statistic:.3f}, "
            #                 f"p={corr.pvalue:.4f}",
            #                 color=region_col
            #             )

            #             results.append([
            #                 counter, brand, group,
            #                 f"Эшелон {ech} по региону",
            #                 kpi, score, n,
            #                 round(corr.statistic, 3),
            #                 round(corr.pvalue, 4),
            #                 "Да" if corr.pvalue < 0.05 else "Нет",
            #                 ""
            #             ])
            #             counter += 1


# =====================================================
# Excel отчёт
# =====================================================
# wb = Workbook()
# ws = wb.active
# ws.title = "Correlation_results"

# ws.append([
#     "№", "Бренд", "Группа", "Тип анализа",
#     "Метрика", "Компетенция",
#     "n", "r", "p", "Значимо", "Комментарий"
# ])

# for row in results:
#     ws.append(row)

# wb.save("correlation_results_final.xlsx")

# print("Excel сохранён: correlation_results_final.xlsx")