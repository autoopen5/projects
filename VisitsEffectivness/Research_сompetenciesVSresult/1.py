from clickhouse_connect import get_client
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt
import plotly.express as px

# clickhouse
client = get_client(host='clickhouse.moscow', username='GrushkoIV', port = '8123', password='jNbrvzd1IcF0Yx5I', database='grushko_iv')

result = f'''SELECT `user_name` AS `user_name`,
       `region` AS `region_n`,
       SUM(DISTINCT cnt_new)/SUM(DISTINCT kol_plan_new)*100 AS `plan_fact`,
       MAX(m_value_agg)/MAX(max_value_agg)*100 AS `score`,
       ((SUM(DISTINCT cnt_new)-SUM(DISTINCT cnt_LY))/SUM(DISTINCT cnt_LY))*100 AS `sales_dynamic`
FROM
  (SELECT *
   FROM
     (SELECT *
      FROM
        (SELECT *,
                SUM(u_value) OVER (PARTITION BY region,
                                                user_name,
                                                manager_,
                                                cat1,
                                                cat2) AS u_value_agg,
                                  SUM(m_value) OVER (PARTITION BY region,
                                                                  user_name,
                                                                  manager_,
                                                                  cat1,
                                                                  cat2) AS m_value_agg,
                                                    AVG(IF(name_questionnaire='Опрос для региональных менеджеров по оценке развития компетенций у сотрудников', max_value_BYCAT, Null)) OVER (PARTITION BY region,
                                                                                                                                                                                                           user_name,
                                                                                                                                                                                                           manager_,
                                                                                                                                                                                                           cat1,
                                                                                                                                                                                                           cat2) AS max_value_agg
         FROM
           (SELECT *,
                   IF (user_name LIKE '%ИДЕАЛЬНЫЙ%',
                       name_NEW,
                       manager_) as manager_,
                      AVG(u_value_BYCAT) OVER (PARTITION BY name_questionnaire,
                                                            manager_,
                                                            cat1,
                                                            cat2) AS avg_u_value,
                                              AVG(m_value_BYCAT) OVER (PARTITION BY name_questionnaire,
                                                                                    manager_,
                                                                                    cat1,
                                                                                    cat2) AS avg_m_value,
                                                                      SUM(IF(user_name LIKE '%ИДЕАЛЬНЫЙ%', value_, NULL)) OVER (PARTITION BY manager_,
                                                                                                                                             cat1,
                                                                                                                                             cat2) AS target_value,
                                                                                                                               user_name as p_dim ,
                                                                                                                               SUM(IF(cat != '', value_, 0)) OVER (PARTITION BY name_questionnaire,
                                                                                                                                                                                region_mmh,
                                                                                                                                                                                region,
                                                                                                                                                                                user_name,
                                                                                                                                                                                cat1,
                                                                                                                                                                                cat2) AS dim_value
            FROM
              (SELECT rank_,
                      IFNULL(user_id, sAMAccountName_NEW) AS user_id,
                      user_name,
                      `date`,
                      id_questionnaire,
                      name_questionnaire,
                      question,
                      answer,
                      value_,
                      Ptext,
                      Plogin,
                      cat,
                      cat_s,
                      user_ld,
                      OU,
                      IFNULL(region, region_new) AS region,
                      dictrict,
                      IFNULL(region_mmh, region_mmh_new) AS region_mmh,
                      cat1,
                      cat2,
                      perfect_value,
                      sAMAccountName,
                      reg_acc,
                      region_mmh_acc,
                      p_level,
                      name_NEW,
                      sAMAccountName_NEW,
                      region_mmh_new,
                      region_new,
                      manager_,
                      perfect_result,
                      u_value,
                      m_value,
                      u_value_BYCAT,
                      m_value_BYCAT,
                      u_max_date,
                      m_max_date,
                      cycle_,
                      max_value_BYCAT,
                      IFNULL(NULLIF(CONCAT(splitByChar('_', IF(LENGTH(splitByChar('/', IfNull(Ptext, '')))>=2, splitByChar('/', IfNull(Ptext, ''))[-2], splitByChar('/', IfNull(Ptext, ''))[-2]))[1], ' ', LEFT(splitByChar('_', IF(LENGTH(splitByChar('/', IfNull(Ptext, '')))>=2, splitByChar('/', IfNull(Ptext, ''))[-2], splitByChar('/', IfNull(Ptext, ''))[-2]))[2], 2), '.', LEFT(splitByChar('_', IF(LENGTH(splitByChar('/', IfNull(Ptext, '')))>=2, splitByChar('/', IfNull(Ptext, ''))[-2], splitByChar('/', IfNull(Ptext, ''))[-2]))[3], 2), '.'), ' ..'), CONCAT(splitByChar(' ', ifNull(manager_new, ''))[1], ' ', leftUTF8(splitByChar(' ', ifNull(manager_new, ''))[2], 1), '.', if(length(splitByChar('.', ifNull(manager_new, '')))>1, splitByChar('.', ifNull(splitByChar(' ', ifNull(manager_new, ''))[2], ''))[-2], leftUTF8(splitByChar(' ', ifNull(manager_new, ''))[-1], 1)), '.')) AS manager_,
                      DataADD,
                      SUM(IF(cat != ''
                             and user_name NOT LIKE '%ИДЕАЛЬНЫЙ%', perfect_value, NULL)) OVER (PARTITION BY `date`,
                                                                                                            id_questionnaire,
                                                                                                            name_questionnaire,
                                                                                                            manager_,
                                                                                                            user_name,
                                                                                                            region_mmh,
                                                                                                            region,
                                                                                                            cat1,
                                                                                                            cat2) AS perfect_result,
                                                                                              SUM(IF(cat != ''
                                                                                                     and user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                     and name_questionnaire='Самооценка компетенций медицинского представителя', value_, Null)) OVER (PARTITION BY `date`,
                                                                                                                                                                                                                   id_questionnaire,
                                                                                                                                                                                                                   name_questionnaire,
                                                                                                                                                                                                                   user_name,
                                                                                                                                                                                                                   cat,
                                                                                                                                                                                                                   cat_s,
                                                                                                                                                                                                                   region_mmh,
                                                                                                                                                                                                                   region) AS u_value,
                                                                                                                                                                                                     SUM(IF(cat != ''
                                                                                                                                                                                                            and user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                                                                                                                            and name_questionnaire='Опрос для региональных менеджеров по оценке развития компетенций у сотрудников', value_, Null)) OVER (PARTITION BY `date`,
                                                                                                                                                                                                                                                                                                                                                       id_questionnaire,
                                                                                                                                                                                                                                                                                                                                                       name_questionnaire,
                                                                                                                                                                                                                                                                                                                                                       user_name,
                                                                                                                                                                                                                                                                                                                                                       cat,
                                                                                                                                                                                                                                                                                                                                                       cat_s,
                                                                                                                                                                                                                                                                                                                                                       region_mmh,
                                                                                                                                                                                                                                                                                                                                                       region) AS m_value,
                                                                                                                                                                                                                                                                                                                                         SUM(IF(cat != ''
                                                                                                                                                                                                                                                                                                                                                and user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                                                                                                                                                                                                                                                                and name_questionnaire='Самооценка компетенций медицинского представителя', value_, Null)) OVER (PARTITION BY name_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                              user_name,
                                                                                                                                                                                                                                                                                                                                                                                                                                                              cat1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                              cat2) AS u_value_BYCAT,
                                                                                                                                                                                                                                                                                                                                                                                                                                                SUM(IF(cat != ''
                                                                                                                                                                                                                                                                                                                                                                                                                                                       and user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                                                                                                                                                                                                                                                                                                                                                                       and name_questionnaire='Опрос для региональных менеджеров по оценке развития компетенций у сотрудников', value_, Null)) OVER (PARTITION BY name_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  user_name,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  cat1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  cat2) AS m_value_BYCAT,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    MAX(IF(user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           and name_questionnaire='Самооценка компетенций медицинского представителя', `date`, Null)) OVER (PARTITION BY user_name,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         region_mmh,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         region) AS u_max_date,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           MAX(IF(user_name NOT LIKE '%ИДЕАЛЬНЫЙ%'
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  and name_questionnaire = 'Опрос для региональных менеджеров по оценке развития компетенций у сотрудников', `date`, Null)) OVER (PARTITION BY user_name,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               region_mmh,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               region) AS m_max_date,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 concat(CEIL(toMonth(`date`)/3), ' цикл ', toYear(`date`)) as cycle_,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 (MAX(IF(cat=''
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         or user_name LIKE '%ИДЕАЛЬНЫЙ%', 0, value_)) OVER (PARTITION BY id_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         name_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         cat1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         cat2)) *(COUNT(DISTINCT IF(cat=''
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    or user_name LIKE '%ИДЕАЛЬНЫЙ%', Null, question)) OVER (PARTITION BY id_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         name_questionnaire,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         cat1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         cat2)) AS max_value_BYCAT
               FROM
                 (SELECT rank() over (
                                      order by user_name) as rank_,
                                     IF(user_name LIKE '%ИДЕАЛЬНЫЙ%', user_id, user_id_part1) as user_id,
                                     CONCAT(splitByChar(' ', ifNull(user_name, ''))[1], ' ', leftUTF8(splitByChar('.', splitByChar(' ', ifNull(user_name, ''))[2])[1], 1), '.', leftUTF8(splitByChar('.', splitByChar(' ', ifNull(user_name, ''))[2])[2], 1), '.') AS user_name,
                                     `date`,
                                     id_questionnaire,
                                     name_questionnaire,
                                     question,
                                     answer,
                                     value_,
                                     MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', Ptext, NULL)) OVER (PARTITION BY user_name) AS Ptext,
                                                                                                                                         MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', Plogin, NULL)) OVER (PARTITION BY user_name) AS Plogin,
                                                                                                                                                                                                                                              cat,
                                                                                                                                                                                                                                              cat_s,
                                                                                                                                                                                                                                              user_ld,
                                                                                                                                                                                                                                              OU,
                                                                                                                                                                                                                                              MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', A.region, NULL)) OVER (PARTITION BY user_name) AS region,
                                                                                                                                                                                                                                                                                                                                                     MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', dictrict, NULL)) OVER (PARTITION BY user_name) AS dictrict,
                                                                                                                                                                                                                                                                                                                                                                                                                                                            MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', A.region_mmh, NULL)) OVER (PARTITION BY user_name) AS region_mmh,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       '' as cat1,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       '' as cat2,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       perfect_value,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       B.sAMAccountName AS sAMAccountName,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       B.region AS reg_acc,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       B.region_mmh AS region_mmh_acc,
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       B.p_level AS p_level
                  FROM
                    (SELECT *,
                            MAX(IF(name_questionnaire = 'Самооценка компетенций медицинского представителя', user_id, NULL)) OVER (PARTITION BY user_name) AS user_id_part1,
                                                                                                                                  1 AS FLAG
                     FROM CLOUD_questionnaire.questionnaire
                     where cat_s NOT IN ('Выявление потребности',
                                         'Знания (медицинские и по продукту)',
                                         'Конкурентная отстройка',
                                         'Открытость',
                                         'Приверженность к этическим ценностям и интересам Компании',
                                         'Соблюдение сроков') ) A
                  LEFT JOIN
                    (SELECT Cat,
                            Cat_s,
                            Value_ AS perfect_value
                     FROM CLOUD_questionnaire.DICT_Perfect_mp
                     WHERE 1 ) P on A.cat = P.Cat
                  and A.cat_s = P.Cat_s
                  LEFT JOIN
                    (SELECT *,
                            1 AS FLAG
                     FROM CLOUD_questionnaire.Users_Access
                     WHERE lowerUTF8(sAMAccountName) = lowerUTF8('GrushkoIV')) B ON A.FLAG = B.FLAG) A
               LEFT JOIN
                 (select CONCAT(splitByChar(' ', ifNull(fio, ''))[1], ' ', leftUTF8(splitByChar(' ', ifNull(fio, ''))[2], 1), '.', if(length(splitByChar('.', ifNull(fio, '')))>1, leftUTF8(splitByChar('.', ifNull(splitByChar(' ', ifNull(fio, ''))[2], ''))[-2], 1), leftUTF8(splitByChar(' ', ifNull(fio, ''))[-1], 1)), '.') as name_NEW,
                         sAMAccountName AS sAMAccountName_NEW,
                         region_mmh as region_mmh_new,
                         region as region_new,
                         manager as manager_new,
                         DataADD
                  from CLOUD_questionnaire.Users_Access
                  limit 1 by name_NEW) B ON A.user_id = B.sAMAccountName_NEW
               or replaceAll(lowerUTF8(A.user_name), 'ё', 'е') = replaceAll(lowerUTF8(B.name_NEW), 'ё', 'е')
               where (1) ))
         where ((p_level = 1)
                or (p_level = 2
                    and (region_mmh = region_mmh_acc
                         or POSITION(lowerUTF8(Plogin), lowerUTF8(sAMAccountName)) != 0))
                or (POSITION(lowerUTF8(Plogin), lowerUTF8('GrushkoIV')) != 0)
                or lowerUTF8('GrushkoIV') not in
                  (select lowerUTF8(sAMAccountName) as sAMAccountName
                   from CLOUD_questionnaire.Users_Access)) ) MAIN
      LEFT JOIN
        (SELECT *
         FROM CLOUD_questionnaire.tmp_kpi) SUB ON MAIN.user_id = SUB.user_ld
      AND MAIN.name_questionnaire = 'Опрос для региональных менеджеров по оценке развития компетенций у сотрудников')) AS `virtual_table`
WHERE `subjmod` IS NOT NULL
  AND `Brick` IS NULL
  AND ((cat != '')
       AND (user_name NOT LIKE '%ИДЕАЛЬНЫЙ%')) AND Brand__= 'Проспекта'
GROUP BY `user_name`,
         `region`
                            '''
df = client.query_df(result)	
print(df.head())


df = df.dropna()
df[['score', 'sales_dynamic']] = df[['score', 'sales_dynamic']].astype(float)

corr = spearmanr(df['score'], df['sales_dynamic'])
print(f'Spearman r={corr.statistic:.3f}, p={corr.pvalue:.4f}')
n = len(df)
# plt.scatter(df['score'], df['plan_fact'])
# plt.show()


fig = px.scatter(
    df,
    x='score',
    y='sales_dynamic',
    hover_data=['user_name', 'region_n'],
    title=f'Оценка vs динамика продаж<br>'
          f'n={n} | Spearman r={corr.statistic:.3f}, p={corr.pvalue:.4f}',
    labels={
        'score': 'Оценка сотрудника',
        'sales_dynamic': 'Динамика региона'
    }
)

fig.add_hline(y=1, line_dash='dash', opacity=0.4)

px.scatter(
    df,
    x='score',
    y='sales_dynamic',
    color='region_n',
    title=f'Проспекта_Общая оценка vs динамика продаж_регионы<br>'
          f'n={n} |Корреляция r={corr.statistic:.3f}, p={corr.pvalue:.4f}',
    hover_data=['user_name']
).show()

# fig.show()


