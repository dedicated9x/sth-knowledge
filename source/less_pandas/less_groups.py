"""******************************************
Filter [rows]
******************************************"""
# series = pd.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
# isodd_filter = (series % 2) == 1
# print(isodd_filter)


"""******************************************
Filter [groups]
******************************************"""
# df = pd.DataFrame.from_dict({
#     'RACE':     ['BLACK',   'HINDU',    'BLACK',    'WHITE',    'HINDU',    'BLACK',    'BLACK',    'WHITE'],
#     'SALARY':   [1200,      150,        3000,       8000,       250,        1400,      2500,       10000]
# })
#
# def remove_poor(group: pd.DataFrame) -> bool:
#     avg_salary = group['SALARY'].mean()
#     return avg_salary > 5000
#
# groups_generator = df.groupby('RACE')
# rich = groups_generator.filter(remove_poor)


"""******************************************
Dataframe -> row
******************************************"""
"""
        1. Czy mozliwy custom? ->                                                                             czemu nie?
"""
# df = pd.DataFrame.from_dict({
#     'x': [1, 2, 3],
#     'y': [6, 5, 4]
# })
#
# geometric_center = df.apply(pd.Series.mean, axis=0)
# print(geometric_center)


"""******************************************
Dataframe(N) -> Series(m) [m << N]
******************************************"""
"""
        1. Czym jest group? ->                                                                                  seriesem
        2. Czym jest SeriesGroupBy? ->                                                          generatorem tych groupow
        3. Czym sie rozni agg() od apply()? ->                         (prawdopodobnie) agg() wymaga, by zwracano skalar
        4. Dlaczego mozna od razu zrobić .mean() ->          wynika to z implementacji elementow GroupBy. Ich zadanie to 
                                                             zaaplikowanie metody i nastepujaca konkatenacja. W tym wy-
                                                             padku aplikuje sie metody do seriesow. A mean() to metoda
                                                             seriesow wlasnie.
"""
# import pandas as pd
# df = pd.DataFrame.from_dict({
#     'IS_RICH':      [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
#     'STUDY_ABROAD': [1, 1, 0, 0, 0, 0, 0, 1, 1, 0]
# })
#
# def calculate_mean(group: pd.Series):
#     mean = group.mean()
#     return mean
#
# seriesgroups_generator = df.groupby('IS_RICH')['STUDY_ABROAD']
#
# summary_v0 = seriesgroups_generator.apply(calculate_mean)
# summary_v1 = seriesgroups_generator.agg(lambda x: x.mean())
# summary_v2 = seriesgroups_generator.mean()
# assert summary_v0.values.tolist() == summary_v1.values.tolist() == summary_v2.values.tolist()
# print(summary_v2)


"""******************************************
Dataframe(n) -> Series(n)
******************************************"""
# df = pd.DataFrame.from_dict({
#     'width': [1, 2, 3],
#     'length': [6, 5, 4]
# })
#
# def area(row: pd.DataFrame):
#     return row['width'] * row['length']
#
# area = df.apply(area, axis=1)
# print(area)


"""******************************************
Dataframe(n) -> Dataframe(n) [group -> group]
******************************************"""
"""
        1. Czym jest jest group? ->                                                                           dataframem
        2. Czym jest DataframeGroupBy? ->                                                  generatorem dataframów (grup)
        3. Jaka magia dzieje sie w DataframeGroupBy.apply() ->                                       konkatenuje resulty 
"""
# df = pd.DataFrame.from_dict({
#     'RACE':     ['BLACK',   'BLACK',    'WHITE',    'BLACK',    'BLACK',    'WHITE'],
#     'SALARY':   [1200,      3000,       8000,        1400,      2500,       10000]
# })
#
# def change_salary_accordingto_race(group: pd.DataFrame):
#     race = group['RACE'].iloc[0]
#     diff = -300 if race == 'BLACK' else 600
#     group['SALARY'] += diff
#     return group
#
# groups_generator = df.groupby('RACE')
# df = groups_generator.apply(change_salary_accordingto_race)


"""******************************************
Dataframe(n) -> Dataframe(n) [row -> row]
******************************************"""
"""
        tak, jak w `Dataframe(n) -> Series(n)`, tylko `return row`
"""

