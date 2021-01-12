import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
path_ = rf"C:\Users\devoted\Documents\RESEARCH\IDEAS\ideas_v3.xlsx"
df = pd.read_excel(path_, engine='openpyxl', nrows=80)
df = df.drop(columns=['IS_B2B', 'NAME', 'COMMENT'])
def simplify_level(level):
    if level in {'none'}:
        return 'none'
    if level in {'TECH', 'IT'}:
        return 'low_tech'
    if level in {'ML', 'HIGH-TECH', 'BIO', 'CV + NLP', 'CV', 'NLP',}:
        return 'high_tech'
df['TECH_LVL_SIMPLIFIED'] = pd.Categorical(df['TECH_LVL'].apply(simplify_level), categories=['none', 'low_tech', 'high_tech'], ordered=True)
df['AWESOMENESS'] = pd.Categorical(df['AWESOMENESS'], categories=['hujowe', 'średnio', 'spoko', 'zajebiste', 'w HUJ zajebiste'], ordered=True)
df['RISK'] = pd.Categorical(df['RISK'].replace('-', 'zerowe'), categories=['zerowe', 'niskie', 'średnie', 'wysokie'], ordered=True)
df['LABOUR_COST'] = pd.Categorical(df['LABOUR_COST'].replace(['SpaceX', 'PayPal'], '>=PayPal'), categories=['akceptowalne', 'na granicy', 'za duże', '>=PayPal'], ordered=True)
df['SALES_COST'] = pd.Categorical(df['SALES_COST'].replace('-', 'OGROMNE').replace('b2b', 'akceptowalne') , categories=['akceptowalne', 'na granicy', 'za duże', 'OGROMNE'], ordered=True)



def resultant_cost(single_row_df: pd.DataFrame):
    map_ = [
        ["akceptowalne",        "50% na granicy",       "50% za duże",          "pieśń przyszłości"],
        ["50% na granicy",      "100% na granicy",      "50% za duże",          "pieśń przyszłości"],
        ["100% za duże",        "100% za duże",         "100% za duże",         "pieśń przyszłości"],
        ["pieśń przyszłości",   "pieśń przyszłości",    "pieśń przyszłości",    "pieśń przyszłości"]
    ]

    map_ = pd.DataFrame(map_, index=df['LABOUR_COST'].values.categories, columns=df['SALES_COST'].values.categories)
    return map_[single_row_df['SALES_COST']][single_row_df['LABOUR_COST']]
df['RESULTANT_COST'] = df.apply(resultant_cost, axis=1)



RESCOST_TO_X = {
    'pieśń przyszłości': -3,
    '100% za duże': -1,
    '50% za duże': 1,
    '100% na granicy': 3,
    '50% na granicy': 5,
    'akceptowalne': 7
}
AWESOMENESS_TO_Y = {
    'hujowe': -3,
    'średnio': -1,
    'spoko': 1,
    'zajebiste': 3,
    'w HUJ zajebiste': 5
}
df['X'] = df.apply(lambda row: RESCOST_TO_X[row['RESULTANT_COST']], axis=1)
df['Y'] = df.apply(lambda row: AWESOMENESS_TO_Y[row['AWESOMENESS']], axis=1)



def shift_positions_inner(subset: pd.DataFrame) -> np.array:
    size = subset.values.shape[0]
    center_x = subset['X'].mean()
    center_y = subset['Y'].mean()
    radius = .5
    std_vertices = matplotlib.patches.RegularPolygon((0, 0), size)._path._vertices[:-1, :]
    new_positions = radius * std_vertices + np.ones(shape=(size, 1)) * np.array([center_x, center_y])
    return new_positions, std_vertices
def shift_position(subset: pd.DataFrame):
    new_positions, std_vertices = shift_positions_inner(subset)
    subset['X'] = new_positions[:, 0]
    subset['Y'] = new_positions[:, 1]
    subset['IS_DEXTER'] = std_vertices[:, 0] > 0
    return subset
df = df.groupby(['RESULTANT_COST', 'AWESOMENESS']).apply(shift_position)


DELTA = .05

class COLORS:
    black = '#0a0a0a'
    dark_grey = '#3b3939'
    light_grey = '#756d6d'
    barely_visible_gray = '#c4bcbc'
    blue = '#4c72b0'
    orange = '#dd8453'
    green = '#017f00'
    pink = '#ffc7ce'
    brown = '#8a4920'


def assign_color_to_techlvl(level):
    if level in {'none'}:
        return COLORS.brown
    if level in {'TECH', 'IT'}:
        return COLORS.pink
    if level in {'ML', 'HIGH-TECH', 'BIO', 'CV + NLP'}:
        return COLORS.green
    if level in {'CV'}:
        return COLORS.blue
    if level in {'NLP'}:
        return COLORS.orange

RISKS_TO_POINTPARAMS = {
    'zerowe':   {'alpha': 1,    'marker': 'o', 'linewidths': 2.5},
    'niskie':   {'alpha': 0.75, 'marker': 'o', 'linewidths': 1.},
    'średnie':  {'alpha': 0.75, 'marker': '.', 'linewidths': 1.5},
    'wysokie':  {'alpha': 0.25, 'marker': '.', 'linewidths': 1.}
}

RISKS_TO_TEXTPARAMS = {
    'zerowe': {'fontsize': 10, 'color': COLORS.black},
    'niskie': {'fontsize': 8, 'color': COLORS.dark_grey},
    'średnie': {'fontsize': 6, 'color': COLORS.light_grey},
    'wysokie': {'fontsize': 6, 'color': COLORS.barely_visible_gray}
}


# TODO wyskalowanie


def plot_our_graph(df_):
    fig, ax = plt.subplots(1, 1)

    for index, row in df_.iterrows():
        horizontal_alignment = 'left' if row['IS_DEXTER'] else 'right'

        ax.scatter(
            x=[row['X']], y=[row['Y']],
            color=assign_color_to_techlvl(row['TECH_LVL']),
            **RISKS_TO_POINTPARAMS[row['RISK']]
        )

        ax.text(
            x=row['X'], y=row['Y'],
            s=f"  {row['ALIAS']}  ",
            ha=horizontal_alignment,
            va='center',
            **RISKS_TO_TEXTPARAMS[row['RISK']]
        )




    ax.axhline(0, color='black')
    ax.axvline(0, color='black')
    ax.set_xticks(list(RESCOST_TO_X.values()))
    ax.set_xticklabels(list(RESCOST_TO_X.keys()))
    ax.set_yticks(list(AWESOMENESS_TO_Y.values()))
    ax.set_yticklabels(list(AWESOMENESS_TO_Y.keys()))

plot_our_graph(df)














# plt.scatter(x=[5], y=[6], color='#43bdd9', **RISKS_TO_PARAMS['zerowe'])
# plt.scatter(x=[5], y=[6], color='#43bdd9', alpha=1, marker='o', linewidths=2.5)
# plt.scatter(x=[6], y=[7], color='#43bdd9', alpha=0.75, marker='o', linewidths=1)
# plt.scatter(x=[7], y=[8], color='#43bdd9', alpha=0.75, marker='.', linewidths=1.5)
# plt.scatter(x=[8], y=[9], color='#43bdd9', alpha=0.25, marker='.', linewidths=1)

