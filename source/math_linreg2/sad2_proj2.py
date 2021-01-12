import pomegranate as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

"""#####################
Model initialization
#####################"""
def initialize_model():
    cloudy = pg.DiscreteDistribution({'T': 0.5, 'F': 0.5})
    sprinkler = pg.ConditionalProbabilityTable([
        ['T', 'T', 0.1],
        ['T', 'F', 0.9],
        ['F', 'T', 0.5],
        ['F', 'F', 0.5],
    ], [cloudy])
    rain = pg.ConditionalProbabilityTable([
        ['T', 'T', 0.8],
        ['T', 'F', 0.2],
        ['F', 'T', 0.2],
        ['F', 'F', 0.8],
    ], [cloudy])
    wet_grass = pg.ConditionalProbabilityTable([
        ['T', 'T', 'T', 0.99],
        ['T', 'T', 'F', 0.01],
        ['T', 'F', 'T', 0.90],
        ['T', 'F', 'F', 0.10],
        ['F', 'T', 'T', 0.90],
        ['F', 'T', 'F', 0.10],
        ['F', 'F', 'T', 0.01],
        ['F', 'F', 'F', 0.99],
    ], [sprinkler, rain])

    cloudy_node = pg.Node(cloudy, name="cloudy")
    sprinkler_node = pg.Node(sprinkler, name="sprinkler")
    rain_node = pg.Node(rain, name="rain")
    wet_grass_node = pg.Node(wet_grass, name="wet_grass")

    model = pg.BayesianNetwork('RAIN')
    model.add_states(rain_node, cloudy_node, sprinkler_node, wet_grass_node)
    model.add_edge(cloudy_node, rain_node)
    model.add_edge(rain_node, wet_grass_node)
    model.add_edge(cloudy_node, sprinkler_node)
    model.add_edge(sprinkler_node, wet_grass_node)
    model.bake()

    return model


model = initialize_model()

"""#######################################################################################
1. Compute probabilities
#######################################################################################"""
nominators_T = {
    'C=T|R=T': model.probability([['T', 'T', 'T', 'T']]),
    'C=T|R=F': model.probability([['F', 'T', 'T', 'T']]),
    'R=T|C=T': model.probability([['T', 'T', 'T', 'T']]),
    'R=T|C=F': model.probability([['T', 'F', 'T', 'T']])
}
nominators_F = {
    'C=F|R=T': model.probability([['T', 'F', 'T', 'T']]),
    'C=F|R=F': model.probability([['F', 'F', 'T', 'T']]),
    'R=F|C=T': model.probability([['F', 'T', 'T', 'T']]),
    'R=F|C=F': model.probability([['F', 'F', 'T', 'T']])
}

def calculate_probabilities_T(nominators_T_, nominators_F_):
    probabilities_T = {}
    for nom_T_key, nom_F_key in zip(['C=T|R=T', 'C=T|R=F', 'R=T|C=T', 'R=T|C=F'],
                                    ['C=F|R=T', 'C=F|R=F', 'R=F|C=T', 'R=F|C=F']):
        nom_T, nom_F = nominators_T_[nom_T_key], nominators_F_[nom_F_key]
        prob_T = nom_T / (nom_T + nom_F)
        probabilities_T[nom_T_key] = prob_T
    return probabilities_T


probabilities_T = calculate_probabilities_T(nominators_T, nominators_F)
"""#######################################################################################
2. Implement the Gibbs sample and draw sequence of 1000
#######################################################################################"""
def get_prob(drawing_var_name, second_var_state):
    return {
        'rain': {
            'T': probabilities_T['R=T|C=T'],
            'F': probabilities_T['R=T|C=F']
        },
        'cloudy': {
            'T': probabilities_T['C=T|R=T'],
            'F': probabilities_T['C=T|R=F']
        }
    }[drawing_var_name][second_var_state]


def draw_sample(rain_seq, cloudy_seq):
    name_to_seq = {'rain': rain_seq, 'cloudy': cloudy_seq}

    drawing_var_name, second_var_name = \
    random.choices([('rain', 'cloudy'), ('cloudy', 'rain')], weights=[0.5, 0.5], k=1)[0]
    second_var_state = name_to_seq[second_var_name][-1]
    prob_true = get_prob(drawing_var_name, second_var_state)
    new_state = random.choices(['T', 'F'], weights=[prob_true, 1 - prob_true], k=1)[0]

    name_to_seq[drawing_var_name].append(new_state)
    name_to_seq[second_var_name].append(second_var_state)


def draw_sequence(length):
    # Starting point
    rain_seq = ['T']
    cloudy_seq = ['T']
    # Run
    for i in range(length-1):
        draw_sample(rain_seq, cloudy_seq)

    df = pd.DataFrame().assign(rain=rain_seq, cloudy=cloudy_seq)
    return df

df = draw_sequence(1000)

"""#######################################################################################
3. Estimate marginal probability of rain
#######################################################################################"""
marginal_R_T = df['rain'].value_counts(normalize=True)['T']


"""#######################################################################################
4. Draw 50000 sample
#######################################################################################"""
max_n = 50000
chain1 = draw_sequence(max_n)
chain2 = draw_sequence(max_n)


"""#######################################################################################
5. Plot relative frequencies of R=T and C=T. Suggest a burn-in time.
#######################################################################################"""
""" wybor -> t = 200 """
def calculate_freqs(chain, t):
    freq_rain = chain[:t]['rain'].value_counts(normalize=True)['T']
    freq_cloudy = chain[:t]['cloudy'].value_counts(normalize=True)['T']
    return freq_rain, freq_cloudy

def plot_frequencies(chain1_, chain2_):
    t_list = [2 * round(elem / 2) for elem in np.logspace(1, 4.68, 150)]
    fig, (ax1, ax2) = plt.subplots(2, 1)

    for ax, chain, label in zip([ax1, ax2], [chain1_, chain2_], ['chain1', 'chain2']):
        freq_rain_list, freq_cloudy_list = list(zip(*(
            [calculate_freqs(chain, t) for t in t_list]
        )))
        ax.scatter(t_list, freq_rain_list, color='b', label="p(R=T)")
        ax.scatter(t_list, freq_cloudy_list, color='k', label="p(C=T)")
        ax.set_xlim(0, 1000)
        ax.set_xlabel(label)
        ax.legend(loc='upper right')

# plot_frequencies(chain1, chain2)

"""#######################################################################################
6. Apply the Gelman test. Plot potential scale reduction factor. Suggest a burn-in time.
#######################################################################################"""
""" wybor -> doubled_n = 500 """
def convert_to_int(chain):
    # Obliczenia wymagają wartosci liczbowych
    return chain.applymap(lambda x: {'T': 1, 'F': 0}[x])


def calculate_psrf(doubled_n, complete_chain1, complete_chain2):
    # Prune the first 1/2.
    n = int(doubled_n / 2)
    chain1 = complete_chain1[n:2*n]
    chain2 = complete_chain2[n:2*n]

    W = 0.5 * (chain1.cov().values + chain2.cov().values)
    matrix_of_averages = pd.concat([
        pd.DataFrame(chain1.mean()).transpose(),
        pd.DataFrame(chain2.mean()).transpose()
    ], axis=0)
    B_slash_n = matrix_of_averages.cov().values
    try:
        eigenvalues, _ = np.linalg.eig(np.linalg.inv(W) * B_slash_n)
    except np.linalg.LinAlgError:
        return 0
    lambda1 = max(eigenvalues)
    psrf = (n - 1) / n + (3 / 2) * lambda1
    return psrf


def plot_psrf_list(psrf_list_, doubled_n_list_):
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_xlim(0, 2000)
    ax1.set_ylim(0.9, 2.5)
    ax1.scatter(doubled_n_list_, psrf_list_)


chain1_int, chain2_int = [convert_to_int(chain) for chain in [chain1, chain2]]
doubled_n_list = [2 * round(elem / 2) for elem in np.logspace(1, 4.68, 150)]
psrf_list = [calculate_psrf(n, chain1_int, chain2_int) for n in doubled_n_list]
# plot_psrf_list(psrf_list, doubled_n_list)


"""#######################################################################################
7. Autocorrelation
#######################################################################################"""







""" testowe prawdopodobienstwa"""
# def get_prob(drawing_var_name, second_var_state):
#     return {
#         'rain': {
#             'T': 0.2,
#             'F': 0.3
#         },
#         'cloudy': {
#             'T': 0.4,
#             'F': 0.5
#         }
#     }[drawing_var_name][second_var_state]
""" sprawdzenie, czy procenty sa takie, jak ustalone wczesniej (dopiero przy 1000 jest ok)"""
# df[df['cloudy'] == 'T']['rain'].value_counts(normalize=True)['T']
# df[df['cloudy'] == 'F']['rain'].value_counts(normalize=True)['T']
# df[df['rain'] == 'T']['cloudy'].value_counts(normalize=True)['T']
# df[df['rain'] == 'F']['cloudy'].value_counts(normalize=True)['T']
"""rzeczywiscie pytaja o kowariancje"""
# test_df = pd.DataFrame.from_dict({
#     'x1': [64.0, 66.0, 68.0, 69.0, 73.0],
#     'x2': [580.0, 570.0, 590.0, 660.0, 600.0],
#     'x3': [29.0, 33.0, 37.0, 46.0, 55.0]
# })
#
# sum_ = np.zeros((3, 3))
# for idx in range(5):
#     x_dash = test_df.mean().values
#     x_i = test_df.iloc[[idx]].values
#     sum_ += (x_i - x_dash).T @ (x_i - x_dash)
#
# print(sum_ / 4)
# print(test_df.cov())


