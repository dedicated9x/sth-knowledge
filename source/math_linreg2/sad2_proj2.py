import pomegranate as pg
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib as pl
import random


root_path = pl.Path(r'C:\Users\devoted\Desktop\sad2_output')
text_output_path = root_path.joinpath('txt_output.txt')

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

with open(text_output_path, 'w') as outfile:
    outfile.write('Task 1.\n')

    outfile.write('Equations and laws we will use:\n')
    outfile.write('P(A|BCD) = P(ABCD)/P(BCD) (cond. probability) \n')
    outfile.write('P(X1, X2, ..., Xn) = product_of_probabilities P(X_i|parents(X_i)) (Bayesian Network\'s rule)\n')
    outfile.write("(a' / b' = a / b and a' + b' = 1) => ( a' = a/(a+b) and b' = b/(a+b) ) (renormalization)\n")

    outfile.write('\n')
    for k, v in nominators_T.items():
        outfile.write("P({},S=T,W=T) =  {:.2f} / some_denominator\n".format(k, v))
    outfile.write('\n')
    for k, v in probabilities_T.items():
        outfile.write("P({},S=T,W=T) =  {:.2f}\n".format(k, v))
    outfile.write('\n')

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

df = draw_sequence(100)

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 2.\n')
    outfile.write('The Gibbs sampler has been implemented.\n\n')

"""#######################################################################################
3. Estimate marginal probability of rain
#######################################################################################"""
marginal_R_T_sample100 = df['rain'].value_counts(normalize=True)['T']

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 3.\n')
    outfile.write('P(R=T|S=T,W=T)_sample100 = {:.2f}\n\n'.format(marginal_R_T_sample100 ))


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
        ax.set_ylim(0, 1)
        ax.set_xlabel(label, labelpad=-10)
        ax.legend(loc='upper right')

    fig.savefig(root_path.joinpath('task5_rel_freq.png'))

plot_frequencies(chain1, chain2)

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 5.\n')
    outfile.write('I think n=300 will be good for burn-in.\n\n')

"""#######################################################################################
6. Apply the Gelman test. Plot potential scale reduction factor. Suggest a burn-in time.
#######################################################################################"""
""" wybor -> doubled_n = 500 """
def convert_to_float(chain):
    # Obliczenia wymagają wartosci liczbowych
    return chain.applymap(lambda x: {'T': 1.0, 'F': 0.0}[x])


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
    ax1.set_xlabel('n*')
    ax1.set_ylabel('PSRF')
    ax1.scatter(doubled_n_list_, psrf_list_)

    fig.savefig(root_path.joinpath('task6_psrf.png'))



chain1_float, chain2_float = [convert_to_float(chain) for chain in [chain1, chain2]]
doubled_n_list = [2 * round(elem / 2) for elem in np.logspace(1, 4.68, 150)]
psrf_list = [calculate_psrf(n, chain1_float, chain2_float) for n in doubled_n_list]
plot_psrf_list(psrf_list, doubled_n_list)


with open(text_output_path, 'a') as outfile:
    outfile.write('Task 6.\n')
    outfile.write('I wrote n* instead of n, because of the Gelman test implementation, where n has been assigned to the half the length.\n')
    outfile.write('Therefore n*=2n indeed.\n')
    outfile.write('I used algorithm directly from: "General Methods for Monitoring Convergence of Iterative Simulations. Brooks, Gelman. 1997".\n')
    outfile.write('I think 2n=500 (n=250) will be good for burn-in.\n')
    outfile.write('Taking into account the result of the task 5 my choice, for burning in point is n=max(300, 250)=300.\n\n')
    

"""#######################################################################################
7. Plot autocorrelation. Suggest an interval.
#######################################################################################"""
""" wybor -> maxlags = 20 """
def plot_acorr(chain):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    for ax, var_full, var_abrr in zip([ax1, ax2], ['rain', 'cloudy'], ['R', 'C']):
        ax.acorr(chain[var_full].values, maxlags=30)
        ax.set_xlabel('k, ' + var_full + f' ({var_abrr})', labelpad=-5)
        ax.set_ylabel('lag k')
        ax.set_xlim(0, 30)
    fig.savefig(root_path.joinpath('task7_corr.png'))

plot_acorr(chain1_float)

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 7.\n')
    outfile.write('I think interval=20 will be good for thinning.\n\n')


"""#######################################################################################
8. Re-estimate marginal probability of rain
#######################################################################################"""
df_v2 = draw_sequence(2600)
df_after_burn_in = df_v2[600:].reset_index()

df_thinned = df_after_burn_in[df_after_burn_in['index'] % 20 == 0].drop(['index'], axis=1)
marginal_R_T_burned_and_thinned = df_thinned['rain'].value_counts(normalize=True)['T']

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 8.\n')
    outfile.write('P(R=T|S=T,W=T)_burned_and_thinned = {:.2f}\n'.format(marginal_R_T_burned_and_thinned))
    outfile.write('We will wait for next task in order to compare this with  "P(R=T|S=T,W=T)_sample100". \n\n')

"""#######################################################################################
9. Compute probability analitically
#######################################################################################"""
nominator_components = model.probability([
    ['T', 'T', 'T', 'T'],
    ['T', 'F', 'T', 'T']
])

denominator_components = model.probability([
    ['T', 'T', 'T', 'T'],
    ['T', 'F', 'T', 'T'],
    ['F', 'T', 'T', 'T'],
    ['F', 'F', 'T', 'T']
])
marginal_R_T_true = sum(nominator_components) / sum(denominator_components)

with open(text_output_path, 'a') as outfile:
    outfile.write('Task 9.\n')
    outfile.write('P(R=T|S=T,W=T)_true = {:.2f}\n'.format(marginal_R_T_true))
    outfile.write('As a summary we can compare probabilities from 2 estimation methods:\n')
    outfile.write('P(R=T|S=T,W=T)_sample100 = {:.2f}\n'.format(marginal_R_T_sample100))
    outfile.write('P(R=T|S=T,W=T)_burned_and_thinned = {:.2f}\n'.format(marginal_R_T_burned_and_thinned))
    outfile.write('As a conclusion I can say, that adding burning-in and thinning gave us better estimate.\n')


