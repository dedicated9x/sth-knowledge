import pomegranate as pg
import pandas as pd
import random

"""#####################
BN
#####################"""
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

probabilities_T = {}
for nom_T_key, nom_F_key in zip(['C=T|R=T', 'C=T|R=F', 'R=T|C=T', 'R=T|C=F'], ['C=F|R=T', 'C=F|R=F', 'R=F|C=T', 'R=F|C=F']):
    nom_T, nom_F = nominators_T[nom_T_key], nominators_F[nom_F_key]
    prob_T = nom_T / (nom_T + nom_F)
    probabilities_T[nom_T_key] = prob_T

"""#####################
GIBBS
#####################"""

def get_prob(drawing_var_name, second_var_state):
    return {
        'rain': {
            'T': 0.2,
            'F': 0.3
        },
        'cloudy': {
            'T': 0.4,
            'F': 0.5
        }
    }[drawing_var_name][second_var_state]

def draw_sample(rain_seq, cloudy_seq):
    name_to_seq = {'rain': rain_seq, 'cloudy': cloudy_seq}

    drawing_var_name, second_var_name = random.choices([('rain', 'cloudy'), ('cloudy', 'rain')], weights=[0.5, 0.5], k=1)[0]
    second_var_state = name_to_seq[second_var_name][-1]
    prob_true = get_prob(drawing_var_name, second_var_state)
    new_state = random.choices(['T', 'F'], weights=[prob_true, 1-prob_true], k=1)[0]

    name_to_seq[drawing_var_name].append(new_state)
    name_to_seq[second_var_name].append(second_var_state)

rain_seq = ['T']
cloudy_seq = ['T']

for i in range(999):
    draw_sample(rain_seq, cloudy_seq)

# TODO sprawdz, czy na pewno prawdopodobienstwa sie zgadzaja

df = pd.DataFrame().assign(rain= rain_seq, cloudy=cloudy_seq)