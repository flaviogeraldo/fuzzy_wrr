import numpy as np
import skfuzzy as fuzz
import networkx as nx
from skfuzzy import control as ctrl
from fractions import Fraction

def FRBS_Multi_Queue(BufferOccupancy1_value, BufferOccupancy2_value, Rate1_value, Rate2_value, tolerance=0.05):
    # Definição das variáveis fuzzy
    BufferOccupancy1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy1')
    BufferOccupancy2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy2')
    Rate1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Rate1')
    Rate2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Rate2')
    dRate = ctrl.Consequent(np.arange(-0.125, 0.126, 0.001), 'dRateProportion')

    # Funções de pertinência
    BufferOccupancy1['Low'] = fuzz.trapmf(BufferOccupancy1.universe, [0, 0, 0.3, 0.4])
    BufferOccupancy1['Moderate'] = fuzz.trapmf(BufferOccupancy1.universe, [0.3, 0.4, 0.6, 0.7])
    BufferOccupancy1['High'] = fuzz.trapmf(BufferOccupancy1.universe, [0.6, 0.7, 1, 1])

    BufferOccupancy2['Low'] = fuzz.trapmf(BufferOccupancy2.universe, [0, 0, 0.3, 0.4])
    BufferOccupancy2['Moderate'] = fuzz.trapmf(BufferOccupancy2.universe, [0.3, 0.4, 0.6, 0.7])
    BufferOccupancy2['High'] = fuzz.trapmf(BufferOccupancy2.universe, [0.6, 0.7, 1, 1])

    Rate1['Low'] = fuzz.trapmf(Rate1.universe, [0, 0, 0.3, 0.4])
    Rate1['Moderate'] = fuzz.trapmf(Rate1.universe, [0.3, 0.4, 0.6, 0.7])
    Rate1['High'] = fuzz.trapmf(Rate1.universe, [0.6, 0.7, 1, 1])

    Rate2['Low'] = fuzz.trapmf(Rate2.universe, [0, 0, 0.3, 0.4])
    Rate2['Moderate'] = fuzz.trapmf(Rate2.universe, [0.3, 0.4, 0.6, 0.7])
    Rate2['High'] = fuzz.trapmf(Rate2.universe, [0.6, 0.7, 1, 1])

    dRate['Decrease'] = fuzz.trimf(dRate.universe, [-0.125, -0.0625, 0])
    dRate['Maintain'] = fuzz.trimf(dRate.universe, [-0.0625, 0, 0.0625])
    dRate['Increase'] = fuzz.trimf(dRate.universe, [0, 0.0625, 0.125])

    # Regras fuzzy
    rule1 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Low'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule2 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Moderate'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule3 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['High'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Decrease'])
    rule4 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Low'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule5 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Moderate'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule6 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['High'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule7 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Low'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Increase'])
    rule8 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Moderate'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])
    rule9 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['High'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), dRate['Maintain'])

    # Criação do sistema de controle
    queue_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    queue_sim = ctrl.ControlSystemSimulation(queue_ctrl)

    # Avaliação do sistema fuzzy
    queue_sim.input['BufferOccupancy1'] = BufferOccupancy1_value
    queue_sim.input['BufferOccupancy2'] = BufferOccupancy2_value
    queue_sim.input['Rate1'] = Rate1_value
    queue_sim.input['Rate2'] = Rate2_value

    queue_sim.compute()

    output = queue_sim.output['dRateProportion']
    NewRate1 = np.minimum(np.maximum(Rate1_value + output, 0.001), 0.999)
    NewRate1 = round(NewRate1,2)
    NewRate2 = round(1 - NewRate1,2)

    # Cálculo dos pesos WRR baseados nas novas taxas de transmissão
    frac1 = Fraction(NewRate1).limit_denominator()
    frac2 = Fraction(NewRate2).limit_denominator()

    wrr_weight1 = frac1.numerator
    wrr_weight2 = frac2.numerator
    total_tokens = frac1.denominator

    # Normalizar os pesos com base em uma tolerância
    weight_ratio = wrr_weight1 / wrr_weight2
    if abs(weight_ratio - round(weight_ratio)) < tolerance:
        factor = round(wrr_weight1 / wrr_weight2)
        wrr_weight1 = factor
        wrr_weight2 = 1
        total_tokens = wrr_weight1 + wrr_weight2
    elif abs((wrr_weight2 / wrr_weight1) - round(wrr_weight2 / wrr_weight1)) < tolerance:
        factor = round(wrr_weight2 / wrr_weight1)
        wrr_weight2 = factor
        wrr_weight1 = 1
        total_tokens = wrr_weight1 + wrr_weight2

    return NewRate1, NewRate2, wrr_weight1, wrr_weight2, total_tokens

