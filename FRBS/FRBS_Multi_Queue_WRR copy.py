import numpy as np
import skfuzzy as fuzz
import networkx as nx
from skfuzzy import control as ctrl
from fractions import Fraction

def FRBS_Multi_Queue_WRR(BufferOccupancy1_value, BufferOccupancy2_value, Rate1_value, Rate2_value, tolerance=0.05):
    # Definição das variáveis fuzzy
    BufferOccupancy1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy1')
    BufferOccupancy2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy2')
    Rate1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Rate1')
    Rate2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'Rate2')
    
    # Consequente como os pesos diretos para o WRR
    WRR_Weight1 = ctrl.Consequent(np.arange(1, 11, 1), 'WRR_Weight1')  # Pesos de 1 a 10 para WRR
    WRR_Weight2 = ctrl.Consequent(np.arange(1, 11, 1), 'WRR_Weight2')

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

    # Funções de pertinência para os pesos WRR
    WRR_Weight1['Low'] = fuzz.trimf(WRR_Weight1.universe, [1, 1, 5])
    WRR_Weight1['Moderate'] = fuzz.trimf(WRR_Weight1.universe, [3, 5, 7])
    WRR_Weight1['High'] = fuzz.trimf(WRR_Weight1.universe, [5, 10, 10])

    WRR_Weight2['Low'] = fuzz.trimf(WRR_Weight2.universe, [1, 1, 5])
    WRR_Weight2['Moderate'] = fuzz.trimf(WRR_Weight2.universe, [3, 5, 7])
    WRR_Weight2['High'] = fuzz.trimf(WRR_Weight2.universe, [5, 10, 10])

    # Regras fuzzy para os pesos do WRR
    rule1 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Low'] & (Rate1['Low'] | Rate1['Moderate'] | Rate1['High']) & (Rate2['Low'] | Rate2['Moderate'] | Rate2['High']), (WRR_Weight1['Moderate'], WRR_Weight2['Moderate']))
    rule2 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Moderate'], (WRR_Weight1['Moderate'], WRR_Weight2['Low']))
    rule3 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['High'], (WRR_Weight1['Low'], WRR_Weight2['High']))
    rule4 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Low'], (WRR_Weight1['Moderate'], WRR_Weight2['Moderate']))
    rule5 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Moderate'], (WRR_Weight1['Moderate'], WRR_Weight2['Moderate']))
    rule6 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['High'], (WRR_Weight1['Low'], WRR_Weight2['High']))
    rule7 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Low'], (WRR_Weight1['High'], WRR_Weight2['Low']))
    rule8 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Moderate'], (WRR_Weight1['Moderate'], WRR_Weight2['Moderate']))
    rule9 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['High'], (WRR_Weight1['Low'], WRR_Weight2['High']))

    # Criação do sistema de controle fuzzy para WRR
    wrr_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    wrr_sim = ctrl.ControlSystemSimulation(wrr_ctrl)

    # Avaliação do sistema fuzzy
    wrr_sim.input['BufferOccupancy1'] = BufferOccupancy1_value
    wrr_sim.input['BufferOccupancy2'] = BufferOccupancy2_value
    wrr_sim.input['Rate1'] = Rate1_value
    wrr_sim.input['Rate2'] = Rate2_value

    wrr_sim.compute()

    # Saída direta como pesos do WRR
    wrr_weight1 = int(wrr_sim.output['WRR_Weight1'])
    wrr_weight2 = int(wrr_sim.output['WRR_Weight2'])
    total_tokens = wrr_weight1 + wrr_weight2

    return wrr_weight1, wrr_weight2, total_tokens
