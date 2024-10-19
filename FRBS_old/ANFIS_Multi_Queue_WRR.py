import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from FRBS.Adjust_Fuzzy_with_ANFIS import Adjust_fuzzy_with_ANFIS
import anfis
import matplotlib.pyplot as plt
#import anfis.membership.membershipfunction
from sklearn.model_selection import train_test_split
from anfis.membership.membershipfunction import MemFuncs


# ANFIS (Adaptive Neuro-Fuzzy Inference System):
# Abordagem para incluir aprendizado nos agentes locais, evoluindo o sistema:
# FRBS (Fuzzy Rule-Based System)

def ANFIS_Multi_Queue_WRR(BufferOccupancy1_value, BufferOccupancy2_value, WRR_Weight1_value, WRR_Weight2_value, total_sum=10, tolerance=0.05, use_anfis=False):
    # Verificar se a soma dos pesos é igual a total_sum
    if WRR_Weight1_value + WRR_Weight2_value != total_sum:
        raise ValueError(f"A soma dos pesos WRR_Weight1 ({WRR_Weight1_value}) e WRR_Weight2 ({WRR_Weight2_value}) deve ser igual a {total_sum}.")
    
    # Definição das variáveis fuzzy
    BufferOccupancy1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy1')
    BufferOccupancy2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy2')
    
    WRR_Weight1 = ctrl.Antecedent(np.arange(1, 10, 1), 'WRR_Weight1')
    Action_WRR_Weight1 = ctrl.Consequent(np.arange(-2, 3, 1), 'Action_WRR_Weight1')

    # Funções de pertinência para BufferOccupancy
    BufferOccupancy1['Low'] = fuzz.trapmf(BufferOccupancy1.universe, [0, 0, 0.3, 0.4])
    BufferOccupancy1['Moderate'] = fuzz.trapmf(BufferOccupancy1.universe, [0.3, 0.4, 0.6, 0.7])
    BufferOccupancy1['High'] = fuzz.trapmf(BufferOccupancy1.universe, [0.6, 0.7, 1, 1])

    BufferOccupancy2['Low'] = fuzz.trapmf(BufferOccupancy2.universe, [0, 0, 0.3, 0.4])
    BufferOccupancy2['Moderate'] = fuzz.trapmf(BufferOccupancy2.universe, [0.3, 0.4, 0.6, 0.7])
    BufferOccupancy2['High'] = fuzz.trapmf(BufferOccupancy2.universe, [0.6, 0.7, 1, 1])

    WRR_Weight1['Low'] = fuzz.trimf(WRR_Weight1.universe, [1, 1, 5])
    WRR_Weight1['Moderate'] = fuzz.trimf(WRR_Weight1.universe, [3, 5, 7])
    WRR_Weight1['High'] = fuzz.trimf(WRR_Weight1.universe, [5, 9, 9])

    Action_WRR_Weight1['Large Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-2, -2, -1])
    Action_WRR_Weight1['Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, -1, 0])
    Action_WRR_Weight1['Maintain'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, 0, 1])
    Action_WRR_Weight1['Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [0, 1, 1])
    Action_WRR_Weight1['Large Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [1, 2, 2])

    # Regras fuzzy ajustadas para incluir os pesos nas condições
    rule1 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Low'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Maintain'])
    rule2 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Moderate'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Maintain'])
    rule3 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['High'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Large Decrement'])
    rule4 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Low'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Maintain'])
    rule5 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Moderate'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Decrement'])
    rule6 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['High'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Large Decrement'])
    rule7 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Low'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Large Increment'])
    rule8 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Moderate'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Increment'])
    rule9 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['High'] & (WRR_Weight1['Low'] | WRR_Weight1['Moderate'] | WRR_Weight1['High']), Action_WRR_Weight1['Large Decrement'])

    wrr_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    
    # Se o ANFIS for ativado, ajustar as funções de pertinência
    if use_anfis:
        wrr_ctrl = Adjust_fuzzy_with_ANFIS(wrr_ctrl, BufferOccupancy1_value, BufferOccupancy2_value)
    
    wrr_sim = ctrl.ControlSystemSimulation(wrr_ctrl)

    # Avaliação do sistema fuzzy
    wrr_sim.input['BufferOccupancy1'] = BufferOccupancy1_value
    wrr_sim.input['BufferOccupancy2'] = BufferOccupancy2_value
    wrr_sim.input['WRR_Weight1'] = WRR_Weight1_value

    wrr_sim.compute()

    # Verifique se o sistema fuzzy gerou as saídas corretamente
    if 'Action_WRR_Weight1' not in wrr_sim.output:
        raise ValueError("A saída 'Action_WRR_Weight1' não foi encontrada no sistema fuzzy.")

    action_wrr_weight1 = int(wrr_sim.output['Action_WRR_Weight1'])
    new_wrr_weight1 = WRR_Weight1_value + action_wrr_weight1
    new_wrr_weight2 = total_sum - new_wrr_weight1

    new_wrr_weight1 = max(1, min(9, new_wrr_weight1))
    new_wrr_weight2 = max(1, min(9, new_wrr_weight2))

    return new_wrr_weight1, new_wrr_weight2, total_sum
