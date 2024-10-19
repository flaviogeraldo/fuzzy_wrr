import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from FRBS.ANFIS_adjust import ANFIS_adjust
from anfis.anfisfile import AnfisClass
from anfis.membership.membershipfunction import MemFuncs


# Função de ajuste dinâmico com ANFIS
def FRBS_WRR(BufferOccupancy1_value, BufferOccupancy2_value, WRR_Weight1_value, WRR_Weight2_value, total_sum=10, tolerance=0.05, use_anfis=False, anfis_system=None):
    # Verificar se a soma dos pesos é igual a total_sum
    if WRR_Weight1_value + WRR_Weight2_value != total_sum:
        raise ValueError(f"A soma dos pesos WRR_Weight1 ({WRR_Weight1_value}) e WRR_Weight2 ({WRR_Weight2_value}) deve ser igual a {total_sum}.")
    
    # Definição das variáveis fuzzy
    BufferOccupancy1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy1')
    BufferOccupancy2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy2')
    
    WRR_Weight1 = ctrl.Antecedent(np.arange(1, 10, 1), 'WRR_Weight1')
    Action_WRR_Weight1 = ctrl.Consequent(np.arange(-2, 3, 1), 'Action_WRR_Weight1')

    # Funções de pertinência ajustadas para usar gaussmf
    BufferOccupancy1['Low'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=0., sigma=1.)
    BufferOccupancy1['Moderate'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=0.5, sigma=0.2)
    BufferOccupancy1['High'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=1., sigma=0.2)

    BufferOccupancy2['Low'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=0., sigma=1.)
    BufferOccupancy2['Moderate'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=0.5, sigma=0.2)
    BufferOccupancy2['High'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=1., sigma=0.2)

    WRR_Weight1['Low'] = fuzz.gaussmf(WRR_Weight1.universe, mean=2, sigma=1)
    WRR_Weight1['Moderate'] = fuzz.gaussmf(WRR_Weight1.universe, mean=5, sigma=1)
    WRR_Weight1['High'] = fuzz.gaussmf(WRR_Weight1.universe, mean=8, sigma=1)

    Action_WRR_Weight1['Large Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-2, -2, -1])
    Action_WRR_Weight1['Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, -1, 0])
    Action_WRR_Weight1['Maintain'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, 0, 1])
    Action_WRR_Weight1['Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [0, 1, 1])
    Action_WRR_Weight1['Large Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [1, 2, 2])

    # Regras fuzzy ajustadas para incluir os pesos nas condições
    rule1 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Low'], Action_WRR_Weight1['Maintain'])
    rule2 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Moderate'], Action_WRR_Weight1['Maintain'])
    rule3 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['High'], Action_WRR_Weight1['Large Decrement'])
    rule4 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Low'], Action_WRR_Weight1['Maintain'])
    rule5 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Moderate'], Action_WRR_Weight1['Decrement'])
    rule6 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['High'], Action_WRR_Weight1['Large Decrement'])
    rule7 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Low'], Action_WRR_Weight1['Large Increment'])
    rule8 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['Moderate'], Action_WRR_Weight1['Increment'])
    rule9 = ctrl.Rule(BufferOccupancy1['High'] & BufferOccupancy2['High'], Action_WRR_Weight1['Large Decrement'])

    wrr_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6, rule7, rule8, rule9])
    
    # Se o ANFIS for ativado, ajustar as funções de pertinência
    if use_anfis and anfis_system is not None:
        wrr_ctrl = ANFIS_adjust(wrr_ctrl, BufferOccupancy1_value, BufferOccupancy2_value, anfis_system)
    
    wrr_sim = ctrl.ControlSystemSimulation(wrr_ctrl)

    # Avaliação do sistema fuzzy
    wrr_sim.input['BufferOccupancy1'] = BufferOccupancy1_value
    wrr_sim.input['BufferOccupancy2'] = BufferOccupancy2_value
    wrr_sim.input['WRR_Weight1'] = WRR_Weight1_value

    wrr_sim.compute()

    action_wrr_weight1 = int(wrr_sim.output['Action_WRR_Weight1'])
    new_wrr_weight1 = WRR_Weight1_value + action_wrr_weight1
    new_wrr_weight2 = total_sum - new_wrr_weight1

    # Garantir que os pesos fiquem dentro dos limites [1, 9]
    new_wrr_weight1 = max(1, min(9, new_wrr_weight1))
    new_wrr_weight2 = max(1, min(9, new_wrr_weight2))

    return new_wrr_weight1, new_wrr_weight2, total_sum
