import anfis
import numpy as np
from sklearn.model_selection import train_test_split
from anfis.anfisfile import AnfisClass, predict
from anfis.membership.membershipfunction import MemFuncs
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from FRBS.plot_mf import show_adjusted_membership_functions, plot_membership_functions

# Exemplo de dados de entrada e saída para treinamento
X = np.array([[0.1, 0.4], [0.5, 0.6], [0.9, 0.7]])  # Exemplo de ocupação de buffer
y = np.array([5, 7, 3])  # Exemplo de pesos WRR esperados

# Divisão dos dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Definir funções de pertinência (membership functions)
mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}],
       ['gaussmf', {'mean': -1., 'sigma': 2.}],
       ['gaussmf', {'mean': -4., 'sigma': 10.}],
       ['gaussmf', {'mean': -7., 'sigma': 7.}]],
      [['gaussmf', {'mean': 1., 'sigma': 2.}],
       ['gaussmf', {'mean': 2., 'sigma': 3.}],
       ['gaussmf', {'mean': -2., 'sigma': 10.}],
       ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]]

# Inicializar funções de pertinência e sistema ANFIS
mfc = MemFuncs(mf)
anfis_system = AnfisClass(X_train, y_train, mfc)

# Treinamento do sistema ANFIS
anfis_system.trainHybridJangOffLine(epochs=100)

# Função para ajustar as funções de pertinência com base no ANFIS
def ANFIS_adjust(fuzzy_system, BufferOccupancy1, BufferOccupancy2, predicted_param):
    """
    Ajusta as funções de pertinência do sistema fuzzy com base nas predições feitas pelo ANFIS.
    
    Args:
        fuzzy_system: Sistema fuzzy no qual as funções de pertinência serão ajustadas.
        BufferOccupancy1: Variável fuzzy 1.
        BufferOccupancy2: Variável fuzzy 2.
        predicted_param: Parâmetro predito pelo ANFIS.
        
    Returns:
        fuzzy_system: O sistema fuzzy com as funções de pertinência ajustadas.
    """
    
    # Ajuste dinâmico das funções de pertinência ajustando 'mean' e 'sigma' para gaussmf
    for var in [BufferOccupancy1, BufferOccupancy2]:
        for label in var.terms:
            current_mf = var.terms[label].mf
            new_mean = current_mf[1] + predicted_param[0]  # Ajuste do mean com base no valor predito
            new_sigma = current_mf[2]  # Mantém sigma fixo ou opcionalmente ajusta
            var.terms[label].mf = fuzz.gaussmf(var.universe, mean=new_mean, sigma=new_sigma)
    
    return fuzzy_system

# Função de teste
def test_adjustment(BufferOccupancy1_value, BufferOccupancy2_value):
    """
    Exemplo de como utilizar a função Adjust_Fuzzy_with_ANFIS para ajustar o sistema fuzzy.
    """
    # Criar variáveis fuzzy
    BufferOccupancy1 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy1')
    BufferOccupancy2 = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'BufferOccupancy2')

    # Definir funções de pertinência iniciais
    BufferOccupancy1['Low'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=0., sigma=1.)
    BufferOccupancy1['Moderate'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=0.5, sigma=0.2)
    BufferOccupancy1['High'] = fuzz.gaussmf(BufferOccupancy1.universe, mean=1., sigma=0.2)

    BufferOccupancy2['Low'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=0., sigma=1.)
    BufferOccupancy2['Moderate'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=0.5, sigma=0.2)
    BufferOccupancy2['High'] = fuzz.gaussmf(BufferOccupancy2.universe, mean=1., sigma=0.2)

    # Definir variável de saída
    Action_WRR_Weight1 = ctrl.Consequent(np.arange(-2, 3, 1), 'Action_WRR_Weight1')
    Action_WRR_Weight1['Large Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-2, -2, -1])
    Action_WRR_Weight1['Decrement'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, -1, 0])
    Action_WRR_Weight1['Maintain'] = fuzz.trimf(Action_WRR_Weight1.universe, [-1, 0, 1])
    Action_WRR_Weight1['Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [0, 1, 1])
    Action_WRR_Weight1['Large Increment'] = fuzz.trimf(Action_WRR_Weight1.universe, [1, 2, 2])

    # Definir regras fuzzy
    rule1 = ctrl.Rule(BufferOccupancy1['Low'] & BufferOccupancy2['Low'], Action_WRR_Weight1['Maintain'])
    rule2 = ctrl.Rule(BufferOccupancy1['Moderate'] & BufferOccupancy2['Moderate'], Action_WRR_Weight1['Increment'])
    
    # Criar sistema fuzzy
    wrr_ctrl = ctrl.ControlSystem([rule1, rule2])
    fuzzy_system = ctrl.ControlSystemSimulation(wrr_ctrl)
    
    # Definir os valores de entrada
    fuzzy_system.input['BufferOccupancy1'] = BufferOccupancy1_value
    fuzzy_system.input['BufferOccupancy2'] = BufferOccupancy2_value

    # Computar o sistema fuzzy
    fuzzy_system.compute()

    plot_membership_functions(BufferOccupancy1, "BufferOccupancy1")
    plot_membership_functions(BufferOccupancy2, "BufferOccupancy2")

    # Fazer a predição com o ANFIS
    predicted_param = predict(anfis_system, np.array([[BufferOccupancy1_value, BufferOccupancy2_value]]))

    # Ajustar o sistema fuzzy
    fuzzy_system = ANFIS_adjust(fuzzy_system, BufferOccupancy1, BufferOccupancy2, predicted_param)

    show_adjusted_membership_functions(BufferOccupancy1, BufferOccupancy2)

    
    return fuzzy_system

# Teste simples
fuzzy_system_after_adjustment = test_adjustment(0.3, 0.6)
print(f"Fuzzy system ajustado: {fuzzy_system_after_adjustment}")
