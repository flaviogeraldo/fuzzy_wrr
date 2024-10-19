import anfis
import numpy as np
from sklearn.model_selection import train_test_split
from anfis.anfisfile import AnfisClass
from anfis.membership.membershipfunction import MemFuncs



# Exemplo de dados de entrada e saída para treinamento
# X representa os valores de ocupação do buffer, e y são os pesos WRR ajustados esperados
X = np.array([[0.1, 0.4], [0.5, 0.6], [0.9, 0.7]])  # Exemplo de ocupação de buffer
y = np.array([[5], [7], [3]])  # Exemplo de pesos WRR esperados

# Divisão dos dados para treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Inicialização do sistema ANFIS
#anfis_system = ANFIS(n_inputs=2, n_rules=5)

# Inicialização do sistema ANFIS com os dados de entrada e saída

# Definir funções de pertinência (membership functions)
mf = [[['gaussmf', {'mean': 0., 'sigma': 1.}], ['gaussmf', {'mean': -1., 'sigma': 2.}], ['gaussmf', {'mean': -4., 'sigma': 10.}], ['gaussmf', {'mean': -7., 'sigma': 7.}]],
      [['gaussmf', {'mean': 1., 'sigma': 2.}], ['gaussmf', {'mean': 2., 'sigma': 3.}], ['gaussmf', {'mean': -2., 'sigma': 10.}], ['gaussmf', {'mean': -10.5, 'sigma': 5.}]]]

# Inicializar funções de pertinência e sistema ANFIS
mfc = MemFuncs(mf)

anfis_system = AnfisClass(X_train, y_train, mfc)


# Treinamento do sistema com dados
#anfis_system.trainHybridJangOffLine(X_train, y_train, epochs=100)

anfis_system.trainHybridJangOffLine(epochs=100)

# Função para ajustar as funções de pertinência com base no ANFIS
def Adjust_Fuzzy_with_ANFIS(fuzzy_system, BufferOccupancy1_value, BufferOccupancy2_value):
    # Fazer predição dos novos parâmetros de pertinência usando o ANFIS treinado
    predicted_weights = anfis_system.predict([[BufferOccupancy1_value, BufferOccupancy2_value]])
    
    # Ajuste dinâmico das funções de pertinência com base nas predições
    for var in [fuzzy_system.input['BufferOccupancy1'], fuzzy_system.input['BufferOccupancy2']]:
        for label in var.terms:
            var.terms[label].mf = [x + predicted_weights[0] for x in var.terms[label].mf]
    return fuzzy_system
