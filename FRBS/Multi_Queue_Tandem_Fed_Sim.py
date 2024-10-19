import numpy as np
from FRBS.FRBS_with_ANFIS_call import FRBS_WRR

def federated_aggregation(wrr_weight1_node1, wrr_weight2_node1, wrr_weight1_node2, wrr_weight2_node2):
 
    aggregated_wrr_weight1 = (wrr_weight1_node1 + wrr_weight1_node2) / 2
    aggregated_wrr_weight2 = (wrr_weight2_node1 + wrr_weight2_node2) / 2
    return aggregated_wrr_weight1, aggregated_wrr_weight2

def Multi_Queue_Tandem_Sim(X1_node1, X2_node1, fuzzy, federated=False):
        
    # Configurações iniciais
    X1_m = np.mean(X1_node1)
    X2_m = np.mean(X2_node1)
    X_m = max(X1_m, X2_m)
    tx_m = 2.5 * X_m
    Qsize = 10 * X1_m

    # Inicializar buffers e métricas para os dois nós
    Co1_node1, Co2_node1 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))
    Co1_node2, Co2_node2 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))

    tx1_node1, tx2_node1 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))
    tx1_node2, tx2_node2 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))

    wrr1_vector_node1, wrr2_vector_node1 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))
    wrr1_vector_node2, wrr2_vector_node2 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))

    Qr1_node1, Qr2_node1 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))
    Qr1_node2, Qr2_node2 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))

    perda1_node1, perda2_node1 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))
    perda1_node2, perda2_node2 = np.zeros(len(X1_node1)), np.zeros(len(X2_node1))

    # Parâmetros iniciais de transmissão e pesos WRR para ambos os nós
    tx1_norm_node1, tx2_norm_node1 = 0.5, 0.5
    tx1_norm_node2, tx2_norm_node2 = 0.5, 0.5
    wrr_weight1_node1, wrr_weight2_node1 = 5, 5
    wrr_weight1_node2, wrr_weight2_node2 = 5, 5

    # Tamanho inicial das filas no nó 1
    Qr1_node1[0] = min(Qsize, max(X1_node1[0] - tx1_node1[0], 0))
    Qr2_node1[0] = min(Qsize, max(X2_node1[0] - tx2_node1[0], 0))

    # Loop de simulação
    for i in range(1, len(X1_node1)):
        # Nó 1 (Fuzzy Logic para Nó 1)
        if fuzzy == 1:
            print(".", end="")
            Qr1_norm_node1 = Qr1_node1[i - 1] / Qsize
            Qr2_norm_node1 = Qr2_node1[i - 1] / Qsize
            new_wrr_weight1_node1, new_wrr_weight2_node1, total_tokens_node1 = FRBS_WRR(
                Qr1_norm_node1, Qr2_norm_node1, wrr_weight1_node1, wrr_weight2_node1,anfis=True)

            # Atualizar pesos WRR para as filas do nó 1
            wrr1_vector_node1[i] = new_wrr_weight1_node1
            wrr2_vector_node1[i] = new_wrr_weight2_node1
            wrr_weight1_node1, wrr_weight2_node1 = new_wrr_weight1_node1, new_wrr_weight2_node1

            # Normalizar as taxas de transmissão com base nos pesos WRR
            tx1_norm_node1 = (new_wrr_weight1_node1 / total_tokens_node1)
            tx2_norm_node1 = (new_wrr_weight2_node1 / total_tokens_node1)

        # Calcular as taxas de transmissão para as filas do nó 1
        tx1_node1[i] = tx1_norm_node1 * tx_m
        tx2_node1[i] = tx2_norm_node1 * tx_m

        # Atualizar tamanho das filas para o nó 1
        Qr1_node1[i] = Qr1_node1[i - 1] + X1_node1[i] - tx1_node1[i]
        Qr2_node1[i] = Qr2_node1[i - 1] + X2_node1[i] - tx2_node1[i]

        # Verificar underflow e overflow nas filas do nó 1
        if Qr1_node1[i] < 0:
            Co1_node1[i] = abs(Qr1_node1[i])
            Qr1_node1[i] = 0
        elif Qr1_node1[i] > Qsize:
            perda1_node1[i] = Qr1_node1[i] - Qsize
            Qr1_node1[i] = Qsize

        if Qr2_node1[i] < 0:
            Co2_node1[i] = abs(Qr2_node1[i])
            Qr2_node1[i] = 0
        elif Qr2_node1[i] > Qsize:
            perda2_node1[i] = Qr2_node1[i] - Qsize
            Qr2_node1[i] = Qsize

        # Saída das filas do Nó 1 alimenta as filas do Nó 2
        X1_node2_input = tx1_node1[i]  # Saída da Fila 1 do Nó 1 vira entrada da Fila 1 do Nó 2
        X2_node2_input = tx2_node1[i]  # Saída da Fila 2 do Nó 1 vira entrada da Fila 2 do Nó 2

        # Nó 2 (Fuzzy Logic para Nó 2)
        if fuzzy == 1:
            Qr1_norm_node2 = Qr1_node2[i - 1] / Qsize
            Qr2_norm_node2 = Qr2_node2[i - 1] / Qsize
            new_wrr_weight1_node2, new_wrr_weight2_node2, total_tokens_node2 = FRBS_WRR(
                Qr1_norm_node2, Qr2_norm_node2, wrr_weight1_node2, wrr_weight2_node2)

            # Atualizar pesos WRR para as filas do nó 2
            wrr1_vector_node2[i] = new_wrr_weight1_node2
            wrr2_vector_node2[i] = new_wrr_weight2_node2
            wrr_weight1_node2, wrr_weight2_node2 = new_wrr_weight1_node2, new_wrr_weight2_node2

            # Normalizar as taxas de transmissão com base nos pesos WRR
            tx1_norm_node2 = (new_wrr_weight1_node2 / total_tokens_node2)
            tx2_norm_node2 = (new_wrr_weight2_node2 / total_tokens_node2)

        # Calcular as taxas de transmissão para as filas do nó 2
        tx1_node2[i] = tx1_norm_node2 * tx_m
        tx2_node2[i] = tx2_norm_node2 * tx_m

        # Atualizar as filas do nó 2 com base nas entradas do nó 1
        Qr1_node2[i] = Qr1_node2[i - 1] + X1_node2_input - tx1_node2[i]
        Qr2_node2[i] = Qr2_node2[i - 1] + X2_node2_input - tx2_node2[i]

        # Verificar underflow e overflow nas filas do nó 2
        if Qr1_node2[i] < 0:
            Co1_node2[i] = abs(Qr1_node2[i])
            Qr1_node2[i] = 0
        elif Qr1_node2[i] > Qsize:
            perda1_node2[i] = Qr1_node2[i] - Qsize
            Qr1_node2[i] = Qsize

        if Qr2_node2[i] < 0:
            Co2_node2[i] = abs(Qr2_node2[i])
            Qr2_node2[i] = 0
        elif Qr2_node2[i] > Qsize:
            perda2_node2[i] = Qr2_node2[i] - Qsize
            Qr2_node2[i] = Qsize

        # Agregação Federated Learning (FedAVG) dos pesos WRR (se a flag federated=True)
        if fuzzy == 1 and federated:
            # Agregar pesos WRR dos dois nós
            aggregated_wrr_weight1, aggregated_wrr_weight2 = federated_aggregation(
                wrr_weight1_node1, wrr_weight2_node1, wrr_weight1_node2, wrr_weight2_node2)

            # Atualizar pesos globais nos dois nós
            wrr_weight1_node1 = wrr_weight1_node2 = aggregated_wrr_weight1
            wrr_weight2_node1 = wrr_weight2_node2 = aggregated_wrr_weight2

    # Corrigindo o cálculo de rho para evitar divisões por zero
    rho1_node1 = np.where(tx1_node1 > 0, (tx1_node1 - Co1_node1) / tx1_node1, 0)
    rho2_node1 = np.where(tx2_node1 > 0, (tx2_node1 - Co2_node1) / tx2_node1, 0)
    rho1_node2 = np.where(tx1_node2 > 0, (tx1_node2 - Co1_node2) / tx1_node2, 0)
    rho2_node2 = np.where(tx2_node2 > 0, (tx2_node2 - Co2_node2) / tx2_node2, 0)

    # Métricas para o nó 1 (filas 1 e 2)
    P1_node1 = np.sum(perda1_node1) / np.sum(X1_node1)
    BO1_node1 = Qr1_node1 / Qsize
    P2_node1 = np.sum(perda2_node1) / np.sum(X2_node1)
    BO2_node1 = Qr2_node1 / Qsize

    # Métricas para o nó 2 (filas 1 e 2)
    P1_node2 = np.sum(perda1_node2) / np.sum(X1_node2_input)
    BO1_node2 = Qr1_node2 / Qsize
    P2_node2 = np.sum(perda2_node2) / np.sum(X2_node2_input)
    BO2_node2 = Qr2_node2 / Qsize

    # Phi para ambos os nós
    phi1_node1 = tx1_node1 / tx_m
    phi2_node1 = tx2_node1 / tx_m
    phi1_node2 = tx1_node2 / tx_m
    phi2_node2 = tx2_node2 / tx_m

    if fuzzy == 1:
        return (P1_node1, P2_node1, tx1_node1, tx2_node1, rho1_node1, rho2_node1,
                BO1_node1, BO2_node1, P1_node2, P2_node2, tx1_node2, tx2_node2,
                rho1_node2, rho2_node2, BO1_node2, BO2_node2, phi1_node1,
                phi2_node1, phi1_node2, phi2_node2, wrr1_vector_node1,
                wrr2_vector_node1, wrr1_vector_node2, wrr2_vector_node2)
    else:
        return (P1_node1, P2_node1, tx1_node1, tx2_node1, rho1_node1,
                rho2_node1, BO1_node1, BO2_node1, P1_node2, P2_node2,
                tx1_node2, tx2_node2, rho1_node2, rho2_node2, BO1_node2,
                BO2_node2, phi1_node1, phi2_node1, phi1_node2, phi2_node2)
