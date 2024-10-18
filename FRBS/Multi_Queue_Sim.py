import numpy as np

from FRBS.FRBS_Multi_Queue_WRR import FRBS_Multi_Queue_WRR

# def FRBS_Multi_Queue(Qr1_norm, Qr2_norm, tx1_norm, tx2_norm):
#     return tx1_norm, tx2_norm

def Multi_Queue_Sim(X1, X2, fuzzy):
    X1_m = np.mean(X1)
    X2_m = np.mean(X2)
    X_m = max(X1_m, X2_m)
    tx_m = 2.5 * X_m
    Qsize = 10 * X1_m

    Co1 = np.zeros(len(X1))
    tx1 = np.zeros(len(X1))
    wrr1_vector = np.zeros(len(X1))
    Qr1 = np.zeros(len(X1))
    perda1 = np.zeros(len(X1))

    Co2 = np.zeros(len(X2))
    tx2 = np.zeros(len(X2))
    wrr2_vector = np.zeros(len(X2))
    Qr2 = np.zeros(len(X2))
    perda2 = np.zeros(len(X2))

    tx1_norm = 0.5
    tx1[0] = tx1_norm * tx_m
    Qr1[0] = min(Qsize, max(X1[0] - tx1[0], 0))
    Co1[0] = abs(min(X1[0] - tx1[0], 0))
    perda1[0] = max(0, (X1[0] - tx1[0]) - Qsize)
    tr1 = 1 if perda1[0] != 0 else 0
    wrr_weight1=5

    tx2_norm = 0.5
    tx2[0] = tx2_norm * tx_m
    Qr2[0] = min(Qsize, max(X2[0] - tx2[0], 0))
    Co2[0] = abs(min(X2[0] - tx2[0], 0))
    perda2[0] = max(0, (X2[0] - tx2[0]) - Qsize)
    tr2 = 1 if perda2[0] != 0 else 0
    print("Executing Fuzzy Rule-Based Inference System",end="")
    wrr_weight2=5

    New_wrr_weight1, New_wrr_weight2, total_tokens = None, None, None

    for i in range(1, len(X1)):
        if fuzzy == 1:
            print(".",end="")
            Qr1_norm = Qr1[i - 1] / Qsize
            Qr2_norm = Qr2[i - 1] / Qsize
            # tx1_norm, tx2_norm, wrr1, wrr2, total_tokens = FRBS_Multi_Queue(Qr1_norm, Qr2_norm, tx1_norm, tx2_norm)
            new_wrr_weight1, new_wrr_weight2, total_tokens = FRBS_Multi_Queue_WRR(Qr1_norm, Qr2_norm, wrr_weight1, wrr_weight2)

#            FRBS_Tandem_WRR(buffer_occupancies_node1, buffer_occupancies_node2, weights_node1, weights_node2)

            wrr1_vector[i]=new_wrr_weight1
            wrr2_vector[i]=new_wrr_weight2
            wrr_weight1=new_wrr_weight1
            wrr_weight2=new_wrr_weight2
            tx1_norm = (new_wrr_weight1/total_tokens) 
            tx2_norm = (new_wrr_weight2/total_tokens) 

        tx1[i] = tx1_norm * tx_m
        tx2[i] = tx2_norm * tx_m

        Qr1[i] = Qr1[i - 1] + X1[i] - tx1[i]
        Qr2[i] = Qr2[i - 1] + X2[i] - tx2[i]

        if Qr1[i] < 0:
            Co1[i] = abs(Qr1[i])
            Qr1[i] = 0

        if Qr2[i] < 0:
            Co2[i] = abs(Qr2[i])
            Qr2[i] = 0

        if Qr1[i] > Qsize:
            perda1[i] = Qr1[i] - Qsize
            Qr1[i] = Qsize
            tr1 += 1

        if Qr2[i] > Qsize:
            perda2[i] = Qr2[i] - Qsize
            Qr2[i] = Qsize
            tr2 += 1

    rho1 = (tx1 - Co1) / tx1
    P1 = np.sum(perda1) / np.sum(X1)
    BO1 = Qr1 / Qsize

    rho2 = (tx2 - Co2) / tx2
    P2 = np.sum(perda2) / np.sum(X2)
    BO2 = Qr2 / Qsize

    phi1 = tx1 / tx_m
    phi2 = tx2 / tx_m

    if fuzzy == 1:
        return P1, P2, tx1, tx2, rho1, rho2, BO1, BO2, phi1, phi2, wrr1_vector, wrr2_vector
    else:
        return P1, P2, tx1, tx2, rho1, rho2, BO1, BO2, phi1, phi2