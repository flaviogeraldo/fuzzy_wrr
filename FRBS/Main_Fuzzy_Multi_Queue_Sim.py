import scipy.io
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 'Agg' é para gráficos sem interface interativa, apenas para salvar arquivos
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import cumfreq
#from Multi_Queue_Sim import Multi_Queue_Sim
from FRBS.Multi_Queue_Tandem_Fed_Sim import Multi_Queue_Tandem_Sim


# dados
data1 = scipy.io.loadmat('dec_pkt_1_100.mat')
X1 = data1['ta'].flatten()[:2000]

data2 = scipy.io.loadmat('dec_pkt_2_100.mat')
X2 = data2['dec2_100'].flatten()[:2000]

# Fuzzy (for "1") and Static (for "0")

# P1_f, P2_f, tx1_f, tx2_f, rho1_f, rho2_f, BO1_f, BO2_f, phi1_f, phi2_f, wrr1, wrr2, total_tokens= Multi_Queue_Sim(X1, X2, 1)
# P1_f, P2_f, tx1_f, tx2_f, rho1_f, rho2_f, BO1_f, BO2_f, phi1_f, phi2_f, wrr1, wrr2= Multi_Queue_Sim(X1, X2, 1)
# P1_s, P2_s, tx1_s, tx2_s, rho1_s, rho2_s, BO1_s, BO2_s, phi1_s, phi2_s = Multi_Queue_Sim(X1, X2, 0)

P1_node1_f, P2_node1_f, tx1_node1_f, tx2_node1_f, rho1_node1_f, rho2_node1_f, BO1_node1_f, BO2_node1_f, P1_node2_f, P2_node2_f, tx1_node2_f, tx2_node2_f, rho1_node2_f, rho2_node2_f, BO1_node2_f, BO2_node2_f, phi1_node1_f, phi2_node1_f, phi1_node2_f, phi2_node2_f, wrr1_vector_node1_f, wrr2_vector_node1_f, wrr1_vector_node2_f, wrr2_vector_node2_f=Multi_Queue_Tandem_Sim(X1,X2,1,federated=False)
P1_node1_ff, P2_node1_ff, tx1_node1_ff, tx2_node1_ff, rho1_node1_ff, rho2_node1_ff, BO1_node1_ff, BO2_node1_ff, P1_node2_ff, P2_node2_ff, tx1_node2_ff, tx2_node2_ff, rho1_node2_ff, rho2_node2_ff, BO1_node2_ff, BO2_node2_ff, phi1_node1_ff, phi2_node1_ff, phi1_node2_ff, phi2_node2_ff, wrr1_vector_node1_ff, wrr2_vector_node1_ff, wrr1_vector_node2_ff, wrr2_vector_node2_ff=Multi_Queue_Tandem_Sim(X1,X2,1,federated=True)
P1_node1_s, P2_node1_s, tx1_node1_s, tx2_node1_s, rho1_node1_s, rho2_node1_s, BO1_node1_s, BO2_node1_s, P1_node2_s, P2_node2_s, tx1_node2_s, tx2_node2_s, rho1_node2_s, rho2_node2_s, BO1_node2_s, BO2_node2_s, phi1_node1_s, phi2_node1_s, phi1_node2_s, phi2_node2_s=Multi_Queue_Tandem_Sim(X1,X2,0,federated=False)

# P1_f2, P2_f2, tx1_f2, tx2_f2, rho1_f2, rho2_f2, BO1_f2, BO2_f2, phi1_f2, phi2_f2, wrr12, wrr22= Multi_Queue_Sim(tx1_f, tx2_f, 1)
# P1_s2, P2_s2, tx1_s2, tx2_s2, rho1_s2, rho2_s2, BO1_s2, BO2_s2, phi1_s2, phi2_s2 = Multi_Queue_Sim(tx1_s, tx2_s, 0)

# Table with statistics (mean buffer occupancy, mean loss rate)
data = {'Parameter': ['BO1', 'BO2', 'P1', 'P2'],
        'Static': [np.mean(BO1_node1_s), np.mean(BO2_node1_s), P1_node1_s, P2_node1_s],
        'Fuzzy': [np.mean(BO1_node1_f), np.mean(BO2_node1_f), P1_node1_f, P2_node1_f],
        'Fed Fuzzy': [np.mean(BO1_node1_ff), np.mean(BO2_node1_ff), P1_node1_ff, P2_node1_ff]}
T = pd.DataFrame(data)
print(T)

# Table with statistics (mean buffer occupancy, mean loss rate)
data = {'Parameter': ['BO1', 'BO2', 'P1', 'P2'],
        'Static': [np.mean(BO1_node2_s), np.mean(BO2_node2_s), P1_node2_s, P2_node2_s],
        'Fuzzy': [np.mean(BO1_node2_f), np.mean(BO2_node2_f), P1_node2_f, P2_node2_f],
        'Fed Fuzzy': [np.mean(BO1_node2_ff), np.mean(BO2_node2_ff), P1_node2_ff, P2_node2_ff]}
T = pd.DataFrame(data)
print(T)

# Plots
plt.figure(1)
plt.step(range(len(BO1_node1_s)), BO1_node1_s, '-b', linewidth=1, label='Static')
plt.step(range(len(BO1_node1_f)), BO1_node1_f, '-.r', linewidth=1, label='Fuzzy')
plt.legend()
plt.ylabel('Buffer Occupancy')
plt.xlabel('Time (x100 ms)')
plt.title('Queue 1 - Node 1')
plt.grid()
plt.gca().tick_params(labelsize=16)

plt.figure(2)
plt.step(range(len(BO2_node1_s)), BO2_node1_s, '-b', linewidth=1, label='Static')
plt.step(range(len(BO2_node1_f)), BO2_node1_f, '-.r', linewidth=1, label='Fuzzy')
plt.legend()
plt.ylabel('Buffer Occupancy')
plt.xlabel('Time (x100 ms)')
plt.title('Queue 2 - Node 1')
plt.grid()
plt.gca().tick_params(labelsize=16)

plt.figure(3)
plt.step(range(len(phi1_node1_f)), phi1_node1_f, '-b', linewidth=1, label='Fuzzy')
plt.step(range(len(phi1_node1_s)), phi1_node1_s, '-.r', linewidth=1, label='Static')
plt.legend()
plt.ylabel('Capacity')
plt.xlabel('Time (x100 ms)')
plt.grid()
plt.gca().tick_params(labelsize=16)

plt.figure(4)
plt.subplot(2, 1, 1)
plt.plot(X1, '-b', linewidth=1, label='dec-pkt-1')
plt.legend()
plt.xlabel('Time (x100 ms)')
plt.ylabel('bytes')
plt.gca().tick_params(labelsize=16)

plt.subplot(2, 1, 2)
plt.plot(X2, '-.r', linewidth=1, label='dec-pkt-2')
plt.legend()
plt.xlabel('Time (x100 ms)')
plt.ylabel('bytes')
plt.gca().tick_params(labelsize=16)

plt.savefig("figure4.png")

# CDF plots
plt.figure(5)

# CDF
def cdfplot(data, linestyle, label):
    data_sorted = np.sort(data)
    cdf_vals = np.arange(1, len(data_sorted)+1) / len(data_sorted)
    plt.step(data_sorted, cdf_vals, linestyle=linestyle, linewidth=1, label=label)

cdfplot(BO1_node1_f, '-', 'Fuzzy (Queue 1 - Node 1)')
cdfplot(BO2_node1_f, '-', 'Fuzzy (Queue 2 - Node 1)')
cdfplot(BO1_node1_ff, '*', 'Fed Fuzzy (Queue 1 - Node 1)')
cdfplot(BO2_node1_ff, '*', 'Fed Fuzzy (Queue 2 - Node 1)')
cdfplot(BO1_node1_s, '--', 'Static (Queue 1 - Node 1)')
cdfplot(BO2_node1_s, '--', 'Static (Queue 2 - Node 1)')

plt.legend(loc='best')
plt.xlabel('x')
plt.ylabel('F(x)')
plt.grid()
plt.gca().tick_params(labelsize=16)
plt.savefig("figure5.png")


# Boxplot data
boxdata = [BO1_node1_s, BO1_node1_f, BO2_node1_s, BO2_node1_f]

# Ploting Boxplot
plt.figure(6)
plt.boxplot(boxdata, patch_artist=True, tick_labels=['BO1_s', 'BO1_f', 'BO2_s', 'BO2_f'])
plt.xticks(rotation=45)  # Rotacionar os rótulos, se necessário
plt.grid(True)

plt.figure(7)
plt.step(range(len(wrr1_vector_node1_f)), wrr1_vector_node1_f, '-b', linewidth=1, label='Weights1 Node 1')
plt.step(range(len(wrr2_vector_node1_f)), wrr2_vector_node1_f, '-.r', linewidth=1, label='Weights2 Node 1')
plt.legend()
plt.ylabel('Weights')
plt.xlabel('Time (x100 ms)')
plt.grid()
plt.gca().tick_params(labelsize=16)
plt.savefig("figure7.png")

plt.figure(8)
plt.subplot(2, 1, 1)
plt.plot(tx1_node1_f, '-b', linewidth=1, label='Queue 1 - Node 1')
plt.legend()
plt.xlabel('Time (x100 ms)')
plt.ylabel('bytes')
plt.gca().tick_params(labelsize=16)

plt.subplot(2, 1, 2)
plt.plot(tx2_node1_f, '-.r', linewidth=1, label='Queue 2 - Node 1')
plt.legend()
plt.xlabel('Time (x100 ms)')
plt.ylabel('bytes')
plt.gca().tick_params(labelsize=16)

plt.savefig("figure8.png")






