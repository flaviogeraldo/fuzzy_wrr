
import numpy as np
from FRBS.FRBS_Multi_Queue_WRR import FRBS_Multi_Queue_WRR

def FRBS_Tandem_WRR(buffer_occupancies_node1, buffer_occupancies_node2, weights_node1, weights_node2):
    # Fuzzy system for Node 1
    new_wrr_weight1_node1, new_wrr_weight2_node1, _ = FRBS_Multi_Queue_WRR(
        buffer_occupancies_node1[0], buffer_occupancies_node1[1], weights_node1[0], weights_node1[1])

    # Fuzzy system for Node 2
    new_wrr_weight1_node2, new_wrr_weight2_node2, _ = FRBS_Multi_Queue_WRR(
        buffer_occupancies_node2[0], buffer_occupancies_node2[1], weights_node2[0], weights_node2[1])

    # Return WRR weights for both nodes
    return (new_wrr_weight1_node1, new_wrr_weight2_node1), (new_wrr_weight1_node2, new_wrr_weight2_node2)
