import pandas as pd
import numpy as np
import math
import data_aggregator as da
from datetime import datetime

#The DCG measure calculation
#Here the list with all documents containing the words of the query is inserted as results list
#It is sorted, to create the list for DCGi
#Then both DCG and DCGi are calculated for all documents
#DCG is divided by DCGi which results in nDCG
def ndcg(result_df):
    dcg_list = []
    dcgi_list = []
    sorted_result_df = result_df.sort_values(by=['target'], ascending=False)
    result_matrix = result_df.as_matrix(['target'])
    sorted_result_matrix = sorted_result_df.as_matrix(['target'])
    for i in range(len(result_matrix)):
        [relevance_rank] = result_matrix[i]
        print relevance_rank
        dcg_rank = ((2**(relevance_rank))-1)/(math.log(1+(1+i),2))
        dcg_list.append(dcg_rank)
    for j in range(len(sorted_result_matrix)):
        relevance_i_rank = sorted_result_matrix[j]
        dcgi_rank = ((2**relevance_i_rank)-1)/(math.log(1+(1+j),2))
        dcgi_list.append(dcgi_rank)
    dcgp = sum(dcg_list)
    dcgi = sum(dcgi_list)
    [ndcg] = dcgp/dcgi
    return ndcg
