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
def ndcg(result_df,k):
    dcg_list = []
    dcgi_list = []
    sorted_result_list = result_df.sort_values(by=['ranking'], ascending = True)
    #sorted_result_list.reverse()
    #print sorted_result_list
    for i, (prop_id, result) in enumerate(result_df.iteritems()):
        if i > k:
            break
        relevance_rank = result
        dcg_rank = ((2**(relevance_rank))-1)/(math.log(1+(1+i),2))
        dcg_list.append(dcg_rank)
    for j, (s_prop_id, s_result) in enumerate(sorted_result_list):
        if j > k:
            break
        relevance_i_rank = s_result
        dcgi_rank = ((2**(relevance_i_rank))-1)/(math.log(1+(1+j),2))
        dcgi_list.append(dcgi_rank)
    dcgp = sum(dcg_list)
    dcgi = sum(dcgi_list)
    ndcg = dcgp/dcgi
    print ndcg
    return ndcg
