from more_itertools import chunked
import numpy as np
import json
import os
from utils import load_data


def compute_recall(
    ground_truth: dict,
    approx_res: dict,
    index_size: int,
    queries: int,
    K: np.array,
    m: int=0,
    ef_construction: int=0,
    ef_search: int=0,
    num_candidates: int=0,
    ):


    performances = {}

    performances["index_size"] = index_size
    performances["queries"] = queries

    try:
        os.mkdir(save_dir)
        print("Save directory created !")
    except:
        print("Save directory already existing !")

    if num_candidates:
        performances["num_candidates"] = num_candidates
    else :
        performances["M"] = m
        performances["ef_construction"] = ef_construction
        performances["ef_search"] = ef_search

    tp_micro = {}
    fp_micro = {}
    fn_micro = {}

    performances["macro_recall"] = {}
    performances["macro_precision"] = {}
    performances["micro_recall"] = {}
    performances["micro_precision"] = {}

    for k in K:
        k = int(k)
        tp_micro[k] = 0
        fp_micro[k] = 0
        fn_micro[k] = 0
        performances["macro_recall"][k] = 0
        performances["macro_precision"][k] = 0
        performances["micro_recall"][k] = 0
        performances["micro_precision"][k] = 0
    
    count = 0
    for id in ground_truth.keys():
        if count >= queries : break
        for k in K :
            k = int(k)
            positive_neighbours = np.isin(approx_res[id]["ids"][1:k+1],ground_truth[id]["ids"][1:k+1])

            tp = np.sum(positive_neighbours.astype(int))
            tp_micro[k] = tp_micro[k] + tp

            fp = k - tp
            fp_micro[k] = fp_micro[k] + fp

            found_neighbours_among_the_expected_ones = np.isin(ground_truth[id]["ids"][1:k+1],approx_res[id]["ids"][1:k+1])
            fn = k - (np.sum(found_neighbours_among_the_expected_ones.astype(int)))
            fn_micro[k]  = fn_micro[k] + fn

            performances["macro_recall"][k] = performances["macro_recall"][k] + tp/(tp+fn)
            performances["macro_precision"][k] = performances["macro_precision"][k] + tp/(tp+fp)
            

        count = count + 1

    
    for k in K:
        k = int(k)
        performances["micro_recall"][k] = tp_micro[k]/(tp_micro[k]+fn_micro[k])
        performances["micro_precision"][k] = tp_micro[k]/(tp_micro[k]+fp_micro[k])
        performances["macro_recall"][k] = performances["macro_recall"][k]/count
        performances["macro_precision"][k] = performances["macro_precision"][k]/count

    return performances


if __name__ == "__main__":
    batch_size = 512
    d = 768
    index_size = 4371343
    queries = 1000
    K = np.array([1,4,10,50,100])
    M = 16
    ef_construction = 100
    ef_search = 128
    num_candidates = 110

    save_dir = "computation_512_recall_index_" + str(index_size) + "_queries_" + str(queries)

    ground_truth = load_data("faiss_saves_index_4371343_queries_1000/512_exact_KNN.json")
    #approx_res = load_data("PCA_300_saves_index_4371343_queries_1000/hnsw_ANN_"+str(M)+"_"+str(ef_construction)+"_"+str(ef_search)+".json")
    approx_res = load_data("elastic_index_4371343_queries_1000/512_ef_hnsw_ANN_ncandidates_110_m_16_efconstruction_100.json")



    complete_perf = compute_recall(ground_truth,
                        approx_res,
                        index_size,
                        queries,
                        K,
                        M,
                        ef_construction,
                        ef_search,
                        )

    with open(save_dir+"/512_num_candidates_"+str(num_candidates)+"_m_"+str(M)+"_ef_construction_"+str(ef_construction)+".json",'w') as f:
        json.dump(complete_perf,f)
