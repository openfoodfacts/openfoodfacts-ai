import pathlib
import time
import tqdm
from more_itertools import chunked
import numpy as np
import psutil
import json
import os
import tqdm

import faiss

from recall_computation import compute_recall
from utils import get_embedding, save_data, load_data

def create_index(d:int, exact : bool, m: int=32, efSearch: int=40, efConstruction: int=40):
    """
    Creates an index object.
    
    Args:
        d: The dimensionality of the embeddings.
        exact: A boolean indicating whether to create an exact index or an approximate index.
        m: The number of connections that each node in the index has to its neighbors.
        efSearch: The maximum number of elements to visit when searching the graph.
        efConstruction: The maximum number of elements to visit when adding an element to the graph.
    
    Returns:
        A tuple containing the index object and the time it took to create the index.
    """

    start_time = time.monotonic()

    if exact :
        index = faiss.index_factory(d,"IDMap,Flat")
    else :
        index_hnsw = faiss.IndexHNSWFlat(d, m, faiss.METRIC_INNER_PRODUCT)
        index_hnsw.hnsw.efSearch = efSearch
        index_hnsw.hnsw.efConstruction = efConstruction
        index = faiss.IndexIDMap(index_hnsw)

    return index, time.monotonic() - start_time 


def build_index(index: faiss.swigfaiss.IndexIDMap, embeddings_path: pathlib.Path, batch_size : int, n_vec : int):
    """
    Builds an index with embeddings.
    
    Args:
        index: The index object.
        embeddings_path: The path to the file containing the embeddings.
        batch_size: The number of embeddings to process at a time.
        n_vec: The total number of embeddings to process.
    
    Returns:
        The time it took to add all the embeddings to the index.
    """

    building_time = 0

    data_gen =  get_embedding(embeddings_path, batch_size, n_vec)

    for batch in tqdm.tqdm(data_gen):
        (embedding_batch, external_id_batch) = batch

        elapsed = time.monotonic()
        index.add_with_ids(embedding_batch.astype('float32'),external_id_batch.astype("int64"))
        building_time = building_time + time.monotonic() - elapsed

    return building_time


def search_index(
    index: faiss.swigfaiss.IndexIDMap, 
    embeddings_path: pathlib.Path, 
    queries: int, 
    batch_size: int,
    K: np.array,
    ):
    """
    Performs k-nearest neighbor search on a set of queries.
    
    Args:
        index: The index object.
        embeddings_path: The path to the file containing the embeddings.
        queries: The number of queries to perform.
        batch_size: The number of queries to process at a time.
        K: The number of nearest neighbors to retrieve for each query.
    
    Returns:
        A tuple containing a dictionary of the nearest neighbors for each query and the mean search time for a query.
    """

    search_time = []

    nearest_neighbours = {}

    data_gen = get_embedding(embeddings_path, batch_size, queries)

    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch
            
        for i in range(len(embeddings_batch)):

            time_before = time.monotonic()

            res = index.search(np.array([embeddings_batch[i]]).astype('float32'),int(K.max())+1) 

            search_time.append(time.monotonic()-time_before)

            res = np.moveaxis(res,1,0)

            query_id = str(external_id_batch[i])

            nearest_neighbours[query_id] = {}

            nearest_neighbours[query_id]["ids"] = res[0][1]
            nearest_neighbours[query_id]["distances"] = res[0][0]

        
    return nearest_neighbours, np.mean(search_time)


def main(
    exact: bool,
    embeddings_path: pathlib.Path, 
    save_dir: str,
    batch_size: int,
    d: int,
    queries: int,
    index_size: int,
    K: np.array,
    M_list: np.array,
    EF_CONSTRUCTION_list: np.array,
    EF_SEARCH_list: np.array,
    ):
    """
    Entry point of the script. Builds an index with embeddings and performs k-nearest neighbor search on a set of queries.
    
    Args:
        exact: A boolean indicating whether to create an exact index or an approximate index.
        embeddings_path: The path to the file containing the embeddings.
        save_dir: The directory to save the performance results and nearest neighbor results.
        batch_size: The number of embeddings/queries to process at a time.
        d: The dimensionality of the embeddings.
        queries: The number of queries to perform.
        index_size: The number of embeddings to index.
        K: The number of nearest neighbors to retrieve for each query.
        M_list: A list of the number of connections that each node in the index has to its neighbors.
        EF_CONSTRUCTION_list: A list of the maximum number of elements to visit when adding an element to the graph.
        EF_SEARCH_list: A list of the maximum number of elements to visit when searching the graph.

    Returns:
        A dictionnary of the characteristics and performances of all indexes studied.
    """

    performances = {}

    performances["index_size"] = index_size
    performances["queries"] = queries

    try:
        os.mkdir(save_dir)
        print("Save directory created !")
    except:
        print("Save directory already existing !")

    perf_file = save_dir + "/exact_perf.json"
    KNN_file = save_dir + "/exact_KNN.json"

    if pathlib.Path(KNN_file).is_file() and pathlib.Path(perf_file).is_file():
        performances["ground_truth"] = load_data(perf_file)
        ground_truth = load_data(KNN_file)
    else:
        values = psutil.virtual_memory()
        ram_available_before = values.available

        performances["ground_truth"] = {}

        print("Creation of FLAT index")

        index, performances["ground_truth"]["creation_time"] = create_index(d, True)

        performances["ground_truth"]["building_time"] = build_index(index, embeddings_path, batch_size, index_size)

        values = psutil.virtual_memory()
        ram_available_after = values.available

        performances["ground_truth"]["ram_used"] = (ram_available_before - ram_available_after)/(1024.0 ** 3)

        print("Search in FLAT index to compute ground truth")

        ground_truth, performances["ground_truth"]["search_time"] = search_index(index, embeddings_path, queries, batch_size, K)

        save_data(KNN_file, ground_truth)

        save_data(perf_file, performances["ground_truth"])

    performances["hnsw"] = {}
    index_number = 0

    for m in M_list :
        m = int(m)
        for ef_construction in EF_CONSTRUCTION_list :
            ef_construction = int(ef_construction)
            for ef_search in EF_SEARCH_list:
                ef_search = int(ef_search)

                print(f"****** {index_number}th HNSW index ******")

                perf_file = save_dir + "/hnsw_perf_" + str(m) + "_" + str(ef_construction) + "_" + str(ef_search) + ".json"
                KNN_file = save_dir + "/hnsw_ANN_" + str(m) + "_" + str(ef_construction) + "_" + str(ef_search) + ".json"

                if pathlib.Path(perf_file).is_file() and pathlib.Path(KNN_file).is_file():
                    hnsw_perf = load_data(perf_file)
                    ground_truth = load_data(KNN_file)

                else :
                    hnsw_perf = {}

                    hnsw_perf["M"] = m
                    hnsw_perf["ef_construction"] = ef_construction
                    hnsw_perf["ef_search"] = ef_search

                    values = psutil.virtual_memory()
                    ram_available_before = values.available

                    print(f"Creation of the {index_number}th jnsw index")

                    index, hnsw_perf["creation_time"] = create_index(
                                                                                    d,
                                                                                    False,
                                                                                    m,
                                                                                    ef_search,
                                                                                    ef_construction,
                                                                                    )

                    hnsw_perf["building_time"] = build_index(index, embeddings_path, batch_size, index_size)

                    values = psutil.virtual_memory()
                    ram_available_after = values.available

                    hnsw_perf["ram_used"] = (ram_available_before - ram_available_after)/(1024.0 ** 3)

                    print(f"Search in the {index_number}th hsnw index")

                    hnsw_nearest_neighbours, hnsw_perf["search_time"] = search_index(index, embeddings_path, queries, batch_size, K)

                    hnsw_perf["metrics"] = compute_recall(
                        ground_truth,
                        hnsw_nearest_neighbours,
                        index_size,
                        queries,
                        K,
                        m,
                        ef_construction,
                        ef_search,
                        )

                    save_data(perf_file, hnsw_perf)
                    save_data(KNN_file, hnsw_nearest_neighbours)
                index_number = index_number + 1
                performances["hnsw"][index_number] = hnsw_perf

    return performances


embeddings_path=pathlib.Path("logos_embeddings_512.hdf5")
batch_size = 512
d = 512  # dimension of the embeddings of the embeddings_path file
index_size = 1000
queries = 10  # number of queries
K = np.array([1,4,10,50,100])

M_array=np.array([5,30])
efConstruction_array=np.array([16])
efSearch_array=np.array([40])


save_dir = "faiss_saves_index_" + str(index_size) + "_queries_" + str(queries)

complete_perf = main(True,
                    embeddings_path,
                    save_dir,
                    batch_size, 
                    d,
                    queries, 
                    index_size, 
                    K, 
                    M_array, 
                    efConstruction_array, 
                    efSearch_array)

with open(save_dir+"/faiss_complete_perf.json",'w') as f:
    json.dump(complete_perf,f)
