import pathlib
import time
import tqdm
import h5py
from more_itertools import chunked
import numpy as np
import psutil
import json

import faiss
import tqdm

def get_embedding(embeddings_path: pathlib.Path, batch_size: int, nb_embeddings: int):
    """
    Get embeddings from an embeddings file.

    Parameters
    ----------
    embeddings_path : pathlib.Path
        Path to the embeddings file.
    batch_size : int
        Number of embeddings to return at once.
    nb_embeddings : int
        Maximum number of embeddings to return.

    Yields
    ------
    embeddings : numpy.ndarray
        Array of embeddings.
    external_ids : numpy.ndarray
        Array of external ids.
    """

    with h5py.File(str(embeddings_path), "r") as f:
        embedding_dset = f["embedding"]
        external_id_dset = f["external_id"]

        for slicing in chunked(range(min(len(embedding_dset), nb_embeddings)), batch_size):
            slicing = np.array(slicing)
            mask = external_id_dset[slicing] == 0

            if np.all(mask):
                break

            mask = ~mask
            yield (
                embedding_dset[slicing][mask],
                external_id_dset[slicing][mask],
            )

def create_index(d:int, exact : bool, m: int=32, efSearch: int=40, efConstruction: int=40):
    """
    Create Index
    -----------

    Create an index using Faiss library based on the input parameters.

    Parameters
    ----------
    d : int
        Number of dimensions of the input vectors.
    exact : bool
        Flag to indicate if the index is exact (True) or approximate (False).
    m : int, optional
        Number of link between vectors stored in HNSW indexes (default 32).
    efSearch : int, optional
        Controls the quality of the search (default 40).
    efConstruction : int, optional
        Controls the quality of the index construction (default 40).

    Returns
    -------
    index : faiss.Index
        Index object.
    create_index_time : int
        Time taken to create the index.
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


def build_index(index: faiss.swigfaiss.IndexIDMap, batch_size : int, n_vec : int):
    """
    Build an index from a hdf5 file containing embeddings and external ids.

    Parameters
    ----------
    index : faiss.swigfaiss.IndexIDMap
        The index to build
    batch_size : int
        The batch size to use when reading the hdf5 file
    n_vec : int
        The number of vectors to read from the hdf5 file

    Returns
    -------
    building_time : float
        The time it took to build the index
    """

    building_time = 0

    data_gen =  get_embedding(pathlib.Path("logos_embeddings.hdf5"), batch_size, n_vec)

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
    Search index with query vectors issued from embedding_path. 

    Parameters
    ----------
    index : faiss.swigfaiss.IndexIDMap
        index to search with
    embeddings_path : pathlib.Path
        path to the file containg the embeddings
    queries : int
        number of queries to perform
    batch_size : int
        number of embedding to search in one batch
    K : np.array
        array of parameters k for which to perform the search


    Returns
    -------
    nearest_neighbours : dict
        dictionary containing distances and ids of the nearest neighbours for each query
    mean_search_time : float
        mean time in seconds for the nearest neighbours search of a single embedding
    """

    search_time = []

    nearest_neighbours = {}

    data_gen = get_embedding(embeddings_path, batch_size, queries)

    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch

        for i in range(len(embeddings_batch)):
            query_vector = embeddings_batch[i]
            query_id = external_id_batch[i]
            
            start_time = time.monotonic()
            res = index.search(np.array([query_vector]).astype('float32'),int(K.max())+1)
            search_time.append(time.monotonic() - start_time)

            res = np.moveaxis(res,1,0)

            nearest_neighbours[query_id] = {}

            nearest_neighbours[query_id]["ids"] = res[0][1]
            nearest_neighbours[query_id]["distances"] = res[0][0]
            

    return nearest_neighbours, np.mean(search_time)


def main(
    embeddings_path: pathlib.Path, 
    batch_size: int,
    d: int,
    queries: int,
    index_size: int,
    K: np.array,
    M_list: np.array,
    EF_CONSTRUCTION_list: np.array,
    EF_SEARCH_list: np.array,
    ):
    '''
    Perform the faiss benchmark by creating, building and searching the various faiss indexes.
    The first one is flat in order to compute the ground truth.
    The other ones are HNSW indexes created with various parameters.

    Parameters
    ----------

    embeddings_path: pathlib.Path
        path to the embeddings file
    batch_size: int
        size of the batch in the index building (number of vectors read from the embeddings file)
    d: int
        dimension of the embeddings
    queries: int
        number of vectors for which we search in the index
    index_size: int
        maximum number of embeddings in the index
    K: np.array
        list of ks for which the evaluation has to be made
    M: np.array
        list of M values for the HNSW index
    ef_construction_list: np.array
        list of ef_construction values
    ef_search_list: np.array
        list of ef_search values

    Returns
    -------

    performances: dict
        performances["index_size"] : int
            the size of the index
        performances["queries"] : int
            the number of queries

        performances[...]["building_time"] : float
            time used in the construction of the ... index
        performances[...]["search_time"] : float
            time used in the search in the ... index
        performances[...]["ram_used"] : float
            memory used by the ... index
        performances["hnsw"][i]["M"] : int
            value of M used for the i-th HNSW index
        performances["hnsw"][i]["ef_construction"] : int
            value of ef_construction used for the i-th HNSW index
        performances["hnsw"][i]["ef_runtime"] : int
            value of ef_runtime used for the i-th HNSW index
        performances["hnsw"][i]["macro_recall"][k] : float
            macro-averaged recall@k for the i-th HNSW index
        performances["hnsw"][i]["macro_precision"][k] : float
            macro-averaged precision@k for the i-th HNSW index
        performances["hnsw"][i]["micro_recall"][k] : float
            micro-averaged recall@k for the i-th HNSW index
        performances["hnsw"][i]["micro_precision"][k] : float
            micro-averaged precision@k for the i-th HNSW index
    '''

    performances = {}

    performances["index_size"] = index_size
    performances["queries"] = queries

    values = psutil.virtual_memory()
    ram_available_before = values.available

    performances["ground_truth"] = {}

    print("Creation of FLAT index")

    index, performances["ground_truth"]["creation_time"] = create_index(d, True)
        
    performances["ground_truth"]["building_time"] = build_index(index, batch_size, index_size)

    values = psutil.virtual_memory()
    ram_available_after = values.available

    performances["ground_truth"]["ram_used"] = ram_available_before - ram_available_after

    print("Search in FLAT index to compute ground truth")

    ground_truth, performances["ground_truth"]["search_time"] = search_index(index, embeddings_path, queries, batch_size, K)
    
    performances["hnsw"] = {}
    index_number = 0

    for m in M_list :
        m = int(m)
        for ef_construction in EF_CONSTRUCTION_list :
            ef_construction = int(ef_construction)
            for ef_search in EF_SEARCH_list:
                ef_search = int(ef_search)

                print(f"****** {index_number}th HNSW index ******")

                performances["hnsw"][index_number] = {}

                performances["hnsw"][index_number]["M"] = m
                performances["hnsw"][index_number]["ef_construction"] = ef_construction
                performances["hnsw"][index_number]["ef_runtime"] = ef_search

                values = psutil.virtual_memory()
                ram_available_before = values.available

                print(f"Creation of the {index_number}th jnsw index")

                index, performances["hnsw"][index_number]["creation_time"] = create_index(
                                                                                d,
                                                                                False,
                                                                                m,
                                                                                ef_search,
                                                                                ef_construction,
                                                                                )

                performances["hnsw"][index_number]["building_time"] = build_index(index, batch_size, index_size)

                values = psutil.virtual_memory()
                ram_available_after = values.available

                performances["hnsw"][index_number]["ram_used"] = ram_available_before - ram_available_after

                print(f"Search in the {index_number}th hsnw index")

                hnsw_nearest_neighbours, performances["hnsw"][index_number]["search_time"] = search_index(index, embeddings_path, queries, batch_size, K)

                tp_micro = {}
                fp_micro = {}
                fn_micro = {}

                performances["hnsw"][index_number]["macro_recall"] = {}
                performances["hnsw"][index_number]["macro_precision"] = {}
                performances["hnsw"][index_number]["micro_recall"] = {}
                performances["hnsw"][index_number]["micro_precision"] = {}

                for k in K:
                    k = int(k)
                    tp_micro[k] = 0
                    fp_micro[k] = 0
                    fn_micro[k] = 0
                    performances["hnsw"][index_number]["macro_recall"][k] = 0
                    performances["hnsw"][index_number]["macro_precision"][k] = 0
                    performances["hnsw"][index_number]["micro_recall"][k] = 0
                    performances["hnsw"][index_number]["micro_precision"][k] = 0
                
                count = 0
                for id in ground_truth.keys():
                    for k in K :
                        k = int(k)
                        positive_neighbours = np.isin(hnsw_nearest_neighbours[id]["ids"][1:k+1],ground_truth[id]["ids"][1:k+1])

                        tp = np.sum(positive_neighbours.astype(int))
                        tp_micro[k] = tp_micro[k] + tp

                        fp = k - tp
                        fp_micro[k] = fp_micro[k] + fp

                        found_neighbours_among_the_expected_ones = np.isin(ground_truth[id]["ids"][1:k+1],hnsw_nearest_neighbours[id]["ids"][1:k+1])
                        fn = k - (np.sum(found_neighbours_among_the_expected_ones.astype(int)))
                        fn_micro[k]  = fn_micro[k] + fn

                        performances["hnsw"][index_number]["macro_recall"][k] = performances["hnsw"][index_number]["macro_recall"][k] + tp/(tp+fn)
                        performances["hnsw"][index_number]["macro_precision"][k] = performances["hnsw"][index_number]["macro_precision"][k] + tp/(tp+fp)
                        

                    count = count + 1

                assert queries == count
                
                for k in K:
                    k = int(k)
                    performances["hnsw"][index_number]["micro_recall"][k] = tp_micro[k]/(tp_micro[k]+fn_micro[k])
                    performances["hnsw"][index_number]["micro_precision"][k] = tp_micro[k]/(tp_micro[k]+fp_micro[k])
                    performances["hnsw"][index_number]["macro_recall"][k] = performances["hnsw"][index_number]["macro_recall"][k]/count
                    performances["hnsw"][index_number]["macro_precision"][k] = performances["hnsw"][index_number]["macro_precision"][k]/count

                index_number = index_number + 1

    return performances


embeddings_path=pathlib.Path("logos_embeddings.hdf5")
batch_size = 512
d = 768  # dimension of the embeddings of the embeddings_path file
index_size = 100000
queries = 1000
K = np.array([1,4,10,50,100])

M_array=np.array([64])
efSearch_array=np.array([512])
efConstruction_array=np.array([1024])

complete_perf = main(embeddings_path, 
                    batch_size, 
                    d,
                    queries, 
                    index_size, 
                    K, 
                    M_array, 
                    efConstruction_array, 
                    efSearch_array)

with open("hnsw_perf.json",'w') as f:
    json.dump(complete_perf,f)

