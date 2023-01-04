import pathlib
import time
import tqdm
import h5py
from more_itertools import chunked
import numpy as np
import psutil
import json
import os
import tqdm
from utils import get_embedding, save_data, load_data
from recall_computation import compute_recall

from redis import Redis
from redis.commands.search.field import VectorField, TagField, NumericField
from redis.commands.search.query import Query



def create_index(
    client : Redis,
    embedding_field_name : str,
    external_id_field_name : str,
    dim : int,
    exact : bool,
    M : int=16,
    EF_CONSTRUCTION : int=200,
    EF_RUNTIME : int=10,
    ):
    """
    Creates an index with either a flat or HNSW structure.
    
    Args:
        client: The redis client object.
        embedding_field_name: The name of the field containing the embeddings.
        external_id_field_name: The name of the field containing the external ids.
        dim: The dimensionality of the embeddings.
        exact: A boolean indicating whether to create an exact index or an approximate index.
        M: The number of connections that each node in the index has to its neighbors.
        EF_CONSTRUCTION: The maximum number of elements to visit when adding an element to the graph.
        EF_RUNTIME: The maximum number of elements to visit when searching the graph.
    
    Returns:
        The time it took to create the index.
    """

    try:
        client.ft("Flat").dropindex()
    except:
        print("No Flat Index to drop")

    try:
        client.ft("HNSW").dropindex()
    except:
        print("No HNSW Index to drop")

    percent = '0' # percentage of index that has been built
    count = 0
    if os.path.exists("percentage_study.txt"): os.remove("percentage_study.txt")

    start_time = time.monotonic()
    if exact : 
        schema = (VectorField(embedding_field_name, "FLAT", {"TYPE": "FLOAT32", 
                                                                    "DIM": dim, 
                                                                    "DISTANCE_METRIC": "COSINE"}),
                        NumericField(external_id_field_name))
        client.ft("Flat").create_index(schema)
        client.ft("Flat").config_set("default_dialect", 2)

        while percent != '1':
            time.sleep(1)
            count = count+1
            percent = client.ft("Flat").info()["percent_indexed"]
            with open("percentage_study.txt","a") as f:
                f.write(f"Percentage of Flat indexed {percent} at time {count*20} sec \n")

    else : 
        schema = (VectorField(embedding_field_name, "HNSW", {"TYPE": "FLOAT32", 
                                                                "DIM": dim, 
                                                                "DISTANCE_METRIC": "COSINE",
                                                                "M":M, 
                                                                "EF_CONSTRUCTION": EF_CONSTRUCTION, 
                                                                "EF_RUNTIME": EF_RUNTIME}),
                    NumericField(external_id_field_name))
        client.ft("HNSW").create_index(schema)
        client.ft("HNSW").config_set("default_dialect", 2)

        while percent != '1':
            time.sleep(1)
            count = count+1
            percent = client.ft("HNSW").info()["percent_indexed"]
            with open("percentage_study.txt","a") as f:
                f.write(f"Percentage of HNSW indexed {percent} at time {count*240} sec \n")

    return time.monotonic() - start_time 


def build_index(
    client : Redis, 
    embedding_path: pathlib.Path,
    embedding_field_name : str, 
    external_id_field_name : str, 
    n_vec : int, 
    batch_size : int,
    ):
    """
    Builds the index by adding the embeddings to the client.
    
    Args:
        client: The redis client object.
        embedding_path: The path to the file containing the embeddings.
        embedding_field_name: The name of the field containing the embeddings.
        external_id_field_name: The name of the field containing the external ids.
        n_vec: The number of embeddings to index.
        batch_size: The number of embeddings to add to the index at a time.
    
    Returns:
        The time it took to build the index.
    """  

    data_gen =  get_embedding(embedding_path, batch_size, n_vec)
    offset = 0
    search_time = []
    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch

        for i in range(len(embeddings_batch)):
            start_time = time.monotonic()
            client.hset(i+offset, mapping = {embedding_field_name: embeddings_batch[i].astype('float32').tobytes(),
                                    external_id_field_name: int(external_id_batch[i])}) 
            search_time.append((time.monotonic() - start_time))
        offset = offset + len(embeddings_batch)
    return np.mean(search_time)


def search_index(
    client: Redis,
    exact: bool, 
    embeddings_path: pathlib.Path, 
    queries: int, 
    batch_size: int,
    embedding_field_name : str,
    K: np.array,
    ):
    """
    Searches the index for the given embeddings and returns the nearest neighbors and search times.
    
    Args:
        client: The redis client object.
        exact: A flag indicating whether to use the exact or approximate index.
        embeddings_path: The path to the file containing the embeddings to search.
        queries: The number of queries to perform.
        batch_size: The size of each batch of queries.
        embedding_field_name: The name of the field containing the embeddings.
        K: The array of number of nearest neighbors to retrieve for each query.
        
    Returns:
        nearest_neighbors: A dictionary containing the nearest neighbors for each query.
        search_time: The mean search time across all queries.
    """

    search_time = []

    nearest_neighbours = {}

    data_gen = get_embedding(embeddings_path, batch_size, queries)

    if exact : idx_name = "Flat"
    else : idx_name = "HNSW"

    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch

        for i in range(len(embeddings_batch)):
            query_vector = embeddings_batch[i].astype('float32')
            query_id = external_id_batch[i]

            assert len(redis_conn.execute_command("FT._LIST")) == 1

            q = Query(f'*=>[KNN $k @{embedding_field_name} $vec_param AS dist]').paging(0,101).sort_by(f'dist')
            
            start_time = time.monotonic()
            res = client.ft(idx_name).search(q, query_params = {'k': int(K.max()+1),'vec_param': query_vector.tobytes()})
            search_time.append(time.monotonic() - start_time)

            query_id = str(query_id)
            nearest_neighbours[query_id] = {}

            nearest_neighbours[query_id]["ids"] = np.array([int(doc.external_id) for doc in res.docs])
            nearest_neighbours[query_id]["distances"] = np.array([float(doc.dist) for doc in res.docs])            

    return nearest_neighbours, np.mean(search_time)


def evaluate(
    embeddings_path: pathlib.Path, 
    save_dir: str,
    batch_size: int,
    client: Redis,
    d: int,
    embedding_field_name: str,
    external_id_field_name: str, 
    queries: int,
    index_size: int,
    K: np.array,
    M_list: np.array,
    EF_CONSTRUCTION_list: np.array,
    EF_RUNTIME_list: np.array,
    ):
    """
    Evaluate the performance of the search index.
    
    This function tests the performance of the search index by building
    and querying both a flat index and an HNSW index. The performance
    of each index is evaluated based on search time and the accuracy
    of the nearest neighbor search results. The results are saved to
    the specified directory.
    
    Args:
        embeddings_path (pathlib.Path): The path to the file containing the
            embeddings to index.
        save_dir (str): The directory to save the results in.
        batch_size (int): The number of embeddings to index at once.
        client (Redis): The Redis client to use for the index.
        d (int): The dimensionality of the embeddings.
        embedding_field_name (str): The name of the field in the index
            to store the embeddings in.
        external_id_field_name (str): The name of the field in the index
            to store the external IDs in.
        queries (int): The number of queries to perform.
        index_size (int): The number of documents in the index.
        K (np.array): An array of the top K nearest neighbors to retrieve
            for each query.
        M_list (np.array): An array of values for the M parameter to test
            in the HNSW index.
        EF_CONSTRUCTION_list (np.array): An array of values for the
            ef_construction parameter to test in the HNSW index.
        EF_RUNTIME_list (np.array): An array of values for the ef_runtime
            parameter to test in the HNSW index.

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

        performances["ground_truth"]["creation_time"] = create_index(client, embedding_field_name, external_id_field_name, d, exact = True)
        
        if client.dbsize() != index_size and client.dbsize() < 4317343:
            performances["ground_truth"]["building_time"] = build_index(client, embeddings_path, embedding_field_name, external_id_field_name, index_size, batch_size)

        values = psutil.virtual_memory()
        ram_available_after = values.available

        performances["ground_truth"]["ram_used"] = (ram_available_before - ram_available_after)/(1024.0 ** 3)

        print("Search in FLAT index to compute ground truth")

        ground_truth, performances["ground_truth"]["search_time"] = search_index(client, True, embeddings_path, queries, batch_size, embedding_field_name, K)
        
        save_data(KNN_file, ground_truth)

        save_data(perf_file, performances["ground_truth"])

    performances["hnsw"] = {}
    index_number = 0

    for m in M_list :
        m = int(m)
        for ef_construction in EF_CONSTRUCTION_list :
            ef_construction = int(ef_construction)
            for ef_runtime in EF_RUNTIME_list:
                ef_runtime = int(ef_runtime)

                print(f"****** {index_number}th HNSW index ******")

                perf_file = save_dir + "/hnsw_perf_" + str(m) + "_" + str(ef_construction) + "_" + str(ef_runtime) + ".json"
                KNN_file = save_dir + "/hnsw_ANN_" + str(m) + "_" + str(ef_construction) + "_" + str(ef_runtime) + ".json"

                if pathlib.Path(perf_file).is_file() and pathlib.Path(KNN_file).is_file():
                    hnsw_perf = load_data(perf_file)
                    ground_truth = load_data(KNN_file)

                else :
                    hnsw_perf = {}

                    hnsw_perf["M"] = m
                    hnsw_perf["ef_construction"] = ef_construction
                    hnsw_perf["ef_runtime"] = ef_runtime

                    print(f"Building of the {index_number}th jnsw index")

                    values = psutil.virtual_memory()
                    ram_available_before = values.available

                    if client.dbsize() != index_size and client.dbsize() < 4317343:
                        hnsw_perf["bulding_time"] = build_index(client, embeddings_path, embedding_field_name, external_id_field_name, index_size, batch_size)

                    values = psutil.virtual_memory()
                    ram_available_after = values.available

                    hnsw_perf["ram_used_for_db"] = (ram_available_before - ram_available_after)/(1024.0 ** 3)

                    values = psutil.virtual_memory()
                    ram_available_before = values.available

                    print(f"Creation of the {index_number}th jnsw index")

                    hnsw_perf["creation_time"] = create_index(
                        client, 
                        embedding_field_name, 
                        external_id_field_name, 
                        d, 
                        False,
                        m,
                        ef_construction,
                        ef_runtime,
                        )

                    values = psutil.virtual_memory()
                    ram_available_after = values.available

                    hnsw_perf["ram_used"] = (ram_available_before - ram_available_after)/(1024.0 ** 3)

                    print(f"Search in the {index_number}th hsnw index")

                    hnsw_nearest_neighbours, hnsw_perf["search_time"] = search_index(client, False, embeddings_path, queries, batch_size, embedding_field_name, K)

                    hnsw_perf["metrics"] = compute_recall(
                        ground_truth,
                        hnsw_nearest_neighbours,
                        index_size,
                        queries,
                        K,
                        m,
                        ef_construction,
                        ef_runtime,
                        )

                    save_data(perf_file, hnsw_perf)
                    save_data(KNN_file, hnsw_nearest_neighbours)
                index_number = index_number + 1
                performances["hnsw"][index_number] = hnsw_perf

    
    return performances


host = "localhost"
port = 6379

redis_conn = Redis(host = host, port = port)

embeddings_path=pathlib.Path("logos_embeddings_512.hdf5")
batch_size = 512
d = 512
index_size = 1000
queries = 10
K = np.array([1,4,10,50,100])

M_array=np.array([5,50])
efConstruction_array=np.array([256])
efRuntime_array=np.array([128])

embedding_field_name = "embedding"
external_id_field_name = "external_id"

save_dir = "redis_saves_index_" + str(index_size) + "_queries_" + str(queries)

complete_perf = evaluate(embeddings_path,
                    save_dir,
                    batch_size, 
                    redis_conn, 
                    d,
                    embedding_field_name, 
                    external_id_field_name, 
                    queries, 
                    index_size, 
                    K, 
                    M_array, 
                    efConstruction_array, 
                    efRuntime_array,
                    )

with open(save_dir+"/redis_complete_perf.json",'w') as f:
    json.dump(complete_perf,f)


