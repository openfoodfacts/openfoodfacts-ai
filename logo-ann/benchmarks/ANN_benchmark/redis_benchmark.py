import pathlib
import time
import tqdm
import h5py
from more_itertools import chunked
import numpy as np
import psutil
import json

from redis import Redis
from redis.commands.search.field import VectorField, TagField, NumericField
from redis.commands.search.query import Query
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


def create_index(
    client : Redis,
    embedding_field_name : str,
    external_id_field_name : str,
    dim : int,
    exact : bool,
    M : int="16",
    EF_CONSTRUCTION : int="200",
    EF_RUNTIME : int="10",
    ):
    """
    Creates an index on the given redis database, with a given schema.

    Parameters:
    ----------
    client: redis.client.Redis
        A Redis client object.
    embedding_field_name: str
        The name of the field containing the embeddings.
    external_id_field_name: str
        The name of the field containing the external ids.
    dim: int
        The dimension of the embeddings.
    exact: bool
        Whether to use exact search or approximate search.
    M: int
        The HNSW parameter M, only used if exact=False.
    EF_CONSTRUCTION: int
        The HNSW parameter EF_CONSTRUCTION, only used if exact=False.
    EF_RUNTIME: int
        The HNSW parameter EF_RUNTIME, only used if exact=False.

    Returns:
    ----------
    float: 
        The time taken to create the index
    """

    client.flushall()

    start_time = time.monotonic()
    if exact : schema = (VectorField(embedding_field_name, "FLAT", {"TYPE": "FLOAT32", 
                                                                    "DIM": dim, 
                                                                    "DISTANCE_METRIC": 
                                                                    "COSINE"}),
                        NumericField(external_id_field_name))

    else : schema = (VectorField(embedding_field_name, "HNSW", {"TYPE": "FLOAT32", 
                                                                "DIM": dim, 
                                                                "DISTANCE_METRIC": "COSINE",
                                                                "M":M, 
                                                                "EF_CONSTRUCTION": EF_CONSTRUCTION, 
                                                                "EF_RUNTIME": EF_RUNTIME}),
                    NumericField(external_id_field_name))

    

    client.ft().create_index(schema)
    client.ft().config_set("default_dialect", 2)

    return time.monotonic() - start_time 


def build_index(
    client : Redis, 
    embedding_field_name : str, 
    external_id_field_name : str, 
    n_vec : int, 
    batch_size : int
    ):
    """
    Builds a Redis database index using a given HDF5 file containing embeddings.
    The database will contain two fields, one for the embedding and one for the external ID.
    
    Parameters
    ----------
    client : Redis
        A Redis client object.
    embedding_field_name : str
        The name of the embeddings field in the database.
    external_id_field_name : str
        The name of the external IDs field in the database.
    n_vec : int
        The dimension of the embeddings.
    batch_size : int
        The number of embeddings in a batch.
        
    Returns
    -------
    float
        The average time of insertion.
    """

    data_gen =  get_embedding(pathlib.Path("logos_embeddings.hdf5"), batch_size, n_vec)
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
    embeddings_path: pathlib.Path, 
    queries: int, 
    batch_size: int,
    embedding_field_name : str,
    K: np.array,
    ):
    """
    Searching nearest neighbours in the index using a list of query vectors from a given path.

    Parameters
    ----------
    client : Redis
        an instance of Redis Client
    embeddings_path : pathlib.Path
        path to the embeddings
    queries : int
        number of queries to be executed 
    batch_size : int
        size of the batch of retrieved embeddings
    K : np.array
        maximum number of nearest neighbours

    Returns
    -------
    nearest_neighbours : dict
        dictionary of nearest neighbours for each query
    mean_search_time : float
        mean search time in ms
    """

    search_time = []

    nearest_neighbours = {}

    data_gen = get_embedding(embeddings_path, batch_size, queries)

    for batch in tqdm.tqdm(data_gen):
        (embeddings_batch, external_id_batch) = batch

        for i in range(len(embeddings_batch)):
            query_vector = embeddings_batch[i]
            query_id = external_id_batch[i]
            q = Query(f'*=>[KNN $k @{embedding_field_name} $vec_param AS dist]').paging(0,101).sort_by(f'dist')
            start_time = time.monotonic()
            res = client.ft().search(q, query_params = {'k': int(K.max()+1),'vec_param': query_vector.tobytes()})
            search_time.append(time.monotonic() - start_time)

            nearest_neighbours[query_id] = {}

            nearest_neighbours[query_id]["ids"] = np.array([int(doc.external_id) for doc in res.docs])
            nearest_neighbours[query_id]["distances"] = np.array([float(doc.dist) for doc in res.docs])
            

    return nearest_neighbours, np.mean(search_time)


def main(
    embeddings_path: pathlib.Path, 
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
    '''
    Perform the redis benchmark by creating, building and searching the various redis indexes.
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

    performances["ground_truth"]["creation_time"] = create_index(client, embedding_field_name, external_id_field_name, d, exact = True)
        
    performances["ground_truth"]["building_time"] = build_index(client, embedding_field_name, external_id_field_name, index_size, batch_size)

    values = psutil.virtual_memory()
    ram_available_after = values.available

    performances["ground_truth"]["ram_used"] = ram_available_before - ram_available_after

    print("Search in FLAT index to compute ground truth")

    ground_truth, performances["ground_truth"]["search_time"] = search_index(client, embeddings_path, queries, batch_size, embedding_field_name, K)
    
    performances["hnsw"] = {}
    index_number = 0

    for m in M_list :
        m = int(m)
        for ef_construction in EF_CONSTRUCTION_list :
            ef_construction = int(ef_construction)
            for ef_runtime in EF_RUNTIME_list:
                ef_runtime = int(ef_runtime)

                print(f"****** {index_number}th HNSW index ******")

                performances["hnsw"][index_number] = {}

                performances["hnsw"][index_number]["M"] = m
                performances["hnsw"][index_number]["ef_construction"] = ef_construction
                performances["hnsw"][index_number]["ef_runtime"] = ef_runtime

                values = psutil.virtual_memory()
                ram_available_before = values.available

                print(f"Creation of the {index_number}th jnsw index")

                performances["hnsw"][index_number]["creation_time"] = create_index(client, 
                                                                                embedding_field_name, 
                                                                                external_id_field_name, 
                                                                                d, 
                                                                                False,
                                                                                m,
                                                                                ef_construction,
                                                                                ef_runtime,
                                                                                )

                performances["hnsw"][index_number]["building_time"] = build_index(client, embedding_field_name, external_id_field_name, index_size, batch_size)

                values = psutil.virtual_memory()
                ram_available_after = values.available

                performances["hnsw"][index_number]["ram_used"] = ram_available_before - ram_available_after

                print(f"Search in the {index_number}th hsnw index")

                hnsw_nearest_neighbours, performances["hnsw"][index_number]["search_time"] = search_index(client, embeddings_path, queries, batch_size, embedding_field_name, K)

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


host = "localhost"
port = 6379

redis_conn = Redis(host = host, port = port)

embeddings_path=pathlib.Path("logos_embeddings.hdf5")
batch_size = 512
d = 768
index_size = 10000000
queries = 10000
K = np.array([1,4,10,50,100])

M_array=np.array([2,4,8])
efRuntime_array=np.array([32,64,128])
efConstruction_array=np.array([64,128,256])

embedding_field_name = "embedding"
external_id_field_name = "external_id"

complete_perf = main(embeddings_path, 
                    batch_size, 
                    redis_conn, 
                    d,
                    embedding_field_name, 
                    external_id_field_name, 
                    queries, index_size, 
                    K, 
                    M_array, 
                    efConstruction_array, 
                    efRuntime_array)

with open("redis_perf.json",'w') as f:
    json.dump(complete_perf,f)

