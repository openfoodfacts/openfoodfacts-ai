import pathlib
import time
import tqdm
from more_itertools import chunked
import numpy as np
import psutil
import json
import os
import tqdm
import warnings
from sklearn.preprocessing import normalize

from elasticsearch import Elasticsearch

from utils import get_embedding, save_data, load_data
from recall_computation import compute_recall


def create_index(es: Elasticsearch, index_name: str, d:int, ef_construction: int, m: int):
  """
  Create an index for the Elasticsearch server and set the mappings for the embedding field.
  
  Parameters:
  es (Elasticsearch): an instance of Elasticsearch class
  index_name (str): the name of the index to create
  d (int): the size of the embeddings
  ef_construction (int): the maximum number of elements to consider during the construction of the index
  m (int): the number of connections that each node in the index has to its neighbors
  
  Returns:
  float: the time taken to create the index
  """

  start_time = time.monotonic()

  mappings = {
          "properties": {
              "external_id": {"type": "integer"},
              "embedding": {
                "type": "dense_vector", 
                "dims": d, 
                "index": True, 
                "similarity": "dot_product", 
                "index_options": {
                  "type": "hnsw",
                  "m": m,
                  "ef_construction": ef_construction,
                  },
                },
      },
  }

  try :
      es.indices.delete(index=index_name)
  except : 
      print("No index existing yet. Let's create one !")

  es.indices.create(index=index_name, mappings=mappings)

  return time.monotonic() - start_time 


def build_index(es: Elasticsearch, index_name: str, embeddings_path: pathlib.Path, d: int, batch_size : int, n_vec : int):
  """
  Builds an Elasticsearch index from a set of embeddings

  Parameters:
  es (Elasticsearch): Elasticsearch object to build index on
  index_name (str): name of index to be created
  embeddings_path (pathlib.Path): path to the file containing embeddings to be added to the index
  d (int): dimension of each embedding
  batch_size (int): number of embeddings to be added to the index at a time
  n_vec (int): total number of embeddings to be added to the index

  Returns:
  building_time (int): time taken to build the index
  """

  building_time = 0

  data_gen = get_embedding(embeddings_path, batch_size, n_vec)

  unique_id = 0

  for batch in tqdm.tqdm(data_gen):
    (embeddings_batch, external_id_batch) = batch
    
    for i in range(len(embeddings_batch)):
      unique_id +=1
      elapsed = time.monotonic()
      vector = embeddings_batch[i].reshape(1,d)
      vector = normalize(vector)
      vector = vector.reshape(d)

      external_id = int(external_id_batch[i])

      doc = {
          "embedding": vector,
      }
      es.index(index=index_name, id=external_id, document=doc)
      building_time = building_time + time.monotonic() - elapsed

  return building_time


def search_index(
  es: Elasticsearch,
  index_name: str, 
  embeddings_path: pathlib.Path, 
  queries: int, 
  batch_size: int,
  K: np.array,
  num_candidates: int,
  ):
  """
  Searches the Elasticsearch index and returns the nearest neighbours of a set of query vectors

  Parameters:
  es (Elasticsearch): Elasticsearch object to search on
  index_name (str): name of the index to search on
  embeddings_path (pathlib.Path): path to the file containing query vectors
  queries (int): number of query vectors
  batch_size (int): number of query vectors to search at a time
  K (np.array): array of top K nearest neighbours to retrieve
  num_candidates (int): number of candidates to be considered during search

  Returns:
  nearest_neighbours (dict): dictionary containing the nearest neighbours for each query vector
  search_time (float): average time taken to search for each query vector
  """

  search_time = []

  nearest_neighbours = {}

  data_gen = get_embedding(embeddings_path, batch_size, queries)

  for batch in tqdm.tqdm(data_gen):
    (embeddings_batch, external_id_batch) = batch
        
    for i in range(len(embeddings_batch)):
      time_before = time.monotonic()
      vector = embeddings_batch[i].reshape(1,d)
      vector = normalize(vector)
      vector = vector.reshape(d)

      request = {
        "knn": {
          "field": "embedding",
          "query_vector": vector,
          "k": int(K.max())+1,
          "num_candidates": num_candidates
        },
      }
      res = es.knn_search(index = index_name, body=request)
      search_time.append(time.monotonic()-time_before)

      query_id = str(external_id_batch[i])

      nearest_neighbours[query_id] = {}

      hits = res['hits']['hits'] 

      nearest_neighbours[query_id]["ids"] = np.array([hits[i]['_source']['external_id'] for i in range(len(hits))])
      nearest_neighbours[query_id]["score"] = np.array([hits[i]['_score'] for i in range(len(hits))])

      
  return nearest_neighbours, np.mean(search_time)


def main(
  es: Elasticsearch,
  index_name: str,
  embeddings_path: pathlib.Path, 
  ground_truth_path: pathlib.Path,
  save_dir: str,
  batch_size: int,
  d: int,
  queries: int,
  index_size: int,
  K: np.array,
  NUM_list: np.array,
  EF_CONSTRUCTION_list: np.array,
  M_list: np.array,
  ):
  """
  Builds and searches an Elasticsearch index to compute the performances of ANN indexes

  Parameters:
  es (Elasticsearch): Elasticsearch object to build and search on
  index_name (str): name of the index to be created and searched on
  embeddings_path (pathlib.Path): path to the file containing embeddings to be added to the index and query vectors to search
  ground_truth_path (pathlib.Path): path to the file containing ground truth nearest neighbours for evaluation
  save_dir (str): directory to save results
  batch_size (int): number of embeddings to be added to the index and query vectors to search at a time
  d (int): dimension of each embedding
  queries (int): number of query vectors
  index_size (int): number of embeddings to be added to the index
  K (np.array): array of top K nearest neighbours to retrieve
  NUM_list (np.array): array of num_candidates values to be considered during search
  EF_CONSTRUCTION_list (np.array): array of ef_construction values to be considered during search
  M_list (np.array): array of M values to be considered during search

  Returns:
  performances (dict): dictionary containing the search performances for various parameter combinations
  """

  performances = {}

  performances["index_size"] = index_size
  performances["queries"] = queries

  try:
    os.mkdir(save_dir)
    print("Save directory created !")
  except:
    print("Save directory already existing !")

  ground_truth = None
  if ground_truth_path.isfile():
    ground_truth = load_data(KNN_file)

  performances["hnsw"] = {}
  index_number = 0

  for num_candidates in NUM_list:
    num_candidates = int(num_candidates)
    for m in M_list:
      m = int(m)
      for ef_construction in EF_CONSTRUCTION_list:
        ef_construction = int(ef_construction)


        print(f"****** {index_number}th HNSW index ******")

        perf_file = save_dir + "/hnsw_perf_ncandidates_" + str(num_candidates) + "_m_" + str(m) + "_efconstruction_" + str(ef_construction) + ".json"
        KNN_file = save_dir + "/hnsw_ANN_ncandidates_" + str(num_candidates) + "_m_" + str(m) + "_efconstruction_" + str(ef_construction) + ".json"

        if pathlib.Path(perf_file).is_file() and pathlib.Path(KNN_file).is_file():
          hnsw_perf = load_data(perf_file)
          hnsw_nearest_neighbours = load_data(KNN_file)

        else :
          hnsw_perf = {}

          hnsw_perf["num_candidates"] = num_candidates

          print(f"Creation of the {index_number}th hnsw index")

          hnsw_perf["creation_time"] = create_index(es, index_name, d, ef_construction, m)

          ram_available_before = psutil.virtual_memory().available

          hnsw_perf["building_time"] = build_index(es, index_name, embeddings_path, d, batch_size, index_size)

          hnsw_perf["ram_used"] = (ram_available_before-psutil.virtual_memory().available)/(1024.0**3)
          

          print(f"Search in the {index_number}th hsnw index")

          hnsw_nearest_neighbours, hnsw_perf["search_time"] = search_index(es, index_name, embeddings_path, queries, batch_size, K, num_candidates)

          hnsw_perf["metrics"] = compute_recall(
            ground_truth,
            hnsw_nearest_neighbours,
            index_size,
            queries,
            K,
            num_candidates,
            )

          save_data(perf_file, performances["hnsw"][index_number])
          save_data(KNN_file, hnsw_nearest_neighbours)
        index_number = index_number + 1


  return performances


if __name__ == '__main__':

  warnings.filterwarnings("ignore")

  es = Elasticsearch(hosts='http://elastic:elastic_pswd@localhost:9200', verify_certs=False)
  es.info().body

  index_name = "logos"

  ground_truth_path = pathlib.Path("faiss_saves_index_1000_queries_10/exact_KNN.json")

  embeddings_path=pathlib.Path("logos_embeddings_512.hdf5")
  batch_size = 512
  d = 512  # dimension of the embeddings of the embeddings_path file
  index_size = 1000
  queries = 10
  K = np.array([1,5,10,50,100])

  NUM_array = np.array([110])
  EF_CONSTRUCTION_list = np.array([100])
  M_list = np.array([8])


  save_dir = "elastic_index_" + str(index_size) + "_queries_" + str(queries)

  complete_perf = main(
    es,
    index_name,
    embeddings_path,
    ground_truth_path,
    save_dir,
    batch_size, 
    d,
    queries, 
    index_size, 
    K, 
    NUM_array,
    EF_CONSTRUCTION_list,
    M_list,
    )

  with open(save_dir+"/elastic_complete_perf.json",'w') as f:
      json.dump(complete_perf,f)
