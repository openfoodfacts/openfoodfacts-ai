# Logo ANN (Approximate Nearest Neighbors)

Approximate Nearest Neighbors system for logo detection and similarity search using embeddings.

## Features

- Logo embedding generation using various models
- ANN index building and querying
- Benchmarking tools for different embedding models
- Streamlit demo interface
- Redis-based benchmarking

## Installation

Install the required dependencies:

```bash
# For basic functionality
pip install -r generation/requirements.txt

# For machine learning models
pip install -r generation/requirements_ml.txt

# For Streamlit demo
pip install -r generation/requirements_streamlit.txt
```

## Usage

### Generate Image Embeddings
```bash
cd generation
python 01_generate_image_dump.py
python 02_generate_embeddings.py
```

### Build ANN Index
```bash
cd generation
python 03_generate_index.py
```

### Run Demo
```bash
cd generation
streamlit run demo_streamlit.py
```

### Run Benchmarks
```bash
cd benchmarks/embedding_models_benchmark
python main.py

cd ../ANN_benchmark
python redis_benchmark.py
```

## Structure

- `generation/` - Core embedding and index generation tools
- `benchmarks/` - Performance benchmarking tools
- `dataset/` - Dataset management utilities