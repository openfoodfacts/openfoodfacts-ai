# Rethinking Table Parsing using Graph Neural Networks

This is a repository containing data generation source code for the arxiv paper 1905.13391 ([link](https://arxiv.org/pdf/1905.13391.pdf)). This paper has been accepted into 
ICDAR 2019. To cite the paper, use:

```
@article{rethinkingGraphs,
  author    = {Qasim, Shah Rukh and Mahmood, Hassan and Shafait, Faisal},
  title     = {Rethinking Table Parsing using Graph Neural Networks},
  journal   = {Accepted into ICDAR 2019},
  volume    = {abs/1905.13391},
  year      = {2019},
  url       = {https://arxiv.org/abs/1905.13391},
  archivePrefix = {arXiv},
  eprint    = {1905.13391},
}
```

# Dataset Generation

* TableGeneration: Contains functionality to generate tables
* TFGeneration: Contains functionality to generate tfrecord files. It uses TableGeneration module to generate tables.
* generate_data: Main script to start dataset generation
* unlv_distribution: a binary file that contains words distribution of UNLV dataset (types of words: numbers, alphabets and words containing special characters)



## How to run
Use the following command to generate tfrecords:

python generate_data.py --filesize num_of_images_per_tfrecord --threads num_of_threads --outpath output_directory_to_store_tfrecords --imagespath path_to_UNLV_images --ocrpath path_to_OCR_groundtruth_UNLV --tablepath path_to_UNLV_tables_ground_truths --visualizeimgs 0_or_1 --visualizebboxes 0_or_1


where,
num_of_images_per_tfrecord: Number of images to store in one tfrecord
num_of_threads: Threads are used to process files in parallel. A single thread generates one single tfrecord file. So 10 threads will generate 10 tfrecord files in parallel.
outpath: Output directory to store generated tfrecords
visualizeimgs: If visualizeimgs=1, the generated images will be stored (along than tfrecords).
visualizebboxes: If visualizebboxes=1, the bounding boxes will be draw to images and those images will be stored separately.

imagespath: Directory containing UNLV dataset images
ocrpath: Directory containing ground truths of characters in UNLV dataset
tablepath: Directory containing ground truths of tables in UNLV dataset

You can download UNLV dataset from this link:
https://drive.google.com/drive/folders/1yES8Se8pyGsvLt92dJFz7z7AJQHjt4GA?usp=sharing 

## Table Generation:

All of the content used in generated tables is extracted from UNLV dataset. We also extracted the distribution of alphabetical words, numbers and special character words from it and used same distribution for our dataset. 

Based on the distribution of words, we generated tables of 4 categories(as mentioned in the paper). The code for generating 4 categories of tables is different than a simple "generate 4 categories" approach.

A table is generated in multiple steps like a lego building block(with each step contributing to generation of table):
1. The data types of columns are defined e.g. which column will contain alphabets, numbers or special character words
2. Some cells are randomly selected for missing data
3. Rows and Column spans are added to table
4. The table can be categorized into two ways based on headers(both categories are equally likely to be chosen):
    -   Table with regular headers(Table with only first row containing headers.)
    -   Table with irregular headers(Table with headers in first row and first column. This category can have multiple row spans for headers of first column.)
5. Table borders are chosen randomly. We define border_categories with 4 possibilities(all four categories are equally likely to be chosen):
    -   All borders
    -   No borders
    -   Borders only under headings
    -   Only internal borders
6. An equivalent HTML code is generated for this table.
7. This HTML coded table is converted to image using selenium.
6. Finally, shear and rotation transformations are applied to the table image.


## TFGeneration

During table generation process, the words are assigned unique IDs. During html-to-image conversion, these words are localized with bounding boxes and transformed on image transformation accordingly.

Based on these words IDs, we compute 3 adjacency matrices:
same_row: if two IDs are sharing a row, the value corresponding to that location will be 1 in that matrix.
same_column: Matrix to show which IDs are sharing column
same_cell: Matrix to show which IDs are sharing cell

Instead of just storing the image, we also store some metadata:
1. image height and width
2. number of words in table
3. table category
4. word IDs
5. adjacency matrix for same cells, same rows and same columns


