#!/bin/zsh

# Set variables
data_path="/Users/baslad01/data_dump/openfoodfacts/yolo"
project_id=56
project_path="/Users/baslad01/PycharmProjects/openfoodfacts-ai/ml_utils/ml_utils_cli"


# Remove the data folder if it exists
if [ -d "$data_path/data" ]; then
    rm -rf "$data_path/data"
fi

alias ml-cli='${CONDA_PREFIX}/bin/python3 ${project_path}/main.py'
# Export the data using the command line
ml-cli datasets export --project-id ${project_id} --from ls --to ultralytics --output-dir ${data_path} --label-names 'product,price-tag' --api-key ${LABEL_STUDIO_KEY}

# Update the path data in data.yaml
sed -i '' "s|path: data|path: ${data_path}/data|1" "${data_path}/data.yaml"