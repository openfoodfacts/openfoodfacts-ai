#!/bin/zsh

# Source the .zshrc file to load environment variables
#source ~/.zshrc
# Set variables
data_path="/Users/baslad01/data_dump/openfoodfacts/yolo"
project_id=56
project_path="/Users/baslad01/PycharmProjects/openfoodfacts-ai/ml_utils/ml_utils_cli"


model_name="${data_path}/model/train10/weights/best.pt"

alias ml-cli='${CONDA_PREFIX}/bin/python3 ${project_path}/main.py'

#ml-cli --help
# Preanotate the data
ml-cli projects add-prediction --project-id ${project_id} --model-name ${model_name} --backend ultralytics --labels 'product' --labels 'price tag' --label-mapping '{"price tag": "price-tag"}' --api-key ${LABEL_STUDIO_KEY}
