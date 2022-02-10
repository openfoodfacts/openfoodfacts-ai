#!/bin/bash

wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-logo-2022-01-21/logo_dataset.tar.gz.1
wget https://github.com/openfoodfacts/openfoodfacts-ai/releases/download/dataset-logo-2022-01-21/logo_dataset.tar.gz.2
cat logo_dataset.tar.gz.* > logo_dataset.tar.gz
tar xvzf logo_dataset.tar.gz