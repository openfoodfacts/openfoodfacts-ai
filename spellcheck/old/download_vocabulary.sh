
#!/bin/bash

# Download list of ingredients
file="./vocabulary/ingredients_fr.txt"
url="https://raw.githubusercontent.com/openfoodfacts/robotoff/master/data/taxonomies/ingredients_fr.txt"
if [ ! -f "$file" ]
then
    echo "File '${file}' not found. Download it."
    wget -O $file $url
fi

# Download list of tokens from ingredients
file="./vocabulary/ingredients_fr_tokens.txt"
url="https://raw.githubusercontent.com/openfoodfacts/robotoff/master/data/taxonomies/ingredients_tokens.txt"
if [ ! -f "$file" ]
then
    echo "File '${file}' not found. Download it."
    wget -O $file $url
fi

# Download and uncompress list of tokens from Wikipedia
gzfile="./vocabulary/fr_tokens_lower.gz"
txtfile="./vocabulary/strings_lower.txt"
url="https://github.com/openfoodfacts/robotoff/raw/master/data/taxonomies/fr_tokens_lower.gz"
if [ ! -f "$txtfile" ]
then
    echo "File '${txtfile}' not found. Uncompress from '${gzfile}'."
    if [ ! -f "$gzfile" ]
    then
        echo "File '${gzfile}' not found. Download it."
        wget -O $gzfile $url
    fi

    echo "Uncompress vocabulary."
    gunzip $gzfile > $txtfile
fi

echo "Download completed."
