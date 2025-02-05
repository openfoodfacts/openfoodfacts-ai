# Number of images to download
n=1000
# Directory to store images
folder="images"
# Codes of filtered images generated with DuckDB
codes="best_image_codes.txt"
# Image urls
urls_file="urls.txt"

# Base URLS
bucket_url="https://openfoodfacts-images.s3.eu-west-3.amazonaws.com/"
off_urls="https://images.openfoodfacts.org/images/products/"

# Create the folder to store images and URLs
mkdir -p "$folder" 
touch "$folder/$urls_file"

# Pre-filter the data once to avoid repeated decompression
zcat data_keys.gz | grep -v ".400.jpg" | grep "1.jpg" > temp_data_keys.txt

# Process each code from best_image_codes.txt
shuf -n "$n" "$codes" | while read -r code; do
    # Format the code into the required pattern 260/012/901/5091
    formatted_code=$(echo "$code" | sed 's|^\(...\)\(...\)\(...\)\(.*\)$|\1\/\2\/\3\/\4|')

    # Search for matching entries in data_keys.gz and pick one random image
    grep "$formatted_code" temp_data_keys.txt | shuf -n 1 | while read -r url; do
        # Construct the filename by stripping the bucket URL and formatting
        filename=$(echo "$url" | sed "s|$bucket_url||" | tr '/' '_' | sed 's|data_||')
        
        # Download the image using wget
        wget -O "$folder/$filename" "$bucket_url$url"

        # Add image url
        image_url=$(echo "$url" | sed 's|data/||')
        echo "$off_urls$image_url" >> "$folder/$urls_file"
    done
done

# Clean
rm temp_filtered_data.txt
