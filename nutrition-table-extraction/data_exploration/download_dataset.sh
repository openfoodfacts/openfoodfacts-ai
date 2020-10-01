#!/bin/bash
echo "Where do you want to put the data? Please enter the base_path:"
read base_path
cd $base_path || { echo "$base_path not found.. Please enter a valid path." ; exit 1; }

echo "Downloading arial.ttf font.."
if [ ! -f "arial.ttf" ]
then
wget "https://www.font-police.com/classique/sans-serif/arial.ttf" -q --show-progress
echo "Done!"
else
echo "arial.ttf already exists!"
fi

echo ""
echo "Downloading the dataset.."
[ ! -d "nutrition-lc-fr-country-fr-last-edit-date-2019-08" ] && wget "https://static.openfoodfacts.org/exports/nutrition-lc-fr-country-fr-last-edit-date-2019-08.tar.gz" -q --show-progress
if [ ! -d "nutrition-lc-fr-country-fr-last-edit-date-2019-08" ]
then
    echo "Done!"
    echo ""
    echo "Unpacking.."
    tar -xzf nutrition-lc-fr-country-fr-last-edit-date-2019-08.tar.gz || { echo 'Unable to unpack the downloaded tar.gz file..' ; exit 1; }
    rm nutrition-lc-fr-country-fr-last-edit-date-2019-08.tar.gz
else
    echo "Data folder already exists!"
fi
echo ""
echo "Starting partitioning the data.."

[ ! -d "image_files" ] && mkdir image_files 
[ ! -d "json_files" ] && mkdir json_files 
[ ! -d "cropped_images" ] && mkdir cropped_images 
[ ! -d "cropped_json_files" ] && mkdir cropped_json_files 
  
mv nutrition-lc-fr-country-fr-last-edit-date-2019-08/*.nutrition.jpg image_files/
mv nutrition-lc-fr-country-fr-last-edit-date-2019-08/*.nutrition.cropped.jpg cropped_images/
mv nutrition-lc-fr-country-fr-last-edit-date-2019-08/*.nutriments.json json_files/
mv nutrition-lc-fr-country-fr-last-edit-date-2019-08/*.nutrition.cropped.json cropped_json_files/

rm -r nutrition-lc-fr-country-fr-last-edit-date-2019-08
echo "Done! Your dataset is ready :)"