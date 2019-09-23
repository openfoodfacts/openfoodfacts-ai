# Training and testing data sets

## Nutrition tables cropping and nutrition facts extraction

### France - French products (20k)

A set of 20k French products with:
* original image containing a nutrition facts table
    * 3596710454181.nutrition.jpg
* rotation angle and bouding box coordinates of the cropped nutrition facts table
    * in the products.csv file
* cropped image of the nutritions facts table
    * 3596710454181.nutrition.cropped.jpg
* Google Cloud Vision resulting json file for the cropped image
    * 3596710454181.nutrition.cropped.jpg.json
* Nutrition values as entered by users in the OFF database
    * 3596710454181.nutriments.json

Location: https://static.openfoodfacts.org/exports/nutrition-lc-fr-country-fr-last-edit-date-2019-08.tar.gz (16.9 Gb)

Command used to generate the test set:
./extract_nutrition_test_set.pl --lc fr --query countries_tags=en:france --query last_edit_dates_tags=2019-08 --dir /srv/off/html/exports/nutrition-lc-fr-country-fr-last-edit-date-2019-08
