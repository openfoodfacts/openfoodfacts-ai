# openfoodfacts-ai

## ❗ Before you read on

* This repository is to track and store all our experimental AI endeavours, models training, and wishlists.

* The [Robotoff repo](https://github.com/openfoodfacts/robotoff) is the place to integrate them into production, and file more trivial issues.

* Most trained Models and useful datasets are attached to [releases of this project](https://github.com/openfoodfacts/openfoodfacts-ai/releases) or [releases on robotoff-models](https://github.com/openfoodfacts/robotoff-models/releases).

  [A google spreasheet](https://docs.google.com/spreadsheets/d/1p2tvA5ySm0RJpTjUwT3fFrDJNJLXNlVTkxxU-izTIMA/edit#gid=0) also tracks active models.


## 🔬 Projects

Here are different experiments.

Nutrition table:

* [Nutrition table detection and extraction (2018 GSoc work by Sagar)](./GSoC2018/table_detection) - integrated in Robotoff, used for the detection part by the Graphnet and TableNet models
* [Nutrition Table Extraction (2020 by Sadok, Yichen and Ramzi)](./nutrition-table-extraction/data_exploration/README.md) - on Graphnet and TableNet

Category prediction:

* deployed

  * [Google.org fellowship (2021) - Category prediction based on ingredients and title](https://github.com/openfoodfacts/off-category-classification/) - deployed

* not deployed:

  * [EM Lyon Category prediction (2020)](./ai-emlyon/README.md)  - not yet evaluated and integrated
  * [Category from OCR prediction, Laure (Laurel16) (2021)](https://github.com/Laurel16/OpenFoodFactsCategorizer) - not yet evaluated and integrated - Categories maybe too general

Logos:

* [Labels and Logo detection (Data 4 Good, by Raphael, Charlotte and Antoine](./data4good_logo_detection/README.md) - code is duplicated and integrated in Robotoff
* logo-ann (related to logos and labels) - classification using approximate KNN search - deployed in [robotoff-ann](https://github.com/openfoodfacts/robotoff-ann)


Spellcheck

* [Spellcheck (by Wauplin)](./spellcheck/README.md) - code is duplicated and integrated in Robotoff

To be documented:

* ocr-cleaning (please add a description)
* object-detection (related to logos and labels)

## 👷 Contributing

You can fork this repository and start your own experiments or use a distinct repository.
Please use a AGPL or more permissive but compatible license.

Do not hesitate to join us on [#robotoff](https://slack.openfoodfacts.org) channel
(or [#computervision](https://slack.openfoodfacts.org) for work relating on images).
We will be happy to help you get data, insights and other useful tips.

* [Our Roadmap for AI and Robotoff](https://wiki.openfoodfacts.org/Artificial_Intelligence/Robotoff/Roadmap)

* [Ideas for research projects for Open Food Facts](https://github.com/openfoodfacts/openfoodfacts-ai/issues)

* [Ideas for applied ML for Open Food Facts](https://github.com/openfoodfacts/robotoff/issues)

* [Proposed ideas for Google's Summer of Code](https://world.openfoodfacts.org/google-summer-of-code)

* [Get the data to start playing with food](https://world.openfoodfacts.org/data)
  (see also datasets in this [project releases](https://github.com/openfoodfacts/openfoodfacts-ai/releases))

## 📚 More documentation

* You can see many [great analysis of Open Food Facts data in notebooks on Kaggle](https://www.kaggle.com/openfoodfacts/world-food-facts)
