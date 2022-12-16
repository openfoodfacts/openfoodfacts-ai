# openfoodfacts-ai

![GitHub language count](https://img.shields.io/github/languages/count/openfoodfacts/openfoodfacts-ai)
![GitHub top language](https://img.shields.io/github/languages/top/openfoodfacts/openfoodfacts-ai)
![GitHub last commit](https://img.shields.io/github/last-commit/openfoodfacts/openfoodfacts-ai)
![Github Repo Size](https://img.shields.io/github/repo-size/openfoodfacts/openfoodfacts-ai)
[![codecov](https://codecov.io/gh/openfoodfacts/robotoff/branch/master/graph/badge.svg?token=BY2T0KXNO1)](https://codecov.io/gh/openfoodfacts/openfoodfacts-ai)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://static.openfoodfacts.org/images/logos/off-logo-horizontal-dark.png?refresh_github_cache=1">
  <source media="(prefers-color-scheme: light)" srcset="https://static.openfoodfacts.org/images/logos/off-logo-horizontal-light.png?refresh_github_cache=1">
  <img height="48" src="https://static.openfoodfacts.org/images/logos/off-logo-horizontal-light.svg">
</picture>


## ❗ Before you read on

* This repository is to track and store all our experimental AI endeavours, models training, and wishlists.

* The [Robotoff repo](https://github.com/openfoodfacts/robotoff) is the place to integrate them into production, and file more trivial issues.

* Most trained Models and useful datasets are attached to [releases of this project](https://github.com/openfoodfacts/openfoodfacts-ai/releases) or [releases on robotoff-models](https://github.com/openfoodfacts/robotoff-models/releases).

  [A Google spreadsheet](https://docs.google.com/spreadsheets/d/1p2tvA5ySm0RJpTjUwT3fFrDJNJLXNlVTkxxU-izTIMA/edit#gid=0) also tracks active models.

## [What can I work on ?](https://github.com/openfoodfacts/openfoodfacts-ai/issues/76)


## 🔬 Projects

Here are different experiments.

### Nutrition table

* [Nutrition table detection and extraction (2018 GSoc work by Sagar)](./GSoC2018/table_detection) - integrated in Robotoff, used for the detection part by the Graphnet and TableNet models
* [Nutrition Table Extraction (2020 by Sadok, Yichen and Ramzi)](./nutrition-table-extraction/data_exploration/README.md) - on Graphnet and TableNet
* Basic nutrition extraction for text tables, already in the Robotoff API

### Category prediction

* deployed

  * [Google.org fellowship (2021) - Category prediction based on ingredients and title](https://github.com/openfoodfacts/off-category-classification/) - deployed

* not deployed:

  * [EM Lyon Category prediction (2020)](./ai-emlyon/README.md)  - not yet evaluated and integrated
  * [Category from OCR prediction, Laure (Laurel16) (2021)](https://github.com/Laurel16/OpenFoodFactsCategorizer) - not yet evaluated and integrated - Categories maybe too general

* on-going project @ https://github.com/openfoodfacts/off-category-classification/issues/2

## Weekly meetings
- We e-meet Mondays at 17:00 Paris Time (16:00 London Time, 21:30 IST, 08:00 AM PT)
- ![Google Meet](https://img.shields.io/badge/Google%20Meet-00897B?logo=google-meet&logoColor=white) Video call link: https://meet.google.com/qvv-grzm-gzb
- Join by phone: https://tel.meet/qvv-grzm-gzb?pin=9965177492770
- Add the Event to your Calendar by [adding the Open Food Facts community calendar to your calendar](https://wiki.openfoodfacts.org/Events)
- [Weekly Agenda](https://drive.google.com/open?id=1RUfmWHjtFVaBcvQ17YfXu6FW6oRFWg-2lncljG0giKI): please add the Agenda items as early as you can. Make sure to check the Agenda items in advance of the meeting, so that we have the most informed discussions possible. 
- The meeting will handle Agenda items first, and if time permits, collaborative bug triage.
- We strive to timebox the core of the meeting (decision making) to 30 minutes, with an optional free discussion/live debugging afterwards.
- We take comprehensive notes in the Weekly Agenda of agenda item discussions and of decisions taken.

### Logos

* [Labels and Logo detection (Data 4 Good, by Raphael, Charlotte and Antoine](./data4good_logo_detection/README.md) - code is duplicated and integrated in Robotoff
* logo-ann (related to logos and labels) - classification using approximate KNN search - deployed in [robotoff-ann](https://github.com/openfoodfacts/robotoff-ann)
* Updating the pre-weighted model to recent publications offers a nice no-effort boost

### Spellcheck

* [Spellcheck (by Wauplin)](./spellcheck/README.md) - code is duplicated and integrated in Robotoff

### To be documented

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

## Contributors

<a href="https://github.com/openfoodfacts/robotoff/graphs/contributors">
  <img alt="List of contributors to this repository" src="https://contrib.rocks/image?repo=openfoodfacts/robotoff" />
</a>
