# Prepare database

This requires CIQUAL codes to be set on products. To do this you need to checkout issue/7918 of openfoodfacts-server and run he update_all_products script with the assign_ciqual_codes option.

# Install dependencies

This project runs using Python3.
Create a virtualenv.
```
cd ./recipe_estimator/api
python -m venv venv 
```

Enter virtualenv.
```
venv/Scripts/activate
```

Install requirements.
```
pip install -r requirements.txt
```

To run the API server:

```
uvicorn main:app --reload
```

To run the app (in a new terminal):
```
cd ./recipe_estimator/app
npm start
```

To test:
http://localhost:3000/product_code

e.g.

http://localhost:3000/0677294998025


# Background Info

To get nutrient types for nutrient_map.csv I used:

```js
db.products.aggregate([
  {
    $project: {
      keys: {
        $map: {
          input: {
            "$objectToArray": "$nutriments"
          },
          in: "$$this.k"
        }
      }
    }
  },
  {
    $unwind: "$keys"
  },
  {
    $group: {
      _id: "$keys",
      count: {
        "$sum": 1
      }
    }
  }
])
```

Need to skip any nutrients where Ciqual value is '-' as this means not known, not zero