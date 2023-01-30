# Install dependencies

This project runs using Python3.
Create a virtualenv.
```
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

Get nutrient types:

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