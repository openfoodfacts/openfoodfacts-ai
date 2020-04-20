from pymongo import MongoClient

# Connect to local Mongo DB
products = MongoClient(host="localhost", port=27017,).off.products
