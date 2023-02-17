import csv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ciqual_ingredients = []
filename = os.path.join(os.path.dirname(__file__), 'Ciqual.csv.0')
with open(filename, newline='', encoding='utf8') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        ciqual_ingredients.append(row)

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/ciqual")
async def ciqual():
    return ciqual_ingredients
