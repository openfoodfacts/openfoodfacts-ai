from fastapi import Body, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ciqual import ciqual_ingredients
from product import get_product
from minimize_nutrient_distance import estimate_recipe

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

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.get("/ciqual")
async def ciqual():
    return ciqual_ingredients

@app.get("/product/{id}")
async def product(id):
    product = get_product(id)
    return product

@app.post("/recipe")
async def recipe(request: Request):
    product = await request.json()
    estimate_recipe(product)
    return product
