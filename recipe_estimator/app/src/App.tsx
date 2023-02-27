import { Table, TableHead, TableRow, TextField, TableBody, TableCell, Typography, Input } from '@mui/material';
import React, { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

import './App.css';

function App() {
  const [product, setProduct] = useState<any>({});

  let location = useLocation();
  useEffect(() => {
    function flattenIngredients(flatIngredients: any[], ingredients: any[], depth: number) {
      for (const ingredient of ingredients) {
        ingredient.depth = depth;
        ingredient.proportion = round(ingredient.proportion);
        flatIngredients.push(ingredient);
        if (ingredient.ingredients) {
          flattenIngredients(flatIngredients, ingredient.ingredients, depth + 1);
        }
      }
    }
    function getProduct(id: string) {
      fetch(`http://localhost:8000/recipe/${id}`)
        .then(res => res.json())
        .then(
          (result) => {
            const flatIngredients: any[] = [];
            flattenIngredients(flatIngredients, result.ingredients, 0);
            result.ingredients = flatIngredients;
            document.title = id + ' - ' + result.name;
            setProduct(result);
          },
          () => {
            setProduct({});
          }
        )
    }
      getProduct(location.pathname.split('/').slice(-1)[0]);
  }, [location]);


  function round(num: number){
    return isNaN(num) ? '-' : Math.round((num + Number.EPSILON) * 100) / 100;
  }

  function getTotal(nutrient_key: string) {
    let total = 0;
    for(const ingredient of product.ingredients) {
      if (!ingredient.ingredients) 
        total += ingredient.proportion * ingredient.nutrients?.[nutrient_key] / 100;
    }
    return total;
  }

  return (
    <div>
      {product.ingredients &&
        <div>
          <div>{product.name}</div>
          <div>{product.ingredients_text}</div>
            <Table size='small'>
              <TableHead>
                <TableRow className='total'>
                  <TableCell><Typography>Ingredient</Typography></TableCell>
                  <TableCell><Typography>CIQUAL Code</Typography></TableCell>
                  <TableCell><Typography>Proportion</Typography></TableCell>
                  {Object.keys(product.nutrients).map((nutrient: string) => (
                    <TableCell key={nutrient}>
                      <Typography>{nutrient}</Typography>
                      <Typography variant="caption">{round(product.nutrients[nutrient].weighting)}</Typography>
                    </TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {product.ingredients.map((ingredient: any, index: number)=>(
                  <TableRow key={index}>
                    <TableCell><Typography sx={{paddingLeft: (ingredient.depth)}}>{ingredient.text}</Typography></TableCell>
                    <TableCell><Typography>{ingredient.ciqual_code}</Typography></TableCell>
                    <TableCell>{!ingredient.ingredients &&
                      <TextField type="number" size='small' value={ingredient.proportion} onChange={(e) => {ingredient.proportion = parseFloat(e.target.value);setProduct({...product});}}/>
                    }
                    </TableCell>
                    {Object.keys(product.nutrients).map((nutrient: string) => (
                      <TableCell key={nutrient}>{!ingredient.ingredients &&
                        <>
                          <Typography variant="caption">{ingredient.nutrients?.[nutrient]}</Typography>
                          <Typography variant="body1">{round(ingredient.proportion * ingredient.nutrients?.[nutrient] / 100)}</Typography>
                        </>
                      }
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
                  <TableRow className='total'>
                    <TableCell colSpan={2}><Typography>Ingredients total</Typography></TableCell>
                    <TableCell><Typography>
                      {round(product.ingredients.reduce((total: number,ingredient: any) => total + (ingredient.ingredients ? 0 : ingredient.proportion), 0))}
                    </Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="body1">{round(getTotal(nutrient_key))}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={3}><Typography>Product total</Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="body1">{round(product.nutrients[nutrient_key].total)}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={2}><Typography>Variance</Typography></TableCell>
                    <TableCell><Typography>{round(Object.keys(product.nutrients).reduce((total: number,nutrient_key: any) => 
                      total + (product.nutrients[nutrient_key].valid 
                        ? product.nutrients[nutrient_key].weighting * Math.abs(getTotal(nutrient_key)- product.nutrients[nutrient_key].total) 
                        : 0), 0))}
                    </Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        {product.nutrients[nutrient_key].valid 
                          ? <>
                            <Typography variant="caption">{round(getTotal(nutrient_key) - product.nutrients[nutrient_key].total)}</Typography>
                            <Typography>{round(product.nutrients[nutrient_key].weighting * (getTotal(nutrient_key)- product.nutrients[nutrient_key].total))}</Typography>
                            </>
                          : <Typography variant="caption">{product.nutrients[nutrient_key].error}</Typography>
                        }
                      </TableCell>
                    ))}
                  </TableRow>
              </TableBody>
            </Table>
        </div>
      }
    </div>
  );
}

export default App;
