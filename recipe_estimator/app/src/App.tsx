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
                <TableRow>
                <TableCell>Ingredient</TableCell>
                <TableCell>CIQUAL Code</TableCell>
                <TableCell>Proportion</TableCell>
                  {Object.keys(product.nutrients).map((nutrient: string) => (
                    <TableCell key={nutrient}>{nutrient}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {product.ingredients.map((ingredient: any, index: number)=>(
                  <TableRow key={index}>
                    <TableCell>{'\u00A0'.repeat(ingredient.depth) + ingredient.text}</TableCell>
                    <TableCell>{ingredient.ciqual_code}</TableCell>
                    <TableCell>
                      <TextField value={round(ingredient.proportion)} onChange={(e) => ingredient.proportion = e.target.value}/>
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
                  <TableRow>
                    <TableCell colSpan={3}>Ingredients total</TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="body1">{round(getTotal(nutrient_key))}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={3}>Product total</TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="caption">{round(product.nutrients[nutrient_key].weighting)}</Typography>
                        <Typography variant="body1">{round(product.nutrients[nutrient_key].total)}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={3}>Variance</TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>{product.nutrients[nutrient_key].valid &&
                        <>
                        <Typography variant="caption">{round(product.nutrients[nutrient_key].weighting * (getTotal(nutrient_key)- product.nutrients[nutrient_key].total))}</Typography>
                        <Typography variant="body1">{round(getTotal(nutrient_key) - product.nutrients[nutrient_key].total)}</Typography>
                        </>}
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
