import { Table, TableHead, TableRow, TextField, TableBody, TableCell } from '@mui/material';
import React, { useState } from 'react';
import './App.css';

function App() {
  const [product, setProduct] = useState<any>({});

  function getProduct(event: React.ChangeEvent<HTMLInputElement>) {
    const id = event.currentTarget.value;
    fetch(`http://localhost:8000/product/${id}`)
      .then(res => res.json())
      .then(
        (result) => {
          setProduct(result);
        },
        () => {
          setProduct({});
        }
      )
  }
  return (
    <div>
      <TextField onChange={getProduct} />
      {product.ingredients &&
        <div>
          <div>{product.name}</div>
          <div>{product.ingredients_text}</div>
            <Table size='small'>
              <TableHead>
                <TableRow>
                  <TableCell>Ingredient</TableCell>
                  {Object.keys(product.nutrients).map((nutrient: string) => (
                    <TableCell>{nutrient}</TableCell>
                  ))}
                </TableRow>
              </TableHead>
              <TableBody>
                {product.ingredients.map((ingredient: any)=>(
                  <TableRow>
                    <TableCell>{ingredient.text}</TableCell>
                    {Object.keys(product.nutrients).map((nutrient: string) => (
                      <TableCell>{ingredient.nutrients?.[nutrient]}</TableCell>
                    ))}
                  </TableRow>
                ))}
              </TableBody>
            </Table>
        </div>
      }
    </div>
  );
}

export default App;
