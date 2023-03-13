import { Table, TableHead, TableRow, TextField, TableBody, TableCell, Typography, Autocomplete, Button} from '@mui/material';
import { useEffect, useRef, useState } from 'react';

interface RecipeProps {
  product: any
}

function ciqualDisplayName(ciqualIngredient: any): string {
  return ciqualIngredient?.alim_nom_eng ? `${ciqualIngredient.alim_nom_eng} (${ciqualIngredient.alim_code})` : ''
}
function flattenIngredients(ingredients: any[], depth = 0): any[] {
  const flatIngredients = [];
  for (const ingredient of ingredients) {
    ingredient.depth = depth;
    ingredient.proportion = round(ingredient.proportion);
    flatIngredients.push(ingredient);
    if (ingredient.ingredients) {
      flatIngredients.push(...flattenIngredients(ingredient.ingredients, depth + 1));
    } else {
      if (!ingredient.options)
        ingredient.options =  ingredient.ciqual_ingredient ? [ingredient.ciqual_ingredient] : [];
      if (ingredient.searchTerm == null)
        ingredient.searchTerm =  ciqualDisplayName(ingredient.ciqual_ingredient);
    }
  }
  return flatIngredients;
}

function round(num: number){
  return isNaN(num) ? '-' : Math.round((num + Number.EPSILON) * 100) / 100;
}

export default function Recipe({product}: RecipeProps) {
  const [ingredients, setIngredients] = useState<any>([]);
  const [myProduct, setMyProduct] = useState<any>(product);

  useEffect(() => {
    if (!myProduct.ingredients)
      return;
    async function fetchData() {
      const results = await (await fetch(`http://localhost:8000/recipe`, {method: 'POST', body: JSON.stringify(myProduct)})).json();
      setIngredients(results.ingredients);
    }
    fetchData();
  }, [myProduct]);

  useEffect(()=>{
    setMyProduct(product);
  }, [product]);

  function getTotal(nutrient_key: string) {
    let total = 0;
    for(const ingredient of ingredients) {
      if (!ingredient.ingredients) 
        total += ingredient.proportion * ingredient.ciqual_ingredient?.[nutrient_key] / 100;
    }
    return total;
  }

  const previousController = useRef<AbortController>();
  
  function getData(searchTerm: string, ingredient: any) {
    if (previousController.current) {
      previousController.current.abort();
    }
    const controller = new AbortController();
    const signal = controller.signal;
    previousController.current = controller;
    fetch("http://localhost:8000/ciqual/" + searchTerm, {signal})
      .then(function (response) {
        return response.json();
      },console.log)
      .then(function (myJson) {
        if (ingredient.ciqual_ingredient && !(myJson.find((i:any) => i.alim_code === ingredient.ciqual_ingredient.alim_code)))
          myJson.push(ingredient.ciqual_ingredient);
        ingredient.options = myJson;
        setIngredients([...ingredients]);
      });
  };
  
  function onInputChange(ingredient:any, value: string, reason: string) {
    ingredient.searchTerm = value;
    if (reason === 'input' && value) {
      getData(value, ingredient);
    } else {
      ingredient.options = ingredient.ciqual_ingredient ? [ingredient.ciqual_ingredient] : [];
      setIngredients([...ingredients]);
    }
  };
  
  function ingredientChange(ingredient: any, value: any) {
    if (value) {
      ingredient.ciqual_ingredient = value;
      ingredient.searchTerm = ciqualDisplayName(value);
      setIngredients([...ingredients]);
    }
  }

  return (
    <div>
      {product.nutrients && ingredients &&
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
                {flattenIngredients(ingredients).map((ingredient: any, index: number)=>(
                  <TableRow key={index}>
                    <TableCell><Typography sx={{paddingLeft: (ingredient.depth)}}>{ingredient.text}</Typography></TableCell>
                    <TableCell>{!ingredient.ingredients &&
                      <Autocomplete
                        id="combo-box-demo"
                        options={ingredient.options}
                        onInputChange={(_event,value,reason) => onInputChange(ingredient,value,reason)}
                        onChange={(_event,value) => ingredientChange(ingredient,value)}
                        value={ingredient.ciqual_ingredient || null}
                        inputValue={ingredient.searchTerm || ''}
                        getOptionLabel={(option:any) => ciqualDisplayName(option)}
                        isOptionEqualToValue={(option:any, value:any) => option.alim_code === value.alim_code}
                        style={{ width: 300 }}
                        renderInput={(params) => (
                          <TextField {...params} size='small'/>
                        )}
                      />
                    }
                    </TableCell>
                    <TableCell>{!ingredient.ingredients &&
                      <TextField type="number" size='small' value={ingredient.proportion} onChange={(e) => {ingredient.proportion = parseFloat(e.target.value);setIngredients([...ingredients]);}}/>
                    }
                    </TableCell>
                    {Object.keys(product.nutrients).map((nutrient: string) => (
                      <TableCell key={nutrient}>{!ingredient.ingredients &&
                        <>
                          <Typography variant="caption">{ingredient.ciqual_ingredient?.[nutrient]}</Typography>
                          <Typography variant="body1">{round(ingredient.proportion * ingredient.ciqual_ingredient?.[nutrient] / 100)}</Typography>
                        </>
                      }
                      </TableCell>
                    ))}
                  </TableRow>
                ))}
                  <TableRow className='total'>
                    <TableCell colSpan={2}><Typography>Ingredients totals</Typography></TableCell>
                    <TableCell><Typography>
                      {round(ingredients.reduce((total: number,ingredient: any) => total + (ingredient.ingredients ? 0 : ingredient.proportion), 0))} %
                    </Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="body1">{round(getTotal(nutrient_key))}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow>
                    <TableCell colSpan={3}><Typography>Quoted product nutrients</Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        <Typography variant="body1">{round(product.nutrients[nutrient_key].total)}</Typography>
                      </TableCell>
                    ))}
                  </TableRow>
                  <TableRow className='total'>
                    <TableCell colSpan={2}><Typography>Variance</Typography></TableCell>
                    <TableCell><Typography>{round(Object.keys(product.nutrients).reduce((total: number,nutrient_key: any) => 
                      total + (!product.nutrients[nutrient_key].error 
                        ? product.nutrients[nutrient_key].weighting * Math.abs(getTotal(nutrient_key)- product.nutrients[nutrient_key].total) 
                        : 0), 0))}
                    </Typography></TableCell>
                    {Object.keys(product.nutrients).map((nutrient_key: string) => (
                      <TableCell key={nutrient_key}>
                        {!product.nutrients[nutrient_key].error 
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
            <Button variant='contained' onClick={()=>setMyProduct({...myProduct, ingredients})}>recalculate</Button>
        </div>
      }
    </div>
  );
}

