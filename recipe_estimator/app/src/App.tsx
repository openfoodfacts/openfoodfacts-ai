import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';

import './App.css';
import Recipe from './Recipe';

function App() {
  const [product, setProduct] = useState<any>({});

  let location = useLocation();
  useEffect(() => {
    function getProduct(id: string) {
      fetch(`http://localhost:8000/product/${id}`)
        .then(res => res.json())
        .then(
          (result) => {
            document.title = id + ' - ' + result.name;
            setProduct(result);
          },
          () => {
            setProduct({});
          }
        )
    }
      getProduct(location.hash.substring(1));
  }, [location]);

  return (
    <>
      <div>{product.name}</div>
      <div>{product.ingredients_text}</div>
      <Recipe product={product} />
    </>
  );
}

export default App;
