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
            fetch(`http://localhost:8000/recipe`, {method: 'POST', body: JSON.stringify(result)})
            .then(res => res.json())
            .then(
              (result) => {
                setProduct(result);
              },() =>{
                setProduct({});
              })
          },
          () => {
            setProduct({});
          }
        )
    }
      getProduct(location.pathname.split('/').slice(-1)[0]);
  }, [location]);

  return (
    <Recipe product={product} />
  );
}

export default App;
