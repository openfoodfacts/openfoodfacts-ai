def extract_values(obj, key):
    """Pull all values of specified key from nested JSON."""
    arr = []
    
    def extract(obj, arr, key):
        """Recursively search for values of key in JSON tree."""
        if isinstance(obj, dict):
            for k, v in obj.items():
                if (k == key) and "children" not in obj:
                    arr.append(v)
                elif isinstance(v, (dict, list)):
                    extract(v, arr, key)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, arr, key)
        return arr
    
    results = extract(obj, arr, key)
    return results
