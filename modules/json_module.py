import json
from typing import Dict


def dict_to_json(dictionary: Dict, path: str):
    """
    Save dict as json
    
    """

    with open(path, 'w') as outfile:
        json.dump(dictionary, outfile)
        
        
def read_json(path: str):
    """
    Load json in a dictionary
    
    """
    
    with open(path, 'r') as json_file:
        return_dict = json.load(json_file)
    return return_dict
