def list_to_txt(list_to_save, file_path):
    with open(file_path, 'w') as f:
        f.write(str(list_to_save))
        
def read_list_from_txt(file_path):
    with open(file_path, 'r') as f:
        arrays_index = eval(f.read())
    return arrays_index