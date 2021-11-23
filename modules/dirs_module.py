import os

def create_directory(directory: str, warn_exists: bool = True):
    
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    else:
        if warn_exists:
            answer = str(input(f"""The directory {directory} that you are trying to write in already exists.
                                  Some files may be overwritten.
                                  Are you sure you want to continue? [y/n]"""))
            
            if answer.lower() == 'y': pass
            else: sys.exit()
        else: pass
