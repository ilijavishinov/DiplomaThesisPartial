import h5py
import dask.array as da

def dask_array_from_h5py(h5py_file_path):
    return da.from_array(h5py.File(h5py_file_path)['data'])
    
