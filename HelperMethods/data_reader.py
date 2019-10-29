import pickle
import re
import struct
import numpy as np
import os

from HelperMethods.layer_help import is_close

def convert_bytes_array(byte_order, byte_type, byte_array, num_bytes):
    #TO DO: Make it so that it can take a byte_array that is possible gargantuan. I.e. it would split it up
    # in reasonable ways and join together or save it in different files or numpy arrays.
    # Maybe a yield would be more reasonable
    NUM_OF_BYTES_IN_TYPE = {"c": 1,
                            "B": 1,
                            "b": 1,
                            "h": 2,
                            "H": 2,
                            "i": 4,
                            "I": 4,
                            "l": 4,
                            "L": 4,
                            "q": 8,
                            "Q": 8,
                            "f": 4,
                            "d": 8}

    assert (isinstance(byte_type, str) and isinstance(byte_order, str) and byte_type in NUM_OF_BYTES_IN_TYPE.keys())
    assert (is_close(num_bytes / NUM_OF_BYTES_IN_TYPE.get(byte_type), num_bytes // NUM_OF_BYTES_IN_TYPE.get(byte_type)))

    num_to_convert = num_bytes // NUM_OF_BYTES_IN_TYPE.get(byte_type)

    return struct.unpack_from(byte_order + byte_type * num_to_convert, byte_array)

def magic_number_converter(byte_seq):
    DATA_TYPE_DICT = {"8": "B",  # Unsigned byte
                      "9": "b",  # signed byte
                      "11": "h",  # signed short
                      "12": "i",  # signed int
                      "13": "f",  # signed float
                      "14": "d"}  # signed double

    NUM_OF_BYTES_IN_TYPE = {"B": 1,
                            "b": 1,
                            "h": 2,
                            "i": 4,
                            "f": 4,
                            "d": 8}

    #Magic number must be a 32 bit integer, i.e. 4 bytes
    #Should be a fatal attempt if it the asserts fail
    assert(isinstance(byte_seq, bytes))
    assert (len(byte_seq) == 4)
    #Byte 2 determines the type


    # Byte 3 is the dimension of the vector/matrix
    return (DATA_TYPE_DICT.get(byte_seq[2].__str__()), NUM_OF_BYTES_IN_TYPE.get(byte_seq[2].__str__()), byte_seq[3])

def convert_idx_data_numpy(file_path):
    returnNpArray = None

    try:
        with open(file_path, 'rb') as f:
            data = f.read()
            # print(try_to_convert_bytes_array(data, 16, len(data), "f"))

            data_type, num_of_bytes_in_type, dim_of_matrix = magic_number_converter(data[0:4])

            if (re.search("\.idx\d-ubyte$", file_path)):
                # Images begin with
                #   32 bit int magic number (2051) -- which determines data type and dimensions of each instance of data
                #   32 bit int number of images
                #   32 bit int number of rows
                #   32 bit int number of columns
                #   unsigned byte pixel
                # So the first byte is kind of irrelevant
                byte_offset = 4 #Offset by the magic number

                sizes_of_data = [int.from_bytes(b, "big", signed= False) for b in [data[byte_offset * i: byte_offset * (i+1)] for i in range(1, dim_of_matrix + 1)]]

                offset_after_data_setup = byte_offset * (dim_of_matrix + 1)
                data_to_convert = data[offset_after_data_setup:]

                returnNpArray = np.asarray(convert_bytes_array(">", "B", data_to_convert, len(data_to_convert)))
                if (len(sizes_of_data) != 1):
                    returnNpArray = returnNpArray.reshape(*sizes_of_data)

    except FileNotFoundError:
        raise FileNotFoundError("Could not find the file, please try again.")

    if (returnNpArray is None):
        raise ValueError("Attempted to return a null dataset.")
    return returnNpArray


def pickle_numpy_array(array, file_path):
    #Pickles numpy array into file_path
    with open(file_path, 'wb') as f:
        pickle.dump(array, f)
        return file_path

def unpickle_numpy_array(file_path):
    # Unpickles numpy_array
    return_array = None
    file_path = os.path.normpath(file_path)
    with open(file_path, 'rb') as f:
        return_array = pickle.load(f)
    if (return_array is not None or isinstance(return_array, np.array)):
        return return_array
    else:
        raise ValueError("Tried to return a null or invalid numpy array instead of valid numpy array after unpickling.")

def convert_and_pickle_idx(file_path):
    #Converts idx file into numpy array and then pickles it
    new_file_path = re.sub(r'\.idx\d-ubyte$', '.pickle', os.path.normpath(file_path))
    if (not os.path.exists(new_file_path)):
        array_to_save = convert_idx_data_numpy(file_path)
        pickle_numpy_array(array_to_save, new_file_path)

def convert_and_pickle_multiple_idx(*file_paths):
    #Will assume full file paths
    for file_path in file_paths:
        convert_and_pickle_idx(file_path)