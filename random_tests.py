import numpy as np
import re
from matplotlib import pyplot as plt
import time
import struct

TRAIN_IMAGE_PATH = "C:\\Users\\Lewis\\PycharmProjects\\ConvNet\\Data\\train-images.idx3-ubyte"
TRAIN_LABEL_PATH = "C:\\Users\\Lewis\\PycharmProjects\\ConvNet\\Data\\train-labels.idx1-ubyte"

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

def byte_array_to_num_array(array, type):
    assert (type in DATA_TYPE_DICT.keys())
    byte_length = NUM_OF_BYTES_IN_TYPE.get(type)
    returnArray = [struct.unpack(type, array[i:i+byte_length]) for i in range(0, len(array), byte_length)]
    return returnArray

def magic_number_converter(byte_seq):
    #Magic number must be a 32 bit integer, i.e. 4 bytes
    #Should be a fatal attempt if it the asserts fail
    assert(isinstance(byte_seq, bytes))
    assert (len(byte_seq) == 4)
    #Byte 2 determines the type


    # Byte 3 is the dimension of the vector/matrix
    return (DATA_TYPE_DICT.get(byte_seq[2].__str__()), NUM_OF_BYTES_IN_TYPE.get(byte_seq[2].__str__()), byte_seq[3])

def try_to_convert_from_bytes(value, byteorder, signed=False):
    # Return an int if int, otherwise converts bytes to an int
    assert (isinstance(value, int) or isinstance(value, bytes))
    if (isinstance(value, int)):
        return value
    elif (isinstance(value, bytes)):
        return int.from_bytes(value, byteorder=byteorder, signed=signed)

def try_to_convert_bytes_array(array, start, end, type):
    # Byte order will be assumed to be "big"
    if (type == 'B'):
        return_array = [try_to_convert_from_bytes(byte, "big") for byte in array[start:end]]
    elif type in DATA_TYPE_DICT.values():
        byte_size = NUM_OF_BYTES_IN_TYPE.get(type)
        return_array = [struct.unpack(">{}".format(type), array[byte_list:byte_list + byte_size])[0] for byte_list in range(start, end, byte_size)]
    if (return_array == None):
        raise ValueError
    return return_array

# print(int.from_bytes(b'\x9C', "big", signed=True))

def convert_to_numpy_and_save(file_path, num_of_dims):
    with open(file_path, 'rb') as f:
        data = f.read()
        # print(try_to_convert_bytes_array(data, 16, len(data), "f"))

        data_type, num_of_bytes_in_type, dim_of_matrix = magic_number_converter(data[0:4])

        returnNpArray = None
        if (re.search("images.idx\d-ubyte", file_path)):
            # Images begin with
            #   32 bit int magic number (2051) --
            #   32 bit int number of images
            #   32 bit int number of rows
            #   32 bit int number of columns
            #   unsigned byte pixel
            # So the first byte is kind of irrelevant
            first_offset = 4
            dim = [int.from_bytes(b, "big", signed= False) for b in [data[first_offset * i: first_offset * (i+1)] for i in range(1, num_of_dims + 1)]]
            offset_after_dims = first_offset * (num_of_dims + 1)
            num_of_images = dim[0] * dim[1] * dim[2]



            # [try_to_convert_from_bytes(data_slice) for data_slice in [offset_after_dims, (num_of_images + offset_after_dims), num_of_bytes_in_type]]
            at = time.time()
            # returnArray = [data[i] for i in range(offset_after_dims, (dim[0]*dim[1]*dim[2]+offset_after_dims))]
            returnArray = try_to_convert_bytes_array(data, offset_after_dims, len(data), 'B')
            print(returnArray)
            print(time.time() - at)
            returnNpArray = np.array(returnArray).reshape(dim[0], dim[1], dim[2])

            plt.imshow(returnNpArray[0,:,:])
            plt.show()


        elif(re.search("labels.idx\d-ubyte", file_path)):
            pass
            # labels begin with
            #   32 bit int magic number (2049)
            #   32 bit int number of images
            #   unsigned byte label
            first_offset = 4


        if (returnNpArray == None):
            raise Exception
        return returnNpArray




print(convert_to_numpy_and_save(TRAIN_IMAGE_PATH, 3))