import numpy as np


"""
tutorial: 
the purpose of the tutorial is to teach basic numpy 

SECTIONS:
 
 1) basic 'for loop' using numpy range declaration
 2) numpy array indexing
 3) tiling arrays
 4) averaging arrays along specific dimensions
 5) matrix multiplication 
"""



"""
SECTION 1) basic 'for loop' using numpy range declaration
----------------------------------------------------------
"""
range_of_numbers = np.arange(0,10); # 0 to 9, in steps of 1
print(range_of_numbers) # prints [0 1 2 3 4 5 6 7 8 9]

for i in range_of_numbers: # print each number on its own line
    print("range_i = " + str(i)) 
    
# equivalently, you can declare range as part of loop
for i in np.arange(0,10):
    print(i)
    
zeros_array = np.zeros([3,3]) # create a 3x3 matrix of zeros
print(zeros_array) # prints 3x3 matrix of zeros

print('row 0 of 3x3: ' + str(zeros_array[0,:])) # print the 0th row
print('shape of row: ' + str(zeros_array[0,:].shape))

print('column 0 of 3x3: ' + str(zeros_array[:,0])) # print 0th col
print('shape of column: ' + str(zeros_array[:,0].shape))

# the reason row and column have the same shape is because
# the indexing functions [:,0] and [0,:] both return 1d vectors


"""
SECTION 2) numy array indexing
----------------------------------------------------------
we already saw some in section 1, but here we go more in-depth
"""
array_3d = np.zeros([3,3,3]) # 3x3x3 cube (27 elements)
print("array_3d shape: "  + str(array_3d.shape))





"""
----------------------------------------------------------
"""





"""
----------------------------------------------------------
"""



