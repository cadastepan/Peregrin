# import os

# # Function for file deletion in a selected folder
# def clean_folder(folder):
#     for filename in os.listdir(folder): # Czcle for deletion of files in selected folder
#         file_path = os.path.join(folder, filename)
#         try:
#             # Check if it's a file - if so, delete it;
#             if os.path.isfile(file_path) or os.path.islink(file_path): 
#                 os.remove(file_path)
#                 print(f"Deleted: {file_path}")
#             # Ceck if it's a directory - if so, skip it;
#             elif os.path.isdir(file_path):
#                 print(f"Skipped directory: {file_path}")
#         except Exception as e:
#             print(f"Failed to delete {file_path}. Reason: {e}")
#     print("All files have been deleted.")

# # Function calculateing a span of an x axis for plotting data with a large number of elements into a hostogram
# def x_span(df):
    # index = 0.08 # definition of an index for x_span calculation
    # num_elements = len(df) # count rows = elements in a dataframe
    # x_span = num_elements * index
    # return x_span



class Multiplication:
    """
    Instantiate a multiplication operation.
    Numbers will be multiplied by the given multiplier.
    
    :param multiplier: The multiplier.
    :type multiplier: int
    """
    
    def __init__(self):
        self.multiplier = 2
    
    def multiply(self, number):
        """
        Multiply a given number by the multiplier.
        
        :param number: The number to multiply.
        :type number: int
    
        :return: The result of the multiplication.
        :rtype: int
        """
        
        return number * self.multiplier

# Instantiate a Multiplication object
multiplication = Multiplication()

# Call the multiply method
print(multiplication.multiply(5))