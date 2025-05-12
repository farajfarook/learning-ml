from zlib import crc32
import numpy as np


# This function is used to split the data into training and testing sets based on a hash of the identifier.
# It ensures that the same identifier will always be in the same set, which is useful for reproducibility.
def is_id_in_test_data(identifier, test_ratio):
    hash = crc32(np.int64(identifier))
    # 2**32 is the maximum value for a 32-bit integer
    # The hash is divided by 2**32 to get a value between 0 and 1
    return hash < test_ratio * 2**32


# This function takes a DataFrame, a test ratio, and the name of the identifier column.
# It returns two DataFrames: one for the training set and one for the testing set.
# The training set contains all rows where the identifier is not in the test set, and vice versa.
def split_train_test_by_id_hash(data, test_ratio, id_column):
    # The id_column is the name of the column that contains the identifiers
    # The data[id_column] selects the column from the DataFrame
    ids = data[id_column]
    # in_test_set is a boolean Series that indicates whether each identifier is in the test set
    # The apply function applies the is_id_in_test_data function to each identifier in the Series
    in_test_set = ids.apply(lambda id: is_id_in_test_data(id, test_ratio))
    # The ~ operator negates the boolean Series, so it selects all rows where in_test_set is False
    # This is equivalent to data.loc[~in_test_set]
    # and data.loc[in_test_set] selects all rows where in_test_set is True
    # data.loc means that we are selecting rows based on the boolean Series
    return data.loc[~in_test_set], data.loc[in_test_set]
