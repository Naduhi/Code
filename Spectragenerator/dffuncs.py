# import numpy as np
import pandas as pd
import numpy as np
from functools import reduce

# from pandas.util._validators import validate_bool_kwarg


def transpose(df, col_as_header=None, header_as_col=False):
    """Transpose the dataframe,
    e.g. from

    | Index | Raman Shift | Sample_1 | Sample_2 | Sample_3 |
    |:-----:|:-----------:|:--------:|:--------:|:--------:|
    |   1   |    500.5    |     1    |     4    |     7    |
    |   2   |    501.0    |     2    |     5    |     8    |
    |   3   |    501.5    |     3    |     6    |     9    |

    to

    | Raman Shift | 500.5 | 501.0 | 501.5 |
    |:-----------:|:-----:|:-----:|:-----:|
    |   Sample_1  |   1   |   2   |   3   |
    |   Sample_2  |   4   |   5   |   6   |
    |   Sample_3  |   7   |   8   |   9   |

    'col_as_header' determines if we want to use a column of the initial
    dataframe as the header of the transposed dataframe. It can be a string
    containing the name of the column to be used, an integer that defines the
    position of the column to be used or 'None', if we just want to transpose
    the dataframe.

    'header_as_col' determines if we want to use the header of the initial
    dataframe as a column of the transposed dataframe. It can be a boolean or
    a string that defines the name of the column.
    """
    if isinstance(col_as_header, str):
        transposed_df = df.set_index(col_as_header).T
    elif isinstance(col_as_header, int):
        transposed_df = df.set_index(df.columns[col_as_header]).T
    elif col_as_header is None:
        transposed_df = df.T

    if isinstance(header_as_col, str):
        transposed_df.index = transposed_df.index.set_names(header_as_col)
        transposed_df.reset_index(inplace=True)
    elif header_as_col:
        transposed_df = transposed_df.reset_index()

    return transposed_df

# def preprocess(
#         df,
#         interpolate=None,
#         smooth=None,
#         snip=None,
#         norm_peak=None,
#         norm_area=None,
#     )
