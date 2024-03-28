import numpy as np
import pandas as pd
from fibermat import Mat


def equal(df1, df2):
    """
    Test the equality between two DataFrames.

    Parameters
    ----------
    df1 : pandas.DataFrame
        First DataFrame.
    df2 : pandas.DataFrame
        Second DataFrame.

    """
    assert np.allclose(df1.values, df2.values)
    assert np.all(df1.index == df2.index)
    assert np.all(df1.columns == df2.columns)
    assert np.all(df1.dtypes == df2.dtypes)
    assert np.all(df1.attrs == df2.attrs)


################################################################################
# Tests
################################################################################

def test_emptyMat():
    """
    Test the initialization of an empty `Mat` object.

    """
    # Optional
    mat = pd.DataFrame(data=[], index=[], columns=[*"lbhxyzuvwGE"], dtype=float)
    mat.attrs = dict(n=0, size=50.)
    equal(Mat(), mat)

    assert Mat().check()
    assert Mat(0).check()
    assert Mat(mat).check()


def test_Mat():
    """
    Test the initialization of a `Mat` object.

    """
    # Optional
    data = np.array([
        [ 25., 1., 1.,  2.4406752 ,  14.5862519 , -19.08627871, 0.73869079, -0.67404445, -0., 1., np.inf],
        [ 25., 1., 1., 10.75946832,   1.44474599, -17.83233563, 0.65127679,  0.75884026, -0., 1., np.inf],
        [ 25., 1., 1.,  5.1381688 ,   3.40222805,  -4.266903  , 0.99052639, -0.1373225 ,  0., 1., np.inf],
        [ 25., 1., 1.,  2.24415915,  21.27983191,  -1.92603189, 0.97697819,  0.21333921, -0., 1., np.inf],
        [ 25., 1., 1., -3.81726003, -21.44819709,   1.09241609, 0.05899562, -0.99825824,  0., 1., np.inf],
        [ 25., 1., 1.,  7.29470565, -20.64353501,   6.99605107, 0.93248535,  0.3612078 ,  0., 1., np.inf],
        [ 25., 1., 1., -3.12063944, -23.98908013,  14.02645881, 0.93863017,  0.34492521, -0., 1., np.inf],
        [ 25., 1., 1., 19.58865004,  16.63099228,  14.95792821, 0.93327912,  0.35915188, -0., 1., np.inf],
        [ 25., 1., 1., 23.18313803,  13.90783755,  22.23344585, 0.17580222,  0.98442551, -0., 1., np.inf],
        [ 25., 1., 1., -5.82792406,  18.50060741,  23.93091711, 0.84124994,  0.54064641, -0., 1., np.inf]
    ])
    index = np.arange(10)

    mat = pd.DataFrame(data=data, index=index, columns=[*"lbhxyzuvwGE"])
    mat.attrs = dict(n=10, size=50.)
    equal(Mat(10), mat)

    assert Mat(mat).check()


################################################################################
# Main
################################################################################

if __name__ == '__main__':

    # Empy mat
    test_emptyMat()

    # Mat initialization
    test_Mat()
