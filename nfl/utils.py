
from datetime import datetime

def safeInt(i):
    '''Attempt to convert strings to integers without raising exceptions
    '''
    try:
        i = int(i)
    except ValueError:
        pass

    return i

def is_listlike(ref):
    '''Return if object is iterable and not a string
    '''
    return hasattr(ref, '__iter__') and not isinstance(ref, str)

def to_seconds(s):
    '''Convert "mm:ss" to seconds
    '''

    (mins,secs) = map(lambda x: int(x), s.split(':'))
    return mins*60 + secs

def to_int_list(s, sep='-'):
    '''Convert hyphenated stats to a list of integers.  "3-8-35" -> [3, 8, 35]
    '''

    return list(map(lambda x: int(x), s.split(sep)))

def current_season():
    '''Return estimated current season based on the clock
    '''
    dt = datetime.now()
    return dt.year if dt.month >= 4 else dt.year-1

vmap_ = {1: 'win', 0: 'loss', -1: 'tie'}
def vmap(v):
    return vmap_.get(v, v)

ivmap_ = {'win': 1, 'loss': 0, 'tie': -1}
def ivmap(v):
    return ivmap_.get(v, v)


def set_dtypes(df, fields):
    '''Set dtypes for DataFrame columns

       fields is a dict of type:fieldname, where fieldname
       can be a string or an array of strings.

       Example:

       set_dtypes(df, {'int16': 'age', 'float32': ['salary', 'income']})
    '''

    for k,v in fields.items():
        if type(v) is not list:
            v = [v]

        for col in v:
            df[col] = df[col].astype(k)

def stack_copy(dst, level, src, key=None):
    '''Copies all columns from src into dst[level].

       level is a first level label of a two-level MultiIndex.
       All columns in src must exist within dst[level]

       src and dst should have similar but not necessarily
       identical indexes

       key is an optional row indexer into dst, in which
       case src should be a Series, with all rows existing
       within dst[level]

       For exanple, given a dst like this:

            overall                division
                win loss  tie  pct      win loss  tie  pct
        CHI     NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN
        PHI     NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN
        TB      NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN

       and a src like this:

              win  loss  tie
        team
        CHI     1     3    0
        TB      2     2    0
        WSH     2     1    0
        DET     1     3    0

       stack_copy(dst, 'overall', src) produces this:

            overall                division
                win loss  tie  pct      win loss  tie  pct
        CHI     1.0  3.0  0.0  NaN      NaN  NaN  NaN  NaN
        PHI     NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN
        TB      2.0  2.0  0.0  NaN      NaN  NaN  NaN  NaN

        and stack_copy(dst, 'overall', src['CHI'], 'TB') produces this:

            overall                division
                win loss  tie  pct      win loss  tie  pct
        CHI     NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN
        PHI     NaN  NaN  NaN  NaN      NaN  NaN  NaN  NaN
        TB      1.0  3.0  0.0  NaN      NaN  NaN  NaN  NaN
    '''

    if key:
        for k,v in src.items():
            dst.loc[key, (level,k)] = v

        return

    for n in src.columns:
        dst[(level,n)] = src[n]
