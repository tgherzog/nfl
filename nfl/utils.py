
from datetime import datetime

def safeInt(i):
    '''Attempt to convert strings to integers without raising exceptions
    '''
    try:
        i = int(i)
    except ValueError:
        pass

    return i

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