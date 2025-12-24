
import math
from .utils import is_listlike

def expand(source, n=1):
    '''Generator that returns each possible expanded sequence from source

       source    a list-like of elements and lists

       n         number of items to return from each sub-list. If negative,
                 then the number of items to *subtract* from each returned
                 sub list

       The number of expanded sequences returned the sum of the binomial
       coeffient of each element: see docs for subset_sequence

       Examples and results:

       >>> for elem in expand_sequence([1, 2, 3]):
       >>>   print(elem)
       [1, 2, 3]

       >>> for elem in expand_sequence([1, [2, 3, 4], [5, 6]]):
       >>>   print(elem)
       [1, 2, 5]
       [1, 2, 6]
       [1, 3, 5]
       [1, 3, 6]
       [1, 4, 5]
       [1, 4, 6]

       >>> for elem in expand_sequence([1, [2, 3, 4], [5, 6], n=2):
       >>>   print(elem)
       [1, 2, 3, 5, 6]
       [1, 2, 4, 5, 6]
       [1, 3, 4, 5, 6]

       >>> for elem in expand_sequence([1, [2, 3, 4], [5, 6]], n=-1):
       >>>   print(elem)
       [1, 2, 3, 5]
       [1, 2, 3, 6]
       [1, 2, 4, 5]
       [1, 2, 4, 6]
       [1, 3, 4, 5]
       [1, 3, 4, 6]

    '''

    def chunk(elems, z, n):
        if len(elems) == 0:
            yield z
        else:
            t = elems[0]
            if is_listlike(t):
                sz = n if n > 0 else len(t)+n

                if n != 0 and sz > 0 and sz <= len(t):
                    for sub in subset(t, sz):
                        yield from chunk(elems[1:], z=z+sub, n=n)
                else:
                    yield from chunk(elems[1:], z=z, n=n)

            else:
                yield from chunk(elems[1:], z=z+elems[:1], n=n)

    yield from chunk(source, [], n=n)

def trim(elems):
    '''Trim empty elements

       NB: deprecated
    '''

    return list(filter(lambda x: is_listlike(x)==False or len(x) > 0, elems))

def reorder(source):
    '''Generator that returns unique orders of the elements in source

       The number of orders returned is equal to sz! where sz=len(source)

       Example:
       >>> for elem in reorder_sequence([1, 2, 3]):
       >>>   print(elem)
       [1, 2, 3]
       [1, 3, 2]
       [2, 1, 3]
       [2, 3, 1]
       [3, 1, 2]
       [3, 2, 1]
    '''

    def chunk(elems, z):
        if len(elems) == 1:
            yield z + elems
        elif len(elems) == 2:
            yield z + elems
            yield z + elems[::-1]
        else:
            for i in range(len(elems)):
                yield from chunk(elems[:i] + elems[i+1:], z + elems[i:i+1])

    yield from chunk(source, [])

def subset(source, n=1):
    '''Generator that returns discrete subsets of a given size

       n is the subset size and must be in the range 0..len(source) inclusive

       The binomial coefficient formula calculates the resulting number of subsets:
         sz = len(source)
         sz! / (n! * (sz-n)!)
    '''

    def chunk(elems, n, z):

        if n == 0:
            yield z

        for i in range(len(elems)):
            yield from chunk(elems[i+1:], n-1, z+elems[i:i+1])

    yield from chunk(source, n, [])

def bc(source, n=1):
    '''Return the binomial coefficient of an array. This is the number
       of discrete subsets of the array of size n (which should be positive)

       Pass expanded=True to calculate the number of subsets returned by the
       expand function. In this case n can be negative
    '''

    sz = len(source)
    if n > 0 and n <= sz:
        return int( math.factorial(sz) / (math.factorial(n) * math.factorial(sz - n)) )

    return 0

def expand_len(source, n=1):
	'''Return the number of expanded arrays that expand will generate. n can be negative 
	'''

	sz = 1
	for elem in source:
		if is_listlike(elem):
			t = bc(elem, n if n > 0 else len(elem)+n)
			if t > 0:
				sz *= t

	return sz


def expand_width(source, n=1):
	'''Return the size o the arrays that expand will generate. n can be negative
	'''

	sz = 0
	for elem in source:
		if is_listlike(elem):
			if n != 0:
				t = n if n > 0 else len(elem)+n
				if t > 0 and t < len(elem):
					sz += t

		else:
			sz += 1

	return sz