from typing import List
import math

class XYPoint():
    """
    A class representing a point in 2D space
    """

    """ The epsilon value for comparing floats. """
    EPSILON = 0.0001
    

    def __init__(self, x: float, y: float):
        """
        Constructor for Point

        :param x: The x coordinate
        :type x: float
        :param y: The y coordinate
        :type y: float
        """
        self.x = x
        self.y = y


    def __str__(self):
        """
        __str__ method override

        :return: The string representation of the Point
        :rtype: str
        """
        return f'[{self.x}, {self.y}]'


    def __repr__(self):
        """
        __repr__ method override

        :return: The string representation of the Point
        :rtype: str
        """
        return f'[{self.x}, {self.y}]'
    
    def __eq__(self, other):
        """
        __eq__ method override

        :param other: The other Point to compare to
        :type other: Point
        :return: True if the Points are equal, False otherwise
        :rtype: bool
        """
        return abs(self.x - other.x) < self.EPSILON and abs(self.y - other.y) < self.EPSILON
    
    def __ne__(self, other):
        """
        __ne__ method override

        :param other: The other Point to compare to
        :type other: Point
        """
        return not self.__eq__(other)
    
    def __hash__(self):
        """
        __hash__ method override

        :return: The hash of the Point
        :rtype: int
        """
        return hash((self.x, self.y))
    

    def __lt__(self, other):
        """
        __lt__ method override

        :param other: The other Point to compare to
        :type other: Point
        :return: True if the Point is less than the other Point, False otherwise
        :rtype: bool
        """
        return self.x < other.x or (self.x == other.x and self.y < other.y)
    

    def __le__(self, other):
        """
        __le__ method override

        :param other: The other Point to compare to
        :type other: Point
        :return: True if the Point is less than or equal to the other Point, False otherwise
        :rtype: bool
        """
        return self.__lt__(other) or self.__eq__(other)
    

    def __gt__(self, other):
        """
        __gt__ method override

        :param other: The other Point to compare to
        :type other: Point
        :return: True if the Point is greater than the other Point, False otherwise
        :rtype: bool
        """
        return self.x > other.x or (self.x == other.x and self.y > other.y)
    

    def __ge__(self, other):
        """
        __ge__ method override

        :param other: The other Point to compare to
        :type other: Point
        :return: True if the Point is greater than or equal to the other Point, False otherwise
        :rtype: bool
        """
        return self.__gt__(other) or self.__eq__(other)
    

    def __add__(self, other):
        """
        __add__ method override

        :param other: The other Point to add to
        :type other: Point
        :return: The sum of the Points
        :rtype: Point
        """
        return XYPoint(self.x + other.x, self.y + other.y)
    

    def __sub__(self, other):
        """
        __sub__ method override

        :param other: The other Point to subtract from
        :type other: Point
        :return: The difference of the Points
        :rtype: Point
        """
        return XYPoint(self.x - other.x, self.y - other.y)
    

    def __mul__(self, other):
        """
        __mul__ method override

        :param other: The other Point to multiply by
        :type other: Point
        :return: The product of the Points
        :rtype: Point
        """
        return XYPoint(self.x * other.x, self.y * other.y)
    

    def __truediv__(self, other):
        """
        __truediv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return XYPoint(self.x / other.x, self.y / other.y)
    

    def __floordiv__(self, other):
        """
        __floordiv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return XYPoint(self.x // other.x, self.y // other.y)
    

    def __mod__(self, other):
        """
        __mod__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The remainder of the Points
        :rtype: Point
        """
        return XYPoint(self.x % other.x, self.y % other.y)
    

    def __pow__(self, other):
        """
        __pow__ method override

        :param other: The other Point to raise to
        :type other: Point
        :return: The power of the Points
        :rtype: Point
        """
        return XYPoint(self.x ** other.x, self.y ** other.y)
    
    
    def __abs__(self):
        """
        __abs__ method override

        :return: The absolute value of the Point
        :rtype: Point
        """
        return XYPoint(abs(self.x), abs(self.y))
    

    def __neg__(self):
        """
        __neg__ method override

        :return: The negated Point
        :rtype: Point
        """
        return XYPoint(-self.x, -self.y)
    

    def __pos__(self):
        """
        __pos__ method override

        :return: The positive Point
        :rtype: Point
        """
        return self.__abs__()
    

    def __invert__(self):
        """
        __invert__ method override

        :return: The inverted Point
        :rtype: Point
        """
        return XYPoint(~self.x, ~self.y)
    

    def __round__(self, n=None):
        """
        __round__ method override

        :param n: The number of decimal places to round to
        :type n: int
        :return: The rounded Point
        :rtype: Point
        """
        return XYPoint(round(self.x, n), round(self.y, n))
    

    def __floor__(self):
        """
        __floor__ method override

        :return: The floored Point
        :rtype: Point
        """
        return XYPoint(math.floor(self.x), math.floor(self.y))
    

    def __ceil__(self):
        """
        __ceil__ method override

        :return: The ceilinged Point
        :rtype: Point
        """
        return XYPoint(math.ceil(self.x), math.ceil(self.y))
    

    def __trunc__(self):
        """
        __trunc__ method override

        :return: The truncated Point
        :rtype: Point
        """
        return XYPoint(math.trunc(self.x), math.trunc(self.y))
    

    def __radd__(self, other):
        """
        __radd__ method override

        :param other: The other Point to add to
        :type other: Point
        :return: The sum of the Points
        :rtype: Point
        """
        return self.__add__(other)
    
    
    def __rsub__(self, other):
        """
        __rsub__ method override

        :param other: The other Point to subtract from
        :type other: Point
        :return: The difference of the Points
        :rtype: Point
        """
        return self.__sub__(other)
    

    def __rmul__(self, other):
        """
        __rmul__ method override

        :param other: The other Point to multiply by
        :type other: Point
        :return: The product of the Points
        :rtype: Point
        """
        return self.__mul__(other)
    

    def __rtruediv__(self, other):
        """
        __rtruediv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return self.__truediv__(other)
    

    def __rfloordiv__(self, other):
        """
        __rfloordiv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return self.__floordiv__(other)
    

    def __rmod__(self, other):
        """
        __rmod__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The remainder of the Points
        :rtype: Point
        """
        return self.__mod__(other)
    

    def __rpow__(self, other):
        """
        __rpow__ method override

        :param other: The other Point to raise to
        :type other: Point
        :return: The power of the Points
        :rtype: Point
        """
        return self.__pow__(other)
    

    def __iadd__(self, other):
        """
        __iadd__ method override

        :param other: The other Point to add to
        :type other: Point
        :return: The sum of the Points
        :rtype: Point
        """
        return self.__add__(other)
    

    def __isub__(self, other):
        """
        __isub__ method override

        :param other: The other Point to subtract from
        :type other: Point
        :return: The difference of the Points
        :rtype: Point
        """
        return self.__sub__(other)
    

    def __imul__(self, other):
        """
        __imul__ method override

        :param other: The other Point to multiply by
        :type other: Point
        :return: The product of the Points
        :rtype: Point
        """
        return self.__mul__(other)
    

    def __itruediv__(self, other):
        """
        __itruediv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return self.__truediv__(other)
    

    def __ifloordiv__(self, other):
        """
        __ifloordiv__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The quotient of the Points
        :rtype: Point
        """
        return self.__floordiv__(other)
    

    def __imod__(self, other):
        """
        __imod__ method override

        :param other: The other Point to divide by
        :type other: Point
        :return: The remainder of the Points
        :rtype: Point
        """
        return self.__mod__(other)
    

    def __ipow__(self, other):
        """
        __ipow__ method override

        :param other: The other Point to raise to
        :type other: Point
        :return: The power of the Points
        :rtype: Point
        """
        return self.__pow__(other)
    

    def __ilshift__(self, other):
        """
        __ilshift__ method override

        :param other: The other Point to shift by
        :type other: Point
        :return: The shifted Point
        :rtype: Point
        """
        return XYPoint(self.x << other.x, self.y << other.y)
    

    def __irshift__(self, other):
        """
        __irshift__ method override

        :param other: The other Point to shift by
        :type other: Point
        :return: The shifted Point
        :rtype: Point
        """
        return XYPoint(self.x >> other.x, self.y >> other.y)


def euclidean_distance(p1: XYPoint, p2: XYPoint) -> float:
    """
    Calculates the Euclidean distance between two points

    :param p1: The first point
    :type p1: Point
    :param p2: The second point
    :type p2: Point
    :return: The Euclidean distance between the points
    :rtype: float
    """
    return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

def manhattan_distance(p1: XYPoint, p2: XYPoint) -> float:
    """
    Calculates the Manhattan distance between two points

    :param p1: The first point
    :type p1: Point
    :param p2: The second point
    :type p2: Point
    :return: The Manhattan distance between the points
    :rtype: float
    """
    return abs(p1.x - p2.x) + abs(p1.y - p2.y)

def cosine_similarity(p1: XYPoint, p2: XYPoint) -> float:
    """
    Calculates the cosine similarity between two points

    :param p1: The first point
    :type p1: Point
    :param p2: The second point
    :type p2: Point
    :return: The cosine similarity between the points
    :rtype: float
    """
    return (p1.x * p2.x + p1.y * p2.y) / (math.sqrt(p1.x**2 + p1.y**2) * math.sqrt(p2.x**2 + p2.y**2))