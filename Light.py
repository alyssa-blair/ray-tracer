from Colour import *
from Coordinate import *

class Light:
    # a light in the scene with name, coordinates, and colour
    def __init__(self,values):
        self.name = str(values[0])
        self.coord = Coordinate(values[1:4])
        self.col = Colour(values[4:7])

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.coord == other.coord and self.col == other.col


