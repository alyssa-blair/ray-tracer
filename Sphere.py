from Colour import *
from Coordinate import *
from Effect import *
from Scale import *

class Sphere:
    # a sphere in the scene with a name, coordinates, 
    # scale, colour, and effects
    def __init__(self,values):
        self.name = str(values[0])
        self.coord = Coordinate(values[1:4])
        self.scale = Scale(values[4:7])
        self.col = Colour(values[7:10])
        self.effect = Effect(values[10:15])

    def __str__(self):
        return self.name
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other):
        return self.coord == other.coord and self.scale == other.scale and self.col == other.col and self.effect == other.effect
