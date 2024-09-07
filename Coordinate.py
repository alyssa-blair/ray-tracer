class Coordinate: 
    # x, y, and z coordinates
    def __init__(self, values): 
        self.x = float(values[0])
        self.y = float(values[1])
        self.z = float(values[2])

    def __str__(self): 
        return f'X-Coordinate: {self.x}, Y-Coordinate: {self.y}, Z-Coordinate: {self.z}'
    
    def __repr__(self):
        return str(self)
    
    def __eq__(self, other): 
        return self.x == other.x and self.y == other.y and self.z == other.z