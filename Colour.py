class Colour: 
    # red, green and blue values
    def __init__(self, values):
        self.r = float(values[0])
        self.g = float(values[1])
        self.b = float(values[2])

    def __str__(self): 
        return f'Red: {self.r}, Green: {self.g}, Blue: {self.b}'
    
    def __repr__(self): 
        return str(self)

    def __eq__(self, other):
        return self.r == other.r and self.g == other.g and self.b == other.b