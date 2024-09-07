
class Scale:
    # Scale of the x, y, and z planes 
    def __init__(self, values): 
        self.sx = float(values[0])
        self.sy = float(values[1])
        self.sz = float(values[2])

    def __str__(self):
        return f'X-Scale: {self.sx}, Y-Scale: {self.sy}, Z-Scale: {self.sz}'
    
    def __repr__(self): 
        return str(self)

    def __eq__(self, other):
        return self.sx == other.sx and self.sy == other.sy and self.sz == other.sz