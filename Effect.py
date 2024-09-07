class Effect:
    # different lighting effects and textures
    def __init__(self, values):
        self.ka = float(values[0])
        self.kd = float(values[1])
        self.ks = float(values[2])
        self.kr = float(values[3])
        self.n = float(values[4])

    def __str__(self): 
        return f'Ambience: {self.ka}, Diffuse: {self.kd}, Specular: {self.ks}, Reflection: {self.kr}, Specular Exponent: {self.n}'
    
    def __repr__(self):
        return str(self)

    def __eq__(self, other):
        return self.ka == other.ka and self.kd == other.kd and self.ks == other.ks and self.kr == other.kr and self.n == other.n
