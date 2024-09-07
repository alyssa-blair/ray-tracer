import sys
import math
import numpy as np
from Sphere import *
from Light import *

spheres = []
lights = []
values = {}
SMALL_NUMBER = 1.e-8


def main(filename):
    with open(filename, "r") as fp:
        line = True
        while line:
            line = fp.readline()
            handle_line(line)

    # gets the resolution of the image
    width = int(values.get("RES")[0])
    height = int(values.get("RES")[1])
    pixels = [0] * 3 * height * width

    outputFile = values.get("OUTPUT")

    # calculate color for each pixel
    k = 0
    for i in range(0, width):
        for j in range(0, height):
            ray = compute_ray(j, i)
            color, count = raytrace(ray, 0)

            # ensure that no colour is over 1
            color = [min(color[0],1), min(color[1],1), min(color[2],1)]

            # convert color code and place it in pixel
            pixels[k] = color[0] * 255
            pixels[k + 1] = color[1] * 255
            pixels[k + 2] = color[2] * 255
            k += 3
    

    save_imageP3(width,height,outputFile[0],pixels)


def handle_line(line: str):
    """
    takes the current line and breaks it up into a token and values
    """
    tokenArr = []
    tokenArr = line.split()
    if len(tokenArr) != 0:
        token = tokenArr[0]
        tokenArr = tokenArr[1:]

        assignVar(token, tokenArr)


def assignVar(token, tokenArr):
    """
    takes the broken up line and adds it to the dictionary
    """
    if token == "LIGHT":
        light = Light(tokenArr)
        lights.append(light)
    elif token == "SPHERE":
        sphere = Sphere(tokenArr)
        spheres.append(sphere)
    else:
        values.update({token: tokenArr})
    
    

def save_imageP3(width, height, filename, pixels):
    # saves image to the ppm file
    maxVal = 255
    print("Saving image", filename, ":", width, "x", height)
    fp = open(filename, "w")
    if not fp:
        print("failed to open output file")
        return
    
    fp.write("P3\n")
    string = "{w} {h}\n".format(w = width, h = height)
    fp.write(string)

    maxStr = "{max}\n".format(max = maxVal)
    fp.write(maxStr)
    k = 0
    for i in range(0, height):
        for j in range(0, width):
            pixel = " {p1} {p2} {p3}".format(p1 = int(pixels[k]), p2 = int(pixels[k+1]), p3 = int(pixels[k+2]))
            fp.write(pixel)
            k += 3
        fp.write("\n")
    
    fp.close()


def raytrace(ray, count):
    """
    traces the route of the given ray, calculating the color
    """
    if (count > 2):
        # if we have reached max recursion, return black
        return [0,0,0], count

    sphere, P = compute_close_intersection(ray)

    # if there is no intersection, return background
    if (len(P) == 0 and count == 0):
        background = values.get("BACK")
        return [float(background[0]), float(background[1]), float(background[2])], count

    elif (len(P) == 0):
        # if there is no intersection and this is not the initial ray, return black
        return [0,0,0], count

    color_local = [0,0,0]

    for light in lights:
        # for each light, check the color of diffuse and specular
        newColor = shadow_ray(light, P, sphere)

        color_local[0] += newColor[0]
        color_local[1] += newColor[1]
        color_local[2] += newColor[2]


    # reflection and ambience variable
    kr = sphere.effect.kr
    ka = sphere.effect.ka

    # if the surface is reflective, raytrace the reflected ray
    reflected_ray = compute_reflected_ray(ray, P, sphere)
    color_reflect, count = raytrace(reflected_ray, count + 1)

    ambient = values.get("AMBIENT")
    
    # add all the colors together
    color = [0,0,0]
    color[0] = ka * float(ambient[0]) * sphere.col.r + color_local[0] + kr * color_reflect[0]
    color[1] = ka * float(ambient[1]) * sphere.col.g + color_local[1] + kr * color_reflect[1]
    color[2] = ka * float(ambient[2]) * sphere.col.b + color_local[2] + kr * color_reflect[2]

    return color, count

def normalize(vec):
    # normalizes the given vector
    for i in range(0,len(vec)):
        vec[i] = vec[i]/np.linalg.norm(vec)
    return vec


def compute_reflected_ray(ray, P, sphere):
    """
    Computes the reflected ray given an incident ray and a point
    """
    c = ray[1]
    
    # compute the normal of the sphere at point P
    N = [P[0] - sphere.coord.x, P[1] - sphere.coord.y, P[2] - sphere.coord.z, 0]
    
    # compute the inverse transformed model matrix
    m = [[sphere.scale.sx,0,0,sphere.coord.x],
         [0,sphere.scale.sy,0,sphere.coord.y],
         [0,0,sphere.scale.sz,sphere.coord.z],
         [0,0,0,1]]
    minv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    invert_matrix(m, minv)
    transpose(minv)

    # multiply the normal by this
    N = list(np.matmul(minv, N))
    
    # normalize the vectors
    N = normalize(N[:3])
    c = normalize(c[:3])
    
    # compute v
    v = [0,0,0]
    v[0] = -2 * (np.dot(N,c)) * N[0] + c[0]
    v[1] = -2 * (np.dot(N,c)) * N[1] + c[1]
    v[2] = -2 * (np.dot(N,c)) * N[2] + c[2]

    line = [P, v]
    return line


def shadow_ray(light, P, sphere):
    """
    checks to see if there is an object blocking the light at this point
    if there is, return black. if not, compute the color value at this point
    """
    # create a vector from the point to the light
    curLight = [light.coord.x-P[0], light.coord.y-P[1],light.coord.z-P[2]]
    line = [P, curLight]

    # check if there is an object between the point and light
    s, point = compute_close_intersection(line)
    if (len(point) != 0):
        # if there is an intersection, return black
        return [0,0,0]

    # diffuse and specular constants
    kd = sphere.effect.kd
    ks = sphere.effect.ks
    
    # get inverse transposed model matrix 
    m = [[sphere.scale.sx,0,0,sphere.coord.x],
         [0,sphere.scale.sy,0,sphere.coord.y],
         [0,0,sphere.scale.sz,sphere.coord.z],
         [0,0,0,1]]
    minv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    invert_matrix(m, minv)
    transpose(minv)

    # Compute the unit normal and light vector
    N = [P[0] - sphere.coord.x, P[1] - sphere.coord.y, P[2] - sphere.coord.z,0]
    L = [light.coord.x - P[0], light.coord.y - P[1], light.coord.z - P[2]]
    N = list(np.matmul(minv, N))[:3]
    L = normalize(L)
    N = normalize(N)

    # compute R and V
    R = [0,0,0]
    R[0] = 2 * (np.dot(L,N)) * N[0] - L[0]
    R[1] = 2 * (np.dot(L,N)) * N[1] - L[1]
    R[2] = 2 * (np.dot(L,N)) * N[2] - L[2]

    
    V = [-P[0], -P[1], -P[2]]

    # normalize R and V
    R = normalize(R)
    V = normalize(V)

    # get the specular exponent of the sphere
    n = sphere.effect.n

    newColor = [0,0,0]
    diffuse = [0,0,0]
    specular = [0,0,0]

    # calculate diffuse
    diffuse[0] = kd * light.col.r * max(0,np.dot(N,L)) * sphere.col.r
    diffuse[1] = kd * light.col.g * max(0,np.dot(N,L)) * sphere.col.g
    diffuse[2] = kd * light.col.b * max(0,np.dot(N,L)) * sphere.col.b

    # calculate specular
    specular[0] = ks * pow(min(max(0,np.dot(R, V)),1),n) * light.col.r
    specular[1] = ks * pow(min(max(0,np.dot(R, V)),1),n) * light.col.g
    specular[2] = ks * pow(min(max(0,np.dot(R, V)),1),n) * light.col.b

    # add diffuse and specular together
    newColor[0] = diffuse[0] + specular[0]
    newColor[1] = diffuse[1] + specular[1]
    newColor[2] = diffuse[2] + specular[2]
   
    
    return newColor
    


def compute_ray(u, v):
    # computes the ray equation at the current pixel
    left = float(values.get("LEFT")[0])
    right = float(values.get("RIGHT")[0])
    top = float(values.get("TOP")[0])
    bottom = float(values.get("BOTTOM")[0])

    width = int(values.get("RES")[0])
    height = int(values.get("RES")[1])

    # compute the pixel location
    uPix = left + right * 2 * u / width
    vPix = -(bottom + top * 2 * v / height)

    n = values.get("NEAR")[0]

    # create the ray
    ray = [uPix, vPix, -int(n)]
    line = [[0,0,0], ray]

    return line


def compute_close_intersection(ray):
    # loops through objects and finds the one with the closest intersection
    # if there is no intersection, return empty array for closeSphere and closePoint
    t = math.inf
    closeSphere = []
    closePoint = []

    for sphere in spheres:        
        # create the transformation matrix
        m = [[sphere.scale.sx,0,0,sphere.coord.x],
             [0,sphere.scale.sy,0,sphere.coord.y],
             [0,0,sphere.scale.sz,sphere.coord.z],
             [0,0,0,1]]
        

        t1, point = compute_current_intersection(ray, m)
        if (ray[0] != [0,0,0]):
            # if the ray did not start at the eye so intersection can be closer
            t1 += 1
        if (t1 < t and t1 > 1.000001):
            # update the closest intersection point and sphere
            t = t1
            closeSphere = sphere
            closePoint = point

    return closeSphere, closePoint

def compute_current_intersection(ray, m):
    # finds if there is an intersection between a ray and an object and returns the lowest value of t
    s = ray[0]
    if (len(s) < 4):
        s.append(1)

    c = ray[1]
    if (len(c) < 4):
        c.append(0)

    # find the transformation matrix
    minv = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    invert_matrix(m, minv)

    # multiply the ray vectors by the transformation matrix
    c2 = list(np.matmul(minv, c))
    s2 = list(np.matmul(minv, s))

    # calculate the polynomial values
    a = c2[0] ** 2 + c2[1] ** 2 + c2[2] ** 2
    b = s2[0] * c2[0] + s2[1] * c2[1] + s2[2] * c2[2]
    d = s2[0] ** 2 + s2[1] ** 2 + s2[2] ** 2 - 1

    sol = (b ** 2) - (a * d)

    if (sol < 0):
        # no intersections
        t1 = math.inf
        return t1, []
    elif (sol == 0.0):
        # one intersection
        t1 = (-b)/(a)
    else:
        # two intersections
        t1 = (-b + math.sqrt(sol))/(a)
        t2 = (-b - math.sqrt(sol))/(a)
        if (t2 < t1 and t2 > 0.000001):
            t1 = t2

    # get the lowest point
    xPoint = s[0] + c[0] * t1
    yPoint = s[1] + c[1] * t1
    zPoint = s[2] + c[2] * t1
    point = [xPoint, yPoint, zPoint]

    return t1, point


def transpose(m):
    # transpose the given matrix
    a = m[0][1]
    b = m[0][2]
    c = m[0][3]

    d = m[1][0]
    e = m[1][2]
    f = m[1][3]

    g = m[2][0]
    h = m[2][1]
    i = m[2][3]

    j = m[3][0]
    k = m[3][1]
    l = m[3][2]

    m[0][1] = d
    m[0][2] = g
    m[0][3] = j

    m[1][0] = a
    m[1][2] = h
    m[1][3] = k

    m[2][0] = b
    m[2][1] = e
    m[2][3] = l

    m[3][0] = c
    m[3][1] = f
    m[3][2] = i

    return m



def invert_matrix (A, Ainv):
    # invert the given matrix
    adjoint(A, Ainv)
    det = det4x4(A)

    if ( abs(det) < SMALL_NUMBER):
        print("invert_matrix: matrix is singular!")
        return

    for i in range(4):
        for j in range(4):
            Ainv[i][j] = Ainv[i][j] / det



def adjoint(into, out):
    a1 = into[0][0]
    b1 = into[0][1]
    c1 = into[0][2] 
    d1 = into[0][3]

    a2 = into[1][0] 
    b2 = into[1][1] 
    c2 = into[1][2] 
    d2 = into[1][3]

    a3 = into[2][0] 
    b3 = into[2][1]
    c3 = into[2][2] 
    d3 = into[2][3]

    a4 = into[3][0] 
    b4 = into[3][1] 
    c4 = into[3][2] 
    d4 = into[3][3]

    out[0][0]  =   det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4)
    out[1][0]  = - det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
    out[2][0]  =   det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
    out[3][0]  = - det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4)
            
    out[0][1]  = - det3x3( b1, b3, b4, c1, c3, c4, d1, d3, d4)
    out[1][1]  =   det3x3( a1, a3, a4, c1, c3, c4, d1, d3, d4)
    out[2][1]  = - det3x3( a1, a3, a4, b1, b3, b4, d1, d3, d4)
    out[3][1]  =   det3x3( a1, a3, a4, b1, b3, b4, c1, c3, c4)
            
    out[0][2]  =   det3x3( b1, b2, b4, c1, c2, c4, d1, d2, d4)
    out[1][2]  = - det3x3( a1, a2, a4, c1, c2, c4, d1, d2, d4)
    out[2][2]  =   det3x3( a1, a2, a4, b1, b2, b4, d1, d2, d4)
    out[3][2]  = - det3x3( a1, a2, a4, b1, b2, b4, c1, c2, c4)
            
    out[0][3]  = - det3x3( b1, b2, b3, c1, c2, c3, d1, d2, d3)
    out[1][3]  =   det3x3( a1, a2, a3, c1, c2, c3, d1, d2, d3)
    out[2][3]  = - det3x3( a1, a2, a3, b1, b2, b3, d1, d2, d3)
    out[3][3]  =   det3x3( a1, a2, a3, b1, b2, b3, c1, c2, c3)



def det3x3( a1,  a2,  a3,  b1,  b2,  b3,  c1, c2,  c3 ):
    ans = a1 * det2x2( b2, b3, c2, c3 ) - b1 * det2x2( a2, a3, c2, c3 ) + c1 * det2x2( a2, a3, b2, b3 )
    return ans


def det2x2(  a,  b,  c,  d):
    ans = a * d - b * c
    return ans


def det4x4(m): 

    a1 = m[0][0] 
    b1 = m[0][1] 
    c1 = m[0][2] 
    d1 = m[0][3]

    a2 = m[1][0] 
    b2 = m[1][1] 
    c2 = m[1][2] 
    d2 = m[1][3]

    a3 = m[2][0] 
    b3 = m[2][1] 
    c3 = m[2][2] 
    d3 = m[2][3]

    a4 = m[3][0] 
    b4 = m[3][1] 
    c4 = m[3][2] 
    d4 = m[3][3]

    ans = a1 * det3x3( b2, b3, b4, c2, c3, c4, d2, d3, d4) 
    - b1 * det3x3( a2, a3, a4, c2, c3, c4, d2, d3, d4)
    + c1 * det3x3( a2, a3, a4, b2, b3, b4, d2, d3, d4)
    - d1 * det3x3( a2, a3, a4, b2, b3, b4, c2, c3, c4)
    
    return ans

  
main(sys.argv[1])

