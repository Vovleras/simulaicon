import math
import numpy as np

#Punto 1: 
# Encuentre las raices mas cercanas a 4.5 y 7.7  al resolver la ecuacion x = tan (x)

#Funcion que evalua un numero x en la expresion: 'tan (x) - x'
def funcion(x):
    tan = math.tan(x)
    return tan - x

#Funcion que calcula la primera derivada de 'tan(x) -x'
def primeraDerivada(x):
    secante = 1/math.cos(x)
    derivada = pow(secante,2)
    return derivada - 1

#Funcion que calcula la raiz de una funcion mediante el metodo de Newton
def metodNewton(x0, it, cifras, error):
    v = funcion(x0)
    if (abs(v)<error):
        return x0
    else:   
        for i in range(1, it+1):
            derivada = primeraDerivada(x0)
            x1 = x0 - (v/derivada)
            v = funcion(x1)
            comp = abs(x1-x0)
            
            if comp < cifras or abs(v) < error:
                
                return x1
            else:
                x0=x1
    
    

#Resultado:         
print("la raiz mas cercana a 4.5 es:", metodNewton(4.5,100,1e-10, 1e-10))
print("la raiz mas cercana a 7.7 es:", metodNewton(7.7,100,1e-10, 1e-10))



#Punto 2: 
# Encuentre las primeras diez raices de la ecuacion x = tan (x)

#Funcion que calcula las primeras diez raices de x = tan(x)
def diezRaices():
    raices = []
    n = 1
    while len(raices) < 10:
        x0 = (math.pi-math.e + 1.1) + n*math.pi
        raiz= metodNewton(x0,100,1e-10, 1e-10)
        raices.append(raiz)
        n += 1
    return raices

#Funcion que muestra las raices encontradas con el metodo de Newton
def mostrarRaices(raices):
    for raiz in raices:
        print(raiz,"\n")
        
#Resultado:         
raicesA = diezRaices()
print("Las primeras diez raíces son: \n")
mostrarRaices(raicesA)

#Punto 13:
#Realizar cinco iteraciones del metodo de Newton en el sistema dado

#Funcion que calcula la funcion F1 dado los parametros 'x y 'y
def evaluarF1(x, y):
    result= 1 + pow(x, 2) - pow(y, 2) + (pow(np.e, x) * np.cos(y))

    return -result

#Funcion que calcula la funcion F2 dado los parametros 'x y 'y
def evaluarF2(x, y):
    result = (2 * x * y) + (pow(np.e, x) * np.sin(y))
    return -result

#Funcion que calcula la matriz jacobiana F1
def jacobiano(x, y):
    df1_dx = 2 * x + np.exp(x) * np.cos(y)  
    df1_dy = -2 * y + (-np.exp(x) * np.sin(y)) 
    
    df2_dx = 2*y + np.exp(x) * np.sin(y)  
    df2_dy = 2 * x + np.exp(x) * np.cos(y)  
    
    return np.array([[df1_dx, df1_dy], [df2_dx, df2_dy]])
    
#Funcion que calcula cinco iteraciones  del metodo de newton de dos ecuaciones no lineales
def newtonRaphson(error):
    
    valorInicial = np.array([-1, 4],dtype=float)
    for _ in range(5):
        
        A = jacobiano(valorInicial[0], valorInicial[1])
        b = np.array([evaluarF1(valorInicial[0],valorInicial[1] ), 
                      evaluarF2(valorInicial[0], valorInicial[1])])
        
        solucion = np.linalg.solve(A, b)
        norma = np.linalg.norm(valorInicial-solucion)
        
        if norma<error:
            return valorInicial
        else:
            valorInicial += solucion
            print(f"Iteracion: {_ + 1}; x= {valorInicial[0]}, y= {valorInicial[1]}")
        
    return valorInicial

print(newtonRaphson(1e-10))

#¿El problema tiene un comportamiento numerico igual al del problema 11?
#Respuesta:
#Se puede observar que  al aplicar el metodo de Newton en el punto 11, la parte real de 'Zn 
# se comporta igual que 'x en  el punto trece. Asi mismo, la parte imaginaria de 'Zn tiene un 
# compartamiento similar que 'y. Por lo que se puede concluir que su comportamiento es parecido.


#punto 11:

def f(z):
    return 1 + z**2 + np.exp(z)

def fDerivada(z):
    return 2 * z + np.exp(z)
def metodo():
    z0 = -1 + 4j
    Zn = z0    
    resultados = []
    for i in range(5):
        fz= f(Zn)
        derivada= fDerivada(Zn)
        Zn = Zn - fz / derivada
        
        resultados.append(Zn)
    for i, z in enumerate(resultados):
        print(f"Iteracion {i}: z = {z}")

metodo()

