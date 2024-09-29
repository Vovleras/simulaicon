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
print("Punto 1\n")      
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
print("\nPunto 2\n")
print("Las primeras diez raíces son: \n")
mostrarRaices(raicesA)



#Punto tres

#Funcion original
def funcionOriginal(x):
    return (1/(x**2))*math.tan(x)

#Realizar la evaluación del x en la funcion dada
def evaluarFuncion(x):
    return ((-2/(x**3)) * math.tan(x) + (1/(x**2)) * (1/(math.cos(x)**2)))

#Realizar la evaluación de x en la derivada de la función
def derivada(x):
    return (6/(x**4)) * math.tan(x) + (-4/(x**3)) * (1/(math.cos(x)**2)) + (2/(x**2)) * ((1/(math.cos(x)**2)) * math.tan(x))

def metodoNewtonTres (xInicial, iteraciones, epsilon, delta):
    v = evaluarFuncion(xInicial)
    if (abs(v) < delta):
        return xInicial
    else:
        for k in range(1, iteraciones):
            xSiguiente = xInicial - (v/derivada(xInicial))

            v = evaluarFuncion(xSiguiente)

            if (abs(xSiguiente - xInicial) < epsilon or abs(v) < delta):
                return xSiguiente
            else:
                xInicial = xSiguiente

    return xInicial

#Pruebas punto tres
print("\nPunto 3\n")
puntoMinimoTres = metodoNewtonTres(1, 50, 0.00001, 0.00001)
print("El punto minimo positivo de la función es ", funcionOriginal(puntoMinimoTres), " cuando x vale ", puntoMinimoTres)

#Punto cuatro
def evaluarFuncionCuatro(x):
    return (x**3 - 5*(x**2) + 3*x -7)

#Realizar la evaluación de x en la derivada de la función
def derivadaCuatro(x):
    return (3*(x**2) - 10*x + 3)

def metodoNewtonCuatro (xInicial, iteraciones, epsilon, delta):
    v = evaluarFuncionCuatro(xInicial)
    if (abs(v) < delta):
        return xInicial
    else:
        for k in range(1, iteraciones):
            xSiguiente = xInicial - (v/derivadaCuatro(xInicial))

            v = evaluarFuncionCuatro(xSiguiente)

            if (abs(xSiguiente - xInicial) < epsilon or abs(v) < delta):
                return xSiguiente
            else:
                xInicial = xSiguiente

    return xInicial

print("\nPunto 4\n")
print("La raiz de la función cuando x0 es igual a 5 es ", metodoNewtonCuatro(5, 10, 0.00001, 0.00001))

#Punto cinco
def evaluarFuncionCinco(x):
    return (2*(x**4) + 24*(x**3) + 61*(x**2) - 16*x +1)

#Realizar la evaluación de x en la derivada de la función
def derivadaCinco(x):
    return (8*(x**3) + 72*(x**2) + 122*x - 16)

def metodoNewtonCinco (xInicial, iteraciones, epsilon, delta):
    v = evaluarFuncionCinco(xInicial)
    if (abs(v) < delta):
        return xInicial
    else:
        for k in range(1, iteraciones):
            xSiguiente = xInicial - (v/derivadaCinco(xInicial))

            v = evaluarFuncionCinco(xSiguiente)

            if (abs(xSiguiente - xInicial) < epsilon or abs(v) < delta):
                return xSiguiente
            else:
                xInicial = xSiguiente

    return xInicial

print("\nPunto 5\n")
print("Siguiendo el método de Newton, las dos raices cercanas a 0.1, son:")
print("x1 = ",metodoNewtonCinco(-0.1, 10, 0.00001, 0.00001))
print("x2 = ",metodoNewtonCinco(0.1, 10, 0.00001, 0.00001))

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

print("\nPunto 13\n")
print(newtonRaphson(1e-10))
print("\n Resultados punto 11\n")

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

#Punto 14

def newton_raphson(f, jacobian, x0, tol=1e-6, max_iter=100):
    # Aplica el método de Newton-Raphson
    x = x0
    for i in range(max_iter):
        J = jacobian(x)
        F = f(x)
        
        # Resuelve J * delta = -F
        delta = np.linalg.solve(J, -F)
        x = x + delta
        
        # Verifica la condición de convergencia
        if np.linalg.norm(delta) < tol:
            return x, i + 1
    
    raise ValueError("No convergió en el número máximo de iteraciones")

# Primer sistema de ecuaciones
def f1(x):
    return np.array([
        x[0] + np.exp(-1 * x[0]) + x[1]**3,
        x[0]**2 + 2 * x[0] * x[1] - x[1]**2 + np.tan(x[0])
    ])

def jacobian1(x):
    return np.array([
        [1 - np.exp(-1 * x[0]), 3 * x[1]**2],
        [2 * x[0] + 2 * x[1] + (1 / (np.cos(x[0])))**2, 2 * x[0] - 2 * x[1]]
    ])

# Segundo sistema de ecuaciones
def f2(x):
    return np.array([
        4 * x[1]**2 + 4 * x[1] + 52 * x[0] - 19,
        169 * x[0]**2 + 3 * x[1]**2 + 111 * x[0] - 10 * x[1] - 10
    ])

def jacobian2(x):
    return np.array([
        [52, 8 * x[1] + 4],
        [338 * x[0]+111, 6 * x[1] - 10]
    ])

# Valores iniciales para ambos sistemas
x0_1 = np.array([1.1, 1.1])
x0_2 = np.array([1.1, 1.1])

# Resolver sistemas
print("\nPunto 14\n")
try:
    raiz2, iteraciones2 = newton_raphson(f2, jacobian2, x0_2)
    print(f"Sistema A - Raíz encontrada: {raiz2}, en {iteraciones2} iteraciones")
except ValueError as e:
    print(f"Sistema A - {e}")
    
try:
    raiz1, iteraciones1 = newton_raphson(f1, jacobian1, x0_1)
    print(f"Sistema b - Raíz encontrada: {raiz1}, en {iteraciones1} iteraciones")
except ValueError as e:
    print(f"Sistema b - {e}")



