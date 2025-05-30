import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np
import matplotlib.pyplot as plt
import re

# --- GaussJordan ---
def GJ(matriz, valores):
    for f in range(len(matriz)-1):
        if f <= (len(matriz[f])-1):
            for c in range(len(matriz[f])):
                matriz[f][c] /= matriz[f][f]
            valores[f] /= matriz[f][f]
        else : continue

        for i in range(f+1, len(matriz)):
            if matriz[i][f] < 0:
                for j in range(len(matriz[i])-1, -1, -1):
                    matriz[i][j] += (-1*matriz[i][f]*matriz[f][j])
                valores[i] += (-1*valores[i]*matriz[f][j])
            elif matriz[i][f] > 0:
                for j in range(len(matriz[i])-1, -1, -1):
                    matriz[i][j] -= (matriz[i][f]*matriz[f][j])
                valores[i] -= (valores[i]*matriz[f][j])

    for f in range(len(matriz)-1, -1, -1):
        for c in range(len(matriz[f])):
            if matriz[f][c] == 1:
                s_piv = matriz[f-1][c]
                for i in range(len(matriz[f])):
                    mult = matriz[f][i]*s_piv
                    v_mult = valores[f]*s_piv
                    matriz[f-1][i] -= mult
                    valores[f] -= v_mult

    return(matriz, valores) 

def menor(m, i, j):
    """Función auxiliar para calcular el menor de una matriz"""
    return [fila[:j] + fila[j+1:] for fila in (m[:i] + m[i+1:])]

def calcular_inversa(matriz):
    """Función que implementa el método de Gauss-Jordan para calcular la inversa"""
    # Verificar si es matriz cuadrada
    if len(matriz) != len(matriz[0]):
        return "Error: La matriz debe ser cuadrada para calcular su inversa"
    
    n = len(matriz)
    
    # Crear matriz identidad
    identidad = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    
    # Matriz aumentada (original | identidad)
    aumentada = [fila + identidad[i] for i, fila in enumerate(matriz)]
    
    # Proceso de eliminación Gauss-Jordan
    for f in range(n):
        # Pivoteo parcial si el elemento diagonal es cero
        if aumentada[f][f] == 0:
            for k in range(f+1, n):
                if aumentada[k][f] != 0:
                    aumentada[f], aumentada[k] = aumentada[k], aumentada[f]
                    break
            else:
                return "La matriz no es invertible (determinante cero)"
        
        # Normalizar fila f
        pivote = aumentada[f][f]
        for j in range(2*n):
            aumentada[f][j] /= pivote
        
        # Eliminar en otras filas
        for i in range(n):
            if i != f and aumentada[i][f] != 0:
                factor = aumentada[i][f]
                for j in range(2*n):
                    aumentada[i][j] -= factor * aumentada[f][j]
    
    # Extraer la inversa (parte derecha de la matriz aumentada)
    inversa = [fila[n:] for fila in aumentada]
    
    return inversa


# --- Funciones para Operaciones con Vectores ---
def producto_escalar():
    num_elementos = simpledialog.askinteger("Producto Escalar", "¿De cuántos elementos son tus vectores?")
    if not num_elementos:
        return
    
    a, b = [], []
    for i in range(num_elementos):
        elemento = simpledialog.askfloat("Vector A", f"Ingrese el elemento {i+1} del vector A:")
        if elemento is None: return
        a.append(elemento)
        
        elemento = simpledialog.askfloat("Vector B", f"Ingrese el elemento {i+1} del vector B:")
        if elemento is None: return
        b.append(elemento)
    
    resultado = np.dot(a, b)
    messagebox.showinfo("Resultado", f"Vector A: {a}\nVector B: {b}\n\nProducto escalar: {resultado}")

def producto_vectorial():
    messagebox.showinfo("Instrucciones", "El producto vectorial requiere vectores de 3 elementos (3D)")
    
    a, b = [], []
    for i in range(3):
        elemento = simpledialog.askfloat("Vector A", f"Ingrese el elemento {i+1} del vector A (3D):")
        if elemento is None: return
        a.append(elemento)
        
        elemento = simpledialog.askfloat("Vector B", f"Ingrese el elemento {i+1} del vector B (3D):")
        if elemento is None: return
        b.append(elemento)
    
    resultado = [
        a[1]*b[2] - a[2]*b[1],
        a[2]*b[0] - a[0]*b[2],
        a[0]*b[1] - a[1]*b[0]
    ]
    messagebox.showinfo("Resultado", f"Vector A: {a}\nVector B: {b}\n\nProducto Vectorial:\n{resultado}")

def triple_producto_escalar():
    messagebox.showinfo("Instrucciones", "El triple producto escalar requiere 3 vectores de 3 elementos (3D)")
    
    a, b, c = [], [], []
    for i in range(3):
        elemento = simpledialog.askfloat("Vector A", f"Ingrese el elemento {i+1} del vector A (3D):")
        if elemento is None: return
        a.append(elemento)
        
        elemento = simpledialog.askfloat("Vector B", f"Ingrese el elemento {i+1} del vector B (3D):")
        if elemento is None: return
        b.append(elemento)
        
        elemento = simpledialog.askfloat("Vector C", f"Ingrese el elemento {i+1} del vector C (3D):")
        if elemento is None: return
        c.append(elemento)
    
    # Calcular producto vectorial B x C
    producto_vec = [
        b[1]*c[2] - b[2]*c[1],
        -(b[0]*c[2] - b[2]*c[0]),
        b[0]*c[1] - b[1]*c[0]
    ]
    
    # Calcular producto escalar A · (B x C)
    resultado = np.dot(a, producto_vec)
    
    messagebox.showinfo("Resultado", 
                      f"Vector A: {a}\nVector B: {b}\nVector C: {c}\n\n"
                      f"B x C = {producto_vec}\n\n"
                      f"Triple Producto Escalar (A·(BxC)): {resultado}")
    
def triple_producto_vectorial():
    messagebox.showinfo("Instrucciones", "El triple producto vectorial requiere 3 vectores de 3 elementos (3D)")
    
    a, b, c = [], [], []
    for i in range(3):
        elemento = simpledialog.askfloat("Vector A", f"Ingrese el elemento {i+1} del vector A (3D):")
        if elemento is None: return
        a.append(elemento)
        
        elemento = simpledialog.askfloat("Vector B", f"Ingrese el elemento {i+1} del vector B (3D):")
        if elemento is None: return
        b.append(elemento)
        
        elemento = simpledialog.askfloat("Vector C", f"Ingrese el elemento {i+1} del vector C (3D):")
        if elemento is None: return
        c.append(elemento)
    
    # Calcular producto vectorial B x C
    d = [
        b[1]*c[2] - b[2]*c[1],
        -(b[0]*c[2] - b[2]*c[0]),
        b[0]*c[1] - b[1]*c[0]
    ]
    
    # Calcular producto vectorial A x (B x C)
    producto_vectorial = [
        a[1]*d[2] - a[2]*d[1],
        -(a[0]*d[2] - a[2]*d[0]),
        a[0]*d[1] - a[1]*d[0]
    ]
    
    messagebox.showinfo("Resultado", 
                      f"Vector A: {a}\nVector B: {b}\nVector C: {c}\n\n"
                      f"B x C = {d}\n\n"
                      f"Triple Producto Vectorial (Ax(BxC)):\n{producto_vectorial}")

# --- Función para Gráfica de Funciones

def graficadora_funciones():
    # Crear ventana de instrucciones
    messagebox.showinfo("Instrucciones",
                      "Ejemplos de funciones válidas:\n"
                      "- x**2 + 3*x - 5\n"
                      "- np.sin(x) + np.cos(2*x)\n"
                      "- np.exp(-x/2)*np.cos(np.pi*x)\n"
                      "- np.log(x) + np.sqrt(x+1)\n\n"
                      "Use 'x' como variable y funciones de numpy (np.sin, np.cos, etc.)")

    while True:
        # Pedir la función a graficar
        funcion_str = simpledialog.askstring("Graficadora", "Ingrese la función (y = ):")
        if not funcion_str:
            return  # Si el usuario cancela

        # Validación básica
        if not re.match(r'^[0-9x\+\-\*\/\^\.\(\)\sincostanlogexpsqrtasinacosatanpi]+$', funcion_str.lower()):
            messagebox.showerror("Error", "La función contiene caracteres no permitidos")
            continue

        try:
            # Reemplazar funciones matemáticas
            funcion_str = funcion_str.replace('sin', 'np.sin')
            funcion_str = funcion_str.replace('cos', 'np.cos')
            funcion_str = funcion_str.replace('tan', 'np.tan')
            funcion_str = funcion_str.replace('log', 'np.log')
            funcion_str = funcion_str.replace('exp', 'np.exp')
            funcion_str = funcion_str.replace('sqrt', 'np.sqrt')
            funcion_str = funcion_str.replace('pi', 'np.pi')

            # Pedir rango de x
            x_min = simpledialog.askfloat("Rango", "Valor mínimo para x:", initialvalue=-10)
            x_max = simpledialog.askfloat("Rango", "Valor máximo para x:", initialvalue=10)
            
            if x_min is None or x_max is None:
                return  # Si el usuario cancela
                
            if x_min >= x_max:
                messagebox.showerror("Error", "El valor mínimo debe ser menor que el máximo")
                continue

            # Generar datos y graficar
            x = np.linspace(x_min, x_max, 1000)
            safe_dict = {'x': x, 'np': np}
            y = eval(funcion_str, {'__builtins__': None}, safe_dict)

            plt.figure(figsize=(10, 6))
            plt.plot(x, y, label=f'y = {funcion_str}', color='blue', linewidth=2)
            plt.title(f'Gráfica de y = {funcion_str}')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.legend()

            # Ajustar límites del eje y
            y_finite = y[np.isfinite(y)]
            if len(y_finite) > 0:
                y_min, y_max = np.min(y_finite), np.max(y_finite)
                margin = 0.1 * (y_max - y_min) if y_max != y_min else 0.5
                plt.ylim(y_min - margin, y_max + margin)

            plt.show()
            break
            
        except Exception as e:
            messagebox.showerror("Error", f"Error al graficar:\n{str(e)}\n"
                                      "Verifique que la función sea válida y use 'x' como variable")

# --- Función para Solución de Sistema de Ecuaciones ---
def solucion_sistemas_ecuaciones():
    NV = 100  # Número máximo de variables
    
    # Crear ventana para ingresar el número de variables
    n = simpledialog.askinteger("Sistema de Ecuaciones", 
                               f"Ingrese el número de variables (1-{NV}):", 
                               minvalue=1, maxvalue=NV)
    if not n:
        return
    
    # Crear matrices
    A = [[0.0]*n for _ in range(n)]
    b = [0.0]*n
    x = [0.0]*n
    
    # Pedir los elementos de la matriz A y el vector b
    for i in range(n):
        for j in range(n):
            A[i][j] = simpledialog.askfloat("Matriz A", 
                                           f"Ingrese A[{i+1}][{j+1}]:")
            if A[i][j] is None: 
                return
        
        b[i] = simpledialog.askfloat("Vector b", f"Ingrese b[{i+1}]:")
        if b[i] is None: 
            return
    
    # Mostrar el sistema ingresado
    sistema_str = "Sistema ingresado:\n"
    for i in range(n):
        for j in range(n):
            sistema_str += f"{A[i][j]:.2f}\t"
        sistema_str += f"| {b[i]:.2f}\n"
    
    # Eliminación Gaussiana con pivoteo parcial
    for f in range(n):
        # Pivoteo si el pivote es 0
        if A[f][f] == 0:
            for k in range(f+1, n):
                if A[k][f] != 0:
                    A[f], A[k] = A[k], A[f]
                    b[f], b[k] = b[k], b[f]
                    break
            else:
                return "El sistema no tiene solución única (pivote cero sin intercambio posible)"

        # Eliminación hacia abajo
        for i in range(f+1, n):
            factor = A[i][f] / A[f][f]
            for j in range(f, n):
                A[i][j] -= factor * A[f][j]
            b[i] -= factor * b[f]

    # Sustitución hacia atrás
    x = [0] * n
    for i in range(n-1, -1, -1):
        suma = sum(A[i][j] * x[j] for j in range(i+1, n))
        x[i] = (b[i] - suma) / A[i][i]
    
    # Preparar el resultado
    resultado_str = sistema_str + "\nSoluciones:\n"
    for i in range(n):
        resultado_str += f"x[{i+1}] = {x[i]:.6f}\n"
    
    # Mostrar el resultado
    messagebox.showinfo("Resultado", resultado_str)

# --- Funciones para Espacios Vectoriales ---
def combinacion_lineal():
            # Crear ventana para ingresar parámetros
    n = simpledialog.askinteger("Combinación Lineal", 
                              "Ingrese el número de vectores:",
                              minvalue=1, maxvalue=10)
    if not n:
        return
    
    dim = simpledialog.askinteger("Dimensión", 
                                "Ingrese la dimensión de los vectores:",
                                minvalue=1, maxvalue=10)
    if not dim:
        return
    
    # Crear matriz aumentada
    matriz = np.zeros((dim, n + 1))
    
    # Ingresar los vectores
    messagebox.showinfo("Instrucciones", "Ingrese los componentes de cada vector:")
    for i in range(n):
        messagebox.showinfo(f"Vector {i+1}", f"Ingrese los componentes del vector {i+1}:")
        for j in range(dim):
            matriz[j, i] = simpledialog.askfloat(f"Vector {i+1}", 
                                               f"Componente {j+1} del vector {i+1}:")
            if matriz[j, i] is None:
                return
    
    # Ingresar el vector a verificar
    messagebox.showinfo("Vector objetivo", "Ingrese el vector a verificar:")
    for j in range(dim):
        matriz[j, n] = simpledialog.askfloat("Vector objetivo",
                                           f"Componente {j+1} del vector objetivo:")
        if matriz[j, n] is None:
            return
    
    # Realizar eliminación Gauss-Jordan
    try:
        rank = 0
        for col in range(n):  # Solo hasta la penúltima columna
            # Encontrar el pivote
            pivot_row = -1
            for row in range(rank, dim):
                if abs(matriz[row, col]) > 1e-10:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Intercambiar filas si es necesario
            if pivot_row != rank:
                matriz[[rank, pivot_row]] = matriz[[pivot_row, rank]]
            
            # Normalizar fila del pivote
            pivot_val = matriz[rank, col]
            matriz[rank, :] = matriz[rank, :] / pivot_val
            
            # Eliminación gaussiana
            for row in range(dim):
                if row != rank and abs(matriz[row, col]) > 1e-10:
                    factor = matriz[row, col]
                    matriz[row, :] -= factor * matriz[rank, :]
            
            rank += 1
            if rank == dim:
                break
        
        # Verificar consistencia del sistema
        solucion = True
        for row in range(rank, dim):
            if abs(matriz[row, n]) > 1e-10:
                solucion = False
                break
        
        # Preparar resultado
        resultado = "=== Matriz Final ===\n"
        resultado += np.array2string(matriz, precision=2, suppress_small=True) + "\n\n"
        
        if solucion:
            resultado += "✅ El vector SÍ es combinación lineal de los vectores dados.\n\n"
            resultado += "Coeficientes posibles:\n"
            
            # Identificar variables básicas y libres
            basic_vars = []
            free_vars = []
            for col in range(n):
                pivot_found = False
                for row in range(dim):
                    if (abs(matriz[row, col] - 1) < 1e-10 and 
                        all(abs(matriz[r, col]) < 1e-10 for r in range(dim) if r != row)):
                        basic_vars.append(col)
                        pivot_found = True
                        break
                if not pivot_found:
                    free_vars.append(col)
            
            # Mostrar solución paramétrica
            for i, col in enumerate(basic_vars):
                resultado += f"α{col+1} = {matriz[i, n]:.4f}"
                for libre in free_vars:
                    coef = -matriz[i, libre] if i < len(basic_vars) else 0
                    resultado += f" + ({coef:.4f})·t{libre+1}"
                resultado += "\n"
            
            for libre in free_vars:
                resultado += f"α{libre+1} = t{libre+1} (variable libre)\n"
        else:
            resultado += "❌ El vector NO es combinación lineal de los vectores dados."
        
        messagebox.showinfo("Resultado", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error:\n{str(e)}")

def dependencia_lineal():
    # Paso 1: Pedir número de vectores y dimensión
    n = simpledialog.askinteger("Dependencia Lineal", 
                              "Ingrese el número de vectores:",
                              minvalue=1, maxvalue=10)
    if not n:
        return
    
    dim = simpledialog.askinteger("Dimensión", 
                                 "Ingrese la dimensión de los vectores:",
                                 minvalue=1, maxvalue=10)
    if not dim:
        return
    
    # Crear matriz para los vectores
    matriz = np.zeros((dim, n))
    
    # Ingresar los vectores
    messagebox.showinfo("Instrucciones", "Ingrese los componentes de cada vector:")
    for i in range(n):
        messagebox.showinfo(f"Vector {i+1}", f"Ingrese los componentes del vector {i+1}:")
        for j in range(dim):
            matriz[j, i] = simpledialog.askfloat(f"Componente {j+1}", 
                                               f"Componente {j+1} del vector {i+1}:")
            if matriz[j, i] is None:
                return
    
    # Realizar eliminación Gauss-Jordan
    try:
        rank = 0
        matriz = matriz.astype(float)  # Asegurar tipo float
        matriz_original = matriz.copy()  # Guardar copia original
        
        for col in range(n):
            # Encontrar el pivote
            pivot_row = -1
            for row in range(rank, dim):
                if abs(matriz[row, col]) > 1e-10:
                    pivot_row = row
                    break
            
            if pivot_row == -1:
                continue
            
            # Intercambiar filas si es necesario
            if pivot_row != rank:
                matriz[[rank, pivot_row]] = matriz[[pivot_row, rank]]
            
            # Normalizar fila del pivote
            pivot_val = matriz[rank, col]
            matriz[rank, :] = matriz[rank, :] / pivot_val
            
            # Eliminación gaussiana
            for row in range(dim):
                if row != rank and abs(matriz[row, col]) > 1e-10:
                    factor = matriz[row, col]
                    matriz[row, :] -= factor * matriz[rank, :]
            
            rank += 1
            if rank == dim:
                break
        
        # Determinar el rango - CORRECCIÓN DEL ERROR AQUÍ
        rango = np.sum([np.any(abs(matriz[i, :]) > 1e-10) for i in range(dim)])
        
        # Preparar resultado
        resultado = "=== Matriz Final ===\n"
        resultado += np.array2string(matriz, precision=2, suppress_small=True) + "\n\n"
        resultado += f"Rango de la matriz: {rango}\n"
        resultado += f"Número de vectores: {n}\n\n"
        
        if rango < n:
            resultado += "🔴 Los vectores son LINEALMENTE DEPENDIENTES\n\n"
            resultado += "Relaciones de dependencia:\n"
            
            # Encontrar relaciones
            for col in range(n):
                es_pivote = False
                for row in range(dim):
                    if (abs(matriz[row, col] - 1) < 1e-10 and 
                        np.all(abs(matriz[np.arange(dim) != row, col]) < 1e-10)):
                        es_pivote = True
                        break
                
                if not es_pivote:
                    coeficientes = -matriz[:, col]
                    ecuacion = f"• v{col+1} = "
                    terminos = []
                    for i in range(col):
                        if abs(coeficientes[i]) > 1e-10:
                            signo = "+" if coeficientes[i] > 0 else "-"
                            terminos.append(f"{signo} {abs(coeficientes[i]):.2f}v{i+1}")
                    if terminos:
                        ecuacion += " ".join(terminos).replace("+", "", 1).lstrip()
                    else:
                        ecuacion += "0"
                    resultado += ecuacion + "\n"
        else:
            resultado += "🟢 Los vectores son LINEALMENTE INDEPENDIENTES"
        
        messagebox.showinfo("Resultado del Análisis", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante el cálculo:\n{str(e)}")

# --- Funciones para Operaciones con Matrices
def suma():
    # Paso 1: Pedir número de matrices a sumar
    num_matrices = simpledialog.askinteger("Suma de Matrices", 
                                         "Ingrese el número de matrices a sumar (mínimo 2):",
                                         minvalue=2, maxvalue=10)
    if not num_matrices:
        return
    
    # Paso 2: Pedir dimensiones de las matrices
    filas = simpledialog.askinteger("Dimensiones", 
                                   "Número de filas de las matrices:",
                                   minvalue=1, maxvalue=10)
    if not filas:
        return
    
    columnas = simpledialog.askinteger("Dimensiones", 
                                     "Número de columnas de las matrices:",
                                     minvalue=1, maxvalue=10)
    if not columnas:
        return
    
    # Paso 3: Ingresar cada matriz
    matrices = []
    for i in range(num_matrices):
        matriz = np.zeros((filas, columnas))
        messagebox.showinfo(f"Matriz {i+1}", f"Ingrese los valores para la matriz {i+1}:")
        
        for f in range(filas):
            for c in range(columnas):
                valor = simpledialog.askfloat("Valor", 
                                            f"Matriz {i+1}, Fila {f+1}, Columna {c+1}:")
                if valor is None:
                    return
                matriz[f, c] = valor
        
        matrices.append(matriz)
        
        # Mostrar matriz ingresada (opcional)
        messagebox.showinfo(f"Matriz {i+1}", 
                          f"Matriz {i+1} ingresada:\n{np.array2string(matriz, precision=2)}")
    
    # Paso 4: Validar dimensiones y sumar
    try:
        # Verificar dimensiones (aunque deberían ser iguales por diseño)
        for i, matriz in enumerate(matrices):
            if matriz.shape != (filas, columnas):
                messagebox.showerror("Error", 
                                   f"La matriz {i+1} tiene dimensiones incorrectas.\n"
                                   f"Esperadas: {(filas, columnas)}, Obtenidas: {matriz.shape}")
                return
        
        # Realizar suma
        suma = np.zeros((filas, columnas))
        for matriz in matrices:
            suma += matriz
        
        # Mostrar resultado
        resultado = "=== Resultado de la Suma ===\n"
        resultado += np.array2string(suma, precision=2, suppress_small=True)
        
        messagebox.showinfo("Resultado", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante la suma:\n{str(e)}")

def resta():
    # Paso 1: Pedir número de matrices a restar
    num_matrices = simpledialog.askinteger("Resta de Matrices", 
                                         "Ingrese el número de matrices a restar (mínimo 2):",
                                         minvalue=2, maxvalue=10)
    if not num_matrices:
        return
    
    # Paso 2: Pedir dimensiones de las matrices
    filas = simpledialog.askinteger("Dimensiones", 
                                  "Número de filas de las matrices:",
                                  minvalue=1, maxvalue=10)
    if not filas:
        return
    
    columnas = simpledialog.askinteger("Dimensiones", 
                                     "Número de columnas de las matrices:",
                                     minvalue=1, maxvalue=10)
    if not columnas:
        return
    
    # Paso 3: Ingresar cada matriz
    matrices = []
    for i in range(num_matrices):
        matriz = np.zeros((filas, columnas))
        messagebox.showinfo(f"Matriz {i+1}", f"Ingrese los valores para la matriz {i+1}:")
        
        for f in range(filas):
            for c in range(columnas):
                valor = simpledialog.askfloat("Valor", 
                                            f"Matriz {i+1}, Fila {f+1}, Columna {c+1}:")
                if valor is None:
                    return
                matriz[f, c] = valor
        
        matrices.append(matriz)
        
        # Mostrar matriz ingresada
        messagebox.showinfo(f"Matriz {i+1}", 
                          f"Matriz {i+1} ingresada:\n{np.array2string(matriz, precision=2)}")
    
    # Paso 4: Validar dimensiones y restar
    try:
        # Verificar dimensiones
        for i, matriz in enumerate(matrices):
            if matriz.shape != (filas, columnas):
                messagebox.showerror("Error", 
                                   f"La matriz {i+1} tiene dimensiones incorrectas.\n"
                                   f"Esperadas: {(filas, columnas)}, Obtenidas: {matriz.shape}")
                return
        
        # Realizar resta (primera matriz menos las demás)
        resta = matrices[0].copy()
        for matriz in matrices[1:]:
            resta -= matriz
        
        # Preparar resultado
        resultado = "=== Operación Realizada ===\n"
        resultado += "Matriz 1 - (Matriz 2"
        for i in range(2, num_matrices):
            resultado += f" + Matriz {i+1}"
        resultado += ")\n\n"
        
        resultado += "=== Resultado de la Resta ===\n"
        resultado += np.array2string(resta, precision=2, suppress_small=True)
        
        messagebox.showinfo("Resultado", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante la resta:\n{str(e)}")

def producto():
    # Paso 1: Pedir número de matrices a multiplicar
    num_matrices = simpledialog.askinteger("Producto de Matrices", 
                                         "Ingrese el número de matrices a multiplicar (mínimo 2):",
                                         minvalue=2, maxvalue=5)  # Limitamos a 5 por complejidad
    if not num_matrices:
        return
    
    # Listas para almacenar dimensiones y matrices
    dimensiones = []
    matrices = []
    
    # Paso 2: Ingresar cada matriz con sus dimensiones
    for i in range(num_matrices):
        # Pedir dimensiones
        messagebox.showinfo(f"Matriz {i+1}", f"Ingrese dimensiones para la Matriz {i+1}")
        filas = simpledialog.askinteger("Dimensiones", 
                                      f"Número de filas de la Matriz {i+1}:",
                                      minvalue=1, maxvalue=10)
        if not filas:
            return
        
        columnas = simpledialog.askinteger("Dimensiones", 
                                         f"Número de columnas de la Matriz {i+1}:",
                                         minvalue=1, maxvalue=10)
        if not columnas:
            return
        
        # Validar compatibilidad de dimensiones
        if i > 0 and dimensiones[i-1][1] != filas:
            messagebox.showerror("Error", 
                               f"No se puede multiplicar:\n"
                               f"Matriz {i} ({dimensiones[i-1][0]}x{dimensiones[i-1][1]}) × "
                               f"Matriz {i+1} ({filas}x{columnas})\n\n"
                               "El número de columnas de la primera matriz debe coincidir\n"
                               "con el número de filas de la segunda matriz.")
            return
        
        dimensiones.append((filas, columnas))
        
        # Ingresar valores de la matriz
        matriz = np.zeros((filas, columnas))
        messagebox.showinfo(f"Matriz {i+1}", f"Ingrese valores para la Matriz {i+1} ({filas}x{columnas})")
        
        for f in range(filas):
            for c in range(columnas):
                valor = simpledialog.askfloat("Valor", 
                                            f"Matriz {i+1}, Fila {f+1}, Columna {c+1}:")
                if valor is None:
                    return
                matriz[f, c] = valor
        
        matrices.append(matriz)
        
        # Mostrar matriz ingresada
        messagebox.showinfo(f"Matriz {i+1}", 
                          f"Matriz {i+1} ingresada:\n{np.array2string(matriz, precision=2)}")
    
    # Paso 3: Realizar la multiplicación secuencial
    try:
        resultado = matrices[0]
        for i in range(1, num_matrices):
            resultado = np.dot(resultado, matrices[i])
        
        # Preparar resultado
        resultado_str = "=== Operación Realizada ===\n"
        resultado_str += "Matriz1"
        for i in range(1, num_matrices):
            resultado_str += f" × Matriz{i+1}"
        resultado_str += "\n\n"
        
        resultado_str += f"=== Dimensiones del resultado: {resultado.shape} ===\n\n"
        resultado_str += "=== Resultado ===\n"
        resultado_str += np.array2string(resultado, precision=4, suppress_small=True)
        
        messagebox.showinfo("Resultado del Producto", resultado_str)
    
    except Exception as e:
        messagebox.showerror("Error", f"Ocurrió un error durante la multiplicación:\n{str(e)}")

def producto_Porescalar():
    # Paso 1: Pedir el escalar
    escalar = simpledialog.askfloat("Producto por Escalar", 
                                  "Ingrese el valor del escalar:")
    if escalar is None:
        return
    
    # Paso 2: Pedir dimensiones de la matriz
    filas = simpledialog.askinteger("Dimensiones", 
                                  "Número de filas de la matriz:",
                                  minvalue=1, maxvalue=10)
    if not filas:
        return
    
    columnas = simpledialog.askinteger("Dimensiones", 
                                     "Número de columnas de la matriz:",
                                     minvalue=1, maxvalue=10)
    if not columnas:
        return
    
    # Paso 3: Ingresar la matriz
    matriz = []
    messagebox.showinfo("Instrucciones", "Ingrese los valores de la matriz:")
    
    for i in range(filas):
        fila = []
        for j in range(columnas):
            valor = simpledialog.askfloat("Valor", 
                                        f"Fila {i+1}, Columna {j+1}:")
            if valor is None:
                return
            fila.append(valor)
        matriz.append(fila)
        
        # Mostrar fila ingresada
        messagebox.showinfo(f"Fila {i+1}", 
                          f"Fila {i+1} ingresada:\n{fila}")
    
    # Paso 4: Realizar el producto por escalar
    try:
        resultado = [[escalar * valor for valor in fila] for fila in matriz]
        
        # Preparar resultado
        res_str = f"=== Producto por escalar ({escalar}) ===\n\n"
        res_str += "Matriz original:\n"
        for fila in matriz:
            res_str += " ".join(f"{x:8.2f}" for x in fila) + "\n"
        
        res_str += f"\nMatriz resultante:\n"
        for fila in resultado:
            res_str += " ".join(f"{x:8.2f}" for x in fila) + "\n"
        
        messagebox.showinfo("Resultado", res_str)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al realizar la operación:\n{str(e)}")

def inversa():
    # Paso 1: Pedir dimensión de la matriz cuadrada
    n = simpledialog.askinteger("Matriz Inversa", 
                              "Ingrese el tamaño de la matriz cuadrada (n x n):",
                              minvalue=2, maxvalue=6)  # Limitamos a 6x6 por complejidad
    if not n:
        return
    
    # Paso 2: Ingresar los valores de la matriz
    matriz = []
    messagebox.showinfo("Instrucciones", f"Ingrese los valores para la matriz {n}x{n}:")
    
    for i in range(n):
        fila = []
        for j in range(n):
            valor = simpledialog.askfloat("Valor", 
                                        f"Fila {i+1}, Columna {j+1}:")
            if valor is None:
                return
            fila.append(valor)
        matriz.append(fila)
        
        # Mostrar fila ingresada
        messagebox.showinfo(f"Fila {i+1}", 
                          f"Fila {i+1} ingresada:\n{fila}")
    
    # Paso 3: Calcular la inversa
    try:
        inversa = calcular_inversa(matriz)
        
        if isinstance(inversa, str):
            messagebox.showerror("Error", inversa)
            return
        
        # Preparar resultado
        resultado = "=== Matriz Original ===\n"
        for fila in matriz:
            resultado += " ".join(f"{x:8.4f}" for x in fila) + "\n"
        
        resultado += "\n=== Matriz Inversa ===\n"
        for fila in inversa:
            resultado += " ".join(f"{x:8.4f}" for x in fila) + "\n"
        
        # Verificar resultado multiplicando ambas matrices
        identidad = np.dot(matriz, inversa)
        resultado += "\n=== Verificación (A × A⁻¹) ===\n"
        resultado += "Esta matriz debería ser aproximadamente la identidad:\n"
        for fila in identidad:
            resultado += " ".join(f"{x:8.4f}" for x in fila) + "\n"
        
        messagebox.showinfo("Resultado", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al calcular la inversa:\n{str(e)}")

def transpuesta():
    # Pedir dimensiones de la matriz
    filas = simpledialog.askinteger("Transpuesta", "Número de filas de la matriz:", minvalue=1)
    if not filas: return
    
    columnas = simpledialog.askinteger("Transpuesta", "Número de columnas de la matriz:", minvalue=1)
    if not columnas: return
    
    # Crear matriz vacía
    matriz = []
    for f in range(filas):
        fila = []
        for c in range(columnas):
            valor = simpledialog.askfloat("Valores de la matriz", 
                                         f"Ingrese valor para fila {f+1}, columna {c+1}:")
            if valor is None: return
            fila.append(valor)
        matriz.append(fila)
    
    # Calcular transpuesta
    transpuesta = []
    for c in range(len(matriz[0])):
        col = []
        for f in range(len(matriz)):
            col.append(matriz[f][c])
        transpuesta.append(col)
    
    # Mostrar resultados
    resultado = "Matriz original:\n"
    for fila in matriz:
        resultado += " ".join(f"{x:8.2f}" for x in fila) + "\n"
    
    resultado += "\nMatriz transpuesta:\n"
    for fila in transpuesta:
        resultado += " ".join(f"{x:8.2f}" for x in fila) + "\n"
    
    messagebox.showinfo("Resultado", resultado)

def determinate():
    # Paso 1: Pedir dimensión de la matriz cuadrada
    n = simpledialog.askinteger("Determinante", 
                              "Ingrese el tamaño de la matriz cuadrada (n x n):",
                              minvalue=2, maxvalue=6)  # Limitamos a 6x6 por complejidad
    if not n:
        return
    
    # Paso 2: Ingresar los valores de la matriz
    matriz = []
    messagebox.showinfo("Instrucciones", f"Ingrese los valores para la matriz {n}x{n}:")
    
    for i in range(n):
        fila = []
        for j in range(n):
            valor = simpledialog.askfloat("Valor", 
                                        f"Fila {i+1}, Columna {j+1}:")
            if valor is None:
                return
            fila.append(valor)
        matriz.append(fila)
        
        # Mostrar fila ingresada
        messagebox.showinfo(f"Fila {i+1}", 
                          f"Fila {i+1} ingresada:\n{matriz[i]}")
    
    # Paso 3: Calcular el determinante
    try:
        det = calcular_determinante(matriz)
        
        # Preparar resultado
        resultado = "=== Matriz Ingresada ===\n"
        for fila in matriz:
            resultado += " ".join(f"{x:8.2f}" for x in fila) + "\n"
        
        resultado += f"\nEl determinante de la matriz es: {det:.4f}"
        
        messagebox.showinfo("Resultado", resultado)
    
    except Exception as e:
        messagebox.showerror("Error", f"Error al calcular el determinante:\n{str(e)}")

def calcular_determinante(m):
    """Función recursiva para calcular el determinante"""
    det = 0
    
    # Verificar si es matriz cuadrada
    if len(m) != len(m[0]):
        raise ValueError("La matriz debe ser cuadrada")
    
    # Caso base para matriz 2x2
    if len(m) == 2:
        return m[0][0]*m[1][1] - m[1][0]*m[0][1]
    
    # Encontrar fila/columna con más ceros para optimización
    filas_nz = [fila.count(0) for fila in m]
    columnas_nz = [sum(1 for fila in m if fila[j] == 0) for j in range(len(m))]
    
    indice_fila = filas_nz.index(max(filas_nz))
    indice_columna = columnas_nz.index(max(columnas_nz))
    
    # Elegir fila o columna con más ceros
    if filas_nz[indice_fila] >= columnas_nz[indice_columna]:
        for col in range(len(m[indice_fila])):
            if m[indice_fila][col] == 0:
                continue  # Saltar ceros para optimizar
            n = menor(m, indice_fila, col)
            det += m[indice_fila][col] * ((-1)**(indice_fila+col)) * calcular_determinante(n)
    else:
        for fil in range(len(m)):
            if m[fil][indice_columna] == 0:
                continue  # Saltar ceros para optimizar
            n = menor(m, fil, indice_columna)
            det += m[fil][indice_columna] * ((-1)**(fil+indice_columna)) * calcular_determinante(n)
                
    return det

# --- Interfaz Gráfica ---
def mostrar_submenu_vectores():
    subventana = tk.Toplevel(root)
    subventana.title("Operaciones con Vectores")
    subventana.geometry("320x280")
    
    tk.Label(subventana, text="Operaciones Vectoriales:", font=("Arial", 12)).pack(pady=10)
    
    opciones = [
        ("Producto Escalar", producto_escalar),
        ("Producto Vectorial", producto_vectorial),
        ("Triple Producto Escalar", triple_producto_escalar),
        ("Triple Producto Vectorial", triple_producto_vectorial)
    ]
    
    for texto, comando in opciones:
        tk.Button(subventana, text=texto, command=comando, width=30, pady=3).pack(pady=2)

def mostrar_submenu_espaciosV():
    subventana = tk.Toplevel(root)
    subventana.title("Espacios Vectoriales")
    subventana.geometry("320x280")
    
    tk.Label(subventana, text="Operaciones con Espacios Vectoriales:", font=("Arial", 12)).pack(pady=10)

    opciones = [
        ("Combinación Lineal", combinacion_lineal),
        ("Determinar Dependencia Lineal", dependencia_lineal)
    ]

    for texto, comando in opciones:
        tk.Button(subventana, text=texto, command=comando, width=30, pady=3).pack(pady=2)

def mostrar_submenu_matrices():
    subventana = tk.Toplevel(root)
    subventana.title("Espacios Vectoriales")
    subventana.geometry("320x280")
    
    tk.Label(subventana, text="Operaciones con Espacios Vectoriales:", font=("Arial", 12)).pack(pady=10)

    opciones = [
        ("Suma", suma),
        ("Resta", resta),
        ("Producto", producto),
        ("Producto por Escalar", producto_Porescalar),
        ("Inversa", inversa),
        ("Transpuesta", transpuesta),
        ("Determinante", determinate)
    ]

    for texto, comando in opciones:
        tk.Button(subventana, text=texto, command=comando, width=30, pady=3).pack(pady=2)

def mostrar_tema(tema):
    if tema == "1. Operaciones con Vectores":
        mostrar_submenu_vectores()

    if tema == "2. Graficadora de Funciones":
         graficadora_funciones()

    if tema == "3. Solución de Sistemas de Ecuaciones":
        solucion_sistemas_ecuaciones()

    if tema == "4. Espacios Vectoriales":
        mostrar_submenu_espaciosV()

    if tema == "5. Operaciones con Matrices":
        mostrar_submenu_matrices()

# --- Configuración principal ---
root = tk.Tk()
root.title("Cálculos de Álgebra Lineal")
root.geometry("450x550")

tk.Label(root, text="Selecciona un tema:", font=("Arial", 14)).pack(pady=15)

temas = [
    "1. Operaciones con Vectores",
    "2. Graficadora de Funciones",
    "3. Solución de Sistemas de Ecuaciones",
    "4. Espacios Vectoriales",
    "5. Operaciones con Matrices"
]

for tema in temas:
    tk.Button(root, text=tema, command=lambda t=tema: mostrar_tema(t), 
             width=35, height=2, font=("Arial", 10)).pack(pady=5)

tk.Button(root, text="Salir", command=root.quit, bg="#ff9999", width=15).pack(pady=15)

root.mainloop()