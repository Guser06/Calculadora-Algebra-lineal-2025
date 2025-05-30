import tkinter as tk
from tkinter import messagebox, simpledialog
import numpy as np

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
    
    # Eliminación gaussiana
    for i in range(n-1):
        for j in range(i+1, n):
            fct = A[j][i]/A[i][i]
            for k in range(n):
                A[j][k] -= fct*A[i][k]
            b[j] -= fct*b[i]
    
    # Sustitución hacia atrás
    for i in range(n-1, -1, -1):
        x[i] = b[i]
        for j in range(n):
            if i != j:
                x[i] -= A[i][j]*x[j]
        x[i] /= A[i][i]
    
    # Preparar el resultado
    resultado_str = sistema_str + "\nSoluciones:\n"
    for i in range(n):
        resultado_str += f"x[{i+1}] = {x[i]:.6f}\n"
    
    # Mostrar el resultado
    messagebox.showinfo("Resultado", resultado_str)

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

def mostrar_tema(tema):
    if tema == "1. Operaciones con Vectores":
        mostrar_submenu_vectores()
    else:
        messagebox.showinfo("Tema Seleccionado", f"Has seleccionado: {tema}")

    if tema == "3. Solución de Sistemas de Ecuaciones":
        solucion_sistemas_ecuaciones()
    else:
        messagebox.showinfo("Tema Seleccionado", f"Has seleccionado: {tema}")

# --- Configuración principal ---
root = tk.Tk()
root.title("Herramientas Matemáticas Avanzadas")
root.geometry("450x400")

tk.Label(root, text="Selecciona un tema:", font=("Arial", 14)).pack(pady=15)

temas = [
    "1. Operaciones con Vectores",
    "2. Graficadora de Funciones",
    "3. Solución de Sistemas de Ecuaciones",
    "4. Combinaciones Lineales",
    "5. Operaciones con Matrices"
]

for tema in temas:
    tk.Button(root, text=tema, command=lambda t=tema: mostrar_tema(t), 
             width=35, height=2, font=("Arial", 10)).pack(pady=5)

tk.Button(root, text="Salir", command=root.quit, bg="#ff9999", width=15).pack(pady=15)

root.mainloop()