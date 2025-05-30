##Funciones
'''
def ingreso_datos(tipo):
    ##Caso ingresar un vector
    if tipo == 'v':
        print("Ingrese el numero de variables del vector")
        n = int(input())
        v = []
        for i in range(n):
            print(f"Ingrese el coeficiente {i} del vector")
            vec = float(input())
            v.append(vec)
        return v
    ##Caso ingresar una matriz
    if tipo == 'm':
        print("Ingrese el numero de filas de la matriz")
        f = int(input())
        print("Ingrese el numero de columnas de la matriz")
        c = int(input())
        m = []
        for i in range(f):
            n = []
            for j in range(c):
                print(f"Ingrese el valor {j+1} de la fila {i+1}")
                val = float(input())
                n.append(val)
            m.append(n)
        return m

def transposicion(matriz):
    n = []
    for c in range(len(matriz[0])):
        col = []
        for f in range(len(matriz)):
            col.append(matriz[f][c])
        n.append(col) 
    return n

##Dependiendo de para que se use la función, se agregarán previamente unos, ceros o lo que convenga
##Al final de cada fila de la matriz
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

#Ejemplos
m = ingreso_datos('m')
print(m)

t = transposicion(m)
print(t)

v = ingreso_datos('v')
print(v)

g = GJ(m, v)
##Bucle para imprimir
for e in g[0]:
    print(f"{e}|{g[1][g[0].index(e)]}")
'''
##Menor de una matriz
def menor(m, i, j):
    n = []
    for f in range(len(m)):
        fn = []
        for c in range(len(m[0])):
            if c != j:
                fn.append(m[f][c])
            else: continue
        if f != i:
            n.append(fn)
        else: continue
    return n

##Determinante de una matriz cuadrada
def Determinante(m):
    det = 0
    ##Verificar dimension de la matriz
    if len(m) != len(m[0]):
        return 'Matriz no valida'
    
    elif len(m) == 2 and len(m[0]) == 2:
        det += (m[0][0]*m[1][1])-(m[1][0]*m[0][1])
        return det
    
    else:
    ##Encontrar los ceros de la matriz
        filas_nz = []
        columnas_nz = []
        for i in range(len(m)):
            fila_nz = 0
            for j in range(len(m[0])):
                if m[i][j] == 0:
                    fila_nz += 1
                else: continue
            filas_nz.append(fila_nz)

        for j in range(len(m[0])):
            columna_nz = 0
            for i in range(len(m)):
                if m[i][j] == 0:
                    columna_nz += 1
                else: continue
            columnas_nz.append(columna_nz)

        ##Encontrar la fila y columna con mas ceros
        indice_fila = 0
        indice_columna = 0
        for f in range(1, len(filas_nz)):
            if filas_nz[f-1] > filas_nz[f] and f <= len(filas_nz):
                indice_fila = f-1
            else: continue
        for c in range(1, len(columnas_nz)):
            if columnas_nz[c-1] > columnas_nz[c] and c <= len(columnas_nz):
                indice_columna = c-1
            else: continue
        ##Calcular el cofactor de la matriz con base a la fila o columna
        ##Que tenga mas ceros
        if filas_nz[indice_fila] >= columnas_nz[indice_columna]:
            for col in range(len(m[indice_fila])):
                n = menor(m=m, i=indice_fila, j = col)
                p = Determinante(n)
                det += m[indice_fila][col]*(pow(-1, indice_fila+col+2))*p
        elif filas_nz[indice_fila] <= columnas_nz[indice_columna]:
            for fil in range(len(m)):
                n = menor(m=m, i = fil, j=indice_columna)
                p = Determinante(n)
                det += m[fil][indice_columna]*(pow(-1, fil+indice_columna+2))*p
                
        return det

##Producto escalar de una matriz
def Prod_Esc_Mat(a, m):
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] *= a
    return m

##Gauss Jordan con valores sustituida con una matriz identidad
def Inversa(matriz):
    ##Verificar dimensión nxn de la matriz
    if len(matriz) != len(matriz[0]):
        return 'Matriz ingresada invalida'
    ##Crear una matriz identidad de las mismas dimensiones de la matriz ingresada 
    Id = []
    for m in range(len(matriz)):
        Id_f = []
        for n in range(len(matriz[m])):
            if n == m:
                Id_f.append(1)
            elif n != m:
                Id_f.append(0)
            else: continue
        Id.append(Id_f)

    # Proceso de Gauss-Jordan
    for f in range(len(matriz)):
        # Si el pivote es 0, buscar fila con valor no nulo en la misma columna y hacer swap
        if matriz[f][f] == 0:
            for k in range(f+1, len(matriz)):
                if matriz[k][f] != 0:
                    matriz[f], matriz[k] = matriz[k], matriz[f]
                    Id[f], Id[k] = Id[k], Id[f]
                    break
            else:
                return 'La matriz no tiene inversa (pivote cero sin posibilidad de intercambio)'

        # Normalizar fila f
        pivote = matriz[f][f]
        for j in range(len(matriz)):
            matriz[f][j] /= pivote
            Id[f][j] /= pivote

        # Eliminar en otras filas
        for i in range(len(matriz)):
            if i != f:
                factor = matriz[i][f]
                for j in range(len(matriz)):
                    matriz[i][j] -= factor * matriz[f][j]
                    Id[i][j] -= factor * Id[f][j]

    return Id

matriz = [[-2, 1, -1, 0], [0, 2, 1, -3], [-1, 0, 0, 2], [1, 1, 4, 1]]
print(f"El determinante de la matriz es {Determinante(matriz)}")

print(f"El producto escalar de la matriz por 4 es {Prod_Esc_Mat(4, matriz)}")

print(f"La inversa de la matriz es {Inversa(matriz)}")