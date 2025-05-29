m = [[1, 2], [-1, 2], [3, 1], [4, 2]]
v = [0, 0, 0, 0]

def transposicion(matriz):
    n = []
    for c in range(len(matriz[0])):
        col = []
        for f in range(len(matriz)):
            col.append(matriz[f][c])
        n.append(col) 
    return n

t = transposicion(m)
print(t)
print()


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

g = GJ(m, v)
##Bucle para imprimir
for e in g[0]:
    print(f"{e}|{g[1][g[0].index(e)]}")

def menor(m, i, j):
    n = []
    for f in range(len(m)):
        fn = []
        for c in range(len(m[0])):
            if c != j:
                fn.append(m[c])
            else: continue
        if f != i:
            n.append(fn)
        else: continue

def cofactor(m):
    ##Verificar dimension de la matriz
    if len(m) == 2:
        for k in range(len(m)):
            for l in range(len(m[k])):

    ##Calcular el determinante de la matriz
    else:
    ##Encontrar los ceros de la matriz
        filas_nz = []
        columnas_nz = []
        for i in range(len(m)):
            fila_nz = 0
            columna_nz = 0
            for j in range(len(m[0])):
                if m[i][j] == 0:
                    fila_nz += 1
                elif m[j][i] == 0:
                    columna_nz += 1
                else: continue
            filas_nz.append(fila_nz)
            columnas_nz.append(columna_nz)
        ##Encontrar la fila y columna con mas ceros
        indice_fila = 0
        indice_columna = 0
        for f in range(len(filas_nz)):
            if filas_nz[f] > filas_nz[f+1] and f+1 <= len(filas_nz):
                indice_fila = f
            else: continue
        for c in range(len(columnas_nz)):
            if columnas_nz[c] > columnas_nz[c+1] and c+1 <= len(columnas_nz):
                indice_columna = c
            else: continue

        if filas_nz[indice_fila] > columnas_nz[indice_columna]:
            
        n = menor(m=m, i=indice_fila, j=indice_columna)
