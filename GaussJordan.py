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