import numpy as np
from mpi4py import MPI


def initTab(row, col):
    row += 2
    col += 2

    matriz = np.random.randint(0,2, size=(row, col), dtype=np.int32)

    for i in range(row):
        matriz[i][0] = 0
        matriz[i][col-1] = 0
        matriz[0][i] = 0
        matriz[row-1][i] = 0

    return matriz

def countVecinos(tabLife, i, j):

    arriba = tabLife[i+1][j] + tabLife[i+1][j-1] + tabLife[i+1][j+1]
    centro = tabLife[i][j-1] + tabLife[i][j+1]
    abajo = tabLife[i-1][j] + tabLife[i-1][j-1] + tabLife[i-1][j+1]

    totalVecinos = arriba + centro + abajo

    return totalVecinos

def gameStart(tabLife: np.ndarray, iterations, rank, totalP, pMaster=0):

    if rank == 0:
        tabLifeIterResult = [tabLife.copy()]
    else:
        tabLifeIterResult = None
    tabSize =  tabLife.shape[0]

    # Calcula el número de filas por proceso
    # Calculate the number of rows per process
    filas_por_proceso = tabSize // totalP
    filas_extra = tabSize % totalP

    # Calcula las filas iniciales y finales para el proceso actual
    # Calculate the initial and final rows for the current process
    start = rank * filas_por_proceso + min(rank, filas_extra) + 1
    end = start + filas_por_proceso + (1 if rank < filas_extra else 0)
    end = end - (3 if rank == (totalP-1) else 0)
    print(rank, start, end, (tabSize- end))
    for k in range(0, iterations):
        arrDicCelUpdate = []

        #Recorro el tablero, y evaluo cuales celulas viven, cuales mueren, y cuales nacen
        #Guardo en una lista los indices de las celulas que tendran esos nuevos valores
        #El tablero se debe de actualizar a lo ultimo, debido a que si lo actualizara en el momento de evaluar
        #los vecinos podrian ser afectados, haciendo que se calcule mal las celulas que viven y mueren
        
        # Traverse the board and evaluate which cells live, die, and are born
        # Store the indices of the cells that will have these new values in a list
        # The board should be updated at the end because if it were updated while evaluating,
        # the neighbors could be affected, causing incorrect calculations of living and dead cells

        for i in range(start, end):
            for j in range(1, len(tabLife)-1):
                cel = tabLife[i][j]
                cantVecinos = countVecinos(tabLife, i, j)
                if cel == 1:
                    if cantVecinos not in (2, 3):
                        arrDicCelUpdate.append({'value': 0, 'row': i, 'col': j})
                else:
                    if cantVecinos == 3:
                        arrDicCelUpdate.append({'value': 1, 'row': i, 'col': j})

        gloablArrDicCelUpdate = comm.gather(arrDicCelUpdate, root=0)

        if rank == 0:

            for arrDicCelUpdate in gloablArrDicCelUpdate:
                for dicCelUpdate in arrDicCelUpdate:

                    row = dicCelUpdate['row']
                    col = dicCelUpdate['col']
                    valueCel = dicCelUpdate['value']

                tabLife[row][col] = valueCel

            tabLifeIterResult.append(tabLife.copy())
        comm.Barrier()
        tabLife = comm.bcast(tabLife, root=0)


    return tabLifeIterResult


comm = MPI.COMM_WORLD
rank = comm.Get_rank()
totalP = comm.Get_size()

iterations = 100
rowYCol = 10

if rank == 0:
    tabLife = initTab(rowYCol,rowYCol)
else:
    tabLife = None

#Los procesos esperan al proceso 0 el cual inicializa el tablero
# The processes wait for process 0, which initializes the board
comm.Barrier()

#reparto el tablero a los demas procesos
# Distribute the board to the other processes
tabLife = comm.bcast(tabLife, root=0)

tabLifeResult = gameStart(tabLife, iterations, rank, totalP)

if (rank == 0) and (tabLifeResult is not None):

    print(tabLifeResult)
