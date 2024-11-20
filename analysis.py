import numpy as np
import pandas as pd
import symnmfmodule as C
import sys
import symnmf
import kmeans
from sklearn.metrics import silhouette_score

#Creats a points assignment table from symNMF algorithm
def assign(H):
    H = np.array(H)
    return np.argmax(H,axis = 1)

#Creats a points assignment table from kMeans algorithm
def kmeans_assign_index (data , currCentroids):
    assignments = [None for i in range(len(data))]
    for j in range(len(data)):
        minDist = float('inf')
        index = -1
        for i in range(len(currCentroids)):
            dist = kmeans.distance(data[j],currCentroids[i])
            if (dist < minDist) :
                minDist = dist
                index = i
        assignments[j] = index
    return assignments

def main(args):
    data = symnmf.read_data(args[1])
    k = int(args[0])
    nmf_assignments = assign(symnmf.symnmf(data,k))
    print("nmf :", "%.4f" % silhouette_score(data,nmf_assignments))

    kmeans_assignments = kmeans_assign_index(data,kmeans.finalCentroids(data,k,0.0001,300))
    kmeans_assignments = np.array(kmeans_assignments)
    print("kmeans :","%.4f" % silhouette_score(data,kmeans_assignments))
    

if __name__ == "__main__":

    main(sys.argv[1:])
