
import math
import copy
import sys

#Reads data from a file and converts each line into a list of floating-point numbers.
def read_data(file_path):
    file = open(file_path)
    data = file.readlines()
    file.close()
    for i in range(len(data)):
        data[i] = data[i].split(",")
    if data[0] == '\n':
        return 0
    for x in data:
        for i in range(len(x)):
            x[i] = float(x[i])
    return data

#Calculates the Euclidean distance between two points.
def distance (point1 ,point2):
    cnt = 0.0
    for i in range(len(point1)):
        cnt += math.pow((point1[i]-point2[i]),2)
    return math.sqrt(cnt)

#Determines if the centroids have converged based on the maximum number of iterations and epsilon
def converge(prevCentroids , currCentroids,curr_itr, Max_itr , epsilon):
    if curr_itr > Max_itr :
        return 1
    for i in range(len(prevCentroids)):
        if distance(prevCentroids[i],currCentroids[i]) >= epsilon:
            return 0
    return 1

#Assigns each data point to the nearest centroid.
def assign (data , currCentroids):
    assignments = [[] for i in range(len(currCentroids))]
    for x in data:
        minDist = float('inf')
        index = -1
        for i in range(len(currCentroids)):
            dist = distance(x,currCentroids[i])
            if (dist < minDist) :
                minDist = dist
                index = i
        assignments[index].append(x)
    return assignments

#Updates centroids by calculating the mean of each assigned cluster.
def update_centroids(assignments,d):
    centroids = []
    for points in assignments:
        centroid = [sum(x[j] for x in points) / len(points) for j in range(d)]
        centroids.append(centroid)

    return centroids

#Executes the KMeans clustering algorithm, 
# returning the final centroids after convergence or reaching the maximum iteration limit.
def finalCentroids(data,k,epsilon,max_iter):
     d= len(data[0])
     centroids = data[:k]
     i = 0
     cnt = 0
     while i==0 :
         assignments = assign(data, centroids)
         prevCentroids = copy.deepcopy(centroids)
         centroids=update_centroids(assignments,d)
         i = converge(prevCentroids,centroids,cnt,max_iter,epsilon)
         cnt+=1
     return centroids
