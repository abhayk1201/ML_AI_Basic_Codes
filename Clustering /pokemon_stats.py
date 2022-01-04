# Abhay Kumar (kumar95)
# CS540 HW4: Hierarchical Agglomerative Clustering (HAC)
import csv
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage


#Load Data
def load_data(filepath):
    """ takes in a string with a path to a CSV file formatted as in the link above, 
    and returns the first 20 data points (without the Generation and Legendary columns 
    but retaining all other columns) in a single structure."""
    
    pokm_data = []
    with open(filepath,'r') as csv_file:
        file_reader = csv.DictReader(csv_file)    
        pokm_data = list(file_reader)[:20]
    for data in pokm_data:
        del data['Generation']
        del data['Legendary']
        for data_key in ['#', 'Total', 'HP', 'Attack', 'Defense', 'Sp. Atk', 'Sp. Def', 'Speed']:
            data[data_key] = int(data[data_key])
    return pokm_data

#Calculate Feature Values
def calculate_x_y(stats):
    """takes in one row from the data loaded from the previous function, calculates the corresponding 
        x, y values for that Pokemon as specified above, and returns them in a single structure.
        This function should return the x, y values in a tuple, formatted as (x, y)."""
    x = stats['Attack'] + stats['Sp. Atk'] + stats['Speed']
    y = stats['Defense'] + stats['Sp. Def'] + stats['HP']
    return (x,y)

#Create input data in required format for linkage
def create_dataset(dataset):
    inp = []
    for data in dataset:
        inp.append((calculate_x_y(data)))
    return inp


#compute linkage distance between clusters
def linkage_dist(cluster1, cluster2):
    """computes Single-linkage Euclidean distance between two clusters, return min dist, 
      and the two points which are closest """
    min_dist = math.inf 
    for pt1 in cluster1:
        for pt2 in cluster2:
            eucl_dist = ((pt1[0]-pt2[0])**2 + (pt1[1]-pt2[1])**2)**0.5  #Euclidean dist (x1-x2)**2 + (y1-y2)**2
            if eucl_dist < min_dist:
                min_dist = eucl_dist
                min_dist_pt1 = pt1
                min_dist_pt2 = pt2
    return min_dist, min_dist_pt1, min_dist_pt2

#Perform HAC
def hac(dataset):
    """performs single linkage hierarchical agglomerative clustering on the Pokemon with the 
      (x,y) feature representation, and returns a data structure representing the clustering."""
    
    actual_dataset = dataset.copy()
    dataset = []
    for data in actual_dataset:
        #if not(math.isinf(data[0]) or math.isinf(data[1])  or math.isnan(data[0]) or math.isnan(data[1])):
        if math.isfinite(data[0]) and  math.isfinite(data[1]): #and isinstance(data[0], int) and isinstance(data[1], int):
            dataset.append(data)
      
    z = []  #output
    current_clusters = []  #list of list [[(x,y), cluster_index], ...., []]
    for i in range(len(dataset)):
        current_clusters.append([[dataset[i]], i])       
    clust_ind = len(dataset)-1

    while len(current_clusters) > 1:
        min_sing_link_dist = math.inf
        clust_ind1 = []
        clust_ind2 = []
        #iterate over every two current cluster pairs
        for i in range(len(current_clusters)):
            for j in range(i+1, len(current_clusters)):
                sing_linkage_dist  = linkage_dist(current_clusters[i][0], current_clusters[j][0])[0]
                if sing_linkage_dist < min_sing_link_dist:
                    min_sing_link_dist = sing_linkage_dist
                    cluster_1 = current_clusters[i][0]
                    cluster_2 = current_clusters[j][0]
                    clust_ind1 = current_clusters[i][1]
                    clust_ind2 = current_clusters[j][1]
                # tie breaking
                elif sing_linkage_dist == min_sing_link_dist:
                    new_clust_ind1 = current_clusters[i][1]
                    new_clust_ind2 = current_clusters[j][1]

                    if new_clust_ind1 < clust_ind1:
                        clust_ind1 = new_clust_ind1
                    elif new_clust_ind1 == clust_ind1:
                        if new_clust_ind2 < clust_ind2:
                            clust_ind2 = new_clust_ind2

        
        #combine the two clusters with the min single-linkage distance
        clust_ind += 1 
        merged_cluster = [cluster_1 + cluster_2, clust_ind]
        
        #Delete clusters which were merged
        current_clusters.remove([cluster_1, clust_ind1])
        current_clusters.remove([cluster_2, clust_ind2])
        
        #Add merged clusters
        current_clusters.append(merged_cluster)
        
        z.append([clust_ind1, clust_ind2, min_sing_link_dist, len(merged_cluster[0])])

    return np.asmatrix(z)
    #return np.array(z)


# Plot the clustering process
def imshow_hac(dataset): 
    """performs single linkage hierarchical agglomerative clustering on the Pokemon 
    with the (x,y) feature representation, and imshow the clustering process."""


    actual_dataset = dataset.copy()
    dataset = []
    for data in actual_dataset:
        if math.isfinite(data[0]) and  math.isfinite(data[1]): #and isinstance(data[0], int) and isinstance(data[1], int):
            dataset.append(data)
    
    
    #scatter plot
    fig = plt.figure(figsize=(9, 6))
    x_pos = [x for x,y in dataset]
    y_pos = [y for x,y in dataset]
    plt.scatter(x_pos, y_pos, c=np.random.rand(len(x_pos),3) )
    
    
    current_clusters = []  #list of list [[(x,y), cluster_index], ...., []]
    for i in range(len(dataset)):
        current_clusters.append([[dataset[i]], i])       
    clust_ind = len(dataset)-1

    while len(current_clusters) > 1:
        min_sing_link_dist = math.inf
        clust_ind1 = []
        clust_ind2 = []
        #iterate over every two current cluster pairs
        for i in range(len(current_clusters)):
            for j in range(i+1, len(current_clusters)):
                sing_linkage_dist,  pt1, pt2  = linkage_dist(current_clusters[i][0], current_clusters[j][0])
                if sing_linkage_dist < min_sing_link_dist:
                    min_sing_link_dist = sing_linkage_dist
                    cluster_1 = current_clusters[i][0]
                    cluster_2 = current_clusters[j][0]
                    clust_ind1 = current_clusters[i][1]
                    clust_ind2 = current_clusters[j][1]
                    point_1, point_2 = pt1, pt2
                # tie breaking
                elif sing_linkage_dist == min_sing_link_dist:
                    new_clust_ind1 = current_clusters[i][1]
                    new_clust_ind2 = current_clusters[j][1]

                    if new_clust_ind1 < clust_ind1:
                        clust_ind1 = new_clust_ind1
                    elif new_clust_ind1 == clust_ind1:
                        if new_clust_ind2 < clust_ind2:
                            clust_ind2 = new_clust_ind2

        #plot results of (m-1) linkage processes.        
        plt.plot([point_1[0], point_2[0]], [point_1[1], point_2[1]])
        plt.pause(0.1)
        
        #combine the two clusters with the min single-linkage distance and Delete clusters which were merged
        clust_ind += 1 
        merged_cluster = [cluster_1 + cluster_2, clust_ind]
        current_clusters.remove([cluster_1, clust_ind1])
        current_clusters.remove([cluster_2, clust_ind2])
        
        #Add merged clusters
        current_clusters.append(merged_cluster)
    plt.pause(5)    
    plt.show()

def random_x_y (m):
    """Given a positive integer m, your aim is to uniformly randomly generate 
     m Pokemons' "x" and "y", which satisfy 0<x<360, 0<y<360, and where x,y are both integers. 
    The output of this random_x_y(m) function will further be used as the input of hac(dataset)."""
    x = np.random.randint(1,high=360, size=m)
    y = np.random.randint(1,high=360, size=m)
    return [(int(x[i]),int(y[i])) for i in range(m)]
    

if __name__=="__main__":
    test_data = load_data('Pokemon.csv')
    test_d1 = create_dataset(test_data)
    imshow_hac(test_d1)