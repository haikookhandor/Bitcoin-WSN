import networkx as nx
import random
# Addition of weights and complete_graph functionality on to the disagreement-minimizing correlation clustering algorithm by Bansal, Blum and Chawla [1].
# Reference: [1] Bansal, Nikhil, Avrim Blum, and Shuchi Chawla.“Correlation Clustering.” Machine Learning 56, no. 1–3 (2004): 89–113.
# Github Repo: {https://github.com/filkry/py-correlation-clustering}

class c_clustering:
    def __init__(self, G, delta = 1.0/44, complete_graph = False):
        self.__G__ = G
        self.__reset_caches__()
        self.__clusters__ = None
        self.__delta__ = delta
        
        self.weights = nx.get_edge_attributes(self.__G__,"Weight") # Weighted added here 
        self.complte_graph = complete_graph # Complete graph added here
        if(self.complte_graph):
            self.fairness = nx.get_node_attributes(G,"fairness")
            self.goodness = nx.get_node_attributes(G,"fairness")

    def __reset_caches__(self):
        self.__G_nodes__ = set(self.__G__.nodes())
        self.__N_plus_cache__ = dict()

    def __remove_cluster__(self, C):
        self.__G__.remove_nodes_from(C)
        self.__reset_caches__()

    def positive_neighbours(self, u):
        if u in self.__N_plus_cache__:
            return self.__N_plus_cache__[u]
        res = set([u])

        for i in self.__G__.neighbors(u):
            if(self.weights[(u,i)] > 0):
                res.add(i)

        if(self.complte_graph):
            for i in nx.non_neighbors(self.__G__, u):
                if(self.fairness[u]*self.goodness[i] > 0):
                    res.add(i)
        self.__N_plus_cache__[u] = res
        return res

    def delta_good(self, v, C, delta):

        Nv = self.positive_neighbours(v)
        return (len(Nv & C) >= (1.0 - delta)*len(C) and len(Nv&(self.__G_nodes__ -C))<=delta*len(C))

    def run(self):
        if self.__clusters__ is None:
            self.__clusters__ = []

        while len(self.__G_nodes__) > 0:
            vs = random.sample(self.__G_nodes__, len(self.__G_nodes__))
            Av = None
            for v in vs:
                Av = self.positive_neighbours(v).copy()
                for x in self.positive_neighbours(v):
                    if not self.delta_good(x, Av, 3 * self.__delta__):
                        Av.remove(x)
                Y = set(y for y in self.__G_nodes__ if self.delta_good(y, Av, 7 * self.__delta__))
                Av = Av | Y
                if len(Av) > 0:
                    break

            if len(Av) == 0:
                break
            print("Cluster: ",len(Av))
            self.__clusters__.append(Av)
            self.__remove_cluster__(Av)

        for v in self.__G_nodes__:
            self.__clusters__.append(set([v]))
        return self.__clusters__