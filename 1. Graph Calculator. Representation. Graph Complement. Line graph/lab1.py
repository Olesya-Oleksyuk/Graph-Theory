import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


class Graph:
    def __init__(self, fname=None, obj_name=None, graph_type=None):
        self.fname = fname
        self.graph_obj = obj_name
        self.graph_type = graph_type
        self._mode = None
        self.valid_data = np.array([])
        self.n_vertices = None
        self.pendant_vertices = np.array([], dtype=int)
        self.pendant_edges = np.array([], dtype=int)
        self.vertices_adj_dict = {}
        self.parent_indexed_edges = None

        if self.fname is not None:
            self._mode = 'From file'
            self.read_from_file()
        elif self.graph_obj is not None:
            self._mode = 'From graph object'
            self.read_from_object()
        else:
            print("\nУкажите имя файла или объект графа...\n")
            raise Exception()

        self.edges_tuples = self.get_edges_tuples()
        self.show_graph_info()

        self.get_vertices_adj_list()
        self.get_vertices_deg()
        self.get_pendant_vertices()

        self.adj_matrix = Adjacency_matrix(self.n_vertices, self.valid_data)
        self.adj_matrix.print()
        self.adj_edge_dict = self.get_edges_adj_list()

        if graph_type == 'Line graph':
            _indexed_edges_obj = Indexed_edges(self.valid_data, graph_type='Line graph')
        else:
            _indexed_edges_obj = Indexed_edges(self.valid_data)
        self.indexed_edges = _indexed_edges_obj.dict

        self.print_id_edges()
        self.adj_edges_matrix = Adjacency_edges_matrix(self.adj_edge_dict, self.indexed_edges)
        self.adj_edges_matrix.print_matrix()

        if self._mode == 'From file':
            self.g_init_view_obj = Graph_plot(self.edges_tuples, 'Исходный граф G')

    def get_graph_obj(self):
        return self.g_init_view_obj

    # сохраняем в объекте типа "Graph" индексированный список indexed_edges объекта Graph родителя
    def get_Line_Graph(self):
        graph_obj = self.g_init_view_obj.get_line_graph()
        graph_obj.parent_indexed_edges = self.indexed_edges
        return graph_obj

    def get_Complement_Graph(self):
        return self.g_init_view_obj.get_graph_complement()

    def lineGraph_get_valid_data(self):
        _valid_data_LG = []
        edge_id = self.parent_indexed_edges
        for edge in self.graph_obj.edges:
            node1 = edge_id[edge[0]]
            node2 = edge_id[edge[1]]
            _valid_data_LG.append((node1, node2))
        return _valid_data_LG

    def print_id_edges(self):
        print('\nБудем считать ребра пронумерованными:')
        print(self.indexed_edges)

    def get_edges_tuples(self):
        edges_list_of_tuples = []
        if self.graph_type == 'Line graph':
            self.valid_data = self.lineGraph_get_valid_data()
            valid_data = self.valid_data
        else:
            valid_data = self.valid_data.tolist()
        for i, j in sorted(valid_data):
            edge_t = (i, j)
            edges_list_of_tuples.append(edge_t)
        return edges_list_of_tuples

    def get_edges_adj_list(self):
        edges_list_of_tuples = []
        if self.graph_type == 'Line graph':
            valid_data = self.valid_data
        else:
            valid_data = self.valid_data.tolist()
        for i, j in sorted(valid_data):
            edge_t = (i, j)
            edges_list_of_tuples.append(edge_t)

        adj_edge_dict = {}
        for i, j in edges_list_of_tuples:
            # фокус
            if self.graph_type == 'Line graph':
                _edge = (i, j)
            else:
                _edge = [i, j]
            _adj_list_per_edge = []
            for edge in valid_data:
                if edge == _edge:
                    continue
                elif i in edge:
                    _adj_list_per_edge.append(tuple(edge))
                elif j in edge:
                    _adj_list_per_edge.append(tuple(edge))
            _key = tuple((i, j))
            adj_edge_dict[_key] = sorted(_adj_list_per_edge)

        print("\nОкрестности рёбер:")
        for edge, adj in adj_edge_dict.items():
            print(f'{edge} = {adj}')
        return adj_edge_dict

    def get_vertices_deg(self):
        self.vertices_deg_dict = {}
        for key in self.vertices_adj_dict:
            self.vertices_deg_dict[key] = len(self.vertices_adj_dict[key])
        self.print_vertices_deg()

    def print_vertices_deg(self):
        print("\nСтепени вершин:")
        for k, v in self.vertices_deg_dict.items():
            print(f"{k} : {v}")

    def read_from_object(self):
        self.valid_data = np.array(list(self.graph_obj.edges))
        if self.graph_type == 'Line graph':
            self.parent_indexed_edges = self.graph_obj.parent_indexed_edges
        _data = list(self.graph_obj.nodes)
        self.n_vertices = len(_data)

    def read_from_file(self):
        self.read_vertices_number()
        self.read_edges()

    def read_vertices_number(self):
        self.file_data = np.genfromtxt(self.fname, max_rows=1, dtype=int)
        self.n_vertices = self.file_data.item()

    def read_edges(self):
        self.file_data = np.genfromtxt(self.fname, skip_header=1, dtype=int)
        self.validation_check()

    def show_graph_info(self):
        if self.graph_type == 'Line graph':
            print(f"\nИсходный граф:\n"
                  f"{self.n_vertices} вершин, {len(self.edges_tuples)} рёбер \n")
        else:
            print(f"\nИсходный граф:\n"
                  f"{self.n_vertices} вершин, {self.valid_data[:, 0].size} рёбер \n")

    def get_pendant_vertices(self, _flag=False):
        for key in self.vertices_adj_dict:
            if len(self.vertices_adj_dict[key]) == 1:
                if _flag:
                    print("\nВисячие вершины:\n")
                    _flag = True
                self.pendant_vertices = np.append(self.pendant_vertices, key)
                print(f"{key}", end=' ')
        if len(self.pendant_vertices) == 0:
            print("\nВисячих вершин нет.")
            print("Висячих рёбер нет.")
        else:
            self.pendant_edges = self.get_pendant_edges()

    def get_pendant_edges(self):
        _pendant_edges = []
        for v1 in self.pendant_vertices:
            _v1_list = self.vertices_adj_dict[v1]
            v2 = _v1_list[0]
            _new_edge = [v1, v2]
            _pendant_edges.append(_new_edge)
        pendant_edges = np.array(_pendant_edges)
        print(pendant_edges)
        return pendant_edges

    def validation_check(self):
        validation = self.sanity_chek()
        if not validation:
            print("\nИсправьте данные в файле и повторите снова...\n")
            raise Exception()
        else:
            print("Корректность данных: подтверждена.")
            self.valid_data = self.file_data
            del self.file_data

    def sanity_chek(self):
        print('Условие проверки корректности данных:\n\t1 <= u, v <= n \n'
              'где u, v - числа, задающие вершины графа, \nn - число вершин.\n')
        condition_matrix = ((self.file_data >= 1) & (self.file_data <= self.n_vertices))
        file_data_masked = np.ma.masked_array(self.file_data, condition_matrix)
        if file_data_masked.all() is np.ma.masked:
            return True
        else:
            nonzero_indices = np.transpose(np.nonzero(file_data_masked))
            print("\nВершины, несоответствующие заданному условию:")
            for i, j in nonzero_indices:
                print(self.file_data[i, j], end='  ')
            return False

    # Get vertices adjacency list
    # valid_data - ndarray
    def get_vertices_adj_list(self):
        for start, end in self.valid_data:  # start = the 1st item in the tuple;  end = the 2nd item in the tuple
            if start in self.vertices_adj_dict:  # Check if Key Exists in the dictionary
                self.vertices_adj_dict[start].append(end)  #
            else:
                self.vertices_adj_dict[start] = [end]  # If there's no such key => a
            # Т.к. граф неориентированный => дублируем процедуру со вторым стобликом
            if end in self.vertices_adj_dict:  # Check if Key Exists in the dictionary
                self.vertices_adj_dict[end].append(start)  #
            else:
                self.vertices_adj_dict[end] = [start]  # If there's no such key => a
        self.print_ver_adj_list()

    def print_ver_adj_list(self):
        print("Окрестность N(u) каждой  вершины u:")
        for k, v in sorted(self.vertices_adj_dict.items()):
            print(f'{k} : {v}')


class Indexed_edges:
    def __init__(self, valid_data, graph_type=None):
        if graph_type == "Line graph":
            self.valid_data = valid_data
        else:
            self.valid_data = valid_data.tolist()
        self.dict = self._make_edges_indexed()

    def _make_edges_indexed(self):
        _edges_indexed_dict = {}
        valid_data = sorted(self.valid_data)
        for i in range(0, len(valid_data)):
            edge_tuple = tuple(valid_data[i])
            _edges_indexed_dict[edge_tuple] = i + 1
        return _edges_indexed_dict

    def get(self):
        return self.dict


class Adjacency_matrix:
    def __init__(self, n_vertices, edges):
        self.n_vertices = n_vertices
        self.edges = edges  # list of lists
        self.adjMatrix = np.zeros((n_vertices, n_vertices), dtype=int)
        self.add_edge()

    def add_edge(self):
        for v1, v2 in self.edges:
            self.adjMatrix[v1 - 1][v2 - 1] = 1
            self.adjMatrix[v2 - 1][v1 - 1] = 1

    def print(self):
        print(f'\nМатрица смежности вершин {self.n_vertices}x{self.n_vertices}:')
        print(self.adjMatrix)


class Adjacency_edges_matrix:
    def __init__(self, edges_dict, id_edges):
        self._matrix_size = len(id_edges)
        self.adjMatrix = np.zeros((self._matrix_size, self._matrix_size), dtype=int)
        self._calc_matrix(edges_dict, id_edges)

    def _calc_matrix(self, edges_dict, id_edges):
        for key, edges_list in edges_dict.items():
            for t in edges_list:
                edge1 = id_edges[key]
                edge2 = id_edges[t]
                self.adjMatrix[edge1 - 1][edge2 - 1] = 1
                self.adjMatrix[edge2 - 1][edge1 - 1] = 1

    def print_matrix(self):
        print(f"\nМатрица смежности рёбер ({self._matrix_size}x{self._matrix_size})")
        print(self.adjMatrix)

    @property
    def matrix(self):
        return self.adjMatrix


class Graph_plot:
    def __init__(self, edges, title):
        self.edges = edges  # must be a list of tuples
        self.title = title  # must be a string
        self.graph_complement = None
        self.line_graph = None

        self.Graph = nx.Graph()
        self.Graph.add_edges_from(self.edges)
        show_graph_plot(self.Graph, self.title)

    def get_graph(self):
        return self.Graph

    def get_graph_complement(self):
        self.graph_complement = nx.complement(self.Graph)
        return self.graph_complement

    def get_line_graph(self):
        self.line_graph = nx.line_graph(self.Graph)
        return self.line_graph


def show_graph_plot(graph, title):
    plt.figure()
    _pos = nx.spring_layout(graph)
    nx.draw_networkx_nodes(graph, _pos, node_size=500, node_color='red')
    nx.draw_networkx_edges(graph, _pos, node_size=500)
    nx.draw_networkx_labels(graph, _pos)
    plt.title(title)
    plt.show()


# print horizontal line of the "char" symbols
def print_hr(char):
    import os
    _size = os.get_terminal_size().columns
    print('\n')
    print(char * _size)


def main():
    print('\nГраф G')
    myGraph = Graph('Lab01-gr5-25.dat')
    # print_hr() doesn't work in pyCharm. Run this script using the prompt
    #print_hr('-')

    complement_g = myGraph.get_Complement_Graph()
    line_g = myGraph.get_Line_Graph()

    show_graph_plot(complement_g, 'Дополнительный граф G')
    show_graph_plot(line_g, 'Рёберный граф L(G)')

    print('Дополнительный граф')
    complementGraph = Graph(obj_name=complement_g)
    #print_hr('-')

    print('Рёберный граф')
    lineGraph = Graph(obj_name=line_g, graph_type='Line graph')
    #print_hr('-')


if __name__ == "__main__":
    main()

# Python program to explain os.get_terminal_size() method
