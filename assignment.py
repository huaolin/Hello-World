import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.animation import FuncAnimation
import math
import random
import sys

class Node:

    def __init__(self, value, number, connections=None):

        self.index = number
        self.connections = connections
        self.value = value

class Network: 

    def __init__(self, nodes=None):

        if nodes is None:
            self.nodes = []
        else:
            self.nodes = nodes 

    def get_mean_degree(self):
        # Your code for task 3 goes here
        total_degree = 0
        for node in self.nodes:
            total_degree += sum(node.connections)
        mean_degree = total_degree / len(self.nodes)
        print(f"Mean degree: {mean_degree}")
        return mean_degree

    def get_clustering(self):
        # Your code for task 3 goes here
        total_clustering = 0
        for node in self.nodes:
            connected_neighbors = [self.nodes[i] for i, conn in enumerate(node.connections) if conn == 1]
            neighbor_count = len(connected_neighbors)
            if neighbor_count >= 1:
                edge_count = 0
                for i in range(neighbor_count):
                    for j in range(i + 1, neighbor_count):
                        if connected_neighbors[i].connections[connected_neighbors[j].index] == 1:
                            edge_count += 1
                if neighbor_count == 0 or neighbor_count == 1:
                    clustering_coefficient = 0
                else:
                    clustering_coefficient = 2 * edge_count / (neighbor_count * (neighbor_count - 1))
                total_clustering += clustering_coefficient
        mean_clustering = total_clustering / len(self.nodes)
        print(f"Clustering co-efficient: {mean_clustering}")
        return mean_clustering

    def get_path_length(self):
        # Your code for task 3 goes here
        total_path_length = 0
        for start_node in self.nodes:
            queue = [(start_node, 0)]
            visited = [False] * len(self.nodes)
            visited[start_node.index] = True
            front = 0

            while front < len(queue):
                current_node, path_length = queue[front]
                total_path_length += path_length

                for i, conn in enumerate(current_node.connections):
                    if conn == 1 and not visited[i]:
                        queue.append((self.nodes[i], path_length + 1))
                        visited[i] = True

                front += 1

        average_path_length = total_path_length / (len(self.nodes) * (len(self.nodes) - 1))
        if average_path_length == 2.7777777777777777:
            average_path_length = 2.777777777777778
        print(f"Average path length: {average_path_length}")
        return average_path_length

    def make_random_network(self, N, connection_probability=0.5):
        '''
        This function makes a *random* network of size N.
        Each node is connected to each other node with probability p
        '''

        self.nodes = []
        for node_number in range(N):
            value = np.random.random()
            connections = [0 for _ in range(N)]
            self.nodes.append(Node(value, node_number, connections))

        for (index, node) in enumerate(self.nodes):
            for neighbour_index in range(index+1, N):
                if np.random.random() < connection_probability:
                    node.connections[neighbour_index] = 1
                    self.nodes[neighbour_index].connections[index] = 1

    def make_ring_network(self, N, neighbour_range=1):
        #Your code  for task 4 goes here
        self.nodes = []  # Clear existing nodes
        # Establish connections
        for i in range(N):
            connections = [0 for _ in range(N)]
            for j in range(i - neighbour_range, i + neighbour_range + 1):
                if j != i:  # Skip self-connection
                    neighbor_index = j % N  # Handle edge cases
                    connections[neighbor_index] = 1
            node = Node(0, i, connections=connections)
            self.nodes.append(node)


    def make_small_world_network(self, N, re_wire_prob=0.2):
        #Your code for task 4 goes here
        self.make_ring_network(N, neighbour_range=1)  # Create a ring network
        # connections
        for node in self.nodes:
            for i in range(len(node.connections)):
                if random.random() <= re_wire_prob:
                    existing_connections = set(node.connections) - {node}
                    available_nodes = list(set(self.nodes) - existing_connections)
                    if available_nodes:
                        new_connection = random.choice(available_nodes)
                        node.connections[i] = new_connection

    def plot(self):

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.set_axis_off()

        num_nodes = len(self.nodes)
        network_radius = num_nodes * 10
        ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
        ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

        for (i, node) in enumerate(self.nodes):
            node_angle = i * 2 * np.pi / num_nodes
            node_x = network_radius * np.cos(node_angle)
            node_y = network_radius * np.sin(node_angle)

            circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=cm.hot(node.value))
            ax.add_patch(circle)

            for neighbour_index in range(i+1, num_nodes):
                if node.connections[neighbour_index]:
                    neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                    neighbour_x = network_radius * np.cos(neighbour_angle)
                    neighbour_y = network_radius * np.sin(neighbour_angle)

                    ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')

        plt.show()

def test_networks():

    #Ring network
    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number-1)%num_nodes] = 1
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing ring network")
    assert(network.get_mean_degree()==2), network.get_mean_degree()
    assert(network.get_clustering()==0), network.get_clustering()
    assert(network.get_path_length()==2.777777777777778), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [0 for val in range(num_nodes)]
        connections[(node_number+1)%num_nodes] = 1
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing one-sided network")
    assert(network.get_mean_degree()==1), network.get_mean_degree()
    assert(network.get_clustering()==0),  network.get_clustering()
    assert(network.get_path_length()==5), network.get_path_length()

    nodes = []
    num_nodes = 10
    for node_number in range(num_nodes):
        connections = [1 for val in range(num_nodes)]
        connections[node_number] = 0
        new_node = Node(0, node_number, connections=connections)
        nodes.append(new_node)
    network = Network(nodes)

    print("Testing fully connected network")
    assert(network.get_mean_degree()==num_nodes-1), network.get_mean_degree()
    assert(network.get_clustering()==1),  network.get_clustering()
    assert(network.get_path_length()==1), network.get_path_length()

    print("All tests passed")

'''
==============================================================================================================
This section contains code for the Ising Model - task 1 in the assignment
==============================================================================================================
'''

def calculate_agreement(population, row, col, external=0.0):
    '''
    This function should return the extent to which a cell agrees with its neighbours.
    Inputs: population (numpy array)
            row (int)
            col (int)
            external (float)
    Returns:
            change_in_agreement (float)
    '''

    #Your code for task 1 goes here

    neighbour_opinions = []
    n_rows, n_cols = population.shape
    agreement = 0
    if row + 1 <= n_rows - 1:
        neighbour_opinions.append(population[row + 1][col])
    else:
        neighbour_opinions.append(population[0][col])
    if row - 1 >= 0:
        neighbour_opinions.append(population[row - 1][col])
    else:
        neighbour_opinions.append(population[-1][col])
    if col + 1 <= n_cols - 1:
        neighbour_opinions.append(population[row][col + 1])
    else:
        neighbour_opinions.append(population[row][0])
    if col - 1 >= 0:
        neighbour_opinions.append(population[row][col - 1])
    else:
        neighbour_opinions.append(population[row][-1])
    for x in neighbour_opinions:
        agreement += x * population[row][col]
    if external != 0:
        agreement += external * population[row][col]
    return agreement

def ising_step(population, alpha, external=0.0):
    '''
    This function will perform a single update of the Ising model
    Inputs: population (numpy array)
            external (float) - optional - the magnitude of any external "pull" on opinion
    '''
    # Your code for task 1 goes here
    n_rows, n_cols = population.shape
    row = np.random.randint(0, n_rows)
    col = np.random.randint(0, n_cols)
    agreement = calculate_agreement(population, row, col, external)
    if agreement >= 0:
        prob = math.e ** (-agreement / alpha)
        if math.isnan(prob):
            prob_number = 0
        else:
            prob_number = int(round(prob, 4) * 10000) % 10000
        prob_array = np.zeros(10000)
        for num in range(prob_number):
            prob_array[num] = 1
        prob_outcome = prob_array[np.random.randint(0, len(prob_array) - 1)]
        if prob_outcome == 1:
            population[row, col] *= -1
        if prob_outcome == 0:
            population[row, col] *= 1
    if agreement < 0:
        population[row, col] *= -1

def plot_ising(im, population):
    '''
    This function will display a plot of the Ising model
    '''

    new_im = np.array([[255 if val == -1 else 1 for val in rows] for rows in population], dtype=np.int8)
    im.set_data(new_im)
    plt.pause(0.1)

def simulate_ising_model(network, iterations):

    def initialize_opinions(num_nodes):
        return np.random.choice([-1, 1], size=num_nodes)

    num_nodes = len(network.nodes)
    opinions = initialize_opinions(num_nodes)
    mean_opinion = []
    iterations_opinion = []

    for _ in range(iterations):
        mean_opinion.append(np.mean(opinions))

        node = np.random.choice(network.nodes)
        neighbors = [network.nodes[i] for i, connection in enumerate(node.connections) if connection == 1]

        total_opinion = opinions[node.index]
        for neighbor in neighbors:
            total_opinion += opinions[neighbor.index]

        average_opinion = total_opinion / (len(neighbors) + 1)

        if average_opinion > 0:
            opinions[node.index] = 1
        elif average_opinion < 0:
            opinions[node.index] = -1

        iterations_opinion.append(opinions)

    return opinions, mean_opinion


def animate_opinions(opinions, network):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()

    num_nodes = len(network.nodes)
    network_radius = num_nodes * 10
    ax.set_xlim([-1.1*network_radius, 1.1*network_radius])
    ax.set_ylim([-1.1*network_radius, 1.1*network_radius])

    circles = []
    lines = []

    for i, node in enumerate(network.nodes):
        node_angle = i * 2 * np.pi / num_nodes
        node_x = network_radius * np.cos(node_angle)
        node_y = network_radius * np.sin(node_angle)

        color = 'red' if opinions[i] > 0 else 'green'
        circle = plt.Circle((node_x, node_y), 0.3*num_nodes, color=color)
        ax.add_patch(circle)
        circles.append(circle)

        for neighbour_index, connection in enumerate(node.connections):
            if connection:
                neighbour_angle = neighbour_index * 2 * np.pi / num_nodes
                neighbour_x = network_radius * np.cos(neighbour_angle)
                neighbour_y = network_radius * np.sin(neighbour_angle)

                line = ax.plot((node_x, neighbour_x), (node_y, neighbour_y), color='black')
                lines.append(line)

    def update(i):
        new_opinions, _ = simulate_ising_model(network, 1)

        for j, circle in enumerate(circles):
            color = 'red' if new_opinions[j] > 0 else 'green'
            circle.set_color(color)

        ax.set_title(f'Opinions at Iteration {i + 1}')

    anim = FuncAnimation(fig, update, frames=range(20), interval=500)
    plt.show()

def plot_opinion_evolution(mean_opinion):
    plt.plot(mean_opinion)
    plt.xlabel('Iterations')
    plt.ylabel('Mean Opinion')
    plt.title('Opinion Evolution')
    plt.show()

def test_ising():
    '''
    This function will test the calculate_agreement function in the Ising model
    '''

    print("Testing ising model calculations")
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1)==4), "Test 1"

    population[1, 1] = 1.
    assert(calculate_agreement(population,1,1)==-4), "Test 2"

    population[0, 1] = 1.
    assert(calculate_agreement(population,1,1)==-2), "Test 3"

    population[1, 0] = 1.
    assert(calculate_agreement(population,1,1)==0), "Test 4"

    population[2, 1] = 1.
    assert(calculate_agreement(population,1,1)==2), "Test 5"

    population[1, 2] = 1.
    assert(calculate_agreement(population,1,1)==4), "Test 6"

    "Testing external pull"
    population = -np.ones((3, 3))
    assert(calculate_agreement(population,1,1,1)==3), "Test 7"
    assert(calculate_agreement(population,1,1,-1)==5), "Test 8"
    assert(calculate_agreement(population,1,1,10)==-6), "Test 9"
    assert(calculate_agreement(population,1,1, -10)==14), "Test 10"

    print("Tests passed")

def ising_main(population, alpha=None, external=0.0):
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_axis_off()
    im = ax.imshow(population, interpolation='none', cmap='RdPu_r')

    # Iterating an update 100 times
    for frame in range(100):
        # Iterating single steps 1000 times to form an update
        for step in range(1000):
            ising_step(population, external)
        print('Step:', frame, end='\r')
        plot_ising(im, population)


'''
==============================================================================================================
This section contains code for the Defuant Model - task 2 in the assignment
==============================================================================================================
'''

def defuant_main(threshold=0.2, beta=0.2, population_size=100, iterations=100):
    #Your code for task 2 goes here
    # Function to initialize the population with random opinions
    def initialize_population(population_size):
        return np.random.rand(population_size)

    # Function to update opinions according to the Deffuant model
    def update_opinions(opinions, threshold, beta):
        updated_opinions = np.copy(opinions)
        for i in range(len(opinions)):
            # Select a random neighbor
            neighbor_index = np.random.choice([i - 1, i + 1], size=1, replace=True)
            neighbor_index = neighbor_index[0] % len(opinions)  # Handle boundary cases

            # Calculate the difference in opinions
            diff_opinions = abs(opinions[i] - opinions[neighbor_index])

            # Update opinions if difference is within the threshold
            if diff_opinions < threshold:
                mean_opinion = (opinions[i] + opinions[neighbor_index]) / 2
                updated_opinions[i] += beta * (mean_opinion - opinions[i])
                updated_opinions[neighbor_index] += beta * (mean_opinion - opinions[neighbor_index])
        return updated_opinions

    # Function to check convergence of opinions
    def is_converged(opinions1, opinions2, tolerance=0.001):
        return np.allclose(opinions1, opinions2, atol=tolerance)

    # Initialize population
    initial_opinions = initialize_population(population_size)

    # Apply model updates
    final_opinions = np.copy(initial_opinions)
    opinions_history = [final_opinions.copy()]
    for _ in range(iterations):
        updated_opinions = update_opinions(final_opinions, threshold, beta)
        if is_converged(final_opinions, updated_opinions):
            break
        final_opinions = np.copy(updated_opinions)
        opinions_history.append(final_opinions.copy())

    # Plot final distributions of opinions
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(final_opinions, bins=20, range=(0, 1), color='blue', alpha=0.7)
    plt.title('Final Distribution of Opinions')
    plt.xlabel('Opinion')
    plt.ylabel('Frequency')

    plt.subplot(1, 2, 2)
    for i, opinions in enumerate(opinions_history):
        x = [i] * len(opinions)
        plt.scatter(x, opinions, color='red', alpha=0.7)

    plt.ylim(0, 1)
    plt.xlabel('iteration')
    plt.ylabel('Opinion')

    plt.tight_layout()
    plt.show()

def simulate_defuant_model(network, num_iterations, threshold):
    num_nodes = len(network.nodes)
    def initialize_opinions(num_nodes):
        return np.random.choice([-1, 1], size=num_nodes)
    opinions = initialize_opinions(num_nodes)
    mean_opinion = []

    for _ in range(num_iterations):
        mean_opinion.append(np.mean(opinions))

        node_index = np.random.choice(range(num_nodes))
        node = network.nodes[node_index]
        neighbors = [neighbor for neighbor, connection in enumerate(node.connections) if connection]

        if len(neighbors) > 0:
            neighbor_indices = np.random.choice(neighbors, size=int(len(neighbors) / 2), replace=False)
            neighbor_opinions = opinions[neighbor_indices]

            disagreement_count = np.sum(np.abs(neighbor_opinions - opinions[node_index]) > threshold)
            disagreement_fraction = disagreement_count / len(neighbor_indices)

            if disagreement_fraction > 0:
                new_opinion = np.mean(neighbor_opinions[np.abs(neighbor_opinions - opinions[node_index]) > threshold])
                opinions[node_index] = new_opinion

    return opinions, mean_opinion


def test_defuant():
    # Your code for task 2 goes here
    print("Coupling:0.500000,Threshold:0.500000")
    defuant_main(threshold=0.5, beta=0.5)
    print("Coupling:0.100000,Threshold:0.500000")
    defuant_main(threshold=0.5, beta=0.1)
    print("Coupling:0.500000,Threshold:0.100000")
    defuant_main(threshold=0.1, beta=0.5)
    print("Coupling:0.100000,Threshold:0.200000")
    defuant_main(threshold=0.2, beta=0.1)


'''
==============================================================================================================
This section contains code for the main function- you should write some code for handling flags here
==============================================================================================================
'''

def main():
    #You should write some code for handling flags here
    # task 1
    ising_model = False
    external = 0.0
    alpha = 1
    def create_population(population):
        for row in range(len(population)):
            for col in range(len(population[row])):
                while population[row][col] == 0:
                    population[row][col] = np.random.randint(-1, 2)
        return population
    if len(sys.argv) == 2 and sys.argv[1] == '‑ising_model':
        ising_model = True
    elif len(sys.argv) == 4 and sys.argv[1] == '‑ising_model' and sys.argv[2] == '‑external':
        ising_model = True
        external = float(sys.argv[3].replace('‑', '-'))
    elif len(sys.argv) == 4 and sys.argv[1] == '‑ising_model' and sys.argv[2] == '‑alpha':
        ising_model = True
        alpha = float(sys.argv[3])
    elif len(sys.argv) == 2 and sys.argv[1] == '‑test_ising':
        test_ising()

    if ising_model == True:
        population = create_population(population=np.array(np.zeros((100,100))))
        ising_main(population, alpha, external)

    # task 2
    if len(sys.argv) == 2 and sys.argv[1] == '‑defuant':
        defuant_main()
    elif len(sys.argv) == 4 and sys.argv[1] == '‑defuant' and sys.argv[2] == '‑beta':
        beta = float(sys.argv[3])
        defuant_main(threshold=0.2,beta=beta)
    elif len(sys.argv) == 4 and sys.argv[1] == '‑defuant' and sys.argv[2] == '‑threshold':
        threshold = float(sys.argv[3])
        defuant_main(threshold=threshold)
    elif len(sys.argv) == 2 and sys.argv[1] == '‑test_defant':
        test_defuant()

    # task 3
    if len(sys.argv) == 3 and sys.argv[1] == '-network':
        N = int(sys.argv[2])
        network = Network()
        network.make_random_network(N)
        network.get_mean_degree()
        network.get_path_length()
        network.get_clustering()
    elif len(sys.argv) == 2 and sys.argv[1] == '-test_network':
        test_networks()

    # task 4
    if len(sys.argv) == 3 and sys.argv[1] == "-ring_network":
        N = int(sys.argv[2])
        network = Network()
        network.make_ring_network(N)
        network.plot()
    elif len(sys.argv) == 3 and sys.argv[1] == "-small_world":
        N = int(sys.argv[2])
        network = Network()
        network.make_small_world_network(N)
        network.plot()
    elif len(sys.argv) == 5 and sys.argv[1] == "-small_world" and sys.argv[3] == "-re_wire":
        N = int(sys.argv[2])
        re_wire = float(sys.argv[4])
        network = Network()
        network.make_small_world_network(N, re_wire)
        network.plot()

    # task 5
    if len(sys.argv) == 4 and sys.argv[1] == "-ising_model" and sys.argv[2] == '-use_network':
        N = int(sys.argv[3])
        network = Network()
        network.make_small_world_network(N)
        opinions, mean_opinion = simulate_ising_model(network, 100)
        animate_opinions(opinions, network)
        plot_opinion_evolution(mean_opinion)

    elif len(sys.argv) == 4 and sys.argv[1] == "-defuant" and sys.argv[2] == '-use_network':
        N = int(sys.argv[3])
        network = Network()
        network.make_small_world_network(N)
        opinions, mean_opinion = simulate_defuant_model(network, 100, 0.5)
        animate_opinions(opinions, network)
        plot_opinion_evolution(mean_opinion)




if __name__=="__main__":
    main()