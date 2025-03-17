import numpy as np
import random
import matplotlib.pyplot as plt

# Set random seed for reproducibility
random.seed(42)

# Define locations, weights, and prizes
locations = np.array([
    [0, 0], [1, 3], [4, 3], [6, 1], [3, 0],
    [7, 8], [9, 2], [2, 7], [8, 3], [4, 5],
    [6, 4], [1, 1], [9, 6], [3, 8], [5, 9],
    [7, 1], [2, 5], [3, 6], [8, 9], [5, 7],
    [9, 9], [2, 4], [4, 7], [6, 6], [8, 8],
    [3, 9], [7, 7], [5, 8], [6, 2], [8, 5],
    [2, 3], [1, 5], [4, 9], [7, 3], [5, 2],
    [6, 9], [8, 7], [3, 2], [4, 6], [7, 6]
])

weights = np.array([
    # Define weights for each location
    0 ,40 ,35 ,55 ,70 ,
    65 ,30 ,45 ,50 ,60 ,
    75 ,55 ,65 ,70 ,85 ,
    25 ,30 ,55 ,40 ,35 ,
    50 ,60 ,45 ,55 ,65 ,
    70 ,35 ,75 ,40 ,50 ,
    30 ,65 ,45 ,60 ,55 ,
    80 ,70 ,30 ,65 ,50
])

prizes = np.array([
    # Define prizes for each location
   0 ,35 ,40 ,55 ,65 ,
   50 ,45 ,70 ,55 ,60 ,
   75 ,30 ,65 ,80 ,85 ,
   40 ,45 ,50 ,60 ,55 ,
   65 ,70 ,35 ,75 ,85 ,
   60 ,50 ,45 ,65 ,55 ,
   70 ,30 ,55 ,80 ,75 ,
   85 ,45 ,60 ,50 ,65 
])

capacity =200

# Calculate distances between locations
def calculate_distance_matrix(locations):
    return np.linalg.norm(locations[:, None] - locations[None,:], axis=2)

distance_matrix = calculate_distance_matrix(locations)

# Implement a routing algorithm that considers capacity constraints
routes = []
visited_all_nodes = set()   # Track all visited nodes

total_prizes = []
best_cost_history = []
average_cost_history = []

# Main loop to generate routes until all nodes are visited
while len(visited_all_nodes) < len(locations)-1:
    current_node =0   # Start from the first node (node '0')
    total_weight = weights[current_node]   # Start with the weight of the starting node
    total_cost =0 # Track travel cost for this route
    route = []   # Start the route at node '0'
    
    total_prize = prizes[current_node] # Add the prize of the starting node
    route.append(current_node) # Start the route with the starting node

    while True:
        best_next_node = None
        best_next_distance = float('inf')

        for next_node in range(1,len(locations)):   # Start from node index '1' to avoid revisiting '0'
            if next_node not in visited_all_nodes:   # Check if the node has been visited
                weight_if_visited = total_weight + weights[next_node]
                if weight_if_visited <= capacity:
                    distance_to_next = distance_matrix[current_node][next_node]
                    if distance_to_next < best_next_distance:
                        best_next_distance = distance_to_next
                        best_next_node = next_node

        if best_next_node is not None:
            total_weight += weights[best_next_node]
            total_prize += prizes[best_next_node]
            total_cost += distance_matrix[current_node][best_next_node] # Add travel cost to next node
            route.append(best_next_node)
            visited_all_nodes.add(best_next_node)   # Mark this node as visited
            current_node = best_next_node
        else:
            break   # Exit loop if no valid next node is found

    total_cost += distance_matrix[current_node][0] # Add cost to return to start (node '0')
    
    route.append(0)   # Return to start (node '0')
    
    routes.append(route)
    
    total_prizes.append(total_prize) # Log total prize for the current route
    
    # Track best and average cost for this generation (route)
    best_cost_history.append(total_cost)
    
# Calculate average costs per generation (in this case per route)
average_cost_history = [
    sum(best_cost_history[:i+1]) / (i+1) for i in range(len(best_cost_history))
]

# Print the routes generated and total prizes
for i in range(len(routes)):
    print(f"Route {i +1}: {routes[i]} (Total Prize: {total_prizes[i]})")

# Function to plot routes and prizes
def plot_routes(locations,routes,total_prizes):
    
    plt.figure(figsize=(12 ,8))
    
    colors=['r', 'g', 'b', 'c', 'm', 'y']
    
    for i in range(len(routes)):
        route = routes[i]
        plt.plot(locations[route][: ,[0]] ,
                 locations[route][: ,[1]] ,'o-',
                 color=colors[i % len(colors)],
                 label=f'Route {i +1} (Prize: {total_prizes[i]})')
    
    plt.title('Generated Routes')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    
    plt.legend()
    
    plt.grid()
    
    plt.show()

# Plotting the routes
plot_routes(locations,routes,total_prizes)

# Plotting the average and best cost values per generation
plt.figure(figsize=(10,6))
plt.plot(best_cost_history,label='Best Cost', marker='o')
plt.plot(average_cost_history,label='Average Cost', marker='x', linestyle='--')
plt.title('Best and Average Costs per Generation')
plt.xlabel('Generation')
plt.ylabel('Cost')
plt.legend()
plt.grid()
plt.show()
