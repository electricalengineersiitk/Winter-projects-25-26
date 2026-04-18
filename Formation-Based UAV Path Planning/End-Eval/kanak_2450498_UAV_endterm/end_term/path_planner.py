import numpy as np
import heapq
import matplotlib.pyplot as plt
from map_setup import start, goal, is_collision, GRID_SIZE, obs_center, obs_rad

def get_dijkstra_path():
    pq = [(0, start)]  # (priority_cost, current_node)
    close_set = set()
    came_from = {}
    g_score = {start: 0}

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == goal:
            # Path found
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1] 
        if current in close_set:
            continue
        close_set.add(current)

       
        for dx, dy in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            
          
            if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
               
                if not is_collision(neighbor[0], neighbor[1], margin=6):
                    
                   
                    step_cost = np.sqrt(dx**2 + dy**2)
                    tentative_g_score = g_score[current] + step_cost

                    if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g_score
                        heapq.heappush(pq, (g_score[neighbor], neighbor))
    
    return None # No path found

if __name__ == "__main__":
  
    path = get_dijkstra_path()
    
    if path:
        for pt in path:
            print(pt)
            
        print(f"Path found with {len(path)} waypoints!")
        plt.figure(figsize=(6,6))
        
    
        circle = plt.Circle(obs_center, obs_rad, color='r', alpha=0.3)
        plt.gca().add_patch(circle)
        
        
        path_np = np.array(path)
        plt.plot(path_np[:, 0], path_np[:, 1], 'b-', label='Dijkstra Path')
        plt.scatter(*start, color='g', label='Start')
        plt.scatter(*goal, color='k', label='Goal')
        
        plt.xlim(0, GRID_SIZE)
        plt.ylim(0, GRID_SIZE)
        plt.legend()
        plt.title("Phase 2: Valid Collision-Free Path")
        plt.show()
    else:
        print("Failed to find a path. Check your obstacle size/margin.")