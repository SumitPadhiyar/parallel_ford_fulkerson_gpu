#include <bits/stdc++.h>
#define milliseconds 1e3

using namespace std;

typedef struct _Node_info{
	u_short parent_index;
	u_int potential_flow;
} Node_info;

void readInput(const char* filename, u_int total_nodes, u_short* residual_capacity) {

	ifstream file;
	file.open(filename);

	if (!file) {
        cout <<  "Error reading file!";
        exit(1);
    }

    string line;
    u_int source, destination;
    u_short capacity;

    while (file) {

        getline(file, line);

        if (line.empty()) {
            continue;
        }

        std::stringstream linestream(line);
        linestream >> source >> destination >> capacity;
        residual_capacity[source * total_nodes + destination] = capacity;
    }

    file.close();
}

__global__ void find_augmenting_path(u_short* residual_capacity, Node_info* node_info, bool* frontier, bool* visited, 
	u_int total_nodes, u_int sink, u_int* locks){

	int node_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(!frontier[sink] && node_id < total_nodes && frontier[node_id]){

		frontier[node_id] = false;
		visited[node_id] = true;

		Node_info *neighbour;
		Node_info current_node_info = node_info[node_id];
		u_int capacity;

		for (u_int i = node_id; i < total_nodes; ++i){
			
			if(frontier[i] || visited[i] || ((capacity = residual_capacity[node_id * total_nodes + i]) <= 0)){
				continue;
			}

			if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
				continue;
			}

			frontier[i] = true;
			locks[i] = 0;

			neighbour = node_info + i;
			neighbour->parent_index = node_id;
			neighbour->potential_flow =  min(current_node_info.potential_flow, capacity);				
		}
		

		for (u_int i = 0; i < node_id; ++i){
			
			if(frontier[i] || visited[i] || ((capacity = residual_capacity[node_id * total_nodes + i]) <= 0)){
				continue;
			}

			if(atomicCAS(locks+i, 0 , 1) == 1 || frontier[i]){
				continue;
			}

			frontier[i] = true;
			locks[i] = 0;

			neighbour = node_info + i;
			neighbour->parent_index = node_id;
			neighbour->potential_flow =  min(current_node_info.potential_flow, capacity);				
		}
	}
}

__global__ void reset(Node_info* node_info, bool* frontier, bool* visited, int source, int total_nodes, u_int* locks){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < total_nodes){
		frontier[id] = id == source;
		visited[id] = false;
		node_info[id].potential_flow = UINT_MAX;
		locks[id] = 0; 
	}
}

__global__ void augment_path(Node_info* node_infos, bool* do_change_capacity , u_int total_nodes, u_short* residual_capacity, u_int bottleneck_flow){
	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
    if(node_id < total_nodes && do_change_capacity[node_id]){
        Node_info* current_node_info = node_infos + node_id;
        residual_capacity[current_node_info->parent_index * total_nodes + node_id] -= bottleneck_flow;
        residual_capacity[node_id * total_nodes + current_node_info->parent_index] += bottleneck_flow; 
    }    
}


void reset_host(bool* frontier, int source, int total_nodes, bool* do_change_capacity){
	frontier[source] = true;
	do_change_capacity[source] = false;

	for (int i = source+1; i < total_nodes; i++) {
		frontier[i] = false;
		do_change_capacity[i] = false;
	}

	for (int i = 0; i < source; i++) {
		frontier[i] = false;		
		do_change_capacity[i] = false;
	}
}

bool is_frontier_empty_or_sink_found(bool* frontier, int N, int sink_pos){
	for (int i = N-1; i > -1; --i) {
		if(frontier[i]){
			return i == sink_pos;
		}
	}
	return true;
}

int main(int argc, char** argv){

	if(argc < 3){
		printf("Specify filename & number of vertices\n");
		return 1;
	}

	u_int N = atoi(argv[2]);
	u_short *residual_capacity;

	size_t matrix_size = N * N * sizeof(u_short);
	residual_capacity = (u_short *)malloc(matrix_size);
	memset(residual_capacity, 0, matrix_size); 

	readInput(argv[1], N, residual_capacity);

	u_int source=0, sink=N-1;
	u_int current_vertex, bottleneck_flow;
	u_int max_flow = 0;

	Node_info* current_node_info;
	u_short* d_residual_capacity;
	u_int* d_locks;
	bool* frontier;
	bool* d_frontier, *d_visited, *d_do_change_capacity, *do_change_capacity;

	Node_info* node_info;
	Node_info* d_node_info;

    clock_t start_time = clock(); 

	size_t node_infos_size = N * sizeof(Node_info);
	node_info = (Node_info*)malloc(node_infos_size);

	size_t vertices_size = N * sizeof(bool);
	frontier = (bool *)malloc(vertices_size);
	do_change_capacity = (bool *)malloc(vertices_size);

	size_t locks_size = N * sizeof(u_int);

	cudaMalloc((void **)&d_residual_capacity, matrix_size);
	cudaMalloc((void **)&d_locks, locks_size);
	cudaMalloc((void **)&d_node_info,node_infos_size);
	cudaMalloc((void **)&d_frontier, vertices_size);
	cudaMalloc((void **)&d_visited, vertices_size);
	cudaMalloc((void **)&d_do_change_capacity, vertices_size);

	cudaMemcpy(d_residual_capacity, residual_capacity, matrix_size, cudaMemcpyHostToDevice);

	bool found_augmenting_path;

	int threads = 256;
	int blocks = ceil(N * 1.0 /threads);

	do{

		// reset visited, frontier, node_info, locks
		reset<<<blocks, threads >>>(d_node_info, d_frontier, d_visited, source, N, d_locks);
		reset_host(frontier, source, N, do_change_capacity);

		while(!is_frontier_empty_or_sink_found(frontier, N, sink)){
				// Invoke kernel
				find_augmenting_path<<< blocks, threads >>>(d_residual_capacity, d_node_info, d_frontier, d_visited, N, sink, d_locks);

				// Copy back frontier from device
				cudaMemcpy(frontier, d_frontier, vertices_size, cudaMemcpyDeviceToHost);
		}

		found_augmenting_path = frontier[sink];

		if(!found_augmenting_path){
			break;
		}

		// copy node_info from device to host
		cudaMemcpy(node_info, d_node_info, node_infos_size, cudaMemcpyDeviceToHost);

		bottleneck_flow = node_info[sink].potential_flow;
		max_flow += bottleneck_flow;

		for(current_vertex = sink; current_vertex != source; current_vertex = current_node_info->parent_index){
			current_node_info = node_info + current_vertex;
			do_change_capacity[current_vertex] = true;
		}

		cudaMemcpy(d_do_change_capacity, do_change_capacity, vertices_size, cudaMemcpyHostToDevice);

		augment_path<<< blocks, threads >>>(d_node_info, d_do_change_capacity, N, d_residual_capacity, bottleneck_flow);

	}while(found_augmenting_path);

	cout << "\nmaxflow " << max_flow << endl;
    double time_taken = ((double)clock() - start_time)/CLOCKS_PER_SEC * milliseconds; // in milliseconds 
	cout << time_taken << " ms for thread size- " << threads << endl;

	free(residual_capacity);
	free(frontier);
	free(node_info);

	cudaFree(d_residual_capacity);
	cudaFree(d_node_info);
	cudaFree(d_frontier);
	cudaFree(d_visited);

	return 0;
}
