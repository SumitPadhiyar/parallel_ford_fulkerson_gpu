#include <bits/stdc++.h>
#include <cuda_runtime.h>

#define forward 0
#define backward 1

typedef struct _Node_info{
	u_short parent_index;
	u_int potential_flow;
} Node_info;

__global__ void find_augmenting_path(u_int** residual_capacity,
		Node_info* node_info,bool* frontier, bool* visited, u_int total_nodes, u_int sink){

	int node_id = blockIdx.x * blockDim.x + threadIdx.x;
	if(!frontier[sink] && node_id < total_nodes && frontier[node_id]){

		frontier[node_id] = false;
		visited[node_id] = true;

		Node_info neighbour;
		u_int capacity;

		for(int i=0; i<total_nodes;i++){

			if(frontier || visited || (capacity = residual_capacity[node_id][i] <= 0)){
				continue;
			}

			frontier[i] = true;
			neighbour = node_info[i];
			neighbour.parent_index = node_id;
			neighbour.potential_flow = min(node_info[node_id].potential_flow, capacity);
		}
	}
}

__global__ void reset(Node_info* node_info,bool* frontier, bool* visited, int source, int total_nodes){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id<total_nodes){
		frontier[id] = id == source;
		visited[id] = false;
		Node_info current_node_info = node_info[id];
		current_node_info.potential_flow = UINT_MAX;
	}
}

bool is_frontier_empty_or_sink_found(bool* frontier, int N, int sink_pos){
	while(N--){
		if(frontier[N]){
			return N == sink_pos;
		}
	}
	return true;
}

int main(){

	unsigned int N = 10;
	unsigned int source=0, sink=N-1;
	unsigned int current_vertex, bottleneck_flow;
	Node_info current_node_info;

	unsigned int max_flow = 0;

	u_int **residual_capacity, **d_residual_capacity;
	char **edge_info_matrix,**d_edge_info_matrix;

	bool* frontier, *visited;
	bool* d_frontier, *d_visited;

	Node_info* node_info;
	Node_info* d_node_info;

	size_t matrix_size = N * N * sizeof(u_int);
	residual_capacity = (u_int **)malloc(matrix_size);

	size_t edge_info_size = N * N * sizeof(char);
	edge_info_matrix = (char **)malloc(edge_info_size);

	size_t node_infos_size = N * sizeof(Node_info);
	node_info = (Node_info*)malloc(node_infos_size);

	size_t vertices_size = N * sizeof(bool);

	frontier = (bool *)malloc(vertices_size);
	visited = (bool *)malloc(vertices_size);

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			edge_info_matrix[i][j] = forward;
		}
	}

	frontier[0] = true;

	//cudaMalloc((char **)d_edge_info_matrix,edge_info_size);

	cudaMalloc((ushort **)d_residual_capacity,matrix_size);

	cudaMalloc((Node_info**)d_node_info,node_infos_size);

	cudaMalloc((bool **)d_frontier, vertices_size);
	cudaMalloc((bool **)d_visited, vertices_size);

	cudaMemcpy(d_residual_capacity, residual_capacity, matrix_size, cudaMemcpyHostToDevice);
	//cudaMemcpy(d_edge_info_matrix, edge_info_matrix, edge_info_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_node_info, node_info, node_infos_size, cudaMemcpyHostToDevice);

	cudaMemcpy(d_frontier, frontier, vertices_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_visited, visited, matrix_size, cudaMemcpyHostToDevice);

	bool found_augmenting_path;

	int threads = 256;
	int blocks = ceil(N * 1.0 /threads);

	do{

		while(!is_frontier_empty_or_sink_found(frontier, N,sink)){

				// Invoke kernel
				find_augmenting_path<<< blocks, threads >>>(d_residual_capacity, d_node_info, d_frontier, d_visited, N, sink);

				// Copy back frontier from device
				cudaMemcpy(frontier, d_frontier, vertices_size, cudaMemcpyDeviceToHost);
		}

		found_augmenting_path = frontier[sink];

		if(!found_augmenting_path){
			continue;
		}

		// copy node_info from device to host
		cudaMemcpy(node_info, d_node_info, node_infos_size, cudaMemcpyDeviceToHost);

		bottleneck_flow = node_info[sink].potential_flow;
		max_flow += bottleneck_flow;

		bool edge_info_changed = false;
		for(current_vertex = sink; current_vertex != source; current_vertex = current_node_info.parent_index){
			current_node_info = node_info[sink];
			if(edge_info_matrix[current_node_info.parent_index][current_vertex] == forward){
				residual_capacity[current_node_info.parent_index][current_vertex] -= bottleneck_flow;
				if(edge_info_matrix[current_node_info.parent_index][current_vertex] != backward){
					edge_info_matrix[current_node_info.parent_index][current_vertex] = backward;
					edge_info_changed = true;
				}
				residual_capacity[current_vertex][current_node_info.parent_index] += bottleneck_flow;
			}else{
				residual_capacity[current_vertex][current_node_info.parent_index] -= bottleneck_flow;
				residual_capacity[current_node_info.parent_index][current_vertex] += bottleneck_flow;
			}
		}

		// copy residual_capacity, edge_info to device
		cudaMemcpy(d_residual_capacity, residual_capacity, matrix_size, cudaMemcpyHostToDevice);

//		if(edge_info_changed){
//			cudaMemcpy(d_edge_info_matrix, edge_info_matrix, edge_info_size, cudaMemcpyHostToDevice);
//		}


		// reset visited, frontier, node_info
		reset<<<blocks, threads >>>(d_node_info,frontier, visited, source, N);

	}while(found_augmenting_path);

	return 0;
}
