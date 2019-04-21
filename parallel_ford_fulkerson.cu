#include <bits/stdc++.h>

#define forward 0
#define backward 1
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

        //cout << source << "\t" << destination << "\t" << capacity << endl;
        residual_capacity[source * total_nodes + destination] = capacity;
    }

    file.close();
}

__global__ void find_augmenting_path(u_short* residual_capacity, Node_info* node_info, bool* frontier, bool* visited, u_int total_nodes, u_int sink){

	int node_id = blockIdx.x * blockDim.x + threadIdx.x;

	if(!frontier[sink] && node_id < total_nodes && frontier[node_id]){

		frontier[node_id] = false;
		visited[node_id] = true;

		Node_info *neighbour;
		Node_info current_node_info = node_info[node_id];
		u_int capacity;

		for(int i=0; i< total_nodes; i++){

			//printf("residual_capacity[%d][%d] - %hu\n", node_id, i, residual_capacity[node_id * total_nodes + i]);

			if(frontier[i] || visited[i] || ((capacity = residual_capacity[node_id * total_nodes + i]) <= 0)){
				continue;
			}

			frontier[i] = true;

			// put an atomic lock
			neighbour = node_info + i;
			neighbour->parent_index = node_id;
			// atomicExch(&(neighbour->potential_flow), min(current_node_info.potential_flow, capacity));
			neighbour->potential_flow =  min(current_node_info.potential_flow, capacity);
			// printf("node_info[node_id].potential_flow, capacity - %u,%hu\n",node_info[node_id].potential_flow, capacity);
			//printf("%d->%d, %u\n",node_id, i, neighbour->potential_flow);

		}
	}
}

__global__ void reset(Node_info* node_info, bool* frontier, bool* visited, int source, int total_nodes){
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if(id < total_nodes){
		frontier[id] = id == source;
		visited[id] = false;
		node_info[id].potential_flow = UINT_MAX;
	}
}

void reset_host(bool* frontier, int source, int total_nodes){
	for (int i = 0; i < total_nodes; i++) {
		frontier[i] = i == source;
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
		printf("Specify filename & number of verices\n");
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
	u_short *d_residual_capacity;
	//char edge_info_matrix[N][N];
	bool* frontier, *visited;
	bool* d_frontier, *d_visited;

	Node_info* node_info;
	Node_info* d_node_info;

    clock_t start_time = clock(); 

	size_t node_infos_size = N * sizeof(Node_info);
	node_info = (Node_info*)malloc(node_infos_size);

	size_t vertices_size = N * sizeof(bool);
	frontier = (bool *)malloc(vertices_size);
	visited = (bool *)malloc(vertices_size);

	for (int i = 0; i < N; ++i) {
		frontier[i] = false;
		visited[i] = false;

		node_info[i].potential_flow = UINT_MAX;

		// for (int j = 0; j < N; ++j) {
		// 	// edge_info_matrix[i][j] = forward;
		// 	residual_capacity[i * N + j] = 0;
		// }
	}

	// residual_capacity[0 * N + 1] = 3;
	// residual_capacity[1 * N + 3] = 2;
	// residual_capacity[1 * N + 2] = 1;
	// residual_capacity[0 * N + 2] = 2;
	// residual_capacity[2 * N + 3] = 3;

	// edge_info_matrix[1][0] = backward;
	// edge_info_matrix[3][1] = backward;
	// edge_info_matrix[2][1] = backward;
	// edge_info_matrix[2][0] = backward;
	// edge_info_matrix[3][2] = backward;

	frontier[0] = true;

	cudaMalloc((void **)&d_residual_capacity, matrix_size);
	cudaMalloc((void **)&d_node_info,node_infos_size);
	cudaMalloc((void **)&d_frontier, vertices_size);
	cudaMalloc((void **)&d_visited, vertices_size);

	cudaMemcpy(d_residual_capacity, residual_capacity, matrix_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_node_info, node_info, node_infos_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_frontier, frontier, vertices_size, cudaMemcpyHostToDevice);
	cudaMemcpy(d_visited, visited, vertices_size, cudaMemcpyHostToDevice);

	bool found_augmenting_path;

	int threads = 256;
	int blocks = ceil(N * 1.0 /threads);

	do{

		// for (int i = 0; i < N; ++i) {
		// 	for (int j = 0; j < N; ++j) {
		// 		printf("residual_capacity[%d][%d] - %hu\n", i, j, residual_capacity[i * N + j]);
		// 	}
		// }

		// reset visited, frontier, node_info
		reset<<<blocks, threads >>>(d_node_info, d_frontier, d_visited, source, N);
		reset_host(frontier, source, N);

		while(!is_frontier_empty_or_sink_found(frontier, N, sink)){
				//printf("is_frontier_empty_or_sink_found\n");
				// Invoke kernel
				find_augmenting_path<<< blocks, threads >>>(d_residual_capacity, d_node_info, d_frontier, d_visited, N, sink);

				// Copy back frontier from device
				cudaMemcpy(frontier, d_frontier, vertices_size, cudaMemcpyDeviceToHost);
		}

		found_augmenting_path = frontier[sink];
		//printf("found_augmenting_path- %d\n", found_augmenting_path);

		if(!found_augmenting_path){
			break;
		}

		// copy node_info from device to host
		cudaMemcpy(node_info, d_node_info, node_infos_size, cudaMemcpyDeviceToHost);

		bottleneck_flow = node_info[sink].potential_flow;
		//pintf("bottleneck_flow, maxflow- %u, %u\n", bottleneck_flow ,max_flow);
		max_flow += bottleneck_flow;
		//cout << "maxflow " << max_flow << endl;
		//printf("maxflow- %u\n", max_flow);

		for(current_vertex = sink; current_vertex != source; current_vertex = current_node_info->parent_index){
			current_node_info = node_info + current_vertex;
			residual_capacity[current_node_info->parent_index * N + current_vertex] -= bottleneck_flow;
			residual_capacity[current_vertex * N + current_node_info->parent_index] += bottleneck_flow;

			// if(edge_info_matrix[current_node_info.parent_index][current_vertex] == forward){
			// 	residual_capacity[current_node_info.parent_index * N + current_vertex] -= bottleneck_flow;
			// 	if(edge_info_matrix[current_node_info.parent_index][current_vertex] != backward){
			// 		edge_info_matrix[current_node_info.parent_index][current_vertex] = backward;
			// 		edge_info_changed = true;
			// 	}
			// 	residual_capacity[current_vertex * N + current_node_info.parent_index] += bottleneck_flow;
			// }else{
			// 	residual_capacity[current_vertex * N + current_node_info.parent_index] -= bottleneck_flow;
			// 	residual_capacity[current_node_info.parent_index * N + current_vertex] += bottleneck_flow;
			// }
		}

		// copy residual_capacity, edge_info to device
		cudaMemcpy(d_residual_capacity, residual_capacity, matrix_size, cudaMemcpyHostToDevice);

	}while(found_augmenting_path);

	cout << "maxflow " << max_flow << endl;
    double time_taken = ((double)clock() - start_time)/CLOCKS_PER_SEC * milliseconds; // in seconds 
	//printf("%f ms",time_taken);
	cout << time_taken << "ms" << endl;

	//free(edge_info_matrix);
	free(residual_capacity);
	free(frontier);
	free(visited);
	free(node_info);

	cudaFree(d_residual_capacity);
	cudaFree(d_node_info);
	cudaFree(d_frontier);
	cudaFree(d_visited);

	return 0;
}
