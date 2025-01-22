
// CUDA kernel function
#include "cuda_runtime.h"
#include "cuda_runtime_api.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <string>
#include <iostream>
#include <algorithm>




void __syncthreads();





__global__ void mlp_forward_cuda_kernel(float *activations, float *weights, float *biases, int *shape, int shape_length, int nr_inputs) {
    //printf("threadIdx.x: %d   threadIdx.y: %d   blockIdx.x: %d   blockIdx.y: %d   blockDim.x: %d   blockDim.y: %d   gridDim.x: %d   gridDim.y: %d\n", 
    //threadIdx.x, threadIdx.y, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, gridDim.x, gridDim.y);
    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    int idx =  threadIdx.x + blockIdx.x * blockDim.x, idy = threadIdx.y + blockIdx.y * blockDim.y;
    int layer_offset_b = 0;
	int layer_offset_weights = 0;
	int layer_offset_input_activations = 0;
    int layer_offset_output_activations = shape[0] * blockDim.y * gridDim.y;
    for (int shape_index = 0; shape_index < shape_length; shape_index++) {
        if (idx < shape[shape_index+1] && idy < nr_inputs) {
            int nr_input_neurons = shape[shape_index];
            int nr_output_neurons = shape[shape_index+1];
            for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++) {
                activations[layer_offset_output_activations + idy * nr_output_neurons + idx] += 
                activations[layer_offset_input_activations + idy * nr_input_neurons + neuron_nr] * 
                weights[layer_offset_weights + idx * nr_input_neurons + neuron_nr];
                // printf("idx: %d idy: %d neuron_nr: %d shape_index: %d   -   %d         /          a * w += z   ;   %f  *  %f  =  %f\n", idx, idy, neuron_nr, shape_index,
                // layer_offset_output_activations + idy * nr_output_neurons + idx,
                // activations[layer_offset_input_activations + idy * nr_input_neurons + neuron_nr],
                // weights[layer_offset_weights + idx * nr_input_neurons + neuron_nr],
                // activations[layer_offset_output_activations + idy * nr_output_neurons + idx]);
            }
            activations[layer_offset_output_activations + idy * nr_output_neurons + idx] += biases[layer_offset_b + idx];
            activations[layer_offset_output_activations + idy * nr_output_neurons + idx] = 1.0 / (1.0 + exp(-activations[layer_offset_output_activations + idy * nr_output_neurons + idx]));
        }

        layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
		layer_offset_b += shape[shape_index + 1];
		layer_offset_input_activations = layer_offset_output_activations;
        layer_offset_output_activations += shape[shape_index + 1] * blockDim.y * gridDim.y;

        __syncthreads(); 
    }
    //printf("cuda kernel end");
}

__global__ void mlp_backward_cuda_kernel(float *activations, float *weights, float *weight_deltas, float *error_terms, int *shape, int shape_length, int nr_weights, int nr_inputs) {
    //int id = blockIdx.x * blockDim.x + threadIdx.x;
    //int idx = threadIdx.x, idy = threadIdx.y;
    int idx =  threadIdx.x + blockIdx.x * blockDim.x, idy = threadIdx.y + blockIdx.y * blockDim.y;
    //int layer_offset_b = 0;
	//int layer_offset_b = 0;
    int layer_offset_b = 0;
	int layer_offset_weights = 0;
	int layer_offset_input_activations = 0;
    int layer_offset_output_activations = shape[0] * blockDim.y * gridDim.y;
    for (int shape_index = 0; shape_index < shape_length-1; shape_index++) {
        if (idx < shape[shape_index] && idy < nr_inputs) {
            error_terms[layer_offset_input_activations + idy * shape[shape_index] + idx] *= activations[layer_offset_input_activations + idy * shape[shape_index] + idx] * 
                (1 - activations[layer_offset_input_activations + idy * shape[shape_index] + idx]);
            
            //bias_deltas[layer_offset_b + idx] += error_terms[layer_offset_input_activations + idy * shape[shape_index] + idx];
        }

        __syncthreads();

        if (idx < shape[shape_index+1] && idy < nr_inputs) {
            //error_terms[layer_offset_activations + id] *= activations[layer_offset_activations + id] * (1 - activations[layer_offset_activations + id]);
            int nr_input_neurons = shape[shape_index];
            int nr_output_neurons = shape[shape_index+1];
            for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++) {
                //std::cout << "id: " << id << " neuron_nr: " << neuron_nr << " shape_index: " << shape_index << std::endl;
                error_terms[layer_offset_output_activations + idy * nr_output_neurons + idx] += 
                error_terms[layer_offset_input_activations + idy * nr_input_neurons + neuron_nr] * weights[layer_offset_weights + nr_output_neurons * neuron_nr + idx];
                //printf("id: %d neuron_nr: %d shape_index: %d     -  er * w  /  %f  *  %f\n", id, neuron_nr, shape_index, 
                //error_terms[layer_offset_activations + neuron_nr],
                //weights[layer_offset_weights + nr_neurons * neuron_nr + id]);
                weight_deltas[idy * nr_weights + layer_offset_weights + nr_output_neurons * neuron_nr + idx] += 
                error_terms[layer_offset_input_activations + idy * nr_input_neurons + neuron_nr] * activations[layer_offset_output_activations + idy * nr_output_neurons + idx];

                // weight_deltas[layer_offset_weights + id * nr_neurons + neuron_nr] += 
                // error_terms[layer_offset_activations + id] * activations[layer_offset_activations + shape[shape_index] + neuron_nr];
            }
        }

        layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
        layer_offset_b += shape[shape_index + 1];
		layer_offset_input_activations = layer_offset_output_activations;
        layer_offset_output_activations += shape[shape_index + 1] * blockDim.y * gridDim.y;

        __syncthreads();
    }
}

// __global__ void mlp_backward_cuda_kernel(float *activations, float *weights, float *weight_deltas, float *error_terms, int *shape, int shape_length) {
//     //int id = blockIdx.x * blockDim.x + threadIdx.x;
//     int id = threadIdx.x;
//     int layer_offset_b = 0;
// 	int layer_offset_weights = 0;
// 	int layer_offset_activations = 0;
//     for (int shape_index = 0; shape_index < shape_length; shape_index++) {
//         if (id < shape[shape_index+1]) {
//             error_terms[layer_offset_activations + id] *= activations[layer_offset_activations + id] * 
//                 (1 - activations[layer_offset_activations + id]);
//         }

//         __syncthreads();

//         if (id < shape[shape_index+1]) {
//             int nr_input_neurons = shape[shape_index];
//             for (int neuron_nr = 0; neuron_nr < nr_input_neurons; neuron_nr++) {
//                 error_terms[layer_offset_activations + shape[shape_index] + id] += 
//                 error_terms[layer_offset_activations + neuron_nr] * weights[layer_offset_weights + id * nr_input_neurons + neuron_nr];

//                 weight_deltas[layer_offset_weights + id * nr_input_neurons + neuron_nr] += 
//                 error_terms[layer_offset_activations + neuron_nr] * activations[layer_offset_activations + shape[shape_index] + id];
//             }
//         }

//         layer_offset_weights += shape[shape_index] * shape[shape_index + 1];
// 		layer_offset_b += shape[shape_index + 1];
// 		layer_offset_activations += shape[shape_index];

//         __syncthreads(); 
//     }
// }

// Wrapper function to call the CUDA kernel
extern "C" void mlp_forward_cuda(float *activations, float *weights, float *biases, int *shape, int shape_length, int nr_neurons, int nr_weights, int nr_biases, int nr_threads, int nr_inputs) {
    // Allocate device memory
    float *d_activations, *d_weights, *d_biases;
    int *d_shape;
    cudaMalloc((void**)&d_activations, (nr_neurons*nr_inputs) * sizeof(float));
    cudaMalloc((void**)&d_weights, nr_weights * sizeof(float));
    cudaMalloc((void**)&d_biases, nr_biases * sizeof(float));
    cudaMalloc((void**)&d_shape, shape_length * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_activations, activations, (nr_neurons*nr_inputs) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, nr_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_biases, biases, nr_biases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, shape_length * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    // int threadsPerBlock = 4;  ----- needed
    //int blocksPerGrid = (nr_threads + threadsPerBlock - 1) / threadsPerBlock;       ------ needed
    //mlp_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_activations, d_weights, d_biases, shape, shape_length);
    //dim3 thread_block_dimension(nr_threads, nr_inputs);
    //std::cout << "before kernel, nr_threads: " << nr_threads << std::endl;
    int threadsPerBlock = 1024;
    int blocksPerGrid = (nr_threads*nr_inputs + threadsPerBlock - 1) / threadsPerBlock;
    //mlp_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_activations, d_weights, d_biases, shape, shape_length);
    dim3 thread_block_dimension(threadsPerBlock);
    dim3 grid_dimension(blocksPerGrid, nr_inputs);
    mlp_forward_cuda_kernel<<<grid_dimension, thread_block_dimension>>>(d_activations, d_weights, d_biases, d_shape, shape_length, nr_inputs);
    //std::cout << "after kernel" << std::endl;
    //my_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_output2, size);
    // Copy result back to host
    cudaMemcpy(activations, d_activations, (nr_neurons*nr_inputs) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_activations);
    cudaFree(d_weights);
    cudaFree(d_biases);
    cudaFree(d_shape);
}

extern "C" void mlp_backward_cuda(float *activations, float *weights, float *weight_deltas, float *error_terms, int *shape, int shape_length, int nr_neurons, int nr_weights, int nr_biases, int nr_threads, int nr_inputs) {
    // Allocate device memory
    float *d_activations, *d_weights, *d_weight_deltas, *d_error_terms;
    int *d_shape;
    cudaMalloc((void**)&d_activations, (nr_neurons*nr_inputs) * sizeof(float));
    cudaMalloc((void**)&d_weights, nr_weights * sizeof(float));
    cudaMalloc((void**)&d_weight_deltas, (nr_weights*nr_inputs) * sizeof(float));
    //cudaMalloc((void**)&d_bias_deltas, nr_biases* sizeof(float));
    cudaMalloc((void**)&d_error_terms, (nr_neurons*nr_inputs) * sizeof(float));
    cudaMalloc((void**)&d_shape, shape_length * sizeof(int));

    // Copy input data to device
    cudaMemcpy(d_activations, activations, (nr_neurons*nr_inputs) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, weights, nr_weights * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight_deltas, weight_deltas, (nr_weights*nr_inputs) * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_bias_deltas, bias_deltas, nr_biases * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_error_terms, error_terms, (nr_neurons*nr_inputs) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_shape, shape, shape_length * sizeof(int), cudaMemcpyHostToDevice);

    // Launch CUDA kernel
    int threadsPerBlock = 1024;
    int blocksPerGrid = (nr_threads*nr_inputs + threadsPerBlock - 1) / threadsPerBlock;
    //mlp_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_activations, d_weights, d_biases, shape, shape_length);
    dim3 thread_block_dimension(threadsPerBlock);
    dim3 grid_dimension(blocksPerGrid, nr_inputs);
    mlp_backward_cuda_kernel<<<grid_dimension, thread_block_dimension>>>(d_activations, d_weights, d_weight_deltas, d_error_terms, d_shape, shape_length, nr_weights, nr_inputs);
    //my_cuda_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, d_output2, size);
    // Copy result back to host
    cudaMemcpy(weight_deltas, d_weight_deltas, (nr_weights*nr_inputs) * sizeof(float), cudaMemcpyDeviceToHost);
    //cudaMemcpy(bias_deltas, d_bias_deltas, nr_biases * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(error_terms, d_error_terms, (nr_neurons*nr_inputs) * sizeof(float), cudaMemcpyDeviceToHost);
    // Free device memory
    cudaFree(d_activations);
    cudaFree(d_weights);
    cudaFree(d_weight_deltas);
    //cudaFree(d_bias_deltas);
    cudaFree(d_error_terms);
    cudaFree(d_shape);
}