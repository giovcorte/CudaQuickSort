#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/**
 * nvcc quicksort.cu -o a.out for compiling.
 * ./a.out n i for executing, where n is the size of the desidered input and i is the type of the input:
 * 0 -> random array.
 * 1 -> ascending array.
 * 2 -> descending array.
 * 3 -> partialy ordered array.
 */

// Convenience procedure to select the minimum value between two number.
#define MIN(a, b) ((a) < (b) ? (a) : (b))

// Number of threads in block. 
// You can tune this value based on your GPU card and input size.
// This algorithm is covenient when the input is big, so the default value is set to an efficient value.
#define THREADS   256

// Max number of sequences to create.
#define MAX_SEQS (1 << 15)

// Partition blocks on the GPU.
// dst array is the auxiliary array to copy the new data, while the src array is the source (original) array.
// blocks array contains the output indexes, pivots and other informations for the blocks.
// seqs array is the state array, to comunicate the two new pivots (one for l-t-p subsequence and one for the g-t-p one) to the host.
__global__ void partition_kernel(int * dst, int * src, int * blocks, int * seqs)
{
    __shared__ int lcs[THREADS]; // Array for every thread (in this block) less-than-pivot item count.
    __shared__ int gcs[THREADS]; // Array for every thread (in this block) greater-than-pivot item count.
    __shared__ int lo; // Lower-than-pivot output start position for this block.
    __shared__ int go; // Greater-than-pivot output start position for this block.

    int * blk = blocks + 4 * blockIdx.x; // Subsequence for this threadblock. Access to the seq of block structure.
    int * seq = seqs + 8 * blk[2]; // Sequence the subsequence belongs to, takes the third int from block struct, the sequence identifier. Select the correct state.

    int tid = threadIdx.x;
    int beg = blk[0] + tid; // Begin of the subsequence for this thread.
    int end = blk[1]; // End of the subsequence for this thread.
    int piv = seq[4]; // Pivot of this sequence.

    int lc = 0, gc = 0; // Count of lower and greter elements than the pivot in the current thread.

    // Each thread count values greater or lesser than the pivot.
    for (int i = beg; i < end; i += THREADS)
    {
        if (src[i] < piv)
        {
            lc++;
        }
        else if (src[i] > piv) 
        {
            gc++;
        }
    }

    lcs[tid] = lc; // Each thread assign his less-than-pivot items count to his position in the shared array.
    gcs[tid] = gc; // Each thread assign his greater-than-pivot items count to his position in the shared array.
    
    __syncthreads(); // Wait for all threads to have populate the less and greater than-pivot counts (shared).

    // First thread performs a simple sequential exclusive prefix scan of the blockwide counts.
    // After this scan the lcs and gcs arrays will contains the start index from which the thread will write less and greater than-pivot values.
    if (tid == 0)
    {
        int sl = lcs[0], sg = gcs[0]; // Initial count of lower-than-pivot items and count of greater-than-pivot items are taken from the first thread.
        lcs[0] = gcs[0] = 0; // Initialize at 0 for the first thread (obviously).
        for (int i = 1; i < THREADS; ++i) // For each thread calculate his two output position.
        {
            int y = lcs[i]; // Copy in a temporany variable "y" the count of lower-than-pivot items of the i-th thread.
            lcs[i] = sl; // The start index for the insertion of less-than-pivot values for the i-th thread is set as the current sl.
            sl += y; // Prepare start-lower index for the next iteration.
            y = gcs[i]; // Now we reuse y copy in a temporany variable the count of greater-than-pivot items of the current thread.
            gcs[i] = sg; // The start index for the insertion of greater-than-pivot values for the i-th thread is set as the current sg.
            sg += y; // Prepare start-greter index for the next iteration.
        }

        // Output index for this block is allocated by first thread as well.
        // Those indexes are sl, total count of items l-t-p in this block, and sg, total count of items g-t-p in this block.
        lo = atomicAdd(&seq[2],  sl); // Count forward from the start of this sequence and get old value as output index.
        go = atomicAdd(&seq[3], -sg) - sg; // Count backward from the end of this sequence and get old value offset with sum as output index.
    }

    __syncthreads(); // wait for the first thread to complete the setup of the output positions for each thread.

    // Calculate output position for this thread.
    int loff = lo + lcs[tid]; // Start index for the l-t-p values for the current thread contextual of his block -> l (start of data) + block l-t-p items count + thread offset.
    int goff = go + gcs[tid]; // Start index for the g-t-p values for the current thread contextual of his block -> r (end of data) - block g-t-p items count + thread offset.

    // Now move data into the output at the correct position in the aux array.
    // Increment i by the number of threads of this block because each thread in a certain block starts the output position from the thread id plus his id.
    for (int i = beg; i < end; i += THREADS)
    {
        if (src[i] < piv) 
        {
            dst[loff++] = src[i];
        }
        else if (src[i] > piv)
        {
            dst[goff++] = src[i];
        }
    }

    // Last block writes out new partition information and their pivot values.
    // We know we are in the last block thanks to the count put at index 6 of state array of this sequence.
    if (tid == 0)
    {
        if (atomicAdd(&seq[6], -1) == 1)
        {
            // Fill in the gap with pivot values
            for (int i = seq[2]; i < seq[3]; ++i)
            {
                dst[i] = piv;
            }

            seq[4] = dst[(seq[0] + seq[2]) / 2]; // Find the pivot for the lower-than-previous pivot section for the call.
            seq[5] = dst[(seq[3] + seq[1]) / 2]; // Find the pivot for the greater-than-previous pivot section for the next call.
        }
    }
}

void quick_sort(int array[], int left, int right);

// Quicksort implementation in CUDA
void gpu_quick_sort(int * array, int length)
{
    int * d_data, * d_aux, * d_state, * d_blocks;

    if (length <= 1) 
    {
        return;
    }

    struct block
    {
        int beg, end, seq, unused;
    };
    
    struct seq
    {
        int beg, end, piv, unused;
    };

    struct state
    {
        int beg1, end1; // Old start and end index of the lower and greater than pivot values.
        int beg2, end2; // New start and end index of the lower and greater than pivot values.
        int piv1, piv2; // Pivots for the l-t-p section and the g-t-p section.
        int num, unused; // Sequence number.
    };

    // Working arrays.
    struct block * blocks = (struct block *)malloc(2 * MAX_SEQS * sizeof(*blocks)); // Array containing the blocks' informations like stard and end indexes to write the resuls.
    struct seq   * work   = (struct seq   *)malloc(2 * MAX_SEQS * sizeof(*work)); // Queue of sequences to be partitioned.
    struct seq   * done   = (struct seq   *)malloc(2 * MAX_SEQS * sizeof(*done)); // Queue of sequences ready to be merged.
    struct state * state  = (struct state *)malloc(2 * MAX_SEQS * sizeof(*state)); // State array for passing data between host and device.

    // Macro to add new sequences.
    #define PUSH_SEQ(buf, num, from, to, pivot) \
        { buf[num].beg = from; buf[num].end = to; buf[num].piv = pivot; num++; }

    int pivot, num_work, num_done;

    // Allocate memory on the GPU.
    cudaMalloc(&d_state,  2 * MAX_SEQS * sizeof(struct state));
    cudaMalloc(&d_blocks, 2 * MAX_SEQS * sizeof(struct block));
    cudaMalloc(&d_data, length * sizeof(int));
    cudaMalloc(&d_aux, length * sizeof(int));

    // Minimum length sequence to partition further.
    int min_length = (length + MAX_SEQS  - 1) / MAX_SEQS;

    // Median-of-three pivot value.
    {
        pivot = array[length - 1];

        int a = array[0];
        int b = array[length / 2];

        if (a > b)
        {
            if (b < pivot) pivot = b;
        }
        else
        {
            if (a < pivot) pivot = a;
        }
    }

    // Copy input data to GPU.
    cudaMemcpy(d_data, array, sizeof(int) * length, cudaMemcpyHostToDevice);

    // First work item.
    work[0].beg = 0;
    work[0].end = length;
    work[0].piv = pivot;
    
    num_work = 1; // Count representing how many work sequences to process are in the work to do queue. At the start there is only one sequence, the whole array.
    num_done = 0; // Count representing how many done sequences there are.

    // Create and partion sequences until they are short enough, or until there is a max number of them.
    while (num_work && (num_work + num_done) < MAX_SEQS)
    {
        // Total length of sequences. Starts with the total length, because the current work is only the first, comprehensive of the whole array.
        int cur_length = 0;
        for (int i = 0; i < num_work; ++i) 
        {
            cur_length += work[i].end - work[i].beg;
        }

        int block_length = (cur_length + MAX_SEQS - 1) / MAX_SEQS; // Size of each blocks to create is length divided the number of maximum allowed sequencesd.

        int num_blocks = 0; // Block count.

        for (int i = 0; i < num_work; ++i)
        {
            // Number of blocks to create for this sequence.
            int block_count = (work[i].end - work[i].beg + block_length - 1) / block_length;
           
            for (int j = work[i].beg; j < work[i].end; j += block_length) // To each block info data we set the start and end indexes, plus his index.
            {
                blocks[num_blocks].beg = j;
                 // Cap to end of current sequence, to not go out of bound with the last element.
                blocks[num_blocks].end = MIN(work[i].end, j + block_length);
                blocks[num_blocks].seq = i;

                num_blocks++;
            }

            // Set up the GPU states, one for each subsequence.
            state[i].beg1 = state[i].beg2 = work[i].beg;
            state[i].end1 = state[i].end2 = work[i].end;
            state[i].piv1 = work[i].piv;
            state[i].num  = block_count;
        }

        // Copy the new state and the new blocks to the GPU.
        cudaMemcpy(d_state, state, num_work * sizeof(state[0]), cudaMemcpyHostToDevice);
        cudaMemcpy(d_blocks, blocks, num_blocks * sizeof(blocks[0]), cudaMemcpyHostToDevice);

        // Launch partition kernel with num_blocks threadblocks.
        partition_kernel<<<num_blocks, THREADS>>>(d_aux, d_data, d_blocks, d_state);

        // Copy new sequences to work buffer. Without this step, sequences that end earlier than others could be left behind.
        cudaMemcpy(d_data, d_aux, sizeof(int) * length, cudaMemcpyDeviceToDevice);

        // Copy back new sequences from the GPU re-using the state array.
        cudaMemcpy(state, d_state, num_work * sizeof(state[0]), cudaMemcpyDeviceToHost);

        int num_new = num_work; // The number of the sequences partitioned is equal to the number of works running, each work partions a sequence.
        num_work = 0; // Reset the work count for the next iteration, the current works are terminated as we exit from the partitioning.

        // Create new sequences if they are long enough, otherwise push them to the done array. 
        // So if the sequences are short to have an advantage to be partitioned, sort them with a serial algorithm.
        // This is the recursion on the l-t-p and g-t-p sides made iterative, we push to the work queue the two sequences and their pivot.
        // For example, after the first iteration, num_work will be incremented to 2 by the two PUSH_SEQ call, because of the initial partitoning (assuming a long enough array).
        for (int i = 0; i < num_new; ++i)
        {
            int beg, end;
            beg = state[i].beg1;
            end = state[i].beg2;
            
            if ((end - beg) >= min_length) 
            {
                PUSH_SEQ(work, num_work, beg, end, state[i].piv1)
            }
            else
            {                           
                PUSH_SEQ(done, num_done, beg, end, state[i].piv1)
            }

            beg = state[i].end2;
            end = state[i].end1;

            if ((end - beg) >= min_length) 
            {
                PUSH_SEQ(work, num_work, beg, end, state[i].piv2)
            }
            else
            {      
                PUSH_SEQ(done, num_done, beg, end, state[i].piv2)
            }
        }
    }

    // Put rest of work into done array.
    // The remaining work is exceeding the max sequences allowed, so we sort it in a serial way.
    for (int i = 0; i < num_work; ++i) 
    {
        done[num_done++] = work[i];
    }

    // Copy back partitioned data from the GPU.
    cudaMemcpy(array, d_data, sizeof(int) * length, cudaMemcpyDeviceToHost);

    // Sort the remaining sequences.
    // beg and end indexes are containing values less or greater tha a pivot, so the sorting is to re-ordinate "correct" data.
    for (int i = 0; i < num_done; ++i) 
    {
        quick_sort(array, done[i].beg, done[i].end - 1);
    }

    #undef PUSH_SEQ

    // Free allocations
    cudaFree(d_state);
    cudaFree(d_data);
    cudaFree(d_aux);
    cudaFree(d_blocks);
    free(blocks);
    free(work);
    free(done);
    free(state);
}

// Serial implementation of QuickSort algorithm.
void quick_sort(int array[], int left, int right)
{
    int i, j, pivot, y;
    i = left;
    j = right;

    pivot = array[(i + j) / 2];

    while (i <= j)
    {
        while (array[i] < pivot && i < right)
        {
            i++;
        }
        while (array[j] > pivot && j > left)
        {
            j--;
        }
        if (i <= j)
        {
            y = array[i];
            array[i] = array[j];
            array[j] = y;
            i++;
            j--;
        }
    }
    // Recursive call for the function to the left part of the array.
    if (j > left)
    {
        quick_sort(array, left, j);
    }

    // Recursive call for the function to the right part of the array.
    if (i < right)
    {
        quick_sort(array, i, right);
    }
}

int main(int argc, char ** argv)
{
    int n = 10000000;

    if (argc > 1)
    {
	    n = atoi(argv[1]);
    }

    int * a, * b;

    a = (int*)malloc(n * sizeof(int));
    b = (int*)malloc(n * sizeof(int));

    srand(time(NULL));
    for (int i = 0; i < n; i++) 
    {
        a[i] = rand() % n;
    }

    printf("Sorting %i elements\n", n);

    for (int i = 0; i < n; ++i)
    {
        b[i] = a[i];
    }

    clock_t start, end;
    double elapsed;

    start = clock();
    gpu_quick_sort(a, n);
    end = clock();
    elapsed = (((double)(end - start)) / CLOCKS_PER_SEC);
    printf("gpu_quick_sort: %g s\n", elapsed);

    start = clock();
    quick_sort(b, 0, n - 1);
    end = clock();
    elapsed = (((double)(end - start)) / CLOCKS_PER_SEC);
    printf("quick_sort: %g s\n", elapsed);

    for (int i = 0; i < n; ++i)
    {
        if (a[i] != b[i])
        {
            puts("Not equal!\n");
            break;
        }
    }

    free(a);
    free(b);

    return 0;
}