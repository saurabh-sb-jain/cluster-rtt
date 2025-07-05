#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>
#include "cuda_runtime.h"
#include "mpi.h"
#include "nccl.h"
#include <omp.h>

#define MPICHECK(cmd)                                                        \
    do                                                                       \
    {                                                                        \
        int e = cmd;                                                         \
        if (e != MPI_SUCCESS)                                                \
        {                                                                    \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e); \
            exit(EXIT_FAILURE);                                              \
        }                                                                    \
    } while (0)

#define CUDACHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        cudaError_t e = cmd;                                              \
        if (e != cudaSuccess)                                             \
        {                                                                 \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, \
                   cudaGetErrorString(e));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

#define NCCLCHECK(cmd)                                                    \
    do                                                                    \
    {                                                                     \
        ncclResult_t r = cmd;                                             \
        if (r != ncclSuccess)                                             \
        {                                                                 \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, \
                   ncclGetErrorString(r));                                \
            exit(EXIT_FAILURE);                                           \
        }                                                                 \
    } while (0)

uint32_t get_str_hash(char *str, int len)
{
    uint32_t hash = 5381;
    for (int i = 0; i < len; i++)
    {
        hash = ((hash << 5) + hash) + str[i];
    }
    return hash;
}

double get_time_delta(struct timespec *start, struct timespec *end)
{
    double delta = 0;
    delta = (double)(end->tv_sec - start->tv_sec) +
            (double)(end->tv_nsec - start->tv_nsec) / 1000000000.0;
    return delta;
}

void parallel_memset(void *buffer, int value, size_t size)
{
#pragma omp parallel
    {
        int num_threads = omp_get_num_threads();
        int thread_id = omp_get_thread_num();
        size_t chunk_size = size / num_threads;
        size_t start = thread_id * chunk_size;
        size_t end = (thread_id == num_threads - 1) ? size : start + chunk_size;
        memset((char *)buffer + start, value, end - start);
    }
}

#define BYTE_PATTERN (0xAAull)

#define DBG_PRINTF(fmt, ...)      \
    if (debug)                    \
    {                             \
        printf(fmt, __VA_ARGS__); \
    }

#define VERB_PRINTF(fmt, ...)     \
    if (verbose)                  \
    {                             \
        printf(fmt, __VA_ARGS__); \
    }

void print_usage(char *cmd)
{
    fprintf(stderr,
            "Usage: %s [-d] [-v] [-s memory size]\n",
            cmd);
    exit(-1);
}

int main(int argc, char *argv[])
{
    uint64_t size = 1 * 1024 * 1024 * 1024ull;
    int myRank, nRanks, localRank = 0;
    char hostname[1024];
    ncclUniqueId id;
    ncclComm_t comm;
    uint64_t *buff, *hostBuff, *tmp;
    uint64_t chkData = 0;
    cudaStream_t s;
    struct timespec start, end, start1, end1;
    double delta = 0, size_mib = 0;
    uint32_t *hostlist, *localRanks;
    uint32_t hostname_hash = 0;
    double *malloc_time, *devmalloc_time, *net_time, *memset_time,
        *host2devcp_time, *dev2hostcp_time, *patchk_time;
    uint64_t *patchk_status;
    bool debug = false, verbose = false;
    char ch;

    while ((ch = getopt(argc, argv, "dvhs:")) != -1)
    {
        switch (ch)
        {
        case 's':
            size = atol(optarg);
            break;
        case 'v':
            verbose = true;
            break;
        case 'd':
            debug = true;
            break;
        case 'h':
        default:
            print_usage(argv[0]);
        }
    }

    size_mib = ((sizeof(uint64_t) * size) / (1024.0 * 1024.0));

    for (int i = 0; i < sizeof(uint64_t); i++)
    {
        chkData |= (BYTE_PATTERN << (i * 8));
    }

    DBG_PRINTF("Check Pattern: 0x%lx\n", chkData);

    MPICHECK(MPI_Init(&argc, &argv));
    MPICHECK(MPI_Comm_rank(MPI_COMM_WORLD, &myRank));
    MPICHECK(MPI_Comm_size(MPI_COMM_WORLD, &nRanks));

    hostlist = (uint32_t *)malloc(nRanks * sizeof(uint32_t));
    localRanks = (uint32_t *)malloc(nRanks * sizeof(uint32_t));
    malloc_time = (double *)malloc(nRanks * sizeof(double));
    devmalloc_time = (double *)malloc(nRanks * sizeof(double));
    net_time = (double *)malloc(nRanks * sizeof(double));
    memset_time = (double *)malloc(nRanks * sizeof(double));
    host2devcp_time = (double *)malloc(nRanks * sizeof(double));
    dev2hostcp_time = (double *)malloc(nRanks * sizeof(double));
    patchk_time = (double *)malloc(nRanks * sizeof(double));
    patchk_status = (uint64_t *)malloc(nRanks * sizeof(uint64_t));

    if (!hostlist || !localRanks || !malloc_time || !devmalloc_time ||
        !net_time || !memset_time || !host2devcp_time || !dev2hostcp_time ||
        !patchk_time || !patchk_status)
    {
        printf("Error: %d allocating memory\n", errno);
        exit(-1);
    }

    gethostname(hostname, 1024);

    hostname_hash = get_str_hash(hostname, strlen(hostname));

    hostlist[myRank] = hostname_hash;

    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostlist,
                           sizeof(uint32_t), MPI_BYTE, MPI_COMM_WORLD));

    for (int i = 0; i < nRanks; i++)
    {
        if (myRank == i)
        {
            break;
        }
        if (hostlist[i] == hostname_hash)
        {
            localRank++;
        }
    }

    localRanks[myRank] = localRank;

    VERB_PRINTF("Hostname: %s Hostname_Hash: %u Rank: %d LocalRank: %d\n", hostname,
                hostname_hash, myRank, localRank);

    if (myRank == 0)
    {
        ncclGetUniqueId(&id);
    }

    MPICHECK(MPI_Bcast((void *)&id, sizeof(id), MPI_BYTE, 0, MPI_COMM_WORLD));

    CUDACHECK(cudaSetDevice(localRank));

    clock_gettime(CLOCK_MONOTONIC, &start);
    CUDACHECK(cudaMalloc((void **)&buff, size * sizeof(uint64_t)));
    clock_gettime(CLOCK_MONOTONIC, &end);
    devmalloc_time[myRank] = get_time_delta(&start, &end);

    DBG_PRINTF("[%s %d %d] device malloc time delta: %f\n", hostname, localRank,
               myRank, get_time_delta(&start, &end));

    CUDACHECK(cudaStreamCreate(&s));

    MPI_Barrier(MPI_COMM_WORLD);

    NCCLCHECK(ncclCommInitRank(&comm, nRanks, id, myRank));

    clock_gettime(CLOCK_MONOTONIC, &start);
    hostBuff = (uint64_t *)malloc(size * sizeof(uint64_t));
    clock_gettime(CLOCK_MONOTONIC, &end);
    malloc_time[myRank] = get_time_delta(&start, &end);
    DBG_PRINTF("[%s %d %d] host malloc time delta: %f\n", hostname, localRank, myRank,
               get_time_delta(&start, &end));

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &start);
    // memset(hostBuff, BYTE_PATTERN, size * sizeof(uint64_t));
    parallel_memset(hostBuff, BYTE_PATTERN, size * sizeof(uint64_t));
    clock_gettime(CLOCK_MONOTONIC, &end);
    memset_time[myRank] = get_time_delta(&start, &end);
    DBG_PRINTF("[%s %d %d] memset time delta: %f\n", hostname, localRank, myRank,
               get_time_delta(&start, &end));

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &start);
    CUDACHECK(cudaMemcpy((void *)buff, (const void *)hostBuff,
                         size * sizeof(uint64_t), cudaMemcpyHostToDevice));
    clock_gettime(CLOCK_MONOTONIC, &end);
    host2devcp_time[myRank] = get_time_delta(&start, &end);
    DBG_PRINTF("[%s %d %d] host2dev memcpy time delta: %f\n", hostname, localRank,
               myRank, get_time_delta(&start, &end));

    MPI_Barrier(MPI_COMM_WORLD);

    clock_gettime(CLOCK_MONOTONIC, &start);
    if (myRank == 0)
    {
        NCCLCHECK(ncclSend(buff, size, ncclUint64, myRank + 1, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        VERB_PRINTF("[%s] Finished sending (%d -> %d)\n", hostname, myRank, myRank + 1);

        NCCLCHECK(ncclRecv(buff, size, ncclUint64, nRanks - 1, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        VERB_PRINTF("[%s] Finished receiving (%d -> %d)\n", hostname, nRanks - 1,
                    myRank);

        clock_gettime(CLOCK_MONOTONIC, &end);
        net_time[myRank] = get_time_delta(&start, &end);

        clock_gettime(CLOCK_MONOTONIC, &start1);
        CUDACHECK(cudaMemcpy((void *)hostBuff, (const void *)buff,
                             size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        clock_gettime(CLOCK_MONOTONIC, &end1);
        dev2hostcp_time[myRank] = get_time_delta(&start1, &end1);
        DBG_PRINTF("[%s %d %d] dev2host memcpy time delta: %f\n", hostname, localRank,
                   myRank, get_time_delta(&start1, &end1));
    }
    else
    {
        NCCLCHECK(ncclRecv(buff, size, ncclUint64, myRank - 1, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        VERB_PRINTF("[%s] Finished receiving (%d -> %d)\n", hostname, myRank - 1,
                    myRank);

        NCCLCHECK(ncclSend(buff, size, ncclUint64, (myRank + 1) % nRanks, comm, s));
        CUDACHECK(cudaStreamSynchronize(s));
        VERB_PRINTF("[%s] Finished sending (%d -> %d)\n", hostname, myRank,
                    (myRank + 1) % nRanks);

        clock_gettime(CLOCK_MONOTONIC, &end);
        net_time[myRank] = get_time_delta(&start, &end);

        clock_gettime(CLOCK_MONOTONIC, &start1);
        CUDACHECK(cudaMemcpy((void *)hostBuff, (const void *)buff,
                             size * sizeof(uint64_t), cudaMemcpyDeviceToHost));
        clock_gettime(CLOCK_MONOTONIC, &end1);
        dev2hostcp_time[myRank] = get_time_delta(&start1, &end1);
        DBG_PRINTF("[%s %d %d] dev2host memcpy time delta: %f\n", hostname, localRank,
                   myRank, get_time_delta(&start1, &end1));
    }

    if (myRank == 0)
    {
        DBG_PRINTF("[%s %d %d] send-recv time: %f\n", hostname, localRank, myRank,
                   get_time_delta(&start, &end));
    }
    else
    {
        DBG_PRINTF("[%s %d %d] recv-send time: %f\n", hostname, localRank, myRank,
                   get_time_delta(&start, &end));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    tmp = hostBuff;
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (int i = 0; i < size; i++)
    {
        if (*tmp != chkData)
        {
            break;
        }
        if (i < size - 1)
        {
            tmp++;
        }
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    patchk_time[myRank] = get_time_delta(&start, &end);
    if (*tmp == chkData)
    {
        DBG_PRINTF("[%s %d %d] pattern check successful, time taken:%f\n", hostname,
                   localRank, myRank, get_time_delta(&start, &end));
        patchk_status[myRank] = 1;
    }
    else
    {
        DBG_PRINTF("[%s %d %d] pattern check failed (0x%lx)\n", hostname, localRank,
                   myRank, *tmp);
        patchk_status[myRank] = 0;
    }

    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, malloc_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, devmalloc_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, memset_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, host2devcp_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, dev2hostcp_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, net_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, patchk_time,
                           sizeof(double), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, patchk_status,
                           sizeof(uint64_t), MPI_BYTE, MPI_COMM_WORLD));
    MPICHECK(MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, localRanks,
                           sizeof(uint32_t), MPI_BYTE, MPI_COMM_WORLD));

    if (myRank == 0)
    {
        printf("Buffer Size: %f MiB\n", size_mib);

        printf("GRank\tLRank\tMallocTime\tDevMallocTime\tMemsetTime\tHost2DevCpTime"
               "\tDev2HostCpTime\tNetworkTime\tPatChkTime\tPatChkStatus\n");
        for (int i = 0; i < nRanks; i++)
        {
            printf("%d\t%d\t%f\t%f\t%f\t%f\t%f\t%f\t%f\t%ld\n", i, localRanks[i],
                   malloc_time[i], devmalloc_time[i], memset_time[i],
                   host2devcp_time[i], dev2hostcp_time[i], net_time[i],
                   patchk_time[i], patchk_status[i]);
        }
        printf("Transfer Time Per Rank: %f secs\n", net_time[0] / nRanks);

        printf("Transfer Time Per Rank/MiB: %f usecs\n",
               (net_time[0] * 1000 * 1000) / (nRanks * size_mib));

        printf("Cluster RTT: %f secs\n", net_time[0]);

        printf("Cluster RTT/MiB: %f usecs\n", (net_time[0] * 1000 * 1000) / size_mib);
    }

    MPICHECK(MPI_Finalize());

    CUDACHECK(cudaFree(buff));

    free(hostBuff);

    free(hostlist);

    ncclCommDestroy(comm);

    return 0;
}