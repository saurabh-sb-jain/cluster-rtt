# cluster-rtt
```
Tool to measure GPU cluster end-to-end round trip time (Rank 0 -> 1 -> 2 -> 3 ... -> N -> 0).
Can also be used to exercise cpu, dram, pcie, nvidia gpu, e-w nic on every rank for a quick sanity test

Dependency: MPI, NCCL, CUDA, Cluster with NVIDIA GPUs connected over network

Compile:
    mpicc -o cluster-rtt cluster-rtt.c -I<path to cuda headers> \
        -I<path to mpi headers> -lnccl -lcudart -L<path to cuda libs> -fopenmp

Usage: ./cluster-rtt [-d] [-v] [-s memory size]
       -d     Enable debugs
       -v     Verbose mode
       -s     Memory size per GPU in units of 8 Bytes.
              For ex: 18253611008 = 136 GiB

Run Example:
    mpirun --map-by node:PE=8 -n 32 -host host1:8,host2:8,host3:8,host4:8 -x OMP_NUM_THREADS=8 \
        /mnt/cl1/training/progs/cluster-rtt -s 18253611008

The above example runs the tool with MPI to measure cluster-rtt and
exercise cpu, dram, pcie, nvidia gpu, e-w nic on every rank of a 4-node cluster
each with 8 H200 GPUs (32 total GPUs) each using 136 GiB memory per GPU

Output: Results from a 4-node (each with 8 H200 GPUs) cluster networked with 400Gbps links 

Buffer Size: 139264.000000 MiB

Transfer Time Per Rank: 4.141678 secs
Transfer Time Per Rank/MiB: 29.739762 usecs
Cluster RTT: 132.533704 secs
Cluster RTT/MiB: 951.672392 usecs
```
