// Arithmancy OpenCL Carry Kernel
__kernel void parallel_carry(__global ulong* limbs, __global uint* carries, int limbs_per_group) {
    int local_id = get_local_id(0);
    int group_id = get_group_id(0);
    int group_size = get_local_size(0); // usually 32 or 64 on Intel
    
    int base_idx = group_id * limbs_per_group;
    
    // Shared memory for carry prefix sum
    __local uint local_carries[16]; 
    
    if (local_id < limbs_per_group) {
        ulong limb = limbs[base_idx + local_id];
        uint c = (local_id == 0) ? carries[group_id] : 0;
        
        // Parallel Scan (Kogge-Stone style) in Local Memory
        local_carries[local_id] = c;
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (int offset = 1; offset < group_size; offset <<= 1) {
            uint temp = 0;
            if (local_id >= offset) temp = local_carries[local_id - offset];
            barrier(CLK_LOCAL_MEM_FENCE);
            local_carries[local_id] += temp;
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        // Apply carries and write back
        limbs[base_idx + local_id] = limb + local_carries[local_id];
    }
}
