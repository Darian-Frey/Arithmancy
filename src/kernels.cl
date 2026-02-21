#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define LIMB_BITS 18
#define LIMB_MASK 0x3FFFF 

__kernel void dwt_squaring(__global double* limbs) {
    int gid = get_global_id(0);
    double val = limbs[gid];
    limbs[gid] = val * val;
}

__kernel void local_carry(__global double* limbs, __global uint* group_carries) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_id = get_group_id(0);
    int wg_size = get_local_size(0);

    __local long local_l[64]; 
    local_l[lid] = (long)limbs[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    if (lid == 0) {
        long c = 0;
        for (int i = 0; i < wg_size; i++) {
            long total = local_l[i] + c;
            local_l[i] = total & LIMB_MASK;
            c = total >> LIMB_BITS;
        }
        group_carries[group_id] = (uint)c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    limbs[gid] = (double)local_l[lid];
}
