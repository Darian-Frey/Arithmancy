#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define M_PI 3.14159265358979323846
#define LIMB_BITS 18
#define LIMB_MASK 0x3FFFF // (1 << 18) - 1

/* Phase 1: Pointwise Squaring (18-bit Precision Protected) */
__kernel void dwt_squaring(__global double* limbs) {
    int gid = get_global_id(0);
    int total_size = get_global_size(0);
    
    // Architect 01: 18-bit limbs fit safely in 53-bit mantissa
    double val = limbs[gid];
    double res = val * val; // Squaring in the frequency domain

    limbs[gid] = res;
}

/* Phase 2: Hierarchical Ripple Carry (Architect 02 & 04 Alignment) */
/* This is the 'Local' pass that stays under 1.0s to avoid Hangcheck */
__kernel void local_carry(__global double* limbs, __global uint* group_carries) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    int group_id = get_group_id(0);
    int wg_size = get_local_size(0);

    __local long local_l[64]; // Architect 02: local storage for ripple
    local_l[lid] = (long)limbs[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Intra-workgroup ripple carry
    if (lid == 0) {
        long c = 0;
        for (int i = 0; i < wg_size; i++) {
            long total = local_l[i] + c;
            local_l[i] = total & LIMB_MASK;
            c = total >> LIMB_BITS;
        }
        // Store overflow for the next kernel pass (Architect 02 Strategy)
        group_carries[group_id] = (uint)c;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    limbs[gid] = (double)local_l[lid];
}
