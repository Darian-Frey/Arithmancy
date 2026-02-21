#pragma OPENCL EXTENSION cl_khr_fp64 : enable

#define M_PI 3.14159265358979323846

__kernel void dwt_squaring(__global double* limbs) {
    int gid = get_global_id(0);
    int total_size = get_global_size(0);
    
    if (gid >= total_size) return;

    double angle = (M_PI * gid) / total_size;
    double weight_re = cos(angle);
    
    // Perform the pointwise squaring in the weighted domain
    double val = limbs[gid] * weight_re;
    double res = val * val;

    limbs[gid] = res / weight_re;
}

__kernel void parallel_carry(__global double* limbs, __global uint* carries, const int limbs_per_group) {
    int lid = get_local_id(0);
    int gid = get_global_id(0);
    
    __local double local_limbs[16];
    local_limbs[lid] = limbs[gid];
    barrier(CLK_LOCAL_MEM_FENCE);

    long current_limb = (long)local_limbs[lid];
    long carry_out = current_limb >> 32;
    long remainder = current_limb & 0xFFFFFFFF;

    limbs[gid] = (double)remainder;
    
    if (lid == 15) {
        carries[get_group_id(0)] = (uint)carry_out;
    }
}
