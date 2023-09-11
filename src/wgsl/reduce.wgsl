
// Dimension     CUDA             WebGPU                                 Description
// -----------------------------------------------------------------------------------------------------------------------
// n/a           Shared Memory    Workgroup Address Space
// n/a           Global Memory    Storage Address Space
// n/a           __syncThreads    storageBarrier or workgroupBarrier
// n/a           Thread           Invocation
// n/a           Warp             n/a
// 1             warpSize         n/a
// n/a           Thread Block     Workgroup
// 
// 3             blockDim         workgroup_size (wgs)
// 3             numBlocks        num_workgroups (nw)
// 3             blockIdx         workgroup_id (wid)                     position of workgroup within the global grid
//                                workgroup_index (wgi)                  wid.x + wid.y*nw.x + (wid.z)*nw.x*nw.y;
// 3             threadIdx        local_invocation_id (liid)             position of invocation within a workgroup grid
//                                local_invocation_index                 unique index within a workgroup, range [0 -> wgs.x*wgs.y*wgs.z - 1]
//                                                                       liid.x + liid.y*wgs.x + liid.z*(wgs.x*wgs.y)
// 3                              global_invocation_id                   position of invocation within global grid
//                                global_invocation_index                unique index within all invocations 
//                                                                       wgi * wgs + liid

const workgroup_size = vec3<u32>(64u,1u,1u);

struct Data {
    data: array<f32>
}

@binding(0) @group(0) var<storage,read> global_input: Data;
@binding(1) @group(0) var<storage,read_write> global_output: Data;

var<workgroup> workgroup_data : array<f32,workgroup_size>;

@compute @workgroup_size(workgroup_size.x,workgroup_size.y,workgroup_size.z)
fn reduce_0(@builtin(local_invocation_id) local_invocation_id: vec3<u32>, 
            @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let tid = local_invocation_id.x;
    let i = workgroup_id.x*workgroup_size.x + local_invocation_id.x;
    
    workgroup_data[tid] = global_input.data[i];
    workgroupBarrier();

    for(var s = 1u; s < workgroup_size.x; s *= 2u){
        if(tid % (2u*s) == 0u){
            workgroup_data[tid] += workgroup_data[tid + s];
        }
        workgroupBarrier();
    }

    if(tid == 0u){
        global_output.data[workgroup_id.x] = workgroup_data[0];
    }
}