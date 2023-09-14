import $, { data } from "jquery";

import shader from "./wgsl/reduce.wgsl";

function reduce_cpu(input: Float32Array): number {
  // TODO: Use Kahan summation?
  var result = 0;

  for (var i = 0; i < input.length; ++i) {
    result += input[i];
  }

  return result;
}

const reduce = (
  device: GPUDevice,
  bind_group_layout: GPUBindGroupLayout,
  compute_pipeline: GPUComputePipeline,
  test_data_size: number,
  test_data_cpu: Float32Array,
  test_data_buffer: GPUBuffer,
  working_buffer_0: GPUBuffer,
  working_buffer_1: GPUBuffer,
  test_correctness: boolean
) => {
  let command_encoder = device.createCommandEncoder({
    label: "command_encoder",
  });

  let compute_pass = command_encoder.beginComputePass({
    label: "compute_pass",
  });

  compute_pass.setPipeline(compute_pipeline);

  // Note: original CUDA does a copy between dispatches instead of swapping
  // buffers...in WebGPU you can't do that, you need multiple passes. I'm assuming
  // swapping buffers is faster but i don't know for sure.

  const get_input_buffer = (pass_num: number): GPUBuffer => {
    if (pass_num == 0) {
      return test_data_buffer;
    } else {
      if (pass_num % 2 == 1) {
        return working_buffer_0;
      } else {
        return working_buffer_1;
      }
    }
  };

  const get_output_buffer = (pass_num: number): GPUBuffer => {
    if (pass_num % 2 == 0) {
      return working_buffer_0;
    } else {
      return working_buffer_1;
    }
  };

  // Keep reducing until data_size is 1
  let data_size = test_data_size;
  let pass_num = 0;
  while (data_size > 1) {
    const input_buffer = get_input_buffer(pass_num);
    const output_buffer = get_output_buffer(pass_num);

    compute_pass.setBindGroup(
      0,
      device.createBindGroup({
        label: "bind_group_0",
        layout: bind_group_layout,
        entries: [
          {
            binding: 0,
            resource: {
              label: "global_input",
              buffer: input_buffer,
              size: data_size * Float32Array.BYTES_PER_ELEMENT,
            },
          },
          {
            binding: 1,
            resource: {
              label: "global_output",
              buffer: output_buffer,
            },
          },
        ],
      })
    );

    compute_pass.dispatchWorkgroups(num_workgroups);

    pass_num++;
    data_size = num_workgroups;
    num_workgroups = data_size / workgroup_size;
  }

  compute_pass.end();

  device.queue.submit([command_encoder.finish()]);

  if (test_correctness) {
    let transfer_command_encoder = device.createCommandEncoder({
      label: "transfer_command_encoder",
    });

    const transfer_buffer = device.createBuffer({
      label: "transfer_buffer",
      size: Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    transfer_command_encoder.copyBufferToBuffer(
      get_output_buffer(pass_num - 1),
      0,
      transfer_buffer,
      0,
      Float32Array.BYTES_PER_ELEMENT
    );

    device.queue.submit([transfer_command_encoder.finish()]);

    transfer_buffer.mapAsync(GPUMapMode.READ).then(() => {
      console.log("CPU Reduce: " + reduce_cpu(test_data_cpu));

      var mapped_buffer = transfer_buffer.getMappedRange(
        0,
        Float32Array.BYTES_PER_ELEMENT
      );

      console.log("GPU Reduce: " + new Float32Array(mapped_buffer)[0]);

      transfer_buffer.unmap();
      transfer_buffer.destroy();
    });
  }
};

function get_workgroup_size_and_num_workgroups(
  vendor: string,
  max_workgroup_size: number,
  max_workgroups: number,
  max_workgroup_storage_size: number,
  num_input_points: number
): { can_perform_calculation: boolean, workgroup_size: number, num_workgroups: number } {
  var subgroup_size = 32;

  const vendor_lower_case = vendor.toLowerCase();

  // This is just a guess based on typical cases - these could be wrong in general
  // or drivers might dynamically change them based on a variety of factors.
  if(vendor_lower_case.includes("nvidia")){
    subgroup_size = 32;
  }
  else if(vendor_lower_case.includes("intel")){
    subgroup_size = 32;
  }
  else if(vendor_lower_case.includes("apple")){
    subgroup_size = 32;
  }
  else if(vendor_lower_case.includes("amd") || vendor_lower_case.includes("ati")){
    subgroup_size = 64;
  }

  const max_invocations_per_dispatch = max_workgroup_size*max_workgroups;

  // We assume the entire input array can be reduced once in a single
  // dispatch.
  if(num_input_points > max_invocations_per_dispatch){
    return { can_perform_calculation: false, workgroup_size: 0, num_workgroups: 0 };
  }

  var workgroup_size = 0;
  var num_workgroups = 0;

  // TODO we need to ensure the maxWorkingGroupMemory constraint is met

  // Shoot for the subgroup size, and then scale up if necessary to meet the 
  // max_workgroups constraint
  workgroup_size = subgroup_size;
  num_workgroups = Math.ceil(num_input_points / workgroup_size);

  while(num_workgroups > max_workgroups){
    workgroup_size *= 2;
    num_workgroups = Math.ceil(num_input_points / workgroup_size);
  }

  // Ensure we can meet the workgroup memory limit constraint
  if(workgroup_size*Float32Array.BYTES_PER_ELEMENT > max_workgroup_storage_size){
    return { can_perform_calculation: false, workgroup_size: 0, num_workgroups: 0 };
  }

  return {
    can_perform_calculation: true,
    workgroup_size,
    num_workgroups
  }
}

const run = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = (await adapter?.requestDevice()) as GPUDevice;
  const max_workgroup_size = device.limits.maxComputeInvocationsPerWorkgroup;
  const max_workgroups = device.limits.maxComputeWorkgroupsPerDimension;

  const workgroup_size = 32;

  // Simplify so that the problem fits exactly into 3 passes
  const num_data_points = workgroup_size * workgroup_size * workgroup_size;

  const test_data = new Float32Array(num_data_points);

  for (var i = 0; i < num_data_points; ++i) {
    test_data[i] = Math.random();
  }

  const test_data_buffer = device.createBuffer({
    label: "test_data_buffer",
    size: num_data_points * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  // Upload test data
  {
    let mapped_test_buffer = new Float32Array(
      test_data_buffer.getMappedRange()
    );
    mapped_test_buffer.set(test_data);
    test_data_buffer.unmap();
  }

  // assumes num_data_points is a multiple of the workgroup_size
  var num_workgroups = num_data_points / workgroup_size;

  const working_buffer_0 = device.createBuffer({
    label: "working_buffer_0",
    size: num_workgroups * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const working_buffer_1 = device.createBuffer({
    label: "working_buffer_1",
    size: (num_workgroups * Float32Array.BYTES_PER_ELEMENT) / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const reduce_shader = device.createShaderModule({
    label: "reduce_shader",
    code: shader,
  });

  const bind_group_layout = device.createBindGroupLayout({
    entries: [
      {
        // global input
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "read-only-storage",
        },
      },
      {
        // global output
        binding: 1,
        visibility: GPUShaderStage.COMPUTE,
        buffer: {
          type: "storage",
        },
      },
    ],
  });

  const compute_pipeline_layout = device.createPipelineLayout({
    bindGroupLayouts: [bind_group_layout],
    label: "compute_pipeline_layout",
  });

  const compute_pipeline = device.createComputePipeline({
    label: "compute_pipeline",
    layout: compute_pipeline_layout,
    compute: {
      module: reduce_shader,
      entryPoint: "reduce_3",
    },
  });
};

run();
