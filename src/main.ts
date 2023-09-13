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

const run = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = (await adapter?.requestDevice()) as GPUDevice;

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
    let mapped_test_buffer = new Float32Array(test_data_buffer.getMappedRange());
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
    size: num_workgroups * Float32Array.BYTES_PER_ELEMENT / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  const transfer_buffer = device.createBuffer({
    label: "transfer_buffer",
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
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
      entryPoint: "reduce_0",
    },
  });

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

  const get_input_buffer = (pass_num: number) : GPUBuffer => {
    if(pass_num == 0){
      return test_data_buffer;
    }
    else {
      if(pass_num % 2 == 1){
        return working_buffer_0;
      }
      else {
        return working_buffer_1;
      }
    }
  };

  const get_output_buffer = (pass_num: number) : GPUBuffer  => {
    if(pass_num % 2 == 0){
      return working_buffer_0;
    }
    else {
      return working_buffer_1;
    }
  };

  // Keep reducing until data_size is 1
  let data_size = num_data_points;
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

  command_encoder.copyBufferToBuffer(
    get_output_buffer(pass_num - 1),
    0,
    transfer_buffer,
    0,
    Float32Array.BYTES_PER_ELEMENT
  );

  device.queue.submit([command_encoder.finish()]);

  transfer_buffer.mapAsync(GPUMapMode.READ).then(() => {
    console.log("CPU Reduce: " + reduce_cpu(test_data));

    var mapped_buffer = transfer_buffer.getMappedRange(
      0,
      Float32Array.BYTES_PER_ELEMENT
    );

    console.log("GPU Reduce: " + new Float32Array(mapped_buffer)[0]);

    transfer_buffer.unmap();
  });
};

run();
