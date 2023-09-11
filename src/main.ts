import $ from "jquery";

import shader from "./wgsl/reduce.wgsl";

function reduce_cpu(input: Float32Array): number {
  // TODO: Use Kahan summation?
  var result = 0;

  for (var i = 0; i < input.length; ++i) result += input[i];

  return result;
}

const run = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = (await adapter?.requestDevice()) as GPUDevice;

  const num_data_points = 16384;

  const test_data = new Float32Array(num_data_points);

  for (var i = 0; i < num_data_points; ++i) {
    test_data[i] = Math.random();
  }

  const input_buffer = device.createBuffer({
    label: "input_buffer",
    size: num_data_points * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  // Upload test data
  {
    let mapped_test_buffer = new Float32Array(input_buffer.getMappedRange());
    mapped_test_buffer.set(test_data);
    input_buffer.unmap();
  }

  const workgroup_size = 64;

  // assumes num_data_points is a multiple of the workgroup_size
  let num_first_pass_workgroups = num_data_points / workgroup_size;

  const output_buffer = device.createBuffer({
    label: "output_buffer",
    size: num_first_pass_workgroups * Float32Array.BYTES_PER_ELEMENT,
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

  compute_pass.dispatchWorkgroups(num_first_pass_workgroups);

  let data_size = num_first_pass_workgroups;

  //while()

  compute_pass.end();

  command_encoder.copyBufferToBuffer(
    output_buffer,
    0,
    transfer_buffer,
    0,
    Float32Array.BYTES_PER_ELEMENT
  );

  device.queue.submit([command_encoder.finish()]);

  transfer_buffer.mapAsync(GPUMapMode.READ).then(()=>{
    console.log("CPU Reduce: " + reduce_cpu(test_data));

    var mapped_buffer = transfer_buffer.getMappedRange(0,Float32Array.BYTES_PER_ELEMENT);

    console.log("GPU Reduce: " + new Float32Array(mapped_buffer)[0]);

    transfer_buffer.unmap();
  });
};

run();
