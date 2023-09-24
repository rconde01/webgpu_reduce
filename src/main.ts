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

function get_workgroup_size_options(
  vendor: string,
  max_workgroup_size: number,
  max_workgroups: number,
  max_workgroup_storage_size: number,
  bytes_per_workgroup_element: number,
  num_input_points: number
): number[] {
  var workgroup_size_options: number[] = [];

  var subgroup_size = 32;

  const vendor_lower_case = vendor.toLowerCase();

  // This is just a guess based on typical cases - these could be wrong in general
  // or drivers might dynamically change them based on a variety of factors.
  if (vendor_lower_case.includes("nvidia")) {
    subgroup_size = 32;
  } else if (vendor_lower_case.includes("intel")) {
    subgroup_size = 32;
  } else if (vendor_lower_case.includes("apple")) {
    subgroup_size = 32;
  } else if (
    vendor_lower_case.includes("amd") ||
    vendor_lower_case.includes("ati")
  ) {
    subgroup_size = 64;
  }

  const max_invocations_per_dispatch = max_workgroup_size * max_workgroups;

  // We assume the entire input array can be reduced once in a single
  // dispatch.
  if (num_input_points > max_invocations_per_dispatch) {
    return workgroup_size_options;
  }

  var workgroup_size = max_workgroup_size;

  while (workgroup_size >= subgroup_size) {
    let first_num_groups = Math.ceil(num_input_points / workgroup_size);

    if (
      first_num_groups <= max_workgroups &&
      workgroup_size * bytes_per_workgroup_element <= max_workgroup_storage_size
    ) {
      workgroup_size_options.push(workgroup_size);
    }

    workgroup_size /= 2;
  }

  return workgroup_size_options;
}

interface TestCase {
  algorithm: string;
  num_points: number;
  workgroup_size: number;
}

let testCases: TestCase[] = [];
let adapter: GPUAdapter | null;
let device: GPUDevice | null;
let vendor: string | undefined;
let max_workgroup_size: number;
let max_workgroups: number;
let max_workgroup_storage_size: number;
let working_buffer_0: GPUBuffer;
let working_buffer_1: GPUBuffer;
let test_data_buffer: GPUBuffer;
let reduce_shader: GPUShaderModule;
let bind_group_layout: GPUBindGroupLayout;
let compute_pipeline_layout: GPUPipelineLayout;
let compute_pipeline: GPUComputePipeline;

let init = async () => {
  adapter = await navigator.gpu?.requestAdapter();
  device = (await adapter?.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits.maxComputeInvocationsPerWorkgroup,
        maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize
    },
  })) as GPUDevice;
  vendor = (await adapter?.requestAdapterInfo())?.vendor;

  max_workgroup_size = device.limits.maxComputeInvocationsPerWorkgroup;
  max_workgroups = device.limits.maxComputeWorkgroupsPerDimension;
  max_workgroup_storage_size = device.limits.maxComputeWorkgroupStorageSize;

  working_buffer_0 = device.createBuffer({
    label: "working_buffer_0",
    size: max_workgroups * max_workgroup_size * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  working_buffer_1 = device.createBuffer({
    label: "working_buffer_1",
    size:
      (max_workgroups * max_workgroup_size * Float32Array.BYTES_PER_ELEMENT) /
      2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  reduce_shader = device.createShaderModule({
    label: "reduce_shader",
    code: shader,
  });

  bind_group_layout = device.createBindGroupLayout({
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

  compute_pipeline_layout = device.createPipelineLayout({
    bindGroupLayouts: [bind_group_layout],
    label: "compute_pipeline_layout",
  });

  let num_points_options: number[] = [
    1_024, 16_384, 131_072, 524_288, 4_194_304,
  ];

  for (let num_points of num_points_options) {
    let workgroup_size_options = get_workgroup_size_options(
      vendor!,
      max_workgroup_size,
      max_workgroups,
      max_workgroup_storage_size,
      Float32Array.BYTES_PER_ELEMENT,
      num_points
    );

    for (let workgroup_size of workgroup_size_options) {
      for (let algo_number of [0, 1, 2, 3]) {
        testCases.push({
          algorithm: "reduce_" + algo_number.toString(),
          num_points,
          workgroup_size,
        });
      }
    }
  }

  postMessagePending++;
  window.postMessage("", "*");
};

const init_case = (device: GPUDevice, test_case: TestCase) => {
  const test_data = new Float32Array(test_case.num_points);

  for (var i = 0; i < test_case.num_points; ++i) {
    test_data[i] = Math.random();
  }

  test_data_buffer = device.createBuffer({
    label: "test_data_buffer",
    size: test_case.num_points * Float32Array.BYTES_PER_ELEMENT,
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

  compute_pipeline = device.createComputePipeline({
    label: "compute_pipeline",
    layout: compute_pipeline_layout,
    compute: {
      module: reduce_shader,
      entryPoint: test_case.algorithm,
      constants: {
        workgroup_size: test_case.workgroup_size,
      },
    },
  });
};

const reduce = (
  device: GPUDevice,
  test_data_size: number,
  workgroup_size: number,
  test_correctness: boolean,
  test_data_cpu: Float32Array | null
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
  let dispatch_num = 0;
  while (data_size > 1) {
    const input_buffer = get_input_buffer(dispatch_num);
    const output_buffer = get_output_buffer(dispatch_num);

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

    let num_workgroups = Math.ceil(data_size / workgroup_size);

    compute_pass.dispatchWorkgroups(num_workgroups);

    dispatch_num++;
    data_size = num_workgroups;
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
      get_output_buffer(dispatch_num - 1),
      0,
      transfer_buffer,
      0,
      Float32Array.BYTES_PER_ELEMENT
    );

    device.queue.submit([transfer_command_encoder.finish()]);

    transfer_buffer.mapAsync(GPUMapMode.READ).then(() => {
      console.log("CPU Reduce: " + reduce_cpu(test_data_cpu!));

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

var postMessagePending = 0;

const casesPerPost = 100;
const maxCaseCount = 10000;

let caseCount = 0;
let testIndex = 0;
let caseStartTime: number;

window.onmessage = () => {
  postMessagePending--;

  const test_case = testCases[testIndex];

  if (caseCount === 0) {
    init_case(device!, test_case);
    caseStartTime = Date.now();
  }

  for (let i = 0; i < casesPerPost; ++i) {
    reduce(
      device!,
      test_case.num_points,
      test_case.workgroup_size,
      false,
      null
    );
  }

  caseCount += casesPerPost;

  if (caseCount >= maxCaseCount) {
    testIndex++;
    caseCount = 0;
    const casesPerSecond = Math.floor(
      (1000.0 * maxCaseCount) / (Date.now() - caseStartTime)
    );
    console.log(
      `${test_case.algorithm} ${test_case.workgroup_size} ${test_case.num_points.toLocaleString()}: ${casesPerSecond.toLocaleString()}`
    );
  }

  if (testIndex === testCases.length) {
    console.log("Tests complete.");
    return;
  }

  if (postMessagePending == 0) {
    postMessagePending++;
    window.postMessage("", "*");
  }
};

init();
