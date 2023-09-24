import $, { data } from "jquery";

import shader from "./wgsl/reduce.wgsl";

function reduceCpu(input: Float32Array): number {
  // TODO: Use Kahan summation?
  var result = 0;

  for (var i = 0; i < input.length; ++i) {
    result += input[i];
  }

  return result;
}

function getWorkgroupSizeOptions(
  vendor: string,
  maxWorkgroupSize: number,
  maxWorkgroups: number,
  maxWorkgroupStorageSize: number,
  bytesPerWorkgroupElement: number,
  numInputPoints: number
): number[] {
  var workgroupSizeOptions: number[] = [];

  var subgroupSize = 32;

  const vendorLowerCase = vendor.toLowerCase();

  // This is just a guess based on typical cases - these could be wrong in general
  // or drivers might dynamically change them based on a variety of factors.
  if (vendorLowerCase.includes("nvidia")) {
    subgroupSize = 32;
  } else if (vendorLowerCase.includes("intel")) {
    subgroupSize = 32;
  } else if (vendorLowerCase.includes("apple")) {
    subgroupSize = 32;
  } else if (
    vendorLowerCase.includes("amd") ||
    vendorLowerCase.includes("ati")
  ) {
    subgroupSize = 64;
  }

  const maxInvocationsPerDispatch = maxWorkgroupSize * maxWorkgroups;

  // We assume the entire input array can be reduced once in a single
  // dispatch.
  if (numInputPoints > maxInvocationsPerDispatch) {
    return workgroupSizeOptions;
  }

  var workgroupSize = maxWorkgroupSize;

  while (workgroupSize >= subgroupSize) {
    let firstNumGroups = Math.ceil(numInputPoints / workgroupSize);

    if (
      firstNumGroups <= maxWorkgroups &&
      workgroupSize * bytesPerWorkgroupElement <= maxWorkgroupStorageSize
    ) {
      workgroupSizeOptions.push(workgroupSize);
    }

    workgroupSize /= 2;
  }

  return workgroupSizeOptions;
}

interface TestCase {
  algorithm: string;
  numPoints: number;
  workgroupSize: number;
}

let testCases: TestCase[] = [];
let adapter: GPUAdapter | null;
let device: GPUDevice | null;
let vendor: string | undefined;
let maxWorkgroupSize: number;
let maxWorkgroups: number;
let maxWorkgroupStorageSize: number;
let workingBuffer0: GPUBuffer;
let workingBuffer1: GPUBuffer;
let testDataBuffer: GPUBuffer;
let reduceShader: GPUShaderModule;
let bindGroupLayout: GPUBindGroupLayout;
let computePipelineLayout: GPUPipelineLayout;
let computePipeline: GPUComputePipeline;

let init = async () => {
  adapter = await navigator.gpu?.requestAdapter();
  device = (await adapter?.requestDevice({
    requiredLimits: {
      maxComputeWorkgroupSizeX: adapter.limits.maxComputeWorkgroupSizeX,
      maxComputeInvocationsPerWorkgroup:
        adapter.limits.maxComputeInvocationsPerWorkgroup,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  })) as GPUDevice;
  vendor = (await adapter?.requestAdapterInfo())?.vendor;

  maxWorkgroupSize = device.limits.maxComputeInvocationsPerWorkgroup;
  maxWorkgroups = device.limits.maxComputeWorkgroupsPerDimension;
  maxWorkgroupStorageSize = device.limits.maxComputeWorkgroupStorageSize;

  workingBuffer0 = device.createBuffer({
    label: "working_buffer_0",
    size: maxWorkgroups * maxWorkgroupSize * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  workingBuffer1 = device.createBuffer({
    label: "working_buffer_1",
    size:
      (maxWorkgroups * maxWorkgroupSize * Float32Array.BYTES_PER_ELEMENT) / 2,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });

  reduceShader = device.createShaderModule({
    label: "reduce_shader",
    code: shader,
  });

  bindGroupLayout = device.createBindGroupLayout({
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

  computePipelineLayout = device.createPipelineLayout({
    bindGroupLayouts: [bindGroupLayout],
    label: "compute_pipeline_layout",
  });

  let numPointsOptions: number[] = [1_024, 16_384, 131_072, 524_288, 4_194_304];

  for (let numPoints of numPointsOptions) {
    let workgroupSizeOptions = getWorkgroupSizeOptions(
      vendor!,
      maxWorkgroupSize,
      maxWorkgroups,
      maxWorkgroupStorageSize,
      Float32Array.BYTES_PER_ELEMENT,
      numPoints
    );

    for (let workgroupSize of workgroupSizeOptions) {
      for (let algoNumber of [0, 1, 2, 3]) {
        testCases.push({
          algorithm: "reduce" + algoNumber.toString(),
          numPoints: numPoints,
          workgroupSize: workgroupSize,
        });
      }
    }
  }

  postMessagePending++;
  window.postMessage("", "*");
};

const initTestCase = (device: GPUDevice, testCase: TestCase) => {
  const testData = new Float32Array(testCase.numPoints);

  for (var i = 0; i < testCase.numPoints; ++i) {
    testData[i] = Math.random();
  }

  if (testDataBuffer !== undefined) {
    testDataBuffer.destroy();
  }

  testDataBuffer = device.createBuffer({
    label: "testDataBuffer",
    size: testCase.numPoints * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
    mappedAtCreation: true,
  });

  // Upload test data
  {
    let mappedTestBuffer = new Float32Array(testDataBuffer.getMappedRange());
    mappedTestBuffer.set(testData);
    testDataBuffer.unmap();
  }

  computePipeline = device.createComputePipeline({
    label: "computePipeline",
    layout: computePipelineLayout,
    compute: {
      module: reduceShader,
      entryPoint: testCase.algorithm,
      constants: {
        workgroup_size: testCase.workgroupSize,
      },
    },
  });
};

function calculateNumDispatches(testCase: TestCase): number {
  let dataSize = testCase.numPoints;

  var numDispatches = 0;

  while (dataSize > 1) {
    let numWorkgroups = Math.ceil(dataSize / testCase.workgroupSize);
    dataSize = numWorkgroups;

    numDispatches++;
  }

  return numDispatches;
}

const reduce = (
  device: GPUDevice,
  testDataSize: number,
  workgroupSize: number,
  testCorrectness: boolean,
  testDataCpu: Float32Array | null
) => {
  let commandEncoder = device.createCommandEncoder({
    label: "commandEncoder",
  });

  let computePass = commandEncoder.beginComputePass({
    label: "computePass",
  });

  computePass.setPipeline(computePipeline);

  // Note: original CUDA does a copy between dispatches instead of swapping
  // buffers...in WebGPU you can't do that, you need multiple passes. I'm assuming
  // swapping buffers is faster but i don't know for sure.

  const getInputBuffer = (dispatchNum: number): GPUBuffer => {
    if (dispatchNum == 0) {
      return testDataBuffer;
    } else {
      if (dispatchNum % 2 == 1) {
        return workingBuffer0;
      } else {
        return workingBuffer1;
      }
    }
  };

  const getOutputBuffer = (dispatchNum: number): GPUBuffer => {
    if (dispatchNum % 2 == 0) {
      return workingBuffer0;
    } else {
      return workingBuffer1;
    }
  };

  // Keep reducing until dataSize is 1
  let dataSize = testDataSize;
  let dispatchNum = 0;
  while (dataSize > 1) {
    const inputBuffer = getInputBuffer(dispatchNum);
    const outputBuffer = getOutputBuffer(dispatchNum);

    computePass.setBindGroup(
      0,
      device.createBindGroup({
        label: "bindGroup0",
        layout: bindGroupLayout,
        entries: [
          {
            binding: 0,
            resource: {
              label: "globalInput",
              buffer: inputBuffer,
              size: dataSize * Float32Array.BYTES_PER_ELEMENT,
            },
          },
          {
            binding: 1,
            resource: {
              label: "globalOutput",
              buffer: outputBuffer,
            },
          },
        ],
      })
    );

    let numWorkgroups = Math.ceil(dataSize / workgroupSize);

    computePass.dispatchWorkgroups(numWorkgroups);

    dispatchNum++;
    dataSize = numWorkgroups;
  }

  computePass.end();

  device.queue.submit([commandEncoder.finish()]);

  if (testCorrectness) {
    let transferCommandEncoder = device.createCommandEncoder({
      label: "transfer_command_encoder",
    });

    const transferBuffer = device.createBuffer({
      label: "transferBuffer",
      size: Float32Array.BYTES_PER_ELEMENT,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });

    transferCommandEncoder.copyBufferToBuffer(
      getOutputBuffer(dispatchNum - 1),
      0,
      transferBuffer,
      0,
      Float32Array.BYTES_PER_ELEMENT
    );

    device.queue.submit([transferCommandEncoder.finish()]);

    transferBuffer.mapAsync(GPUMapMode.READ).then(() => {
      console.log("CPU Reduce: " + reduceCpu(testDataCpu!));

      var mappedBuffer = transferBuffer.getMappedRange(
        0,
        Float32Array.BYTES_PER_ELEMENT
      );

      console.log("GPU Reduce: " + new Float32Array(mappedBuffer)[0]);

      transferBuffer.unmap();
      transferBuffer.destroy();
    });
  }
};

let postMessagePending = 0;

const casesPerPost = 100;
const maxCaseCount = 10000;

let caseCount = 0;
let testIndex = 0;
let caseStartTime: number;

window.onmessage = () => {
  postMessagePending--;

  if (postMessagePending == 0) {
    const testCase = testCases[testIndex];

    if (caseCount === 0) {
      initTestCase(device!, testCase);
      caseStartTime = Date.now();
    }

    for (let i = 0; i < casesPerPost; ++i) {
      reduce(
        device!,
        testCase.numPoints,
        testCase.workgroupSize,
        false,
        null
      );
    }

    caseCount += casesPerPost;

    if (caseCount >= maxCaseCount) {
      caseCount = 0;
      const casesPerSecond = Math.floor(
        (1000.0 * maxCaseCount) / (Date.now() - caseStartTime)
      );
      const numDispatches = calculateNumDispatches(testCases[testIndex]);
      console.log(
        `${testCase.algorithm} ${
          testCase.workgroupSize
        } ${testCase.numPoints.toLocaleString()} ${numDispatches}: ${casesPerSecond.toLocaleString()}`
      );

      testIndex++;
    }

    if (testIndex === testCases.length) {
      console.log("Tests complete.");
      return;
    }

    postMessagePending++;
    window.postMessage("", "*");
  }
};

init();
