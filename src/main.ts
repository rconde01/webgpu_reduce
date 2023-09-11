import $ from "jquery";

const run = async () => {
  const adapter = await navigator.gpu?.requestAdapter();
  const device = (await adapter?.requestDevice()) as GPUDevice;

  const num_data_points = 16384;

  const test_data = new Float32Array(num_data_points);

  for (var i = 0; i < num_data_points; ++i) {
    test_data[i] = Math.random();
  }

  const input_buffer = device.createBuffer({
    size: num_data_points * Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.STORAGE,
    label: "input_buffer",
    mappedAtCreation: true,
  });

  // Upload test data
  {
    let mapped_test_buffer = new Float32Array(input_buffer.getMappedRange());
    mapped_test_buffer.set(test_data);
    input_buffer.unmap();
  }

  const workgroup_size = 64;

  const intermediate_buffer = device.createBuffer({
    // assumes num_data_points is a multiple of the workgroup_size
    size: (num_data_points / workgroup_size) * Float32Array.BYTES_PER_ELEMENT, 
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    label: "intermediate_buffer"
  });

  const transfer_buffer = device.createBuffer({
    size: Float32Array.BYTES_PER_ELEMENT,
    usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    label: "transfer_buffer"
  });
};

run();
