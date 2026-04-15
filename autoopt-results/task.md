# 2026-04-15-0800-cache-codeword-buffer-across-segments

Cache the RS codeword `DeviceBuffer` in the GPU device across segment prove() calls, avoiding the cold `cudaMallocAsync` that triggers a real `cudaMalloc` for the first segment. At APC 300, seg 0's `rs_code_matrix` takes 57ms while seg 1's takes 0ms — the entire gap is the first-time 192MB allocation through the CUDA async memory pool. By caching the codeword buffer after each segment and reusing it for the next, every segment gets 0ms allocation. Expected saving: ~57ms at APC 300.
