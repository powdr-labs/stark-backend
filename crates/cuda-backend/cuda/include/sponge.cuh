#pragma once

#include "fp.h"
#include "poseidon2.cuh"
#include <cstdint>

// Must match the Rust DeviceSpongeState struct layout
struct DeviceSpongeState {
    Fp state[CELLS];      // WIDTH = 16
    uint32_t absorb_idx;
    uint32_t sample_idx;
};

static_assert(sizeof(DeviceSpongeState) == CELLS * sizeof(Fp) + 2 * sizeof(uint32_t),
              "DeviceSpongeState size mismatch with Rust");

// Sponge operations matching DuplexSponge behavior

__device__ inline void sponge_observe(DeviceSpongeState& sponge, Fp value) {
    sponge.state[sponge.absorb_idx] = value;
    sponge.absorb_idx += 1;
    if (sponge.absorb_idx == CELLS_RATE) {
        poseidon2::poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
}

__device__ inline Fp sponge_sample(DeviceSpongeState& sponge) {
    if (sponge.absorb_idx != 0 || sponge.sample_idx == 0) {
        poseidon2::poseidon2_mix(sponge.state);
        sponge.absorb_idx = 0;
        sponge.sample_idx = CELLS_RATE;
    }
    sponge.sample_idx -= 1;
    return sponge.state[sponge.sample_idx];
}
