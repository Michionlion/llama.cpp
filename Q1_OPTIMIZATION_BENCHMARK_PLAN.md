# Q1_0 CPU optimization benchmark plan

This protocol validates the x86 Q1_0 direct-dot and four-row repack
candidate in this checkout. It keeps every build and result under the Q1
tree and separates three effects:

1. The unmodified source baseline.
2. The AVX-512 VBMI/VNNI direct dot with CPU repacking disabled.
3. The direct dot plus the Q1_0 four-row GEMV/GEMM repack.

The core build, correctness, quality-smoke, and 27B commands below were
exercised during the authorized 2026-07-31 optimization session. Optional
cross-compiler and full-corpus steps remain future work. This is also the
exact protocol for a future rerun; do not run it again without a new
explicit benchmark authorization.

## Isolation and provenance

- Q1 source: `/home/saejin/projects/inference/bonsai/q1/llama.cpp`
- Q1 build directories: `/home/saejin/projects/inference/bonsai/q1/build-q1-*`
- Q1 result directories: `/home/saejin/projects/inference/bonsai/q1/results-q1-*`
- Q1 model: `/home/saejin/projects/inference/bonsai/models/Bonsai-27B-Q1_0.gguf`
- Never use `/home/saejin/projects/inference/bonsai/build` for Q1.
- Never write results into `/home/saejin/projects/inference/bonsai/results`.
- Never edit, configure, switch, merge, rebase, or build the Q2 checkout at
  `/home/saejin/projects/inference/bonsai/llama.cpp`.
- A frozen Q2 binary may be run read-only only when the user separately
  authorizes a Q1-versus-Q2 runtime comparison.
- Use a fresh build and result name for every compiler, revision, or flag
  change. Never reuse a CMake cache.

The Q1 branch was originally expected at
`b4ca032ae3729516943884786de4ae39fba0bbca`. Before this candidate was
implemented, it had advanced externally to
`e41889057ee570929991c83cf8832d21ac114f3f`. That later revision is the
correct source baseline for isolating this patch. Results against it must
not be described as an isolated comparison against `b4ca032`.

## Fresh baseline, direct-only, and repack builds

The following sequence archives all five candidate source files, reverses
them only long enough to configure and build the baseline, and restores
them before configuring either candidate. The trap is required.

```bash
set -euo pipefail

Q1_ROOT=/home/saejin/projects/inference/bonsai/q1
Q1_SRC=$Q1_ROOT/llama.cpp
Q1_MODEL=/home/saejin/projects/inference/bonsai/models/Bonsai-27B-Q1_0.gguf
BASE_COMMIT=e41889057ee570929991c83cf8832d21ac114f3f
BASE_SHORT=e418890
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)
RESULTS=$Q1_ROOT/results-q1-vbmi-vnni-repack-$BASE_SHORT-$RUN_ID
BASE_BUILD=$Q1_ROOT/build-q1-baseline-$BASE_SHORT-$RUN_ID
DIRECT_BUILD=$Q1_ROOT/build-q1-direct-vbmi-vnni-$BASE_SHORT-$RUN_ID
CAND_BUILD=$Q1_ROOT/build-q1-vbmi-vnni-repack-$BASE_SHORT-$RUN_ID
PATCH=$RESULTS/q1-vbmi-vnni-repack-candidate.patch
SOURCE_FILES=(
    ggml/src/ggml-cpu/arch-fallback.h
    ggml/src/ggml-cpu/arch/x86/quants.c
    ggml/src/ggml-cpu/arch/x86/repack.cpp
    ggml/src/ggml-cpu/repack.cpp
    ggml/src/ggml-cpu/repack.h
)

test "$(git -C "$Q1_SRC" rev-parse HEAD)" = "$BASE_COMMIT"
test ! -e "$RESULTS"
test ! -e "$BASE_BUILD"
test ! -e "$DIRECT_BUILD"
test ! -e "$CAND_BUILD"
mkdir -p "$RESULTS"

git -C "$Q1_SRC" status --short --branch > "$RESULTS/source-status-before.txt"
git -C "$Q1_SRC" diff --binary "$BASE_COMMIT" -- "${SOURCE_FILES[@]}" > "$PATCH"
test -s "$PATCH"
sha256sum "$Q1_MODEL" "$PATCH" > "$RESULTS/input-sha256.txt"
grep -q '^17ef842e47450caeb8eaa3ebfbbab5d2f2278b62b79be107985fb69a2f819aa0 ' \
    "$RESULTS/input-sha256.txt"

git -C "$Q1_SRC" apply --reverse --check "$PATCH"
git -C "$Q1_SRC" apply --reverse "$PATCH"

restore_candidate() {
    if git -C "$Q1_SRC" apply --check "$PATCH" >/dev/null 2>&1; then
        git -C "$Q1_SRC" apply "$PATCH"
    fi
}
trap restore_candidate EXIT INT TERM

test -z "$(git -C "$Q1_SRC" diff --name-only "$BASE_COMMIT" -- "${SOURCE_FILES[@]}")"

COMMON_CMAKE_ARGS=(
    -G Ninja
    -DCMAKE_BUILD_TYPE=Release
    -DGGML_NATIVE=ON
    -DGGML_OPENMP=ON
    -DGGML_BLAS=OFF
    -DGGML_CCACHE=OFF
    -DGGML_LTO=OFF
    -DLLAMA_CURL=OFF
    -DLLAMA_BUILD_SERVER=OFF
    -DLLAMA_BUILD_TESTS=ON
    -DLLAMA_BUILD_TOOLS=ON
)
TARGETS=(
    llama-bench
    llama-cli
    llama-perplexity
    test-quantize-fns
    test-backend-ops
    test-quantize-perf
)

cmake -S "$Q1_SRC" -B "$BASE_BUILD" "${COMMON_CMAKE_ARGS[@]}" \
    -DGGML_CPU_REPACK=ON \
    2>&1 | tee "$RESULTS/configure-q1-baseline.txt"
cmake --build "$BASE_BUILD" --target "${TARGETS[@]}" --parallel 8 \
    2>&1 | tee "$RESULTS/build-q1-baseline.txt"

restore_candidate
trap - EXIT INT TERM
git -C "$Q1_SRC" apply --reverse --check "$PATCH"

cmake -S "$Q1_SRC" -B "$DIRECT_BUILD" "${COMMON_CMAKE_ARGS[@]}" \
    -DGGML_CPU_REPACK=OFF \
    2>&1 | tee "$RESULTS/configure-q1-direct.txt"
cmake --build "$DIRECT_BUILD" --target "${TARGETS[@]}" --parallel 8 \
    2>&1 | tee "$RESULTS/build-q1-direct.txt"

cmake -S "$Q1_SRC" -B "$CAND_BUILD" "${COMMON_CMAKE_ARGS[@]}" \
    -DGGML_CPU_REPACK=ON \
    2>&1 | tee "$RESULTS/configure-q1-repack.txt"
cmake --build "$CAND_BUILD" --target "${TARGETS[@]}" --parallel 8 \
    2>&1 | tee "$RESULTS/build-q1-repack.txt"

git -C "$Q1_SRC" diff --check
git -C "$Q1_SRC" status --short --branch > "$RESULTS/source-status-after.txt"
lscpu > "$RESULTS/lscpu.txt"
cc --version > "$RESULTS/cc-version.txt"
c++ --version > "$RESULTS/cxx-version.txt"
cmake --version > "$RESULTS/cmake-version.txt"
ninja --version > "$RESULTS/ninja-version.txt"
```

If link-time optimization is part of the deployment build, repeat the
entire three-build sequence with new names and `-DGGML_LTO=ON`. Do not
compare an LTO baseline with a non-LTO candidate.

## Compiler and ISA variants

The primary comparison is a matched GCC native pair. In fresh Q1 build
directories, also compile and run the correctness tests for:

1. AVX2 with AVX-512, VNNI, and VBMI disabled.
2. AVX-512 VNNI with VBMI disabled.
3. AVX-512F/BW/DQ/VNNI/VBMI enabled.
4. Clang for all three variants, if Clang is installed.
5. A second Intel or AMD VBMI/VNNI host when available.

The first two variants prove that the unchanged AVX2 direct path and
generic repack aliases still build and run. Example configurations:

```bash
AVX2_BUILD=$Q1_ROOT/build-q1-avx2-fallback-$BASE_SHORT-$RUN_ID
NOVBMI_BUILD=$Q1_ROOT/build-q1-avx512-no-vbmi-$BASE_SHORT-$RUN_ID
FULLISA_BUILD=$Q1_ROOT/build-q1-avx512-vbmi-vnni-$BASE_SHORT-$RUN_ID

cmake -S "$Q1_SRC" -B "$AVX2_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_AVX_VNNI=OFF \
    -DGGML_AVX512=OFF \
    -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=OFF \
    -DGGML_CPU_REPACK=ON \
    -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF

cmake -S "$Q1_SRC" -B "$NOVBMI_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=OFF \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_CPU_REPACK=ON \
    -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF

cmake -S "$Q1_SRC" -B "$FULLISA_BUILD" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX=ON -DGGML_AVX2=ON -DGGML_FMA=ON -DGGML_F16C=ON \
    -DGGML_AVX512=ON \
    -DGGML_AVX512_VBMI=ON \
    -DGGML_AVX512_VNNI=ON \
    -DGGML_CPU_REPACK=ON \
    -DGGML_OPENMP=ON -DGGML_BLAS=OFF -DGGML_CCACHE=OFF \
    -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_TOOLS=OFF

for build in "$AVX2_BUILD" "$NOVBMI_BUILD" "$FULLISA_BUILD"; do
    cmake --build "$build" --target test-quantize-fns test-backend-ops --parallel 8
    "$build/bin/test-quantize-fns" -v
    "$build/bin/test-backend-ops" test -b CPU -o MUL_MAT -p 'type_a=q1_0' -j 1
done
```

Use `CC=clang CXX=clang++` and new directory names for Clang. If either
binary is absent, record that fact and do not silently substitute GCC.

## Direct and repacked kernel correctness

Run the stock tests first:

```bash
"$CAND_BUILD/bin/test-quantize-fns" -v \
    > "$RESULTS/q1-candidate-test-quantize-fns.txt" 2>&1
"$CAND_BUILD/bin/test-backend-ops" test -b CPU -o MUL_MAT \
    -p 'type_a=q1_0' -j 1 \
    > "$RESULTS/q1-candidate-test-backend-ops-q1.txt" 2>&1
"$CAND_BUILD/bin/test-backend-ops" test -b CPU -j 8 --output csv \
    > "$RESULTS/q1-candidate-test-backend-ops-full.csv" \
    2> "$RESULTS/q1-candidate-test-backend-ops-full.stderr"
```

`test-quantize-fns` exercises one deterministic direct dot with a loose
binary-quantization tolerance. `test-backend-ops` covers many shapes but
uses the same backend as the code under test; it does not independently
prove the repacked Q1 or Q8 byte layout. Neither test proves that the
optimized runtime route was selected.

Create the following dedicated comparator. It initializes ggml before
using half conversion, covers arbitrary Q1 bits, signed Q1 scales, the
full Q8 byte range including -128, GEMV fused-pack and tail cases, and
the actual four-byte Q1 and Q8 interleaves used at runtime.

```bash
cat > "$RESULTS/q1-kernel-compare.cpp" <<'CPP'
#include "ggml-cpu.h"
#include "ggml-cpu/quants.h"
#include "ggml-cpu/repack.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t rng_state = 0x12345678U;

static uint32_t next_u32() {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 17;
    rng_state ^= rng_state << 5;
    return rng_state;
}

static ggml_half random_scale() {
    float value = 0.0025f * (1 + next_u32() % 100);
    if (next_u32() & 1) {
        value = -value;
    }
    return ggml_fp32_to_fp16(value);
}

static void fill_q1(block_q1_0 & block) {
    block.d = random_scale();
    for (uint8_t & value : block.qs) {
        value = (uint8_t) next_u32();
    }
}

static void fill_q8(block_q8_0 & block) {
    block.d = random_scale();
    for (int8_t & value : block.qs) {
        value = (int8_t) ((int) (next_u32() % 256) - 128);
    }
}

static bool check_value(
        const char * label, int test, int index, float actual, float expected,
        float & max_abs, float & max_rel) {
    const float abs_error = std::fabs(actual - expected);
    const float rel_error = abs_error / std::max(1.0f, std::fabs(expected));
    max_abs = std::max(max_abs, abs_error);
    max_rel = std::max(max_rel, rel_error);
    if (!std::isfinite(actual) || rel_error > 2.0e-4f) {
        std::fprintf(
            stderr,
            "%s failure: test=%d index=%d actual=%g expected=%g abs=%g rel=%g\n",
            label, test, index, actual, expected, abs_error, rel_error);
        return false;
    }
    return true;
}

static bool test_direct(float & max_abs, float & max_rel) {
    for (int test = 0; test < 10000; ++test) {
        const int n = QK1_0 * (1 + next_u32() % 32);
        const int nb1 = n / QK1_0;
        const int nb8 = n / QK8_0;
        std::vector<block_q1_0> weights(nb1);
        std::vector<block_q8_0> activations(nb8);

        for (block_q1_0 & block : weights) {
            fill_q1(block);
        }
        for (block_q8_0 & block : activations) {
            fill_q8(block);
        }

        float actual = 0.0f;
        float expected = 0.0f;
        ggml_vec_dot_q1_0_q8_0(
            n, &actual, 0, weights.data(), 0, activations.data(), 0, 1);
        ggml_vec_dot_q1_0_q8_0_generic(
            n, &expected, 0, weights.data(), 0, activations.data(), 0, 1);
        if (!check_value("direct", test, 0, actual, expected, max_abs, max_rel)) {
            return false;
        }
    }
    return true;
}

static void pack_q1(
        const std::vector<block_q1_0> & source,
        std::vector<block_q1_0x4> & packed, int nb1, int nc) {
    for (int row = 0; row < nc; row += 4) {
        for (int l = 0; l < nb1; ++l) {
            block_q1_0x4 & out = packed[(row / 4) * nb1 + l];
            for (int j = 0; j < 4; ++j) {
                const block_q1_0 & in = source[(row + j) * nb1 + l];
                out.d[j] = in.d;
                for (int k = 0; k < QK1_0 / QK8_0; ++k) {
                    std::memcpy(out.qs + k * 16 + j * 4, in.qs + k * 4, 4);
                }
            }
        }
    }
}

static void pack_q8(
        const std::vector<block_q8_0> & source,
        std::vector<block_q8_0x4> & packed, int nb8, int nr) {
    for (int row = 0; row < nr; row += 4) {
        for (int l = 0; l < nb8; ++l) {
            block_q8_0x4 & out = packed[(row / 4) * nb8 + l];
            for (int m = 0; m < 4; ++m) {
                const block_q8_0 & in = source[(row + m) * nb8 + l];
                out.d[m] = in.d;
                for (int k = 0; k < QK8_0 / 4; ++k) {
                    std::memcpy(out.qs + k * 16 + m * 4, in.qs + k * 4, 4);
                }
            }
        }
    }
}

static bool test_repack(float & max_abs, float & max_rel) {
    for (int test = 0; test < 500; ++test) {
        const int n = QK1_0 * (1 + next_u32() % 16);
        const int nc = 4 * (1 + next_u32() % 5);
        const int nr = 4 * (1 + next_u32() % 3);
        const int nb1 = n / QK1_0;
        const int nb8 = n / QK8_0;

        std::vector<block_q1_0> weights(nc * nb1);
        std::vector<block_q1_0x4> packed_weights((nc / 4) * nb1);
        std::vector<block_q8_0> activations(nr * nb8);
        std::vector<block_q8_0x4> packed_activations((nr / 4) * nb8);
        std::vector<float> actual(nr * nc);
        std::vector<float> expected(nr * nc);

        for (block_q1_0 & block : weights) {
            fill_q1(block);
        }
        for (block_q8_0 & block : activations) {
            fill_q8(block);
        }
        pack_q1(weights, packed_weights, nb1, nc);
        pack_q8(activations, packed_activations, nb8, nr);

        ggml_gemv_q1_0_4x4_q8_0(
            n, actual.data(), nc, packed_weights.data(), activations.data(), 1, nc);
        for (int col = 0; col < nc; ++col) {
            ggml_vec_dot_q1_0_q8_0_generic(
                n, &expected[col], 0, weights.data() + col * nb1, 0,
                activations.data(), 0, 1);
            if (!check_value(
                    "gemv", test, col, actual[col], expected[col], max_abs, max_rel)) {
                return false;
            }
        }

        ggml_gemm_q1_0_4x4_q8_0(
            n, actual.data(), nc, packed_weights.data(),
            packed_activations.data(), nr, nc);
        for (int row = 0; row < nr; ++row) {
            for (int col = 0; col < nc; ++col) {
                const int index = row * nc + col;
                ggml_vec_dot_q1_0_q8_0_generic(
                    n, &expected[index], 0, weights.data() + col * nb1, 0,
                    activations.data() + row * nb8, 0, 1);
                if (!check_value(
                        "gemm", test, index, actual[index], expected[index],
                        max_abs, max_rel)) {
                    return false;
                }
            }
        }
    }
    return true;
}

int main() {
    ggml_cpu_init();
    float max_abs = 0.0f;
    float max_rel = 0.0f;

    if (!test_direct(max_abs, max_rel) || !test_repack(max_abs, max_rel)) {
        return 1;
    }
    std::printf(
        "direct_cases=10000 repack_cases=500 max_abs=%g max_rel=%g\n",
        max_abs, max_rel);
    return 0;
}
CPP

c++ -O2 -std=c++17 \
    -I"$Q1_SRC/ggml/include" \
    -I"$Q1_SRC/ggml/src" \
    "$RESULTS/q1-kernel-compare.cpp" \
    -L"$CAND_BUILD/bin" \
    -Wl,-rpath,"$CAND_BUILD/bin" \
    -lggml-cpu -lggml-base -lggml \
    -o "$CAND_BUILD/bin/q1-kernel-compare"

"$CAND_BUILD/bin/q1-kernel-compare" \
    > "$RESULTS/q1-candidate-kernel-compare.txt"
```

Inspect the generated instructions and stack use:

```bash
objdump -drwC -Mintel --disassemble=ggml_vec_dot_q1_0_q8_0 \
    "$CAND_BUILD/bin/libggml-cpu.so" \
    > "$RESULTS/q1-candidate-direct-assembly.txt"
objdump -drwC -Mintel --disassemble=ggml_gemv_q1_0_4x4_q8_0 \
    "$CAND_BUILD/bin/libggml-cpu.so" \
    > "$RESULTS/q1-candidate-gemv-assembly.txt"
objdump -drwC -Mintel --disassemble=ggml_gemm_q1_0_4x4_q8_0 \
    "$CAND_BUILD/bin/libggml-cpu.so" \
    > "$RESULTS/q1-candidate-gemm-assembly.txt"
grep -q vpmultishiftqb "$RESULTS/q1-candidate-direct-assembly.txt"
grep -q vpdpbusd "$RESULTS/q1-candidate-direct-assembly.txt"
grep -q vpmultishiftqb "$RESULTS/q1-candidate-gemv-assembly.txt"
grep -q vpdpbusd "$RESULTS/q1-candidate-gemm-assembly.txt"
```

### Q8_0 4x4 activation quantizer

The Q1 repack uses `ggml_quantize_mat_q8_0_4x4`. On x86, compare its
AVX2 implementation byte-for-byte with the generic scalar routine,
including half-way values that distinguish `roundf` from nearest-even
rounding:

```bash
cat > "$RESULTS/q1-quantize-4x4-compare.cpp" <<'CPP'
#include "ggml-cpu.h"
#include "ggml-cpu/repack.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

static uint32_t state = 0x31415926U;

static uint32_t next_u32() {
    state ^= state << 13;
    state ^= state >> 17;
    state ^= state << 5;
    return state;
}

int main() {
    ggml_cpu_init();
    uint64_t values = 0;

    {
        constexpr int k = 32;
        std::vector<float> source(4 * k, 0.0f);
        std::vector<block_q8_0x4> generic(k / QK8_0);
        std::vector<block_q8_0x4> optimized(k / QK8_0);

        for (int row = 0; row < 4; ++row) {
            source[row * k + 0] = 127.0f;
            source[row * k + 1] = 0.5f;
            source[row * k + 2] = -0.5f;
            source[row * k + 3] = 1.5f;
            source[row * k + 4] = -1.5f;
        }

        ggml_quantize_mat_q8_0_4x4_generic(source.data(), generic.data(), k);
        ggml_quantize_mat_q8_0_4x4(source.data(), optimized.data(), k);
        const size_t bytes = optimized.size() * sizeof(optimized[0]);
        if (std::memcmp(generic.data(), optimized.data(), bytes) != 0) {
            std::fputs("half-way rounding mismatch\n", stderr);
            return 1;
        }
        values += 4 * k;
    }

    for (int test = 0; test < 2000; ++test) {
        const int k = 32 * (1 + next_u32() % 64);
        std::vector<float> source(4 * k);
        std::vector<block_q8_0x4> generic(k / QK8_0);
        std::vector<block_q8_0x4> optimized(k / QK8_0);

        for (float & value : source) {
            value = (int32_t) (next_u32() & 0xffff) / 1024.0f - 32.0f;
        }

        ggml_quantize_mat_q8_0_4x4_generic(source.data(), generic.data(), k);
        ggml_quantize_mat_q8_0_4x4(source.data(), optimized.data(), k);
        const size_t bytes = optimized.size() * sizeof(optimized[0]);
        if (std::memcmp(generic.data(), optimized.data(), bytes) != 0) {
            std::fprintf(stderr, "mismatch: test=%d k=%d\n", test, k);
            return 1;
        }
        values += 4 * k;
    }

    constexpr int k = 5120;
    constexpr int iterations = 2000;
    std::vector<float> source(4 * k);
    std::vector<block_q8_0x4> output(k / QK8_0);
    for (float & value : source) {
        value = (int32_t) (next_u32() & 0xffff) / 1024.0f - 32.0f;
    }

    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_quantize_mat_q8_0_4x4_generic(source.data(), output.data(), k);
    }
    auto stop = std::chrono::steady_clock::now();
    const double generic_us =
        std::chrono::duration<double, std::micro>(stop - start).count() / iterations;

    start = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; ++i) {
        ggml_quantize_mat_q8_0_4x4(source.data(), output.data(), k);
    }
    stop = std::chrono::steady_clock::now();
    const double optimized_us =
        std::chrono::duration<double, std::micro>(stop - start).count() / iterations;

    std::printf(
        "cases=2001 values=%llu generic_us=%.3f optimized_us=%.3f speedup=%.3fx\n",
        (unsigned long long) values, generic_us, optimized_us,
        generic_us / optimized_us);
}
CPP

c++ -O2 -std=c++17 \
    -I"$Q1_SRC/ggml/include" \
    -I"$Q1_SRC/ggml/src" \
    "$RESULTS/q1-quantize-4x4-compare.cpp" \
    -L"$CAND_BUILD/bin" \
    -Wl,-rpath,"$CAND_BUILD/bin" \
    -lggml-cpu -lggml-base -lggml \
    -o "$CAND_BUILD/bin/q1-quantize-4x4-compare"

taskset -c 0 "$CAND_BUILD/bin/q1-quantize-4x4-compare" \
    > "$RESULTS/q1-candidate-quantize-4x4-compare.txt"
```

This comparator is required because `test-quantize-fns` does not invoke
the four-row activation quantizer, and backend tolerances do not prove
that its byte layout and rounding match the scalar implementation.

## Kernel performance

`test-quantize-perf` measures the direct dot, not the repacked model route.
Pin it to one physical core, retain its five internal warmups, and repeat
the entire command five times:

```bash
for label in baseline direct repack; do
    case "$label" in
        baseline) build=$BASE_BUILD ;;
        direct)   build=$DIRECT_BUILD ;;
        repack)   build=$CAND_BUILD ;;
    esac
    for repetition in 1 2 3 4 5; do
        OMP_NUM_THREADS=1 taskset -c 0 \
            "$build/bin/test-quantize-perf" \
            --type q1_0 --op vec_dot_q -3 -i 500
    done > "$RESULTS/q1-$label-direct-dot-perf-5x.txt" 2>&1
done
```

Report minimum and mean cycles per 32 values for every working-set size.
Do not interpret the synthetic GB/s field as end-to-end model bandwidth.

## Matched 27B pp128 and tg32

Primary controls:

- Eight threads on the eight visible physical cores.
- `OMP_PROC_BIND=spread` and `OMP_PLACES=cores`.
- CPU only with `-ngl 0`.
- BLAS disabled at configure time.
- Flash attention on.
- F16 K and V caches.
- Batch and ubatch both 128.
- Poll level zero.
- One built-in warmup and five measured repetitions.
- Both build orders.

```bash
run_q1_27b() {
    label=$1
    build=$2
    order=$3
    OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
        "$build/bin/llama-bench" \
        -m "$Q1_MODEL" \
        -t 8 -r 5 \
        -p 128 -n 32 \
        -b 128 -ub 128 \
        -ngl 0 -fa on \
        -ctk f16 -ctv f16 \
        --poll 0 -o json \
        > "$RESULTS/q1-$label-27b-pp128-tg32-$order.json" \
        2> "$RESULTS/q1-$label-27b-pp128-tg32-$order.stderr"
}

run_q1_27b repack   "$CAND_BUILD"  order1
run_q1_27b direct   "$DIRECT_BUILD" order1
run_q1_27b baseline "$BASE_BUILD"  order1
run_q1_27b baseline "$BASE_BUILD"  order2
run_q1_27b direct   "$DIRECT_BUILD" order2
run_q1_27b repack   "$CAND_BUILD"  order2
```

Map row-count sensitivity separately:

```bash
for label in direct repack; do
    if test "$label" = direct; then
        build=$DIRECT_BUILD
    else
        build=$CAND_BUILD
    fi
    OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
        "$build/bin/llama-bench" \
        -m "$Q1_MODEL" \
        -t 8 -r 3 \
        -p 1,8,32,128 \
        -n 32 \
        -b 128 -ub 128 \
        -ngl 0 -fa on \
        -ctk f16 -ctv f16 \
        --poll 0 -o json \
        > "$RESULTS/q1-$label-27b-row-sweep.json" \
        2> "$RESULTS/q1-$label-27b-row-sweep.stderr"
done
```

Only the 27B Q1 model was available on 2026-07-31. If genuine smaller
Q1 GGUFs become available, add matched `pp128`, `pp512`, `tg32`, and
`tg128` runs. Q2 models are not substitutes for a smaller Q1 workload.

Thread count, flash attention, ubatch, and BLAS are separate deployment
experiments. Change one at a time and give each a distinct output name:

```bash
for threads in 1 2 4 8; do
    OMP_NUM_THREADS=$threads OMP_PROC_BIND=spread OMP_PLACES=cores \
        "$CAND_BUILD/bin/llama-bench" \
        -m "$Q1_MODEL" \
        -t "$threads" -r 3 \
        -p 128 -n 32 \
        -b 128 -ub 128 \
        -ngl 0 -fa on \
        -ctk f16 -ctv f16 \
        --poll 0 -o json \
        > "$RESULTS/q1-repack-27b-threads-$threads.json"
done
```

When a read-only Q2 runtime comparison is explicitly authorized, use the
frozen Q2 binary and keep its output in the Q1-labeled result directory:

```bash
Q2_BENCH=/home/saejin/projects/inference/bonsai/build-q2-attn/bin/llama-bench
Q2_MODEL=/home/saejin/projects/inference/bonsai/models/Ternary-Bonsai-27B-Q2_g64.gguf

test -x "$Q2_BENCH"
test -s "$Q2_MODEL"
OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
    "$Q2_BENCH" \
    -m "$Q2_MODEL" \
    -t 8 -r 5 \
    -p 128 -n 32 \
    -b 128 -ub 128 \
    -ngl 0 -fa on \
    -ctk f16 -ctv f16 \
    --poll 0 -o json \
    > "$RESULTS/q1-comparison-readonly-q2-27b-pp128-tg32.json" \
    2> "$RESULTS/q1-comparison-readonly-q2-27b-pp128-tg32.stderr"
```

Do not configure, rebuild, or modify Q2 to make this comparison.

## Q1 quality and perplexity gates

Start with a local one-chunk smoke test, saving baseline logits and
comparing the candidate against them:

```bash
PPL_INPUT=$Q1_SRC/README.md
BASE_LOGITS=$RESULTS/q1-baseline-readme-c512-chunk1.logits
COMMON_PPL_ARGS=(
    -m "$Q1_MODEL"
    -c 512
    -t 8
    -tb 8
    -b 512
    -ub 512
    -ngl 0
    -fa on
)

"$BASE_BUILD/bin/llama-perplexity" "${COMMON_PPL_ARGS[@]}" \
    -f "$PPL_INPUT" --chunks 1 \
    --save-all-logits "$BASE_LOGITS" \
    > "$RESULTS/q1-baseline-readme-c512-chunk1-ppl.txt" 2>&1

"$CAND_BUILD/bin/llama-perplexity" "${COMMON_PPL_ARGS[@]}" \
    -f "$PPL_INPUT" --chunks 1 \
    --kl-divergence-base "$BASE_LOGITS" \
    --kl-divergence \
    > "$RESULTS/q1-repack-readme-c512-chunk1-kld.txt" 2>&1
```

For a GEMV change, also force batch one so every evaluated logit uses
the single-row decode kernel:

```bash
GEMV_BASE_LOGITS=$RESULTS/q1-baseline-readme-c128-batch1.logits
GEMV_PPL_ARGS=(
    -m "$Q1_MODEL"
    -f "$Q1_SRC/README.md"
    -c 128
    -t 8
    -tb 8
    -b 1
    -ub 1
    -ngl 0
    -fa on
    --chunks 1
)

"$BASE_BUILD/bin/llama-perplexity" "${GEMV_PPL_ARGS[@]}" \
    --save-all-logits "$GEMV_BASE_LOGITS" \
    > "$RESULTS/q1-baseline-readme-c128-batch1-ppl.txt" 2>&1
"$CAND_BUILD/bin/llama-perplexity" "${GEMV_PPL_ARGS[@]}" \
    --kl-divergence-base "$GEMV_BASE_LOGITS" \
    --kl-divergence \
    > "$RESULTS/q1-candidate-readme-c128-batch1-kld.txt" 2>&1
```

After the smoke gate passes, repeat with Wikitext-2. Download it only
inside the Q1 result directory:

```bash
WIKI_ZIP=$RESULTS/wikitext-2-raw-v1.zip
WIKI_DIR=$RESULTS/wikitext
WIKI_FILE=$WIKI_DIR/wikitext-2-raw/wiki.test.raw
curl -L \
    https://huggingface.co/datasets/ggml-org/ci/resolve/main/wikitext-2-raw-v1.zip \
    -o "$WIKI_ZIP"
mkdir -p "$WIKI_DIR"
python3 -m zipfile -e "$WIKI_ZIP" "$WIKI_DIR"
test -s "$WIKI_FILE"

PPL_INPUT=$WIKI_FILE
BASE_LOGITS=$RESULTS/q1-baseline-wikitext-c512.logits
"$BASE_BUILD/bin/llama-perplexity" "${COMMON_PPL_ARGS[@]}" \
    -f "$PPL_INPUT" --chunks -1 \
    --save-all-logits "$BASE_LOGITS" \
    > "$RESULTS/q1-baseline-wikitext-c512-ppl.txt" 2>&1
"$CAND_BUILD/bin/llama-perplexity" "${COMMON_PPL_ARGS[@]}" \
    -f "$PPL_INPUT" --chunks -1 \
    --kl-divergence-base "$BASE_LOGITS" \
    --kl-divergence \
    > "$RESULTS/q1-repack-wikitext-c512-kld.txt" 2>&1
```

Quality gates:

- Dedicated comparator: zero failures.
- Perplexity ratio: within 0.1 percent of the baseline.
- Mean KLD: at most `5e-4`.
- Maximum KLD: at most `0.01`.
- RMS probability delta: at most 1 percent.
- Same-top-token rate: at least 99 percent.
- Baseline and candidate perplexity intervals overlap at two combined
  standard errors.

Also compare deterministic generation, keeping stdout and diagnostic
stderr separate:

```bash
PROMPT='Explain why the sky appears blue in three concise sentences.'
for label in baseline repack; do
    if test "$label" = baseline; then
        build=$BASE_BUILD
    else
        build=$CAND_BUILD
    fi
    OMP_NUM_THREADS=8 OMP_PROC_BIND=spread OMP_PLACES=cores \
        "$build/bin/llama-cli" \
        -m "$Q1_MODEL" \
        -p "$PROMPT" \
        -n 96 -t 8 -s 1234 --temp 0 \
        -ngl 0 -fa on \
        > "$RESULTS/q1-$label-greedy.stdout" \
        2> "$RESULTS/q1-$label-greedy.stderr"
done
cmp "$RESULTS/q1-baseline-greedy.stdout" "$RESULTS/q1-repack-greedy.stdout"
```

The Q1-versus-Q2 model quality tradeoff is a separate format evaluation.
These gates only prove that the CPU optimization preserves the existing
Q1 model behavior.

## Results obtained on 2026-07-31

Host: AMD EPYC 9645 under KVM with eight visible physical cores. Primary
builds used GCC 14.2.0, Release `-O3`, native ISA, OpenMP on, BLAS off,
LTO off, CPU only, flash attention on, F16 K/V caches, batch 128, ubatch
128, and five repetitions.

Correctness:

- `test-quantize-fns`: zero failures; reported Q1 dot error `0.211374`.
- Focused Q1 CPU `MUL_MAT`: 45/45 supported cases passed.
- Full CPU backend suite: all 17,403 supported cases passed; 3,004
  unsupported cases were skipped. Of 362 Q1-containing rows, 215 were
  supported and passed.
- Dedicated comparator: 10,000 direct cases and 500 repack cases passed;
  maximum absolute error was `9.15527e-05` and maximum relative error was
  `2.98023e-05`.
- Explicit AVX2 and AVX-512-without-VBMI builds compiled and passed the
  quantization and focused Q1 tests.
- Clang was not installed, so Clang coverage remains open.
- Assembly contains `vpmultishiftqb`, `vpdpbusd`, and `vpsubd`. The final
  GEMM stack frame is 0x48 bytes with no accumulator spills.

One-chunk Q1 quality smoke test:

- Direct baseline perplexity: `4.7456 +/- 0.86923`.
- Repack candidate perplexity: `4.747260 +/- 0.868898`.
- Perplexity ratio: `1.000339`.
- Mean KLD: `0.000282 +/- 0.000048`.
- Maximum KLD: `0.007172`.
- RMS probability delta: `0.671 percent`.
- Same-top-token rate: `99.608 percent`.

End-to-end Q1 decomposition:

| Q1 build | pp128 tokens/s | tg32 tokens/s |
| --- | ---: | ---: |
| Current source, repack disabled | 15.2069 +/- 0.1234 | 8.3909 +/- 0.1799 |
| Current source, Q1 repack enabled | 29.1456 +/- 0.3953 | 11.7214 +/- 0.3125 |
| Repack speedup | 1.916x | 1.397x |

The same-source comparison isolates the Q1 repack. Against the older
committed Q1 result, the complete direct-plus-repack candidate was about
3.06x faster in pp128 and 1.86x faster in tg32, but that historical run
used different flash-attention and batch controls and is not the primary
matched claim.

Read-only Q2 comparison:

| IPO-matched 27B build | pp128 tokens/s | tg32 tokens/s |
| --- | ---: | ---: |
| Finalized Q2 | 26.6538 +/- 0.4803 | 9.6507 +/- 0.0980 |
| Q1 candidate | 28.6565 +/- 0.2339 | 11.0382 +/- 0.7648 |
| Q1 advantage | 7.51 percent | 14.38 percent |

The faster non-LTO Q1 deployment build reached 29.1456 pp128 and 11.7214
tg32. LTO was neutral-to-negative for Q1 on this host, consistent with
the previous Q2 work. The Q1 GGUF is 3.792 GB versus 7.574 GB for Q2,
but attention, activation traffic, scheduling, and non-matrix kernels
prevent model throughput from scaling directly with weight size.

### Follow-up activation quantizer result

Commit `4e246ff7f` was used as the clean checkpoint for a second
optimization cycle. The retained follow-up adds an AVX2
`ggml_quantize_mat_q8_0_4x4` implementation for Q1 prompt activations.
It preserves the scalar routine's round-away-from-zero behavior and
four-byte interleave exactly.

- Randomized quantizer comparator: 2,000 cases and 8,336,512 values,
  with zero byte mismatches.
- Isolated `k=5120` quantizer: about 59.3 us scalar versus 4.44 us AVX2,
  a 13.4x speedup.
- Full CPU backend suite: all 17,403 supported rows passed; all 215
  supported Q1 rows passed.
- AVX-only generic fallback, AVX2-only, and AVX-512-without-VBMI builds
  compiled and passed 45/45 focused Q1 cases.

Prompt-size sweep:

| Workload | Checkpoint tokens/s | AVX2 quantizer tokens/s | Gain |
| --- | ---: | ---: | ---: |
| pp8 | 22.5660 | 26.9433 | 19.40 percent |
| pp32 | 28.2860 | 29.9770 | 5.98 percent |
| pp128 | 29.3071 | 30.6624 | 4.62 percent |
| pp512 | 29.5031 | 30.9419 | 4.88 percent |

Across two five-repetition pp128 orderings, the checkpoint averaged
29.1378 tokens/s and the retained candidate averaged 30.8910 tokens/s,
a 6.02 percent gain. The quantizer is not used by the single-row decode
path, so tg32 is considered unchanged.

The exact-rounding quality check against checkpoint logits reported a
perplexity ratio of `1.000001`, mean KLD `-0.000001`, maximum KLD
`0.000047`, RMS probability delta `0.000 percent`, and a
`100.000 percent` same-top-token rate.

The following experiments were rejected:

- Signed-byte VNNI: this CPU lacks `AVX-VNNI-INT8` and therefore has no
  `VPDPBSSD` instruction.
- Sixteen-output GEMV fusion: tg32 regressed 18.4 percent.
- Paired eight-output GEMM: pp8 regressed 8.2 percent and pp32 regressed
  3.3 percent for only marginal large-prompt gains.
- Nearest-even activation rounding: faster, but exceeded the perplexity,
  maximum-KLD, and RMS probability gates.
- AVX-512 exact activation quantization: 25 percent faster than AVX2 in
  isolation but produced no measurable end-to-end gain.

### Twelve-output decode tile result

Commit `2823d4495` was used as the clean checkpoint for a third
optimization cycle. The retained follow-up extends the AVX-512 Q1 GEMV
from an eight-output primary tile to a twelve-output primary tile. It
processes three adjacent four-row packs together, sharing each Q8 load
and correction across twelve output channels. The established eight- and
four-output loops handle remainders, and GEMM is unchanged.

Decode-only 27B results from two five-repetition orderings:

| Ordering | Checkpoint tg32 | Twelve-output tg32 | Gain |
| --- | ---: | ---: | ---: |
| Candidate then checkpoint | 12.0560 | 13.3907 | 11.07 percent |
| Checkpoint then candidate | 11.8961 | 13.2075 | 11.02 percent |
| Pooled | 11.9761 | 13.2991 | 11.05 percent |

In the combined pp128/tg32 protocol, pp128 was unchanged within noise:
the checkpoint pooled to 30.2748 tokens/s and the candidate pooled to
30.4256 tokens/s. Combined-run tg32 was noisier, but still pooled to
11.6734 tokens/s for the candidate versus 11.2953 tokens/s for the
checkpoint. Use the isolated decode pair for the kernel claim.

A pinned `n=5120, nc=320` kernel harness produced identical checksums
and measured the following ten-run means:

| Build mode | Checkpoint GEMV | Twelve-output GEMV | Gain |
| --- | ---: | ---: | ---: |
| Release | 15.5380 us | 15.1499 us | 2.56 percent |
| Release with IPO | 15.6224 us | 15.1652 us | 3.02 percent |

The dedicated kernel comparator passed 10,000 direct and 500 repacked
cases with maximum absolute error `9.15527e-05` and maximum relative
error `2.98023e-05`. The full CPU backend suite passed all 17,403
supported rows and all 215 supported Q1 rows. AVX-only, AVX2-only, and
AVX-512-without-VBMI builds each passed 45/45 focused Q1 cases.

A batch-1, 128-token quality pass forced the GEMV path for every
evaluated logit. It reported PPL `2.2637` for both builds, zero KL
divergence, zero RMS probability delta, and a 100 percent same-top-token
rate. The candidate completed the pass in 10.98 seconds versus 13.50
seconds for the checkpoint.

Using the frozen Q2 measurements as a historical reference, the current
combined-run Q1 result is about 14.2 percent faster for pp128 and 21.0
percent faster for tg32. The isolated Q1 decode result is 37.8 percent
faster than the frozen Q2 tg32 result. These are cross-session
comparisons, not a new same-session Q1-versus-Q2 claim.

Two correction-hoisting variants were rejected:

- Precomputing scalar Q8 corrections in a thread-local array regressed
  pooled tg32 by about 6.0 percent.
- Using a negative correction as the VNNI accumulator required register
  copies for each destination and regressed pooled tg32 by about 3.5
  percent.

## Final handoff and future work

Candidate source files:

- `ggml/src/ggml-cpu/arch/x86/quants.c`
- `ggml/src/ggml-cpu/arch/x86/repack.cpp`
- `ggml/src/ggml-cpu/repack.cpp`
- `ggml/src/ggml-cpu/repack.h`
- `ggml/src/ggml-cpu/arch-fallback.h`

The direct path expands Q1 bits to unsigned 0/2 codes with AVX-512 VBMI
and computes `dot(2*b, q8) - sum(q8)` with VNNI. The repack stores four
Q1 rows in four-byte chunks aligned with each Q8_0 block. Its AVX-512
GEMV shares Q8 loads and corrections across twelve output channels,
with eight- and four-output remainder paths. Its 4x4 GEMM shares them
across a four-activation tile. Q1 scales are applied after four Q8
subblocks. The follow-up AVX2 quantizer loads and scales four activation
rows in parallel, packs their signed bytes directly into the same
four-byte interleave, and exactly preserves scalar round-away-from-zero
behavior.

Runtime selection requires AVX-512, VNNI, VBMI, and a row count divisible
by four for the Q1 repack. The activation quantizer uses AVX2 when
available and the generic scalar implementation otherwise. Existing AVX2,
AVX, SSSE3, generic, and non-x86 direct paths are unchanged.
Non-qualifying tensors and architectures retain the established generic
or existing repack fallbacks.

Remaining risks are Clang and cross-CPU coverage, wide-vector frequency
behavior, full-corpus perplexity, and performance on shapes unlike the
27B model. Sixteen-output Q1 fusion was not robust on this host, while
the retained twelve-output tile stayed below its register-pressure
failure point. A future independently reviewable optimization should
examine Q1-specific work chunking or a correction sidecar populated
during activation quantization, without changing the on-disk Q1 format.
The rejected per-call correction prepass shows that either idea needs
its own traffic analysis, layout proof, and benchmark pair.

Before rerunning this protocol, the user must explicitly authorize a
Q1 benchmark window that permits new Q1 build/result directories,
compilation, model loading, and CPU/memory use. Running the frozen Q2
binary additionally requires explicit read-only Q2 runtime authorization.
No future authorization should imply permission to edit or rebuild Q2.
