// Cost of GDRCopy pin+map: per 64 KiB page (as the proxy does on a first add) and
// for a whole buffer at once; and of an 8-byte read-modify-write through the BAR.
#include <cuda_runtime.h>
#include <gdrapi.h>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <vector>
static double us(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
    return std::chrono::duration<double, std::micro>(b - a).count();
}
int main() {
    const size_t total = 256ull << 20, page = GPU_PAGE_SIZE;
    void *dptr = nullptr;
    if (cudaMalloc(&dptr, total + page) != cudaSuccess) { printf("cudaMalloc failed\n"); return 1; }
    uintptr_t base = (reinterpret_cast<uintptr_t>(dptr) + page - 1) & ~(page - 1);
    gdr_t g = gdr_open();
    if (!g) { printf("gdr_open failed\n"); return 1; }
    // Per page, first 256 pages (16 MiB).
    std::vector<double> t;
    std::vector<gdr_mh_t> mhs; std::vector<void *> bars;
    for (int i = 0; i < 256; ++i) {
        auto a = std::chrono::steady_clock::now();
        gdr_mh_t mh{}; void *bar = nullptr;
        if (gdr_pin_buffer(g, base + i * page, page, 0, 0, &mh) || gdr_map(g, mh, &bar, page)) { printf("page %d failed\n", i); return 1; }
        auto b = std::chrono::steady_clock::now();
        t.push_back(us(a, b)); mhs.push_back(mh); bars.push_back(bar);
    }
    double sum = 0, mx = 0; for (double x : t) { sum += x; mx = x > mx ? x : mx; }
    printf("per-page pin+map: first %.1f us, mean %.1f us, max %.1f us (256 pages)\n", t[0], sum / t.size(), mx);
    // RMW cost through a mapping.
    uint64_t *w = reinterpret_cast<uint64_t *>(bars[1]);
    auto a = std::chrono::steady_clock::now();
    for (int i = 0; i < 10000; ++i) { uint64_t v; gdr_copy_from_mapping(mhs[1], &v, w, 8); v += 1; gdr_copy_to_mapping(mhs[1], w, &v, 8); }
    auto b = std::chrono::steady_clock::now();
    printf("8-byte RMW through the BAR: %.2f us\n", us(a, b) / 10000);
    for (int i = 0; i < 256; ++i) { gdr_unmap(g, mhs[i], bars[i], page); gdr_unpin_buffer(g, mhs[i]); }
    // Whole buffers at once.
    for (size_t sz : {16ull << 20, 238ull << 20}) {
        auto c = std::chrono::steady_clock::now();
        gdr_mh_t mh{}; void *bar = nullptr;
        int r1 = gdr_pin_buffer(g, base, sz, 0, 0, &mh);
        int r2 = r1 ? -1 : gdr_map(g, mh, &bar, sz);
        auto d = std::chrono::steady_clock::now();
        printf("whole-buffer pin+map of %zu MiB: %s %.1f us\n", sz >> 20, (r1 || r2) ? "FAILED" : "ok", us(c, d));
        if (!r2) gdr_unmap(g, mh, bar, sz);
        if (!r1) gdr_unpin_buffer(g, mh);
    }
    gdr_close(g);
    return 0;
}
