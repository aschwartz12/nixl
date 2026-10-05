#include <stdio.h>
#include <infiniband/verbs.h>
#include <infiniband/efadv.h>
int main(void) {
    int n = 0;
    struct ibv_device **list = ibv_get_device_list(&n);
    for (int i = 0; i < n && i < 2; ++i) {
        struct ibv_context *ctx = ibv_open_device(list[i]);
        struct efadv_device_attr attr = {0};
        if (!ctx || efadv_query_device(ctx, &attr, sizeof(attr))) { printf("%s: query failed\n", ibv_get_device_name(list[i])); continue; }
        printf("%s: caps=0x%x rdma_read=%d rnr_retry=%d rdma_write=%d unsolicited_write_recv=%d cq_ext_mem_dmabuf=%d comp_cntr=%d max_rdma_size=%u\n",
               ibv_get_device_name(list[i]), attr.device_caps, !!(attr.device_caps & (1 << 0)), !!(attr.device_caps & (1 << 1)),
               !!(attr.device_caps & (1 << 3)), !!(attr.device_caps & (1 << 4)), !!(attr.device_caps & (1 << 5)), !!(attr.device_caps & (1 << 6)),
               attr.max_rdma_size);
        ibv_close_device(ctx);
    }
    printf("devices=%d\n", n);
    return 0;
}
