| Metric | base | fenceoff | receiver | rxnoflush | rxrail | sender |
|---|---|---|---|---|---|---|
| ping-pong 8 B (us) | 114.4 (113.9-114.8, n=3) | 58.4 (58.0-58.6, n=3) | 58.7 (58.2-58.8, n=3) | 57.7 (57.6-58.3, n=3) | 58.7 (58.6-59.0, n=3) | 114.4 (113.9-114.8, n=3) |
| put 8 B (us) | 38.5 (38.3-38.7, n=3) | 38.4 (38.1-38.4, n=3) | 38.8 (38.5-38.9, n=3) | 38.3 (38.2-38.4, n=3) | 39.0 (38.6-39.0, n=3) | 38.4 (37.9-38.8, n=3) |
| put+signal 8 B at sender (us) | 75.9 (75.6-76.1, n=3) | 48.6 (48.4-48.8, n=3) | 49.1 (49.1-49.4, n=3) | 48.5 (48.4-48.6, n=3) | 49.3 (49.0-49.3, n=3) | 76.1 (75.0-76.5, n=3) |
| put+signal 64 KiB at sender (us) | 87.8 (87.4-88.3, n=3) | 52.8 (51.8-52.8, n=3) | 60.9 (60.1-61.4, n=3) | 60.5 (59.9-61.0, n=3) | 60.6 (60.4-61.0, n=3) | 87.9 (87.8-88.5, n=3) |
| pipelined 1 MiB, 1 ch (Gbit/s) | 356.7 (356.7-356.9, n=3) | 356.6 (356.4-357.1, n=3) | 355.0 (353.2-355.1, n=3) | 354.9 (354.6-355.1, n=3) | 354.8 (354.8-355.3, n=3) | 356.8 (356.6-356.9, n=3) |
| pipelined 64 KiB, 1 ch (Gbit/s) | 118.6 (118.5-118.7, n=3) | 118.6 (118.3-119.0, n=3) | 117.0 (112.1-117.2, n=3) | 117.2 (117.1-117.6, n=3) | 117.0 (116.5-117.5, n=3) | 118.5 (118.3-118.8, n=3) |
| pipelined 8 KiB, 1 ch (Gbit/s) | 15.8 (15.7-15.9, n=3) | 15.9 (15.8-16.0, n=3) | 15.9 (15.5-15.9, n=3) | 15.8 (15.8-15.9, n=3) | 15.9 (15.8-15.9, n=3) | 15.9 (15.9-15.9, n=3) |
| batch-signal 64 KiB, 1 ch (Gbit/s) | 95.7 (95.1-95.8, n=3) | 118.6 (118.4-119.7, n=3) | 111.5 (108.3-113.0, n=3) | 113.0 (110.6-113.3, n=3) | 112.0 (111.9-112.9, n=3) | 95.3 (95.0-95.7, n=3) |
| pipelined 1 MiB, 4 ch (Gbit/s) | 389.6 (389.5-389.8, n=3) | 389.7 (389.7-389.8, n=3) | 389.7 (294.1-389.7, n=3) | 389.7 (389.7-389.7, n=3) | 389.7 (389.7-389.7, n=3) | 389.7 (389.7-389.7, n=3) |
| pipelined 64 KiB, 4 ch (Gbit/s) | 290.9 (290.2-291.4, n=3) | 291.2 (291.1-291.5, n=3) | 261.3 (207.6-273.0, n=3) | 287.4 (287.1-287.8, n=3) | 287.8 (286.6-287.8, n=3) | 291.4 (291.3-291.5, n=3) |
| pipelined 8 KiB, 4 ch (Gbit/s) | 58.6 (57.7-58.7, n=3) | 58.6 (58.2-58.6, n=3) | 40.5 (37.1-40.6, n=3) | 58.6 (58.5-58.6, n=3) | 58.2 (57.6-58.3, n=3) | 58.6 (58.6-58.7, n=3) |
| ordering check, 4 ch (signals/s) | 47828 (46784-49559, n=3) | 56949 (53708-57428, n=3) | 65571 (64890-65593, n=3) | 56949 (56026-57633, n=3) | 58019 (56910-59386, n=3) | 48346 (47926-49014, n=3) |
| ordering check, profiled (signals/s) | 47459 (46573-47795, n=3) | 53105 (52507-53867, n=3) | 63840 (63796-66740, n=3) | 54709 (53988-58106, n=3) | 56357 (52671-56824, n=3) | 46967 (46693-47994, n=3) |
| target cores busy, ordering check | - | 0.60 (0.58-0.61, n=3) | 2.05 (1.90-2.48, n=3) | 0.66 (0.65-0.75, n=3) | 0.70 (0.66-0.71, n=3) | 0.49 (0.47-0.50, n=3) |
| sender cores busy, ordering check | - | 2.04 (1.99-2.11, n=3) | 1.50 (1.49-1.62, n=3) | 2.00 (1.92-2.08, n=3) | 1.97 (1.92-2.09, n=3) | 2.06 (1.98-2.07, n=3) |
| target cores busy, 16 KiB puts + host writes | - | - | 0.63 (0.55-0.65, n=3) | 0.21 (0.21-0.22, n=3) | 0.21 (0.21-0.22, n=3) | - |
| host 1 MiB writes, proxy idle (Gbit/s) | 200.5 (199.6-200.8, n=3) | 201.5 (200.6-204.8, n=3) | 177.0 | 198.9 (198.4-202.9, n=3) | 202.7 (199.0-203.6, n=3) | 200.9 (199.1-202.7, n=3) |
| host 64 KiB writes, proxy idle (Gbit/s) | 180.7 (180.4-180.9, n=3) | 180.9 (180.6-185.5, n=3) | 173.7 | 177.4 (177.3-179.6, n=3) | 179.2 (179.1-179.4, n=3) | 180.6 (180.5-180.7, n=3) |
| host 1 MiB writes during device 16 KiB puts (Gbit/s) | - | 198.8 (195.0-201.2, n=3) | 187.7 (185.2-189.0, n=3) | 194.6 (194.5-196.3, n=3) | 198.4 (183.9-199.1, n=3) | 195.2 (185.8-197.6, n=3) |
| device 16 KiB puts, 4 ch, during host writes (Gbit/s) | - | 58.2 (57.7-58.6, n=3) | 48.8 (48.4-49.8, n=3) | 58.0 (56.2-58.7, n=3) | 57.9 (56.6-62.4, n=3) | 58.4 (58.3-62.7, n=3) |
rx-step4-ep-7683462.log === mode=base install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-efa-cpu-proxy-ctr-install params=efa_proxy_ordering=sender: per-run means ['9.43', '9.70', '8.68'] median 9.43 GB/s
rx-step4-ep-7683462.log === mode=sender install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=sender: per-run means ['8.23', '9.32', '8.43'] median 8.43 GB/s
rx-step4-ep-7683462.log === mode=receiver install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=receiver: per-run means ['5.74', '5.85', '5.78'] median 5.78 GB/s
rx-step4-ep-7683462.log === mode=rxrail install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail: per-run means ['8.54', '7.91', '8.47'] median 8.47 GB/s
rx-step4-ep-7683462.log === mode=rxnoflush install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=receiver,efa_proxy_rx_thread=rail,efa_proxy_rx_flush=0: per-run means ['7.75', '8.38', '8.39'] median 8.38 GB/s
rx-step4-ep-7683462.log === mode=fenceoff install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=sender,efa_proxy_test_fence_off=1: per-run means ['12.57', '13.69'] median 13.13 GB/s
rx-step4-ep-default-7683844.log === mode=base install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-efa-cpu-proxy-ctr-install params=efa_proxy_ordering=sender: per-run means ['14.31', '14.31', '14.17', '14.08', '14.14'] median 14.17 GB/s
rx-step4-ep-default-7683844.log === mode=sender install=/lustre/fsw/portfolios/network/projects/network_research_advdev/users/aschwartz/efa/build/nixl-rxorder-ctr-install params=efa_proxy_ordering=sender: per-run means ['14.27', '14.19', '14.11', '14.21', '14.29'] median 14.21 GB/s
