| Metric | base |
|---|---|
| ping-pong 8 B (us) | 117.0 (116.6-117.3, n=3) |
| put 8 B (us) | 38.9 (38.8-39.1, n=3) |
| put+signal 8 B at sender (us) | 77.7 (77.4-77.7, n=3) |
| put+signal 64 KiB at sender (us) | 89.8 (89.7-89.9, n=3) |
| pipelined 1 MiB, 1 ch (Gbit/s) | 356.6 (356.4-356.6, n=3) |
| pipelined 64 KiB, 1 ch (Gbit/s) | 117.9 (117.8-118.1, n=3) |
| pipelined 8 KiB, 1 ch (Gbit/s) | 15.7 (15.7-15.8, n=3) |
| batch-signal 64 KiB, 1 ch (Gbit/s) | 94.3 (94.3-95.2, n=3) |
| pipelined 1 MiB, 4 ch (Gbit/s) | 389.7 (389.7-389.8, n=3) |
| pipelined 64 KiB, 4 ch (Gbit/s) | 290.4 (277.3-290.7, n=3) |
| pipelined 8 KiB, 4 ch (Gbit/s) | 59.0 (58.3-59.0, n=3) |
| ordering check, 4 ch (signals/s) | 46913 (46623-47146, n=3) |
| host 1 MiB writes, proxy idle (Gbit/s) | 199.7 (198.9-200.1, n=3) |
| host 64 KiB writes, proxy idle (Gbit/s) | 182.2 (181.2-182.5, n=3) |
rx-step1-ep-7682251.log === install=default channels=4 workers=4 params=: per-run means ['8.89', '8.34', '7.83'] median 8.34 GB/s
