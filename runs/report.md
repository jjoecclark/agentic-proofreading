# Outer-harness report

report_version 2. Best harness: **h003**. Total spend in the ledger: **$91.18** (4102 requests).

## Rounds

Edit = merge + split tasks; no-edit = negative tasks (right answer: none).

| round | harness | train all | train edit | train no-edit | val all | val merge | val split | val no-merge | val no-split | meta $ | probe $ | eval $ | kept | why |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | h000 | 39/163 (24%) | 39/99 (39%) | 0/63 (0%) | 22/86 (26%) | 22/25 (88%) | 0/25 (0%) | 0/25 (0%) | 0/11 (0%) | 2.78 | 3.70 | 1.42 | yes | first harness |
| 1 | h001 | 84/163 (52%) | 31/100 (31%) | 53/63 (84%) | 43/86 (50%) | 16/25 (64%) | 0/25 (0%) | 18/25 (72%) | 9/11 (82%) | 4.28 | 4.43 | 0.54 | yes | val hits 43 vs 22 (+21 >= 3) |
| 2 | h002 | 90/163 (55%) | 39/100 (39%) | 51/63 (81%) | 45/86 (52%) | 18/25 (72%) | 0/25 (0%) | 17/25 (68%) | 10/11 (91%) | 2.45 | 2.27 | 18.77 | no | val hits 45 vs 43 (+2 < +3) |
| 3 | h003 | 92/163 (56%) | 32/100 (32%) | 60/63 (95%) | 52/86 (60%) | 19/25 (76%) | 0/25 (0%) | 22/25 (88%) | 11/11 (100%) | 15.59 | 9.46 | 0.00 | yes | val hits 52 vs 43 (+9 >= 3) |
