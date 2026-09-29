# External execution timeout

The caller terminated this run after 3600000ms, with 29 passed points and
llama-fa-b512-r2-i1024-c4 incomplete. The original manifest's `running` state and
all partial files are intentionally preserved, not represented as a passed run.
No owned servers remained after timeout (process/container checks performed).

The sibling `2026-09-29-inferencex-local-resumed` run verifies identities and
workloads, inherits only the 29 passed points, and runs the remaining seven fresh.
Use that combined manifest for results. The interrupted point is excluded because
it is incomplete, not because of its performance. Future whole-matrix invocations
use an explicit three-hour outer tool timeout; individual stall limits remain.
