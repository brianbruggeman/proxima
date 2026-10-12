# independent shared-weight owner admission

status: ADMIT

The revision adds the missing count-bearing TASKS row 7. It binds the three new
owner cases plus eight CPU and three Metal recurrent cases to 14 executions,
caller/planner/floor/clippy counts, 48 live four-arm forward records, eight
independent method comparisons, and a fresh 12-request configured checkpoint
(TASKS.md:14-27). The exact nextest selection and native no-wrapper config are
present at TASKS.md:16-20.

The lifetime model places the completed binding and mapping in one Arc-owned
`ModelWeights` destructor; it explicitly forbids `strong_count` polling and
keeps graph descriptors, KV, prompt caches, and expert state independent
(WEIGHT-OWNER-SPEC.md:21-24, 28-40, 62-69). Descriptor reuse accepts only
recurrence/readout differences and checks before graph construction
(WEIGHT-OWNER-SPEC.md:52-61). The concurrent final-drop case is CPU-only and
states there are no active Metal residency or checkpoint-mapping registrations;
the Metal survivor/drop case executes and drops on its owning thread
(WEIGHT-OWNER-SPEC.md:94-118). Thus it does not claim cross-thread cleanup for
thread-local Metal resources.

The fixture specifies BF16-owned decode paths, literal input data and expected
vectors, pointer/codec identity, distinct-owner and invalid-descriptor controls
(WEIGHT-OWNER-SPEC.md:80-118). Production integration reuses the existing HF
constructor, admission, graph builder, and lab arm vector
(WEIGHT-OWNER-SPEC.md:45-73). No unresolved admission condition remains in this
slice.

The later validation amendment separates strict production-library clippy
from all-target clippy. Only the all-target run may grant command-line
`expect_used` and `unwrap_used` exceptions; source-level lint suppression is
forbidden and raw failures are retained (WEIGHT-OWNER-SPEC.md:120-145). The
native selection now requires all 14 selected cases to execute and a separate
excluded-by-filter count. TASKS.md now includes exact production-library and
all-target clippy commands with their zero-diagnostic requirements. The task
row's clippy criterion is bound to that command section, closing the earlier
row-level omission.
