# Senate2

## About the dataset
Each row is one US Senate election. [user, 2026-09-16] Rows are identified by state + year. [user, 2026-09-16] The file has 1390 rows and 17 columns. [profile] The file keeps every unit. Every US Senate election from 1914 to 2010 with needed returns. [user, 2026-09-16] Missing values: Values are missing where the next election's returns or the candidate's service record were not available; that has nothing to do with who won. [user, 2026-09-16] Nothing not in the file affected both who got the change and the outcome. [user, 2026-09-16] Units that got the change could not affect the outcomes of those that did not. [user, 2026-09-16]

## What changed
**Holding a Senate seat.** It reached a Democrat, for six years after winning an election. [user, 2026-09-16] Who got the change was decided by a cutoff on a score: A Democrat won if the margin of victory was at or above zero. The score is margin and the cutoff is 0; units above it got the change, a unit exactly at the cutoff included. No column records receipt separately: the rule itself is the change. A unit could not change what the rule looked at. In the file: 750 rows on the treated side of 0.0 on 'margin', 640 on the other. [profile] [user, 2026-09-16]

## About each column
**state** — state name. Set at the change. [user, 2026-09-16]

**year** — year of the election. Set at the change. [user, 2026-09-16]

**margin** — Democratic margin of victory at the election, in percentage points. Set at the change. [user, 2026-09-16]

**vote** — Democratic vote share at the next election for the same seat. Measured after the change. Missing for 93 rows. [user, 2026-09-16]

**class** — Senate class of the seat. Fixed before the change. [user, 2026-09-16]

**termshouse** — terms served in the House. Set at the change. Missing for 282 rows. [user, 2026-09-16]

**termssenate** — terms served in the Senate. Set at the change. Missing for 282 rows. [user, 2026-09-16]

**dopen** — 1 if the seat was open (no incumbent running). Set at the change. Missing for 10 rows. [user, 2026-09-16]

**population** — state population. Set at the change. [user, 2026-09-16]

**presdemvoteshlag1** — Democratic vote share in the state at the presidential election. Fixed before the change. Missing for 3 rows. [user, 2026-09-16]

**demvoteshlag1** — Democratic vote share at the state's previous Senate election. Fixed before the change. Missing for 41 rows. [user, 2026-09-16]

**demvoteshlag2** — Democratic vote share at the state's second previous Senate election. Fixed before the change. Missing for 82 rows. [user, 2026-09-16]

**demvoteshfor1** — Democratic vote share at the state's next Senate election for the other seat. Measured after the change. Missing for 49 rows. [user, 2026-09-16]

**demwinprv1** — 1 if the Democrat won the previous Senate election. Fixed before the change. Missing for 41 rows. [user, 2026-09-16]

**demwinprv2** — 1 if the Democrat won the second previous Senate election. Fixed before the change. Missing for 82 rows. [user, 2026-09-16]

**dmidterm** — 1 if a midterm year. Set at the change. [user, 2026-09-16]

**dpresdem** — 1 if the sitting president was a Democrat. Set at the change. [user, 2026-09-16]
