# Senate3

## About the dataset
Each row is one US Senate election. [user, 2026-09-17] Rows are identified by state + year. [user, 2026-09-17] The file has 1390 rows and 17 columns. [profile] The file keeps every unit. Every election in the period with the needed returns is in the file. [user, 2026-09-17] Missing values: The next election's returns or the candidate's service record were not available. [user, 2026-09-17] Nothing not in the file affected both who got the change and the outcome. [user, 2026-09-17] Units that got the change could not affect the outcomes of those that did not. [user, 2026-09-17]

## What changed
**Winning the seat.** It reached a Democrat, for six years. [user, 2026-09-17] Who got the change was decided by a cutoff on a score: A Democrat won if the Democratic margin of victory was at or above zero. The score is margin and the cutoff is 0; units above it got the change, a unit exactly at the cutoff included. No column records receipt separately: the rule itself is the change. A unit could not change what the rule looked at. In the file: 750 rows on the treated side of 0.0 on 'margin', 640 on the other. [profile] [user, 2026-09-17]

## About each column
**state** — the state name. Set at the change. [user, 2026-09-17]

**year** — year of the election. Set at the change. [user, 2026-09-17]

**margin** — Democratic margin of victory at the election, in points, minus 100 to 100. Set at the change. The change could not have moved it. [user, 2026-09-17]

**vote** — Democratic vote share at the next election for the same seat six years later. Measured after the change. The change could have moved it. Missing for 93 rows. [user, 2026-09-17]

**class** — Senate class of the seat, 1 to 3, which fixes the years it is contested. Fixed before the change. [user, 2026-09-17]

**termshouse** — terms served in the House as of the election. Set at the change. Missing for 282 rows. [user, 2026-09-17]

**termssenate** — terms served in the Senate as of the election. Set at the change. Missing for 282 rows. [user, 2026-09-17]

**dopen** — 1 if the seat was open, no incumbent running. Set at the change. Missing for 10 rows. [user, 2026-09-17]

**population** — state population in the election year. Set at the change. [user, 2026-09-17]

**presdemvoteshlag1** — Democratic vote share in the state at the presidential election before this one. Fixed before the change. Missing for 3 rows. [user, 2026-09-17]

**demvoteshlag1** — Democratic vote share at the state's previous one Senate election. Fixed before the change. Missing for 41 rows. [user, 2026-09-17]

**demvoteshlag2** — Democratic vote share at the state's previous two Senate elections. Fixed before the change. Missing for 82 rows. [user, 2026-09-17]

**demvoteshfor1** — Democratic vote share at the state's next Senate election two years later, for the other seat. Measured after the change. Missing for 49 rows. [user, 2026-09-17]

**demwinprv1** — 1 if the Democrat won the previous one Senate election. Fixed before the change. Missing for 41 rows. [user, 2026-09-17]

**demwinprv2** — 1 if the Democrat won the previous two Senate elections. Fixed before the change. Missing for 82 rows. [user, 2026-09-17]

**dmidterm** — 1 if a midterm year. Set at the change. [user, 2026-09-17]

**dpresdem** — 1 if the sitting president was a Democrat. Set at the change. [user, 2026-09-17]
