# US Senate Elections 1914 to 2010

## About the dataset
Each row is one US Senate election: a state and a year. There are 1,390 rows over 50 states and 49 election years from 1914, the first year of popular Senate elections, to 2010. A state appears 16 to 33 times, once for each election of either of its two seats, so rows within a state are related and state can serve as a cluster. The file is not sampled by side of the cutoff; it holds every election in the period with the needed returns. Data are from Cattaneo, Frandsen and Titiunik's 2015 study and ship inside the rdrobust Python package as data/rdrobust_RDsenate.csv under GPL-3; senate.csv is that table written out unchanged. [rdrobust package data; Cattaneo, Frandsen and Titiunik 2015]

## What changed
**A Democrat wins the seat at election t.** The score is margin, the Democratic margin of victory at election t in percentage points of the vote. The cutoff is 0. A margin at or above 0 means the Democrat won the seat and the party holds it for the six-year term; a margin below 0 means the Democrat lost. No row sits exactly at 0. Margins run continuously from -100 to 100: 50 elections were decided by less than two points against the Democrat and 52 by less than two points in the Democrat's favour, so elections sit just below and just above the cutoff. There is no take-up column: treatment is the cutoff rule itself, 1 when margin is at or above 0. The outcome is vote, the Democratic vote share at election t+2, the next election for the same seat six years later. [Cattaneo, Frandsen and Titiunik 2015]

## About each column
**state** - State name. Fixed. Repeated across elections, so usable as a cluster. [rdrobust package docs]

**year** - Year of election t, 1914 to 2010. [rdrobust package docs]

**margin** - The score. Democratic margin of victory at election t, in percentage points of the vote, from -100 to 100. The cutoff is 0: at or above 0 the Democrat won and the row is treated; below 0 the Democrat lost. Set by the election t result. [rdrobust package docs; Cattaneo, Frandsen and Titiunik 2015]

**vote** - The outcome. Democratic vote share at election t+2, the next election for the same seat, in percent. Measured six years after the score. Missing for 93 rows. [rdrobust package docs]

**class** - Senate class of the seat, 1 to 3. The class fixes which years the seat is contested: class 1 in years ending 2 modulo 6, class 2 in years ending 4, class 3 in years ending 0. Fixed before election t. [Cattaneo, Frandsen and Titiunik 2015; rdrobust package docs]

**termshouse** - Number of terms served in the House, per the package docstring, which does not say whose terms. A characteristic at election t. Missing for 282 rows. [rdrobust package docs]

**termssenate** - Number of terms served in the Senate, per the package docstring, which does not say whose terms. A characteristic at election t. Missing for 282 rows. [rdrobust package docs]

**dopen** - 1 if the seat was open at election t, that is no incumbent ran. Known before the vote. Missing for 10 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**population** - State population in the election year. Fixed before the vote. [rdrobust package docs]

**presdemvoteshlag1** - Democratic vote share in the state at the presidential election before t (t-1), in percent. Fixed before the score. Missing for 3 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**demvoteshlag1** - Democratic vote share at the state's previous Senate election (t-1), in percent. A lag of an earlier election, fixed before the score. Missing for 41 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**demvoteshlag2** - Democratic vote share at the state's Senate election two before t (t-2), in percent. A lag, fixed before the score. Missing for 82 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**demvoteshfor1** - Democratic vote share at election t+1, the state's next Senate election two years after t, which is for the other seat. Measured after the score, so it is a later outcome, not a covariate. Missing for 49 rows. [Cattaneo, Frandsen and Titiunik 2015]

**demwinprv1** - 1 if the Democrat won the state's previous Senate election (t-1). Derived from that earlier result, fixed before the score. Missing for 41 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**demwinprv2** - 1 if the Democrat won the state's Senate election two before t (t-2). Derived from that earlier result, fixed before the score. Missing for 82 rows. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**dmidterm** - 1 if election t was a midterm election, that is the year is not a presidential year. Set by the calendar. [Cattaneo, Idrobo and Titiunik 2024, Table 2.2]

**dpresdem** - 1 if the sitting president at election t was a Democrat. In the file it is 1 for 1914 to 1920 and 1934 to 1940 and 0 for 1922 to 1932, matching the president's party. Fixed before the vote. [rdrobust package docs; file check]
