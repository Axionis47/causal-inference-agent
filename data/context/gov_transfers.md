# Uruguay Government Transfers

## About the dataset
Each row is one household from a 2007 survey in Uruguay, taken after the PANES emergency cash transfer programme of 2005 to 2007. The file keeps only households whose eligibility score sits within a narrow band around the programme's cutoff, 1,948 of them, so it is the households closest to the line on either side. There is no household identifier and 21 rows repeat exactly. Households are treated as independent. Data are from Manacorda, Miguel and Vigorito's 2011 study, distributed with the causaldata package. [docs/gov_transfers/readme.md] The survey sample was drawn by side of the cutoff, two eligible households for every ineligible one, so the count of rows on each side says nothing about manipulation of the score. [Manacorda, Miguel and Vigorito 2011]

## What changed
**PANES cash transfer, 2005 to 2007.** The government paid a monthly transfer to poor households. Eligibility was decided by a score predicted from household characteristics before the programme; households with a score below a fixed cutoff were eligible, households above were not. The rule was applied mechanically, and take-up among the eligible was near universal, so in this file every household below the cutoff received the transfer and no household above it did. [docs/gov_transfers/readme.md]

## About each column
**Income_Centered** — The eligibility score minus the cutoff, so the cutoff is at zero. Negative means eligible. Computed before the programme from household characteristics, and not something households could adjust afterwards. Values run from about minus 0.02 to plus 0.02, the band kept in the file. [readme.md]

**Education** — Years of schooling of the household head. Recorded in the survey. Missing for 51 households. [readme.md] The study treats the head's years of schooling as a characteristic fixed before the programme and uses it as a balance check. [Manacorda, Miguel and Vigorito 2011]

**Age** — Age of the household head at the survey. [readme.md] The study treats the head's age as a characteristic fixed before the programme and uses it as a balance check. [Manacorda, Miguel and Vigorito 2011]

**Participation** — 1 if the household received the transfer, 0 if not. Determined by the score: 1 for every household with Income_Centered below zero, 0 for every household above. [readme.md]

**Support** — The respondent's stated support for the government in the 2007 survey, scored 0, 0.5, or 1. Measured after the programme. [readme.md]
