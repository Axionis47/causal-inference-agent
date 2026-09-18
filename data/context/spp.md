# Ser Pilo Paga Scholarship, Colombia

## About the dataset
Each row is one Colombian student who took the SABER 11 school-leaving exam in the autumn of 2014, the first cohort of the Ser Pilo Paga (SPP) programme. The file keeps only students whose exam score cleared the programme's merit cutoff and whose household was in the SISBEN welfare register, 23,132 rows, with no identifier; 362 rows repeat exactly, which is expected with nine coarse columns. The file is not sampled by side of the cutoff; the cut is on the exam score and register membership, not on the wealth index. Students are treated as independent. Data are from Londoño-Vélez, Rodríguez and Sánchez's 2020 study, as used in Cattaneo, Idrobo and Titiunik's 2024 Extensions; the file CIT_2024_CUP_fuzzy.csv comes from the rdpackages-replication GitHub repository CIT_2024_CUP, which has no licence file. [readme.md; Cattaneo, Idrobo and Titiunik 2024]

## What changed
**SPP scholarship offer, 2014.** SPP paid full tuition at a high-quality university for students who both scored in the top 9 percent on SABER 11 and came from a poor household, measured by the SISBEN wealth index against a region-specific cutoff. The score is X1, the SISBEN index minus the student's cutoff, in index points, with the cutoff at 0. In this file a student with X1 at or above 0 is eligible and T is 1; below 0 the student is not eligible and T is 0. No row sits exactly at 0. Scores run continuously across the cutoff: 473 students sit within one index point below it and 407 within one point at or above it. Eligibility is not receipt: D records whether the student actually took up the scholarship. No ineligible student received it, and about 59 percent of eligible students did, so noncompliance is one-sided. The outcome Y is enrolment in a higher-education institution right after the offer. [Cattaneo, Idrobo and Titiunik 2024; file check]

## About each column
**X1** - The score. The student's SISBEN wealth index minus the cutoff for their region, in index points; the index runs from 0, poorest, to 100, richest, and X1 runs from -43.5 to 56.2. The cutoff is 0: at or above 0 the student is eligible; below 0 not. The index comes from a household survey taken before the exam, so students could not adjust it. Repeated values are common, 12,918 distinct among 23,132 rows. [Cattaneo, Idrobo and Titiunik 2024; file check]

**T** - Eligibility, 1 or 0. Equal to 1 exactly when X1 is at or above 0 in every row; it is the cutoff rule, not take-up. [CIT_2024_CUP_fuzzy.py; file check]

**D** - Take-up. 1 if the student actually received the SPP scholarship, 0 if not. 0 for every student with X1 below 0 and 1 for about 59 percent of students with X1 at or above 0, so noncompliance is one-sided. [Cattaneo, Idrobo and Titiunik 2024; file check]

**Y** - The outcome. 1 if the student enrolled in any higher-education institution immediately after the offer, 0 if not. Measured after the score. Because it is 0 or 1, an average is the share enrolled and an effect is a difference in shares, not a percentage-point count of students. [Cattaneo, Idrobo and Titiunik 2024]

**icfes_female** - 1 if the student reported being female on the exam registration. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024]

**icfes_age** - The student's age at the exam, in years, from the registration. Fixed before the score. Missing for 60 rows. [Cattaneo, Idrobo and Titiunik 2024]

**icfes_urm** - 1 if the student reported belonging to an ethnic minority on the registration. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024]

**icfes_stratum** - The residential stratum of the student's household, 1 poorest to 6 richest, from the registration. Fixed before the score. Missing for 59 rows. [Cattaneo, Idrobo and Titiunik 2024]

**icfes_famsize** - Number of people in the student's household, from the registration, 1 to 12. Fixed before the score. Missing for 59 rows. [Cattaneo, Idrobo and Titiunik 2024]
