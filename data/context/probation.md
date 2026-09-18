# Academic Probation and Next-Term GPA

## About the dataset
Each row is one first-year student at one of three campuses of a large Canadian university, entering between 1996 and 2005, 40,582 rows, with no identifier; 166 rows repeat exactly, which is expected with a coarse score and few columns. Only students who came back for a second term are in the file, because the outcome is the next term's GPA; a student who left after the first year has no row. The file is not sampled by side of the cutoff. Students are treated as independent. Data are from Lindo, Sanders and Oreopoulos's 2010 study, as used in Cattaneo, Idrobo and Titiunik's 2024 Extensions; the file CIT_2024_CUP_discrete.csv comes from the rdpackages-replication GitHub repository CIT_2024_CUP, which has no licence file. [readme.md; Cattaneo, Idrobo and Titiunik 2024]

## What changed
**Placement on academic probation after the first year.** A student whose first-year GPA fell below the campus cutoff, 1.5 on campuses 1 and 2 and 1.6 on campus 3, was placed on probation and had to raise their GPA the next term or face suspension. The score is X, the campus cutoff minus the student's first-year GPA, in GPA points, so the three campuses share a cutoff of 0. X at or above 0 means the GPA was at or below the cutoff and the student was placed on probation; X below 0 means no probation. T records this rule exactly. The 208 students whose GPA equalled the cutoff sit at X = -0.000005, a rounding residue, on the untreated side with T = 0. There is no take-up column: probation is the rule itself. Students sit just either side of the cutoff: 1,038 within 0.1 GPA points below it and 719 within 0.1 at or above it. The outcome is nextGPA, the student's GPA in the term right after. [Cattaneo, Idrobo and Titiunik 2024; file check]

## About each column
**X** - The score. The campus probation cutoff minus the student's first-year GPA, in GPA points, from -2.8 to 1.6 in steps of 0.01, so only 429 distinct values and many ties. The cutoff is 0: at or above 0 the student was placed on probation and is treated; below 0 not. The 208 students with GPA equal to the cutoff are stored as -0.000005 and count as below. Fixed by first-year grades, before probation. [Cattaneo, Idrobo and Titiunik 2024; file check]

**T** - 1 if the student was placed on probation, 0 if not. Equal to 1 exactly when X is at or above 0 in every row; it is the cutoff rule, not take-up. [CIT_2024_CUP_discrete.py; file check]

**nextGPA** - The outcome. The student's GPA in the term right after the probation decision, on the 0 to 4.3 scale. Observed only for students who returned. Measured after the score. [Cattaneo, Idrobo and Titiunik 2024]

**hsgrade_pct** - The percentile of the student's average grade in standard high-school classes, 1 to 100. Fixed before university entry. [Cattaneo, Idrobo and Titiunik 2024]

**totcredits_year1** - Number of credits the student enrolled in during the first year, 3 to 6.5 in half steps. Chosen at the start of the first year, before the GPA was known. [Cattaneo, Idrobo and Titiunik 2024]

**age_at_entry** - The student's age at entry, 17 to 21. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024]

**male** - 1 if the student is male. Fixed. [Cattaneo, Idrobo and Titiunik 2024]

**bpl_north_america** - 1 if the student was born in North America. Fixed. [Cattaneo, Idrobo and Titiunik 2024]

**loc_campus1** - 1 if the student was at campus 1, where the cutoff is a GPA of 1.5. Exactly one of the three campus columns is 1 in every row. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024; file check]

**loc_campus2** - 1 if the student was at campus 2, where the cutoff is a GPA of 1.5. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024]

**loc_campus3** - 1 if the student was at campus 3, where the cutoff is a GPA of 1.6. Fixed before the score. [Cattaneo, Idrobo and Titiunik 2024]
