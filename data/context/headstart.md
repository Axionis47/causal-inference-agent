# Head Start and Child Mortality

## About the dataset
Each row is one US county, 2,810 rows across 50 states coded 1 to 51, with a county code in oldcode. The score is povrate60, the county's poverty rate in the 1960 census, in percent, from 15.2 to 93.1. The outcome is mortality of children aged 5 to 9 from causes that Head Start's health services could affect, over 1973 to 1983, in deaths per 100,000. The file is not sampled by side of the cutoff; it holds every county with data. 6 rows lack the score and 25 lack the outcome. Data are from Ludwig and Miller's 2007 study as re-analysed by Cattaneo, Titiunik and Vazquez-Bare 2017; the file headstart.csv comes from the rdpackages-replication GitHub repository CTV_2017_JPAM, which has no licence file. [readme.md; Cattaneo, Titiunik and Vazquez-Bare 2017]

## What changed
**Head Start grant-writing help, 1965.** In 1965 the federal Office of Economic Opportunity offered the 300 poorest counties help to apply for Head Start funding. The cutoff on the 1960 poverty rate was set at 59.1984 so that exactly 300 counties fell at or above it. A county with povrate60 at or above 59.1984 got the help and is treated; a county below did not. No county sits exactly at the cutoff. Poverty rates run continuously across it: 169 counties sit within five points below the cutoff and 141 within five points at or above it. There is no take-up column: treatment is the cutoff rule itself. The score was computed from 1960 census data, five years before the rule, so counties could not adjust it. [Cattaneo, Titiunik and Vazquez-Bare 2017]

## About each column
**oldcode** - County code, one per row. An identifier. [CTV_2017_JPAM.py]

**state** - State number, 1 to 51. Fixed. [CTV_2017_JPAM.py]

**povrate60** - The score. County poverty rate from the 1960 census, in percent. The cutoff is 59.1984: at or above it the county got Head Start help and is treated; below it the county did not. Fixed in 1960, before the programme. Missing for 6 rows. [Cattaneo, Titiunik and Vazquez-Bare 2017]

**mort_age59_related_postHS** - The outcome. Deaths of children aged 5 to 9 from causes Head Start's health services could affect, 1973 to 1983, per 100,000. Measured after the programme. Many zeros, maximum 136. Missing for 25 rows. [Cattaneo, Titiunik and Vazquez-Bare 2017]

**mort_age59_injury_postHS** - Deaths of children aged 5 to 9 from injuries, 1973 to 1983, per 100,000. A later outcome that Head Start should not affect; the paper uses it as a placebo outcome, not as a covariate. Missing for 25 rows. [Cattaneo, Titiunik and Vazquez-Bare 2017]

**mort_age59_all_postHS** - Deaths of children aged 5 to 9 from all causes, 1973 to 1983, per 100,000. A later outcome. Missing for 25 rows. [CTV_2017_JPAM.py; column name]

**mort_age25plus_related_postHS** - Deaths of people aged 25 and over from the same Head Start related causes, in the post period, per 100,000. A later outcome for an age group the programme did not serve. Missing for 25 rows. [column name]

**mort_age25plus_injuries_postHS** - Deaths of people aged 25 and over from injuries, in the post period, per 100,000. A later outcome. Missing for 26 rows. [column name]

**mort_age59_related_preHS** - Deaths of children aged 5 to 9 from Head Start related causes in the years before the programme began, per 100,000. Fixed before treatment; the paper uses it as a placebo outcome and as a balance check. Missing for 2 rows. [Cattaneo, Titiunik and Vazquez-Bare 2017]

**mort_wh_age59_related_postHS** - The main outcome for white children only, 1973 to 1983, per 100,000. A later outcome. Missing for 25 rows. [column name]

**mort_bl_age59_related_postHS** - The main outcome for black children only, 1973 to 1983, per 100,000. A later outcome. Missing for 438 rows, counties with too few black children. [column name]

**census1960_pop** - County population in the 1960 census. Fixed before the programme; a pre-period characteristic. Missing for 6 rows. [CTV_2017_JPAM.py]

**census1960_pctsch1417** - Share of those aged 14 to 17 in school in 1960, in percent. Pre-period characteristic. Missing for 28 rows. [CTV_2017_JPAM.py; column name]

**census1960_pctsch534** - Share of those aged 5 to 34 in school in 1960, as a fraction from 0 to 0.75. Pre-period characteristic. Missing for 27 rows. [CTV_2017_JPAM.py; column name]

**census1960_pctsch25plus** - A schooling measure for those aged 25 and over in 1960, in percent, per the column name. Pre-period characteristic. Missing for 28 rows. [CTV_2017_JPAM.py; column name]

**census1960_pop1417** - Number of residents aged 14 to 17 in 1960. Pre-period characteristic. Missing for 27 rows. [CTV_2017_JPAM.py; column name]

**census1960_pop534** - Number of residents aged 5 to 34 in 1960. Pre-period characteristic. Missing for 27 rows. [CTV_2017_JPAM.py; column name]

**census1960_pop25plus** - Number of residents aged 25 and over in 1960. Pre-period characteristic. Missing for 27 rows. [CTV_2017_JPAM.py; column name]

**census1960_pcturban** - Share of the county living in urban areas in 1960, in percent. Pre-period characteristic. Missing for 22 rows. [CTV_2017_JPAM.py; column name]

**census1960_pctblack** - Share of the county that was black in 1960, in percent. Pre-period characteristic. Missing for 22 rows. [CTV_2017_JPAM.py; column name]

**census1990_pop** - County population in the 1990 census. Measured 25 years after the programme, so not fixed before the score. Missing for 4 rows. [CTV_2017_JPAM.py]

**census1990_pop1824** - Share of the 1990 population aged 18 to 24, as a fraction. Measured after the programme. Missing for 9 rows. [CTV_2017_JPAM.py; column name]

**census1990_pop2534** - Share of the 1990 population aged 25 to 34, as a fraction. Measured after the programme. Missing for 9 rows. [CTV_2017_JPAM.py; column name]

**census1990_pop3554** - Share of the 1990 population aged 35 to 54, as a fraction. Measured after the programme. Missing for 9 rows. [CTV_2017_JPAM.py; column name]

**census1990_pop55plus** - Share of the 1990 population aged 55 and over, as a fraction. Measured after the programme. Missing for 9 rows. [CTV_2017_JPAM.py; column name]

**census1990_pcturban** - Share of the county living in urban areas in 1990, as a fraction. Measured after the programme. Missing for 4 rows. [CTV_2017_JPAM.py; column name]

**census1990_pctblack** - Share of the county that was black in 1990, as a fraction. Measured after the programme. Missing for 4 rows. [CTV_2017_JPAM.py; column name]

**census1990_percapinc** - Per capita income in 1990, in dollars. Measured after the programme. Missing for 4 rows. [CTV_2017_JPAM.py; column name]
