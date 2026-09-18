# Card and Krueger Minimum Wage

## About the dataset
Each row is one fast food restaurant surveyed twice in 1992: once in February and March, before New Jersey raised its minimum wage, and once in November and December, after. Restaurants are Burger King, KFC, Wendy's, and Roy Rogers outlets in New Jersey and in eastern Pennsylvania just across the border. This file is a trimmed version of the original survey, 384 restaurants with the two employment counts and the state. There is no restaurant identifier, and 18 rows repeat exactly, which is plausible with only three columns but means individual restaurants cannot be told apart. Restaurants compete locally but the study treats each as its own unit. [docs/card_krueger/readme.md]

## What changed
**New Jersey minimum wage rise, 1 April 1992.** New Jersey raised its state minimum wage from 4.25 to 5.05 dollars an hour. Every employer in New Jersey was covered, and none in Pennsylvania, where the minimum stayed at 4.25. The change was a state law, decided by the legislature; which restaurants got it was decided entirely by which side of the state line they sat on. Both waves of the survey bracket the change. [docs/card_krueger/readme.md]

## About each column
**state** — 1 for New Jersey, 0 for Pennsylvania. Where the restaurant is, and therefore whether the wage rise applied to it. Fixed. About 80 percent of rows are New Jersey. [readme.md]

**total_emp_feb** — Full-time equivalent employment in February and March 1992, before the change. Counted as full-time staff plus managers plus half of part-time staff. [readme.md]

**total_emp_nov** — The same count in November and December 1992, after the change. [readme.md]
