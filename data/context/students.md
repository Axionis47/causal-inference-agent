# Students Performance

## About the dataset
Each row is one student's results from the May 2026 exam at one school. Every enrolled student who sat the exam is included; absentees are not. There is no student identifier in the file, so rows cannot be tied back to a person. Students sit individually, so one student's result does not affect another's. Exported from the school register on 1 June 2026. [docs/students/readme.md]

## What changed
**Prep course, 1 March to 12 April 2026.** The school ran a six-week test preparation course before the exam. The counsellor offered places first to students on free or reduced lunch and to students whose parents hold no degree; remaining places were open to anyone who asked. Completion was recorded by the counsellor. [docs/students/prep_course_memo.md]

## About each column
**gender** — Recorded at enrolment, before the exam. [readme.md]

**race/ethnicity** — The school's own grouping, letters A to E; the mapping is not published. Recorded at enrolment. [readme.md]

**parental level of education** — Highest qualification held by either parent, as declared at enrolment. Six levels from some high school to master's degree. Known before the exam. [readme.md]

**lunch** — Standard or free/reduced. Free/reduced is assigned by the district when household income is below the federal threshold, so it is the school's low-income indicator. Set before the exam and not by the school. [lunch_policy.md]

**test preparation course** — Whether the student completed the prep course. Who got a place depended on lunch status and parental education, per the counsellor's offering rule above; after that, uptake was voluntary. So this column is a decision, not a measurement, and it was shaped by lunch and parental education. Fixed before the exam. [prep_course_memo.md]

**math score, reading score, writing score** — The exam marks, 0 to 100, recorded at the sitting. No missing values. [readme.md]
