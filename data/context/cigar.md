# State Cigarette Sales 1963 to 1992

## About the dataset
Each row is one US state in one year. 46 states are observed every year from 1963 to 1992, 1,380 rows, no gaps. Years are written as two digits, 63 to 92. States are numbered 1 to 51 in alphabetical order of the standard state list with five states missing, so the numbers are not contiguous; state 5 is California. The first column is an unnamed row index left over from an R export and carries no information. States are treated as separate markets, though cigarettes bought in a neighbouring state with lower prices are a known leak, which is why the minimum neighbouring price is included. Data are Baltagi and Levin's panel, distributed in the Ecdat package. [docs/cigar/readme.md]

## What changed
**California Proposition 99, effective January 1989.** California voters passed a 25 cent per pack tax increase, with the revenue funding an anti-smoking campaign. It applied to the whole state and no other state adopted anything comparable in the period. It was decided by a ballot vote, and which state got it was simply California. The file has 26 years before the change and 4 after. Other states raised their own taxes by smaller amounts at various times, which shows up in the price column. [docs/cigar/readme.md]

## About each column
**Unnamed: 0** — Row index from the export. Ignore. [readme.md]

**state** — State number, see above. Fixed. [readme.md]

**year** — Two-digit year, 63 to 92. [readme.md]

**price** — Average retail price of a pack in that state and year, in cents, including taxes. Set by manufacturers, retailers, and state tax, and moves with them. [readme.md]

**pop** — State population in thousands. [readme.md]

**pop16** — State population aged 16 and over, in thousands. [readme.md]

**cpi** — National consumer price index, 1983 equals 100. Same for every state in a year. [readme.md]

**ndi** — Per capita disposable income in the state, in current dollars. [readme.md]

**sales** — Packs sold per capita in the state that year, from tax-paid sales. The year's result. [readme.md]

**pimin** — Lowest average price among the state's neighbouring states, in cents. A measure of how cheap it is to buy across the border. [readme.md]
