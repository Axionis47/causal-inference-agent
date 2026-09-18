# Fast Food Marketing Campaign

## About the dataset
Each row is one store's sales for one week. A fast food chain tested three promotions for a new menu item across 137 stores in 10 markets over four weeks. Every store appears in all four weeks, so there are 548 rows and each store plus week identifies a row. Stores are in separate locations and do not share customers or stock. Sales are in thousands of dollars. Source is the IBM Watson Analytics sample used for this test. [docs/marketing/readme.md]

## What changed
**Promotion test, weeks 1 to 4.** Each store ran one of three promotions for the new item from the first week of the test, and kept the same promotion for all four weeks. Which promotion a store got was assigned at random by the chain's marketing team; it did not depend on the store's market, size, or age. There is no week before the test in the file, so every row is under a promotion. [docs/marketing/test_plan.md]

## About each column
**MarketID** — Which of the 10 markets the store belongs to. Fixed per store. [readme.md]

**MarketSize** — Small, medium, or large, the chain's classification of the market. Fixed per store and set before the test. [readme.md]

**LocationID** — Store number. Fixed. [readme.md]

**AgeOfStore** — Years since the store opened, at the start of the test. Fixed per store, known before the test. [readme.md]

**Promotion** — 1, 2, or 3. Which promotion the store ran. Assigned at random by the marketing team before week 1 and held constant for all four weeks. A decision, and one that did not depend on anything else in the file. [test_plan.md]

**week** — Test week, 1 to 4. [readme.md]

**SalesInThousands** — Sales of the new item in that store in that week, in thousands of dollars. The week's result. [readme.md]
