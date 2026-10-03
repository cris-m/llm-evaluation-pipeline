# RAG Test Questions

Use these to check whether retrieval finds the right chunk(s) and the LLM answers correctly.
Types: SIMPLE = one fact, MULTI = needs 2+ chunks/files, OVERRIDE = a specific rule beats a general one,
TRICKY = similar-sounding facts, UNANSWERABLE = the correct answer is "not in the documents".

| # | Type | Question | Expected answer | Source file(s) |
|---|------|----------|-----------------|----------------|
| 1 | SIMPLE | How much is the daily meal allowance on business trips? | $40 per day | expenses_and_equipment |
| 2 | SIMPLE | Which days are mandatory in-office? | Tuesdays and Thursdays | general_policies |
| 3 | SIMPLE | How long is leave for a primary caregiver? | 16 weeks, fully paid | general_policies |
| 4 | SIMPLE | When does an employee hired on March 18 get health insurance? | April 1 | general_policies |
| 5 | SIMPLE | What is the 401(k) match? | 100% up to 4% of salary | compensation_and_performance |
| 6 | TRICKY | Do I need a receipt for a $25 purchase? | Receipts are required only ABOVE $25, so exactly $25 does not; it must still be logged | expenses_and_equipment |
| 7 | TRICKY | How many vacation days do I get after 4 years? | 20 days per year | general_policies |
| 8 | TRICKY | Do sick days come out of my vacation balance? | No, 10 sick days are separate | general_policies |
| 9 | TRICKY | Does the company closure between Dec 25 and Jan 1 use my vacation days? | No | general_policies |
| 10 | OVERRIDE | How much can I spend on an AI training certification? | $500 per year, within the $1,200 budget | security_and_ai_usage, learning_and_development |
| 11 | OVERRIDE | Do I need manager approval for a $200 AI tool subscription? | Yes, approval from BOTH the IT Security Lead and manager (AI rule overrides the general rule) | security_and_ai_usage, it_and_support |
| 12 | MULTI | I want a $400 conference ticket. Who approves it? | Direct manager via HR portal ($300-$1,200); conference travel is NOT covered by the learning budget | learning_and_development, expenses_and_equipment |
| 13 | MULTI | I spent $350 on a Claude Pro subscription personally. Can I be reimbursed? | No, personal AI subscriptions are not reimbursed | security_and_ai_usage |
| 14 | MULTI | A new hire wants to use the stipend for a desk. Do they need receipts and is it taxed? | No receipts needed; it is taxable income | expenses_and_equipment |
| 15 | MULTI | I accidentally pasted firmware code into an external AI tool. What now? | Report to the security team immediately (it is prohibited) | security_and_ai_usage |
| 16 | MULTI | A part-time employee wants a $700 course. Is it within budget? | No, the part-time budget is $600 | learning_and_development |
| 17 | MULTI | Which office has the demo lab and what are its hours? | Austin, 7:00 AM to 7:00 PM weekdays | offices_and_facilities |
| 18 | MULTI | I was rated 2 in December. Do I get a bonus and a promotion? | Neither (rated 1 or 2 is not bonus eligible; promotion needs 4+) | compensation_and_performance |
| 19 | UNANSWERABLE | What is the company's stock option vesting schedule? | Not in the documents | none |
| 20 | UNANSWERABLE | Does Acme allow pets in the office? | Not in the documents | none |
| 21 | UNANSWERABLE | What is the CEO's name? | Not in the documents | none |
