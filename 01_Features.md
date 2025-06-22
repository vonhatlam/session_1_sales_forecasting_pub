# Part 1: The Foundation - Preparing Time-Series Data for AI

## Goal
Explain that financial data is often time-dependent, and we need to engineer specific features to help our models understand it.

---

## Feature Type 1: Cyclical Features

### What You Need to Know
Many financial datasets have predictable cycles that repeat over time:
- **Daily cycles**: Stock trading volume patterns
- **Weekly cycles**: Retail banking transactions 
- **Monthly cycles**: Credit card spending around payday
- **Quarterly cycles**: Business earnings reports
- **Yearly cycles**: Seasonal spending patterns

### The Problem with Regular Dates
A standard calendar date like "2025-06-17" is just a number to a machine. The AI doesn't naturally understand:
- That one day follows another in a cycle
- That December and January are actually close together in a yearly cycle
- That Monday and Sunday are consecutive days of the week

### How Cyclical Features Work
We transform dates into sine and cosine waves that represent cycles. This mathematical approach helps the model understand that:
- December (month 12) is close to January (month 1) in a yearly cycle
- Sunday (day 7) connects to Monday (day 1) in a weekly cycle
- 11:59 PM is very close to 12:01 AM in a daily cycle

### Simple Analogy
Think of a clock face. The numbers 1 and 12 are far apart numerically (11 numbers between them), but on the actual clock, they sit right next to each other. Cyclical features present time to the AI in this same "circular" way.

### Financial Example
**Predicting cash demand for an ATM network**
- ATMs need more cash on Fridays (people getting money for weekends)
- Different demand patterns on weekdays vs weekends
- Monthly cycles around payday dates
- Cyclical features help the model learn these repeating patterns

---

## Feature Type 2: Lag Features

### What You Need to Know
The past often holds the best clues to predict the future. A lag feature is simply a past value of the same variable we're trying to predict.

### How It Works
You create new columns of data where each row contains values from previous time periods:
- **Lag 1**: Yesterday's value
- **Lag 7**: Same day last week's value  
- **Lag 30**: Same day last month's value

This gives the model explicit "memory" of what happened recently.

### Why This Matters
Instead of the model trying to figure out patterns on its own, lag features directly show:
- Recent trends and momentum
- How current values relate to past performance
- Seasonal patterns from equivalent time periods

### Financial Example
**Predicting credit card default risk**
- **Lag 1**: Payment status last month (immediate history)
- **Lag 3**: Average balance over last three months (recent trend)
- **Lag 12**: Payment behavior same month last year (seasonal pattern)

### Advanced Lag Features: Rolling Averages
Beyond simple lag values, rolling averages provide smoothed insights:

- **7-day average**: Average of the last 7 days (reduces daily noise)
- **30-day average**: Average of the last 30 days (shows monthly trends)
- **90-day average**: Average of the last 90 days (reveals quarterly patterns)

### Why Rolling Averages Matter
Rolling averages act like a "smoothing filter" that:
- **Reduces noise**: Daily fluctuations don't distort the bigger picture
- **Reveals trends**: Shows the underlying direction of movement
- **Provides stability**: Less sensitive to one-time events or outliers

### Business Analogy
Think of rolling averages like looking at your business performance through different lenses:
- **Daily view**: Like checking your bank balance every day (lots of ups and downs)
- **Weekly view**: Like reviewing your weekly sales report (clearer patterns emerge)
- **Monthly view**: Like analyzing your monthly P&L statement (major trends become obvious)

### Financial Example
**Predicting stock price movements**
- **7-day average**: Shows short-term momentum (recent investor sentiment)
- **30-day average**: Reveals medium-term trends (market direction)
- **200-day average**: Indicates long-term market health (bull vs bear market)

When the 7-day average crosses above the 30-day average, it often signals upward momentum - a classic trading signal used by professional investors.

These lag features become powerful predictors because past payment behavior strongly indicates future risk.