WITH daily_stats AS (
    SELECT order_date, SUM(amount) as daily_amounts, COUNT(*) as daily_counts
    FROM orders
    GROUP BY order_date
)

SELECT ROUND(AVG(daily_counts),2) as avg_daily_orders,
        ROUND(AVG(daily_amounts),2) as avg_daily_revenue,
        MAX(daily_counts) as busiest_day_orders
FROM daily_stats;