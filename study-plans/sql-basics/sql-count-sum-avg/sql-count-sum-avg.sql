SELECT category, 
    COUNT(*) as total_sales, 
    SUM(amount) as total_revenue,
    ROUND(AVG(discount),2) as avg_discount
FROM sales
GROUP BY category
ORDER BY total_revenue DESC, category ASC;