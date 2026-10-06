SELECT customer, COUNT(*) as total_orders, SUM(amount) as total_spent 
FROM orders 
GROUP BY customer 
HAVING total_orders >= 2
ORDER BY total_spent DESC;