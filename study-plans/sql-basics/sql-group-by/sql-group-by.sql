SELECT customer, COUNT(customer) as total_orders, SUM(amount) as total_spent
FROM orders 
GROUP BY customer
ORDER BY total_spent DESC;