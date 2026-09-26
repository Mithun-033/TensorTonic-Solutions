SELECT customer_name, COUNT(DISTINCT product) AS unique_products FROM orders GROUP BY customer_name ORDER BY unique_products DESC, customer_name ASC;
