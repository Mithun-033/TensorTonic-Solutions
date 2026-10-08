SELECT model_name, dataset, accuracy,
    RANK() OVER(PARTITION BY dataset ORDER BY accuracy DESC) as accuracy_rank,
    DENSE_RANK() OVER(PARTITION BY dataset ORDER BY accuracy DESC) as accuracy_dense_rank
FROM model_metrics
ORDER BY dataset ASC, accuracy DESC, model_name ASC;  