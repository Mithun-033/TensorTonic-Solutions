SELECT s.segment_name, m.metric_name
FROM segments s
CROSS JOIN metrics m
ORDER BY s.segment_name ASC, m.metric_name ASC;
