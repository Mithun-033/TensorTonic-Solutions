SELECT department, 
        COUNT(*) as total_tickets,
        SUM(CASE WHEN status = 'open' THEN 1 ELSE 0 END) as open_count,
        SUM(CASE WHEN status = 'in_progress' THEN 1 ELSE 0 END) as in_progress_count,
        SUM(CASE WHEN status = 'closed' THEN 1 ELSE 0 END) as closed_count
FROM tickets
GROUP BY department
ORDER BY total_tickets DESC, department ASC;