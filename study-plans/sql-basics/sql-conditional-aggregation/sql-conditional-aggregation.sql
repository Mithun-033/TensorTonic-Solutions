SELECT department, 
        COUNT(*) as total_tickets,
        COUNT(CASE WHEN status = 'open' THEN 1 END) as open_count,
        COUNT(CASE WHEN status = 'in_progress' THEN 1 END) as in_progress_count,
        COUNT(CASE WHEN status = 'closed' THEN 1 END) as closed_count
FROM tickets
GROUP BY department
ORDER BY total_tickets DESC, department ASC;