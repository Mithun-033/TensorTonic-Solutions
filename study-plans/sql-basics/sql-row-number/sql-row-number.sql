SELECT username, segment, engagement_score,
    ROW_NUMBER() OVER (
        PARTITION BY segment
        ORDER BY engagement_score DESC, username ASC
    ) as activity_rank
FROM user_activity 
ORDER BY segment ASC, activity_rank ASC;