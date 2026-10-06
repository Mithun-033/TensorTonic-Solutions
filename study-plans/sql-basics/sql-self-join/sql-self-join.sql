SELECT u.username, COALESCE(r.username, 'organic') as referrer_name 
FROM user_referrals u 
LEFT JOIN user_referrals r ON u.referred_by = r.id ORDER BY u.username ASC;