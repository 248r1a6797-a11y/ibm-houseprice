-- ============================================================
-- HouseWise AI — Script 12: Temporal Sales Velocity & Seasonality
-- Business Question: How do transaction volume and price seasonality
-- behave across calendar months and sale years in the Ames market?
-- ============================================================

SELECT
    MoSold AS calendar_month,
    CASE MoSold
        WHEN 1 THEN 'January'
        WHEN 2 THEN 'February'
        WHEN 3 THEN 'March'
        WHEN 4 THEN 'April'
        WHEN 5 THEN 'May'
        WHEN 6 THEN 'June'
        WHEN 7 THEN 'July'
        WHEN 8 THEN 'August'
        WHEN 9 THEN 'September'
        WHEN 10 THEN 'October'
        WHEN 11 THEN 'November'
        WHEN 12 THEN 'December'
    END AS month_name,
    COUNT(*) AS total_transactions,
    ROUND(AVG(SalePrice), 0) AS mean_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS mean_price_per_sqft,
    ROUND(SUM(SalePrice) / 1000000.0, 2) AS aggregate_volume_millions
FROM properties
GROUP BY MoSold
ORDER BY MoSold ASC;
