-- ============================================================
-- HouseWise AI — Script 06: Renovation & Remodel Valuation Premium
-- Business Question: What is the price and price/sqft premium of remodeled
-- homes compared to unremodeled homes within older vintage cohorts?
-- ============================================================

WITH RemodelStatus AS (
    SELECT
        PID,
        SalePrice,
        GrLivArea,
        YearBuilt,
        YearRemodAdd,
        CASE WHEN YearRemodAdd > YearBuilt THEN 1 ELSE 0 END AS is_remodeled,
        CASE
            WHEN YearBuilt >= 1980 THEN 'Post-1980'
            WHEN YearBuilt >= 1950 THEN '1950-1979'
            ELSE 'Pre-1950'
        END AS age_era
    FROM properties
)
SELECT
    age_era,
    CASE WHEN is_remodeled = 1 THEN 'Remodeled' ELSE 'Original Condition' END AS remodel_status,
    COUNT(*) AS total_sales,
    ROUND(AVG(SalePrice), 0) AS mean_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS mean_price_per_sqft
FROM RemodelStatus
GROUP BY age_era, is_remodeled
ORDER BY age_era ASC, is_remodeled DESC;
