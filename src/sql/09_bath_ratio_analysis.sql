-- ============================================================
-- HouseWise AI — Script 09: Bathroom-to-Bedroom Ratio Valuation Impact
-- Business Question: How does bathroom abundance relative to bedroom
-- count impact property market value and liquidity?
-- ============================================================

WITH BathMetrics AS (
    SELECT
        PID,
        SalePrice,
        BedroomAbvGr,
        (FullBath + 0.5 * HalfBath + COALESCE(BsmtFullBath, 0) + 0.5 * COALESCE(BsmtHalfBath, 0)) AS total_bathrooms,
        GrLivArea
    FROM properties
    WHERE BedroomAbvGr > 0
)
SELECT
    BedroomAbvGr AS bedroom_count,
    ROUND(AVG(total_bathrooms), 1) AS avg_bathrooms,
    COUNT(*) AS transaction_volume,
    ROUND(AVG(SalePrice), 0) AS avg_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / total_bathrooms), 0) AS avg_price_per_bathroom
FROM BathMetrics
WHERE BedroomAbvGr BETWEEN 1 AND 5
GROUP BY BedroomAbvGr
ORDER BY BedroomAbvGr ASC;
