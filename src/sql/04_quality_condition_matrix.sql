-- ============================================================
-- HouseWise AI — Script 04: Overall Quality vs Condition Matrix
-- Business Question: How do physical construction quality and maintenance
-- condition interact to determine realized transaction prices?
-- ============================================================

SELECT
    OverallQual AS construction_quality,
    COUNT(*) AS sample_size,
    ROUND(AVG(SalePrice), 0) AS avg_sale_price,
    ROUND(MIN(SalePrice), 0) AS min_sale_price,
    ROUND(MAX(SalePrice), 0) AS max_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS avg_price_per_sqft,
    ROUND(AVG(OverallCond), 2) AS avg_condition_rating
FROM properties
GROUP BY OverallQual
ORDER BY OverallQual DESC;
