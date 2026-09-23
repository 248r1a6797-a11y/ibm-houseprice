-- ============================================================
-- HouseWise AI — Script 07: Basement Finished Area Valuation
-- Business Question: How does finished basement area contribute to total
-- property value relative to above-ground gross living area?
-- ============================================================

SELECT
    CASE
        WHEN TotalBsmtSF = 0 OR TotalBsmtSF IS NULL THEN 'No Basement'
        WHEN BsmtFinSF1 = 0 OR BsmtFinSF1 IS NULL THEN 'Unfinished Basement'
        WHEN (BsmtFinSF1 + IFNULL(BsmtFinSF2, 0)) >= 1000 THEN 'Large Finished Basement (>=1000 SF)'
        ELSE 'Moderate Finished Basement (<1000 SF)'
    END AS basement_category,
    COUNT(*) AS property_count,
    ROUND(AVG(TotalBsmtSF), 0) AS avg_bsmt_sqft,
    ROUND(AVG(GrLivArea), 0) AS avg_above_ground_sqft,
    ROUND(AVG(SalePrice), 0) AS avg_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS avg_price_per_above_ground_sqft
FROM properties
GROUP BY basement_category
ORDER BY avg_sale_price DESC;
