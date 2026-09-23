-- ============================================================
-- HouseWise AI — Script 05: Construction Era & Age Depreciation
-- Business Question: How does property vintage impact pricing and
-- what is the average annual depreciation across historical cohorts?
-- ============================================================

WITH VintageCohorts AS (
    SELECT
        PID,
        SalePrice,
        GrLivArea,
        YearBuilt,
        YrSold,
        (YrSold - YearBuilt) AS age_at_sale,
        CASE
            WHEN YearBuilt >= 2000 THEN 'Modern (2000-2010)'
            WHEN YearBuilt >= 1980 THEN 'Late 20th Century (1980-1999)'
            WHEN YearBuilt >= 1950 THEN 'Post-War Boom (1950-1979)'
            ELSE 'Historic / Pre-War (< 1950)'
        END AS construction_era
    FROM properties
)
SELECT
    construction_era,
    COUNT(*) AS property_count,
    ROUND(AVG(age_at_sale), 1) AS avg_age_at_sale_years,
    ROUND(AVG(SalePrice), 0) AS avg_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS avg_price_per_sqft,
    ROUND(MIN(SalePrice), 0) AS min_price,
    ROUND(MAX(SalePrice), 0) AS max_price
FROM VintageCohorts
GROUP BY construction_era
ORDER BY avg_age_at_sale_years ASC;
