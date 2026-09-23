-- ============================================================
-- HouseWise AI — Script 10: Neighborhood Volatility & Quartile Tiers
-- Business Question: Which neighborhoods demonstrate highest price
-- dispersion and how do submarkets segment into investment tiers?
-- Technique: NTILE(4) Window Function + Price Range
-- ============================================================

WITH NeighborhoodStats AS (
    SELECT
        Neighborhood,
        COUNT(*) AS sales_volume,
        ROUND(AVG(SalePrice), 0) AS avg_price,
        ROUND(MIN(SalePrice), 0) AS min_price,
        ROUND(MAX(SalePrice), 0) AS max_price,
        ROUND(MAX(SalePrice) - MIN(SalePrice), 0) AS price_spread
    FROM properties
    GROUP BY Neighborhood
    HAVING COUNT(*) >= 25
)
SELECT
    Neighborhood,
    sales_volume,
    avg_price,
    min_price,
    max_price,
    price_spread,
    NTILE(4) OVER (ORDER BY avg_price DESC) AS price_quartile_tier
FROM NeighborhoodStats
ORDER BY price_quartile_tier ASC, avg_price DESC;
