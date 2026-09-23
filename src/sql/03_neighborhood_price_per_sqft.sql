-- ============================================================
-- HouseWise AI — Script 03: Neighborhood Price Per Sqft Ranking
-- Business Question: Which neighborhoods command the highest price per
-- square foot, and how does sales volume distribute across submarkets?
-- Technique: Aggregation + Common Table Expression + DENSE_RANK()
-- ============================================================

WITH NeighborhoodMetrics AS (
    SELECT
        Neighborhood,
        COUNT(*) AS transaction_count,
        ROUND(AVG(SalePrice), 0) AS avg_sale_price,
        ROUND(AVG(GrLivArea), 0) AS avg_living_area_sqft,
        ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS avg_price_per_sqft,
        ROUND(SUM(SalePrice) / 1000000.0, 2) AS total_volume_millions
    FROM properties
    WHERE GrLivArea > 0
    GROUP BY Neighborhood
)
SELECT
    DENSE_RANK() OVER (ORDER BY avg_price_per_sqft DESC) AS price_sqft_rank,
    Neighborhood,
    transaction_count,
    avg_sale_price,
    avg_living_area_sqft,
    avg_price_per_sqft,
    total_volume_millions
FROM NeighborhoodMetrics
ORDER BY price_sqft_rank ASC;
