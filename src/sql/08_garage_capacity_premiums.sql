-- ============================================================
-- HouseWise AI — Script 08: Garage Car Capacity & Pricing Elasticity
-- Business Question: What is the marginal pricing premium of additional
-- garage car stalls across Ames properties?
-- ============================================================

SELECT
    CAST(COALESCE(GarageCars, 0) AS INTEGER) AS garage_car_stalls,
    COUNT(*) AS total_properties,
    ROUND(AVG(COALESCE(GarageArea, 0)), 1) AS avg_garage_area_sqft,
    ROUND(AVG(SalePrice), 0) AS mean_sale_price,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS mean_price_per_sqft,
    ROUND(
        AVG(SalePrice) - LAG(AVG(SalePrice), 1, AVG(SalePrice)) OVER (ORDER BY CAST(COALESCE(GarageCars, 0) AS INTEGER)),
        0
    ) AS marginal_step_premium_usd
FROM properties
WHERE CAST(COALESCE(GarageCars, 0) AS INTEGER) <= 4
GROUP BY CAST(COALESCE(GarageCars, 0) AS INTEGER)
ORDER BY garage_car_stalls ASC;
