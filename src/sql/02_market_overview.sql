-- ============================================================
-- HouseWise AI — Script 02: Macro Market Overview & Capitalization
-- Business Question: What is the total transaction volume, aggregate
-- market capitalization, and central tendency of property sale prices?
-- ============================================================

SELECT
    COUNT(*) AS total_properties_sold,
    ROUND(SUM(SalePrice) / 1000000.0, 2) AS total_market_volume_millions_usd,
    ROUND(AVG(SalePrice), 2) AS mean_sale_price_usd,
    MIN(SalePrice) AS min_sale_price_usd,
    MAX(SalePrice) AS max_sale_price_usd,
    ROUND(AVG(GrLivArea), 1) AS avg_above_ground_living_area_sqft,
    ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea), 2) AS avg_price_per_sqft,
    MIN(YrSold) AS earliest_sale_year,
    MAX(YrSold) AS latest_sale_year
FROM properties;
