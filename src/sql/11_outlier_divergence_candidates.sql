-- ============================================================
-- HouseWise AI — Script 11: Micro-Market Valuation Divergence Candidates
-- Business Question: Which individual properties exhibit the largest
-- valuation divergence (potential bargains or overpriced risks) compared
-- to their specific neighborhood average price-per-square-foot?
-- Technique: AVG() OVER (PARTITION BY Neighborhood) Window Function
-- ============================================================

WITH MicroMarketBaseline AS (
    SELECT
        PID,
        Neighborhood,
        GrLivArea,
        OverallQual,
        YearBuilt,
        SalePrice,
        ROUND(CAST(SalePrice AS REAL) / GrLivArea, 2) AS property_ppsf,
        ROUND(AVG(CAST(SalePrice AS REAL) / GrLivArea) OVER (PARTITION BY Neighborhood), 2) AS neighborhood_avg_ppsf
    FROM properties
    WHERE GrLivArea > 0
),
DivergenceScored AS (
    SELECT
        PID,
        Neighborhood,
        GrLivArea,
        OverallQual,
        YearBuilt,
        SalePrice,
        property_ppsf,
        neighborhood_avg_ppsf,
        ROUND(((property_ppsf - neighborhood_avg_ppsf) / neighborhood_avg_ppsf) * 100.0, 2) AS ppsf_divergence_pct
    FROM MicroMarketBaseline
)
SELECT
    PID,
    Neighborhood,
    OverallQual,
    YearBuilt,
    GrLivArea,
    SalePrice,
    property_ppsf,
    neighborhood_avg_ppsf,
    ppsf_divergence_pct,
    CASE
        WHEN ppsf_divergence_pct <= -25.0 THEN 'Deep Value Candidate'
        WHEN ppsf_divergence_pct >= 35.0 THEN 'Premium / Overpriced Alert'
        ELSE 'Normal Range'
    END AS divergence_classification
FROM DivergenceScored
WHERE ABS(ppsf_divergence_pct) >= 25.0
ORDER BY ppsf_divergence_pct ASC
LIMIT 30;
