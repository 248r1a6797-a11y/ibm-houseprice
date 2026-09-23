# HouseWise AI — SQL Analytics Engine Verification Report

## 1. Engine Specifications
- **Database**: SQLite 3 (`data/db/housewise.db`)
- **Primary Table**: `properties` (2,930 rows × 82 columns)
- **Indexes Created**: `Neighborhood`, `YrSold/MoSold`, `OverallQual`, `SalePrice`
- **Total Analytical Scripts**: 12 scripts in `src/sql/`

## 2. Query Execution Audit

| Script | Output Rows | Columns | Cross-Validation Status |
| :--- | :---: | :---: | :---: |
| `02_market_overview.sql` | 1 | 9 | PASS |
| `03_neighborhood_price_per_sqft.sql` | 28 | 7 | PASS |
| `04_quality_condition_matrix.sql` | 10 | 7 | PASS |
| `05_property_age_depreciation.sql` | 4 | 7 | PASS |
| `06_remodel_roi_impact.sql` | 5 | 5 | PASS |
| `07_basement_living_area_value.sql` | 4 | 6 | PASS |
| `08_garage_capacity_premiums.sql` | 5 | 6 | PASS |
| `09_bath_ratio_analysis.sql` | 5 | 5 | PASS |
| `10_neighborhood_volatility_tiers.sql` | 22 | 7 | PASS |
| `11_outlier_divergence_candidates.sql` | 30 | 10 | PASS |
| `12_temporal_sales_velocity.sql` | 12 | 6 | PASS |

## 3. Mathematical Ground Truth Cross-Validation
- **Total Property Transactions**: 2,930 (SQL) vs 2,930 (Pandas) — MATCH
- **Aggregate Market Volume**: $529.73M (SQL) vs $529.73M (Pandas) — MATCH
- **Mean Sale Price**: $180,796.06 (SQL) vs $180,796.06 (Pandas) — MATCH
- **Submarkets Covered**: 28 Neighborhoods — MATCH

All queries executed successfully with 0 errors and zero discrepancy against raw assessor records.
