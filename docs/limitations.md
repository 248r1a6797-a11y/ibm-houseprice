# System & Model Limitations Documentation

## 1. Dataset Scope & Geographic Boundaries
- **Municipal Isolation**: The underlying dataset represents residential sales within the municipal boundaries of Ames, Iowa. The pricing dynamics reflect a Midwestern university community and do NOT generalize to major metropolitan centers (e.g. New York, Chicago, San Francisco, Mumbai) without local re-training.
- **Historical Temporal Constraint**: The sales occurred between January 2006 and July 2010. Consequently, the price levels reflect the macroeconomic environment of the U.S. subprime mortgage crisis and do not reflect post-2020 inflation, modern building material costs, or contemporary interest rate surges.

## 2. Statistical & Machine Learning Limitations
- **Prediction vs. Causality**: High feature attribution (e.g. SHAP values for living area or quality) indicates statistical association within historical assessor records. It does NOT prove that modifying a specific feature will causally increase the sale price by that exact dollar amount.
- **Heteroscedasticity on High-Value Custom Estates**: Residual variance expands significantly for properties valued above $350,000 or exceeding 4,000 sq ft. For these properties, the model's point estimate must be reviewed by a certified professional appraiser.
- **Non-Normal Sale Conditions**: In cases of foreclosures, estate liquidations, or family property transfers, the historical transaction price was disconnected from physical property characteristics, resulting in higher residual error.

## 3. Responsible AI & Governance Limitations
- **Prohibited Uses**: This system is built strictly for comparative market analysis, analytical decision support, and educational demonstration. It must **NOT** be used as an automated decision engine for mortgage denial, loan underwriting, insurance pricing, or tenant background screening.
- **Omitted Macroeconomic Variables**: The model does not observe live interest rates, local property tax rate adjustments, or unemployment fluctuations.
