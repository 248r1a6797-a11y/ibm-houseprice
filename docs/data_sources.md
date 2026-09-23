# Data Sources & Provenance Documentation

## Dataset Overview
- **Dataset Name**: Ames Housing Dataset
- **Original Author / Creator**: Dean De Cock (Professor of Statistics, Truman State University)
- **Primary Publication**: Journal of Statistics Education (JSE), Volume 19, Number 3 (2011)
- **Article Title**: *Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project*
- **Authoritative Source URL**: `http://jse.amstat.org/v19n3/decock/`
- **Official Documentation**: `http://jse.amstat.org/v19n3/decock/DataDocumentation.txt`
- **Download Date**: September 23, 2026
- **License / Usage**: Open access academic and educational research data archive (American Statistical Association). Freely usable for educational, benchmarking, and machine learning research purposes.

## Dataset Specifications
- **Observations (Rows)**: 2,930
- **Total Variables (Columns)**: 82 (including target variable `SalePrice` and identifier variables `Order` and `PID`)
- **Target Variable**: `SalePrice` (Continuous USD $\$$ sale price)
- **Geographic Scope**: Ames, Iowa, USA (incorporating 28 distinct neighborhoods across the municipality)
- **Time Period**: Residential property sales transacted between January 2006 and July 2010
- **Nature of Records**: Real-estate tax assessor records maintained by the Ames Assessor's Office for computing assessed values and recording property transfer values.

## Data Limitations & Ethical Governance
1. **Temporal Horizon**: Covers 2006 to 2010. Captures the market dynamics preceding and during the 2007–2008 subprime mortgage financial crisis; prices do not reflect post-2010 inflation, macroeconomic interest shifts, or modern housing cost surges.
2. **Geographic Specificity**: Reflects the local demographic and microeconomic dynamics of Ames, Iowa (a college town hosting Iowa State University). Models trained on this dataset must not be presented as generalizable to metropolitan centers (e.g., New York, San Francisco, London, Mumbai).
3. **Absence of Macroeconomic Drivers**: The dataset lacks exogenous variables such as prevailing mortgage interest rates, regional unemployment, inflation rates, and local school district testing scores.
4. **Special Assessor Notes**: The author documentation highlights 5 specific observations with Gross Living Area $> 4000\text{ sq ft}$ that distort linear fits (three partial sales and two unusually large custom homes). Our data pipeline audits and flags these cases explicitly.
