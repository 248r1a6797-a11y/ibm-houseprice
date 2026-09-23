# Ames Housing Dataset — Comprehensive Data Dictionary

**Total Records**: 2,930  
**Total Features**: 82  
**Target Variable**: `SalePrice` (Continuous USD $\$$)  

This data dictionary provides precise metadata, statistical profiles, roles, and business interpretations for every variable in the dataset based on the authoritative documentation by Dean De Cock.

| Column Name | Type | Missing (%) | Unique | Example Values | Modeling Role | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `Order` | int64 | 0.0% (0) | 2930 | 1, 2, 3 | Identifier (Excluded) | Observation index order |
| `PID` | int64 | 0.0% (0) | 2930 | 526301100, 526350040, ... | Identifier (Excluded) | Parcel identification number (tax assessor ID) |
| `MSSubClass` | int64 | 0.0% (0) | 16 | 20, 60, 120 | Numerical Feature | Identifies the type of dwelling involved in the sale |
| `MSZoning` | str | 0.0% (0) | 7 | RL, RH, FV | Numerical Feature | Identifies the general zoning classification of the sale |
| `LotFrontage` | float64 | 16.7% (490) | 128 | 141.0, 80.0, 81.0 | Numerical Feature | Linear feet of street connected to property |
| `LotArea` | int64 | 0.0% (0) | 1960 | 31770, 11622, 14267 | Numerical Feature | Lot size in square feet |
| `Street` | str | 0.0% (0) | 2 | Pave, Grvl | Numerical Feature | Type of road access to property |
| `Alley` | str | 93.2% (2732) | 2 | Pave, Grvl | Numerical Feature | Type of alley access to property |
| `LotShape` | str | 0.0% (0) | 4 | IR1, Reg, IR2 | Numerical Feature | General shape of property (Reg, IR1, IR2, IR3) |
| `LandContour` | str | 0.0% (0) | 4 | Lvl, HLS, Bnk | Numerical Feature | Flatness of the property |
| `Utilities` | str | 0.0% (0) | 3 | AllPub, NoSewr, NoSeWa | Numerical Feature | Type of utilities available |
| `LotConfig` | str | 0.0% (0) | 5 | Corner, Inside, CulDSac | Numerical Feature | Lot configuration (Corner, Inside, CulDSac) |
| `LandSlope` | str | 0.0% (0) | 3 | Gtl, Mod, Sev | Numerical Feature | Slope of property |
| `Neighborhood` | str | 0.0% (0) | 28 | NAmes, Gilbert, StoneBr | Numerical Feature | Physical locations within Ames city limits |
| `Condition1` | str | 0.0% (0) | 9 | Norm, Feedr, PosN | Numerical Feature | Proximity to various conditions (arterial, railroad) |
| `Condition2` | str | 0.0% (0) | 8 | Norm, Feedr, PosA | Numerical Feature | Proximity to secondary conditions |
| `BldgType` | str | 0.0% (0) | 5 | 1Fam, TwnhsE, Twnhs | Numerical Feature | Type of dwelling (1Fam, 2FmCon, Duplex, Twnhs) |
| `HouseStyle` | str | 0.0% (0) | 8 | 1Story, 2Story, 1.5Fin | Numerical Feature | Style of dwelling (1Story, 2Story, etc.) |
| `OverallQual` | int64 | 0.0% (0) | 10 | 6, 5, 7 | Numerical Feature | Rates the overall material and finish of the house (1-10) |
| `OverallCond` | int64 | 0.0% (0) | 9 | 5, 6, 7 | Numerical Feature | Rates the overall condition of the house (1-10) |
| `YearBuilt` | int64 | 0.0% (0) | 118 | 1960, 1961, 1958 | Temporal / Feature Engineering | Original construction date |
| `YearRemod/Add` | int64 | 0.0% (0) | 61 | 1960, 1961, 1958 | Temporal / Feature Engineering | Remodel date (same as construction date if no remodeling) |
| `RoofStyle` | str | 0.0% (0) | 6 | Hip, Gable, Mansard | Numerical Feature | Type of roof (Gable, Hip, Gambrel, Mansard, Flat) |
| `RoofMatl` | str | 0.0% (0) | 8 | CompShg, WdShake, Tar&Grv | Numerical Feature | Roof material (CompShg, Tar&Grv, WdShake, etc.) |
| `Exterior1st` | str | 0.0% (0) | 16 | BrkFace, VinylSd, Wd Sdng | Numerical Feature | Exterior covering on house |
| `Exterior2nd` | str | 0.0% (0) | 17 | Plywood, VinylSd, Wd Sdng | Numerical Feature | Exterior covering on house (if more than one material) |
| `MasVnrType` | str | 60.6% (1775) | 4 | Stone, BrkFace, BrkCmn | Numerical Feature | Masonry veneer type |
| `MasVnrArea` | float64 | 0.8% (23) | 445 | 112.0, 0.0, 108.0 | Numerical Feature | Masonry veneer area in square feet |
| `ExterQual` | str | 0.0% (0) | 4 | TA, Gd, Ex | Numerical Feature | Evaluates the quality of the material on the exterior |
| `ExterCond` | str | 0.0% (0) | 5 | TA, Gd, Fa | Numerical Feature | Evaluates the present condition of the material on the exterior |
| `Foundation` | str | 0.0% (0) | 6 | CBlock, PConc, Wood | Numerical Feature | Type of foundation (PConc, CBlock, BrkTil, etc.) |
| `BsmtQual` | str | 2.7% (80) | 5 | TA, Gd, Ex | Numerical Feature | Evaluates the height of the basement |
| `BsmtCond` | str | 2.7% (80) | 5 | Gd, TA, Po | Numerical Feature | Evaluates the general condition of the basement |
| `BsmtExposure` | str | 2.8% (83) | 4 | Gd, No, Mn | Numerical Feature | Refers to walkout or garden level basement walls |
| `BsmtFinType1` | str | 2.7% (80) | 6 | BLQ, Rec, ALQ | Numerical Feature | Rating of basement finished area |
| `BsmtFinSF1` | float64 | 0.0% (1) | 995 | 639.0, 468.0, 923.0 | Numerical Feature | Type 1 finished square feet |
| `BsmtFinType2` | str | 2.8% (81) | 6 | Unf, LwQ, BLQ | Numerical Feature | Rating of basement finished area (if multiple types) |
| `BsmtFinSF2` | float64 | 0.0% (1) | 274 | 0.0, 144.0, 1120.0 | Numerical Feature | Type 2 finished square feet |
| `BsmtUnfSF` | float64 | 0.0% (1) | 1137 | 441.0, 270.0, 406.0 | Numerical Feature | Unfinished square feet of basement area |
| `TotalBsmtSF` | float64 | 0.0% (1) | 1058 | 1080.0, 882.0, 1329.0 | Numerical Feature | Total square feet of basement area |
| `Heating` | str | 0.0% (0) | 6 | GasA, GasW, Grav | Numerical Feature | Type of heating |
| `HeatingQC` | str | 0.0% (0) | 5 | Fa, TA, Ex | Numerical Feature | Heating quality and condition |
| `CentralAir` | str | 0.0% (0) | 2 | Y, N | Numerical Feature | Central air conditioning (Y/N) |
| `Electrical` | str | 0.0% (1) | 5 | SBrkr, FuseA, FuseF | Numerical Feature | Electrical system |
| `1stFlrSF` | int64 | 0.0% (0) | 1083 | 1656, 896, 1329 | Numerical Feature | First Floor square feet |
| `2ndFlrSF` | int64 | 0.0% (0) | 635 | 0, 701, 678 | Numerical Feature | Second floor square feet |
| `LowQualFinSF` | int64 | 0.0% (0) | 36 | 0, 390, 362 | Numerical Feature | Low quality finished square feet (all floors) |
| `GrLivArea` | int64 | 0.0% (0) | 1292 | 1656, 896, 1329 | Numerical Feature | Above grade (ground) living area square feet |
| `BsmtFullBath` | float64 | 0.1% (2) | 4 | 1.0, 0.0, 2.0 | Numerical Feature | Basement full bathrooms |
| `BsmtHalfBath` | float64 | 0.1% (2) | 3 | 0.0, 1.0, 2.0 | Numerical Feature | Basement half bathrooms |
| `FullBath` | int64 | 0.0% (0) | 5 | 1, 2, 3 | Numerical Feature | Full bathrooms above grade |
| `HalfBath` | int64 | 0.0% (0) | 3 | 0, 1, 2 | Numerical Feature | Half baths above grade |
| `BedroomAbvGr` | int64 | 0.0% (0) | 8 | 3, 2, 1 | Numerical Feature | Bedrooms above grade (does NOT include basement bedrooms) |
| `KitchenAbvGr` | int64 | 0.0% (0) | 4 | 1, 2, 3 | Numerical Feature | Kitchens above grade |
| `KitchenQual` | str | 0.0% (0) | 5 | TA, Gd, Ex | Numerical Feature | Kitchen quality |
| `TotRmsAbvGrd` | int64 | 0.0% (0) | 14 | 7, 5, 6 | Numerical Feature | Total rooms above grade (does not include bathrooms) |
| `Functional` | str | 0.0% (0) | 8 | Typ, Mod, Min1 | Numerical Feature | Home functionality (Typ, Min, Maj, etc.) |
| `Fireplaces` | int64 | 0.0% (0) | 5 | 2, 0, 1 | Numerical Feature | Number of fireplaces |
| `FireplaceQu` | str | 48.5% (1422) | 5 | Gd, TA, Po | Numerical Feature | Fireplace quality |
| `GarageType` | str | 5.4% (157) | 6 | Attchd, BuiltIn, Basment | Numerical Feature | Garage location |
| `GarageYrBlt` | float64 | 5.4% (159) | 103 | 1960.0, 1961.0, 1958.0 | Temporal / Feature Engineering | Year garage was built |
| `GarageFinish` | str | 5.4% (159) | 3 | Fin, Unf, RFn | Numerical Feature | Interior finish of the garage |
| `GarageCars` | float64 | 0.0% (1) | 6 | 2.0, 1.0, 3.0 | Numerical Feature | Size of garage in car capacity |
| `GarageArea` | float64 | 0.0% (1) | 603 | 528.0, 730.0, 312.0 | Numerical Feature | Size of garage in square feet |
| `GarageQual` | str | 5.4% (159) | 5 | TA, Fa, Gd | Numerical Feature | Garage quality |
| `GarageCond` | str | 5.4% (159) | 5 | TA, Fa, Gd | Numerical Feature | Garage condition |
| `PavedDrive` | str | 0.0% (0) | 3 | P, Y, N | Numerical Feature | Paved driveway (Y, P, N) |
| `WoodDeckSF` | int64 | 0.0% (0) | 380 | 210, 140, 393 | Numerical Feature | Wood deck area in square feet |
| `OpenPorchSF` | int64 | 0.0% (0) | 252 | 62, 0, 36 | Numerical Feature | Open porch area in square feet |
| `EnclosedPorch` | int64 | 0.0% (0) | 183 | 0, 170, 184 | Numerical Feature | Enclosed porch area in square feet |
| `3SsnPorch` | int64 | 0.0% (0) | 31 | 0, 238, 224 | Numerical Feature | Three season porch area in square feet |
| `ScreenPorch` | int64 | 0.0% (0) | 121 | 0, 120, 144 | Numerical Feature | Screen porch area in square feet |
| `PoolArea` | int64 | 0.0% (0) | 14 | 0, 144, 480 | Numerical Feature | Pool area in square feet |
| `PoolQC` | str | 99.6% (2917) | 4 | Ex, Gd, TA | Numerical Feature | Pool quality |
| `Fence` | str | 80.5% (2358) | 4 | MnPrv, GdPrv, GdWo | Numerical Feature | Fence quality |
| `MiscFeature` | str | 96.4% (2824) | 5 | Gar2, Shed, Othr | Numerical Feature | Miscellaneous feature not covered in other categories |
| `MiscVal` | int64 | 0.0% (0) | 38 | 0, 12500, 500 | Numerical Feature | Value of miscellaneous feature ($) |
| `MoSold` | int64 | 0.0% (0) | 12 | 5, 6, 4 | Temporal / Feature Engineering | Month Sold (MM) |
| `YrSold` | int64 | 0.0% (0) | 5 | 2010, 2009, 2008 | Temporal / Feature Engineering | Year Sold (YYYY) |
| `SaleType` | str | 0.0% (0) | 10 | WD , New, COD | Numerical Feature | Type of sale (WD, CWD, New, COD, etc.) |
| `SaleCondition` | str | 0.0% (0) | 6 | Normal, Partial, Family | Numerical Feature | Condition of sale (Normal, Abnorml, AdjLand, Alloca, Family, Partial) |
| `SalePrice` | int64 | 0.0% (0) | 1032 | 215000, 105000, 172000 | Target (SalePrice) | Sale price in USD ($) — Target Variable |