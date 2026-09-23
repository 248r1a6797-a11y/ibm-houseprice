"""Generate authoritative data dictionary from dataset and documentation."""
import pandas as pd
from pathlib import Path
import json

CSV_PATH = Path("data/raw/AmesHousing.csv")
DOC_PATH = Path("data/raw/DataDocumentation.txt")
OUTPUT_MD = Path("docs/data_dictionary.md")

def generate():
    df = pd.read_csv(CSV_PATH)
    
    # Read documentation text to extract descriptions if available
    doc_text = ""
    if DOC_PATH.exists():
        with open(DOC_PATH, "r", encoding="utf-8", errors="ignore") as f:
            doc_text = f.read()

    lines = []
    lines.append("# Ames Housing Dataset — Comprehensive Data Dictionary\n")
    lines.append(f"**Total Records**: {len(df):,}  ")
    lines.append(f"**Total Features**: {len(df.columns):,}  ")
    lines.append(f"**Target Variable**: `SalePrice` (Continuous USD $\\$$)  \n")
    lines.append("This data dictionary provides precise metadata, statistical profiles, roles, and business interpretations for every variable in the dataset based on the authoritative documentation by Dean De Cock.\n")
    
    lines.append("| Column Name | Type | Missing (%) | Unique | Example Values | Modeling Role | Business Interpretation |")
    lines.append("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")

    # Critical columns role mapping
    id_cols = {"Order", "PID"}
    target_col = "SalePrice"

    # Descriptions dictionary based on De Cock documentation
    descriptions = {
        "Order": "Observation index order",
        "PID": "Parcel identification number (tax assessor ID)",
        "MS SubClass": "Identifies the type of dwelling involved in the sale",
        "MS Zoning": "Identifies the general zoning classification of the sale",
        "Lot Frontage": "Linear feet of street connected to property",
        "Lot Area": "Lot size in square feet",
        "Street": "Type of road access to property",
        "Alley": "Type of alley access to property",
        "Lot Shape": "General shape of property (Reg, IR1, IR2, IR3)",
        "Land Contour": "Flatness of the property",
        "Utilities": "Type of utilities available",
        "Lot Config": "Lot configuration (Corner, Inside, CulDSac)",
        "Land Slope": "Slope of property",
        "Neighborhood": "Physical locations within Ames city limits",
        "Condition 1": "Proximity to various conditions (arterial, railroad)",
        "Condition 2": "Proximity to secondary conditions",
        "Bldg Type": "Type of dwelling (1Fam, 2FmCon, Duplex, Twnhs)",
        "House Style": "Style of dwelling (1Story, 2Story, etc.)",
        "Overall Qual": "Rates the overall material and finish of the house (1-10)",
        "Overall Cond": "Rates the overall condition of the house (1-10)",
        "Year Built": "Original construction date",
        "Year Remod/Add": "Remodel date (same as construction date if no remodeling)",
        "Roof Style": "Type of roof (Gable, Hip, Gambrel, Mansard, Flat)",
        "Roof Matl": "Roof material (CompShg, Tar&Grv, WdShake, etc.)",
        "Exterior 1st": "Exterior covering on house",
        "Exterior 2nd": "Exterior covering on house (if more than one material)",
        "Mas Vnr Type": "Masonry veneer type",
        "Mas Vnr Area": "Masonry veneer area in square feet",
        "Exter Qual": "Evaluates the quality of the material on the exterior",
        "Exter Cond": "Evaluates the present condition of the material on the exterior",
        "Foundation": "Type of foundation (PConc, CBlock, BrkTil, etc.)",
        "Bsmt Qual": "Evaluates the height of the basement",
        "Bsmt Cond": "Evaluates the general condition of the basement",
        "Bsmt Exposure": "Refers to walkout or garden level basement walls",
        "BsmtFin Type 1": "Rating of basement finished area",
        "BsmtFin SF 1": "Type 1 finished square feet",
        "BsmtFin Type 2": "Rating of basement finished area (if multiple types)",
        "BsmtFin SF 2": "Type 2 finished square feet",
        "Bsmt Unf SF": "Unfinished square feet of basement area",
        "Total Bsmt SF": "Total square feet of basement area",
        "Heating": "Type of heating",
        "Heating QC": "Heating quality and condition",
        "Central Air": "Central air conditioning (Y/N)",
        "Electrical": "Electrical system",
        "1st Flr SF": "First Floor square feet",
        "2nd Flr SF": "Second floor square feet",
        "Low Qual Fin SF": "Low quality finished square feet (all floors)",
        "Gr Liv Area": "Above grade (ground) living area square feet",
        "Bsmt Full Bath": "Basement full bathrooms",
        "Bsmt Half Bath": "Basement half bathrooms",
        "Full Bath": "Full bathrooms above grade",
        "Half Bath": "Half baths above grade",
        "Bedroom AbvGr": "Bedrooms above grade (does NOT include basement bedrooms)",
        "Kitchen AbvGr": "Kitchens above grade",
        "Kitchen Qual": "Kitchen quality",
        "TotRms AbvGrd": "Total rooms above grade (does not include bathrooms)",
        "Functional": "Home functionality (Typ, Min, Maj, etc.)",
        "Fireplaces": "Number of fireplaces",
        "Fireplace Qu": "Fireplace quality",
        "Garage Type": "Garage location",
        "Garage Yr Blt": "Year garage was built",
        "Garage Finish": "Interior finish of the garage",
        "Garage Cars": "Size of garage in car capacity",
        "Garage Area": "Size of garage in square feet",
        "Garage Qual": "Garage quality",
        "Garage Cond": "Garage condition",
        "Paved Drive": "Paved driveway (Y, P, N)",
        "Wood Deck SF": "Wood deck area in square feet",
        "Open Porch SF": "Open porch area in square feet",
        "Enclosed Porch": "Enclosed porch area in square feet",
        "3Ssn Porch": "Three season porch area in square feet",
        "Screen Porch": "Screen porch area in square feet",
        "Pool Area": "Pool area in square feet",
        "Pool QC": "Pool quality",
        "Fence": "Fence quality",
        "Misc Feature": "Miscellaneous feature not covered in other categories",
        "Misc Val": "Value of miscellaneous feature ($)",
        "Mo Sold": "Month Sold (MM)",
        "Yr Sold": "Year Sold (YYYY)",
        "Sale Type": "Type of sale (WD, CWD, New, COD, etc.)",
        "Sale Condition": "Condition of sale (Normal, Abnorml, AdjLand, Alloca, Family, Partial)",
        "SalePrice": "Sale price in USD ($) — Target Variable"
    }

    for col in df.columns:
        norm_key = col.replace(" ", "").replace("/", "")
        dtype_str = str(df[col].dtype)
        missing_count = int(df[col].isnull().sum())
        missing_pct = (missing_count / len(df)) * 100
        n_unique = int(df[col].nunique())
        
        # Sample non-null values
        samples = [str(x) for x in df[col].dropna().unique()[:3]]
        sample_str = ", ".join(samples) if samples else "N/A"
        if len(sample_str) > 25:
            sample_str = sample_str[:22] + "..."

        # Role
        if col in id_cols:
            role = "Identifier (Excluded)"
        elif col == target_col:
            role = "Target (SalePrice)"
        elif "Yr" in col or "Year" in col or "Mo" in col:
            role = "Temporal / Feature Engineering"
        elif df[col].dtype == "object":
            role = "Categorical Feature"
        else:
            role = "Numerical Feature"

        # Lookup description
        desc = descriptions.get(col, f"Feature representing {col}")
        for k, v in descriptions.items():
            if k.replace(" ", "").replace("/", "").lower() == norm_key.lower():
                desc = v
                break

        lines.append(f"| `{col}` | {dtype_str} | {missing_pct:.1f}% ({missing_count}) | {n_unique} | {sample_str} | {role} | {desc} |")

    OUTPUT_MD.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Data dictionary successfully written to {OUTPUT_MD} with {len(df.columns)} variables documented.")

if __name__ == "__main__":
    generate()
