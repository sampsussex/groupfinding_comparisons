import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple
from astropy.io import fits
from astropy.table import Table
from typing import Optional

# ------------------------------------------------------------------
# Generic readers
# ------------------------------------------------------------------

def read_table(file: str) -> pd.DataFrame:
    if file.endswith(".fits"):
        with fits.open(file, memmap=False) as hdul:
            return Table(hdul[1].data).to_pandas()
    elif file.endswith(".parquet"):
        return pd.read_parquet(file)
    else:
        return pd.read_csv(file, sep=r"\s+", comment="#", names=["galaxy_id", "group_id"])


# ------------------------------------------------------------------
# Standardisers
# ------------------------------------------------------------------

def standardise_membership(
    df: pd.DataFrame,
    galaxy_col_candidates=("galaxy_id", "uberID", "ids", "id_galaxy_sky", "CATAID", "UberID"),
    group_col_candidates=("group_id", "GroupID", "group_id_nessie", "group_id_sussex", "id_fof", "group_id_finder"),
    group_name="group_id",
) -> pd.DataFrame:
    df = df.copy()

    galaxy_col = next((c for c in galaxy_col_candidates if c in df.columns), None)
    group_col = next((c for c in group_col_candidates if c in df.columns), None)

    if galaxy_col is None:
        raise ValueError(f"Could not identify galaxy ID column. Columns are: {list(df.columns)}")
    if group_col is None:
        raise ValueError(f"Could not identify group ID column. Columns are: {list(df.columns)}")

    out = df.rename(columns={galaxy_col: "galaxy_id", group_col: group_name})
    return out[["galaxy_id", group_name]].copy()


def standardise_group_properties(
    df: pd.DataFrame,
    group_col_candidates=("group_id", "GroupID", "id_fof", "ids"),
) -> pd.DataFrame:
    df = df.copy()
    group_col = next((c for c in group_col_candidates if c in df.columns), None)
    if group_col is None:
        raise ValueError(f"Could not identify group ID column in group table. Columns are: {list(df.columns)}")
    if group_col != "group_id":
        df = df.rename(columns={group_col: "group_id"})
    return df


# ------------------------------------------------------------------
# Specific loaders
# ------------------------------------------------------------------

def load_gama_data(file: str) -> pd.DataFrame:
    return read_table(file)


def load_sharks_data(file: str) -> pd.DataFrame:
    df = read_table(file).copy()

    rename_map = {}
    if "id_galaxy_sky" in df.columns:
        rename_map["id_galaxy_sky"] = "galaxy_id"
    elif "ids" in df.columns:
        rename_map["ids"] = "galaxy_id"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "galaxy_id" not in df.columns:
        raise ValueError("Could not find galaxy ID column in Sharks data.")

    return df


def load_sharks_groups(sharks_data: pd.DataFrame) -> pd.DataFrame:
    group_col = "id_fof"
    host_id_col = "id_group_sky"
    mass_col = "mass_virial_hosthalo"
    mag_col = "mag_abs_r_VST"
    stellar_mass_col = "stellar_mass"
    gals = sharks_data.copy()


    gals["is_bcg"] = False

    ungrouped = gals[group_col].eq(-1)
    gals.loc[ungrouped, "is_bcg"] = True

    valid_grouped = (
        gals[group_col].notna()
        & gals[group_col].ne(-1)
        & gals[mag_col].notna()
        & gals[stellar_mass_col].notna()
    )

    # Bcg centres....
    bcg_idx = gals.loc[valid_grouped].groupby(group_col)[mag_col].idxmin()
    gals.loc[bcg_idx, "is_bcg"] = True

    bcg_broadcast_cols = ["ra", "dec", "redshift_observed", mag_col, stellar_mass_col]

    for col in bcg_broadcast_cols:
        gals[f"{col}_bcg"] = gals[col]

    bcg_rows = gals.loc[
        gals["is_bcg"] & gals[group_col].ne(-1),
        [group_col] + bcg_broadcast_cols
    ].drop_duplicates(subset=[group_col])

    for col in bcg_broadcast_cols:
        mapper = bcg_rows.set_index(group_col)[col]
        gals.loc[gals[group_col].ne(-1), f"{col}_bcg"] = (
            gals.loc[gals[group_col].ne(-1), group_col].map(mapper)
        )

    gals["log_stellar_mass_bcg"] = np.log10(
        gals["stellar_mass_bcg"].where(gals["stellar_mass_bcg"] > 0)
    )

    gals["fof_halo_mass"] = np.nan

    grouped = gals[group_col].notna() & gals[group_col].ne(-1)
    fof_mass = (
        gals.loc[grouped, [group_col, host_id_col, mass_col]]
        .dropna()
        .drop_duplicates(subset=[group_col, host_id_col])
        .groupby(group_col, sort=False)[mass_col]
        .sum()
    )

    gals.loc[grouped, "fof_halo_mass"] = gals.loc[grouped, group_col].map(fof_mass)
    gals.loc[ungrouped, "fof_halo_mass"] = gals.loc[ungrouped, mass_col]
    gals["log_fof_halo_mass"] = np.log10(gals["fof_halo_mass"].where(gals["fof_halo_mass"] > 0))



    M_sun_r = 4.63
    gals["L"] = 10.0 ** (-0.4 * (gals[mag_col] - M_sun_r))

    # Only sum real groups; leave ungrouped as self-values
    gals["group_L"] = gals["L"]
    real_group_L = gals.loc[grouped].groupby(group_col)["L"].sum()
    gals.loc[grouped, "group_L"] = gals.loc[grouped, group_col].map(real_group_L)
    gals["log_group_L"] = np.log10(gals["group_L"].where(gals["group_L"] > 0))

    gals["n_group_fof"] = 1
    real_group_n = gals.loc[grouped].groupby(group_col).size()
    gals.loc[grouped, "n_group_fof"] = gals.loc[grouped, group_col].map(real_group_n).astype(int)

    gals["group_stellar_mass"] = gals["stellar_mass"]
    real_group_sm = gals.loc[grouped].groupby(group_col)["stellar_mass"].sum()
    gals.loc[grouped, "group_stellar_mass"] = gals.loc[grouped, group_col].map(real_group_sm)
    gals["log_group_stellar_mass"] = np.log10(
        gals["group_stellar_mass"].where(gals["group_stellar_mass"] > 0)
    )

    group_cols = [
    "ra_bcg",
    "dec_bcg",
    "redshift_observed_bcg",
    f"{mag_col}_bcg",
    "stellar_mass_bcg",
    "log_stellar_mass_bcg",
    "fof_halo_mass",
    "log_fof_halo_mass",
    "group_L",
    "log_group_L",
    "n_group_fof",
    "group_stellar_mass",
    "log_group_stellar_mass",]

    groups = (
        gals.loc[gals[group_col].ne(-1)]
        .groupby(group_col, sort=False)[group_cols]
        .first()
        .reset_index()
    )
    del gals
    
    # rename id_fof to group_id
    groups = groups.rename(columns={group_col: "group_id"})
    
    return groups

def load_membership_with_optional_gama_mapping(
    file: str,
    output_group_name: str,
    gama_id_mapping_file: Optional[str] = None,
) -> pd.DataFrame:
    df = read_table(file).copy()

    # If this is a GAMA-style membership table with CATAID rather than uberID,
    # convert to uberID using the supplied mapping.
    if "CATAID" in df.columns and gama_id_mapping_file is not None:
        mapping = read_table(gama_id_mapping_file)
        if "CATAID" not in mapping.columns or "uberID" not in mapping.columns:
            raise ValueError("GAMA mapping file must contain CATAID and uberID")
        df = df.merge(mapping[["CATAID", "uberID"]], on="CATAID", how="inner")

    df = standardise_membership(df, group_name=output_group_name)

    # Prefer uberID if present after mapping
    if "uberID" in df.columns:
        df = df.rename(columns={"uberID": "galaxy_id"})

    return df


def load_group_properties(file: str) -> pd.DataFrame:
    df = read_table(file)
    return standardise_group_properties(df)


# ------------------------------------------------------------------
# Bijective matching
# ------------------------------------------------------------------

def bijective_group_mapping(
    group_ids_1: List[int],
    group_ids_2: List[int],
    min_group_size: int = 2,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return matched IDs (from catalogue 1, catalogue 2) for bijective matches.
    """
    group_ids_1 = np.asarray(group_ids_1)
    group_ids_2 = np.asarray(group_ids_2)

    if len(group_ids_1) != len(group_ids_2):
        raise ValueError("Group ID arrays must have the same length")

    valid = (
        pd.notna(group_ids_1)
        & pd.notna(group_ids_2)
        & (group_ids_1 != -1)
        & (group_ids_2 != -1)
    )

    g1 = group_ids_1[valid]
    g2 = group_ids_2[valid]

    count_1 = Counter(g1)
    count_2 = Counter(g2)

    valid_groups_1 = [gid for gid, n in count_1.items() if n >= min_group_size]

    matched_1 = []
    matched_2 = []

    for gid1 in valid_groups_1:
        idx = np.where(g1 == gid1)[0]
        overlaps = g2[idx]

        if len(overlaps) == 0:
            continue

        overlap_counts = Counter(overlaps)
        n1 = count_1[gid1]

        best_gid2 = None
        best_score = -1.0
        best_q1 = 0.0
        best_q2 = 0.0

        for gid2, n_overlap in overlap_counts.items():
            n2 = count_2[gid2]
            q1 = n_overlap / n1
            q2 = n_overlap / n2
            score = q1 * q2

            if score > best_score:
                best_score = score
                best_gid2 = gid2
                best_q1 = q1
                best_q2 = q2

        if best_q1 > 0.5 and best_q2 > 0.5:
            matched_1.append(gid1)
            matched_2.append(best_gid2)

    return np.asarray(matched_1), np.asarray(matched_2)


# ------------------------------------------------------------------
# Catalogue builders
# ------------------------------------------------------------------

def load_group_set_gama(
    gama_file: str = "/Users/sp624AA/Downloads/gama3/groupfinding_gama4_processed.parquet",
    nessie_members_file: str = "/Users/sp624AA/Downloads/gama3/G3CGalv11.fits",
    sussex_members_file: str = "/Users/sp624AA/Downloads/groupfinder_results/gama/gal_groups_gama.parquet",
    nessie_groups_file: str = "/Users/sp624AA/Downloads/gama3/G3CFoFGroupv11.fits",
    sussex_groups_file: str = "/Users/sp624AA/Downloads/groupfinder_results/gama/gal_groups_gama_properties.parquet",
    gama_id_mapping_file: str = "/Users/sp624AA/Downloads/gama3/gkvGamaIIMatchesv01.fits",
):
    gama = load_gama_data(gama_file).copy()

    nessie_members = load_membership_with_optional_gama_mapping(
        nessie_members_file,
        output_group_name="group_id_nessie",
        gama_id_mapping_file=gama_id_mapping_file,
    )

    sussex_members = load_membership_with_optional_gama_mapping(
        sussex_members_file,
        output_group_name="group_id_sussex",
        gama_id_mapping_file=gama_id_mapping_file,
    )

    nessie_groups = load_group_properties(nessie_groups_file)
    sussex_groups = load_group_properties(sussex_groups_file)

    if "uberID" not in gama.columns:
        raise ValueError("GAMA galaxy catalogue must contain uberID")

    gama = gama.merge(
        nessie_members,
        left_on="uberID",
        right_on="galaxy_id",
        how="left",
    ).drop(columns=["galaxy_id"])

    gama = gama.merge(
        sussex_members,
        left_on="uberID",
        right_on="galaxy_id",
        how="left",
    ).drop(columns=["galaxy_id"])

    bij = bijective_group_mapping(
        gama["group_id_nessie"].to_numpy(),
        gama["group_id_sussex"].to_numpy(),
    )

    return gama, nessie_groups, sussex_groups, bij


def load_group_set_sharks_like_gama(
    sharks_galaxy_file: str = "/Users/sp624AA/Downloads/mocks/gama_like_from_groupfinding_cat.parquet",
    #sharks_group_file: str = "/Users/sp624AA/Downloads/group_finding_mocks/groups_shark.parquet",
    nessie_members_file: str = "/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/nessie_membership.parquet",
    sussex_members_file: str = "/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/gal_groups_sharks_gama_like.parquet",
    nessie_groups_file: str = "/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/nessie_groups.parquet",
    sussex_groups_file: str = "/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/gal_groups_sharks_gama_like_properties.parquet",
):
    sharks_data = load_sharks_data(sharks_galaxy_file).copy()
    sharks_groups = load_sharks_groups(sharks_data).copy()
    nessie_members = load_membership_with_optional_gama_mapping(
        nessie_members_file,
        output_group_name="group_id_nessie",
        gama_id_mapping_file=None,
    )
    sussex_members = load_membership_with_optional_gama_mapping(
        sussex_members_file,
        output_group_name="group_id_sussex",
        gama_id_mapping_file=None,
    )
    nessie_groups = load_group_properties(nessie_groups_file).copy()
    sussex_groups = load_group_properties(sussex_groups_file).copy()

    sharks_data = sharks_data.merge(nessie_members, on="galaxy_id", how="left")
    sharks_data = sharks_data.merge(sussex_members, on="galaxy_id", how="left")

    # restrict to real Sharks groups
    sharks_groups = sharks_groups[sharks_groups["group_id"] != -1].copy()

    truth_col = "id_fof"
    if "id_fof" in sharks_data.columns and "group_id" not in sharks_data.columns:
        sharks_data = sharks_data.rename(columns={"id_fof": "group_id_truth"})
        truth_col = "group_id_truth"
    elif "group_id_truth" in sharks_data.columns:
        truth_col = "group_id_truth"

    if "id_fof" in sharks_groups.columns and "group_id" not in sharks_groups.columns:
        sharks_groups = sharks_groups.rename(columns={"id_fof": "group_id"})
        sharks_groups = standardise_group_properties(sharks_groups)

    truth_ids = sharks_data[truth_col].to_numpy()

    nessie_to_truth = bijective_group_mapping(
        sharks_data["group_id_nessie"].to_numpy(),
        truth_ids,
    )
    sussex_to_truth = bijective_group_mapping(
        sharks_data["group_id_sussex"].to_numpy(),
        truth_ids,
    )

    # invert mapping so truth id -> recovered id
    truth_to_nessie = pd.Series(nessie_to_truth[0], index=nessie_to_truth[1])
    truth_to_sussex = pd.Series(sussex_to_truth[0], index=sussex_to_truth[1])

    sharks_groups["matched_nessie_id"] = sharks_groups["group_id"].map(truth_to_nessie)
    sharks_groups["matched_sussex_id"] = sharks_groups["group_id"].map(truth_to_sussex)

    nessie_groups = nessie_groups.add_prefix("nessie_").rename(columns={"nessie_group_id": "matched_nessie_id"})
    sussex_groups = sussex_groups.add_prefix("sussex_").rename(columns={"sussex_group_id": "matched_sussex_id"})

    sharks_groups = sharks_groups.merge(nessie_groups, on="matched_nessie_id", how="left")
    sharks_groups = sharks_groups.merge(sussex_groups, on="matched_sussex_id", how="left")

    return sharks_data, sharks_groups, nessie_groups, sussex_groups