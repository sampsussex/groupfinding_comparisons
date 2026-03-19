import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple


def load_sussex_groups(file):
    # group_id\tcentre_ra\tcentre_dec\tcentre_redshift
    # group_luminosity\tstellar_mass_total
    # stellar_mass_3_biggest\tbcg_abs_mag\tmultiplicity
    # bcg_is_red\tr_200
    groups = pd.read_csv(file, delim_whitespace=True)
    return groups


def load_nessie_groups(file):
    # should be dat file as well]
    # Should have the columns; 
    # ids
    # iter_ras
    # iter_decs
    # iter_redshifts
    # iter_idxs
    # median_redshifts,
    # distances
    # r50s
    # r100s
    # rsigmas
    # multiplicity
    # velocity_dispersion_gap
    # velocity_dispersion_gap_err
    # raw_masses
    # estimated_masses
    # vd_corrected_masses
    groups = pd.read_csv(file, delim_whitespace=True)
    return groups


def load_sharks_groups(groups_file, sharks_data):
    # Will contain loads of columns, and id_fof as group column, and id_galaxy_sky as galaxy id column.
    group_properties = pd.read_parquet(groups_file)
    # Find all id_fofs in sharks data
    group_properties = group_properties[group_properties['id_fof'].isin(sharks_data['group_id_fof'])]

    return group_properties


def load_nessie_membership(file):
    # should be dat file as well, with galaxy_id, and group_id columns
    df = pd.read_csv(file, delim_whitespace=True)
    if 'group_id' in df.columns():
        df.rename(columns = {'group_id': 'group_id_nessie'}, inplace=True)

    if 'ids' in df.columns():
        df.rename(columns = {'ids': 'galaxy_id'}, inplace=True)
    
    if 'id_group_sky' in df.columns():
       df.rename(columns = {'id_group_sky': 'galaxy_id'}, inplace=True)

    df['galaxy_id'] = df['galaxy_id'].astype(np.int64)

    return df


def load_sussex_membership(file):
    # should be dat file as well, with galaxy_id, and group_id columns
    # change group name to group_id_sam if called group_id
    df = pd.read_csv(file, delim_whitespace=True)
    if 'group_id' in df.columns():
        df.rename(columns = {'group_id': 'group_id_sussex'}, inplace=True)

    if 'ids' in df.columns():
        df.rename(columns = {'ids': 'galaxy_id'}, inplace=True)
    
    if 'id_group_sky' in df.columns():
       df.rename(columns = {'id_group_sky': 'galaxy_id'}, inplace=True)

    df['galaxy_id'] = df['galaxy_id'].astype(np.int64)

    return df

def load_sharks_data(file):
    # Will contain loads of columns, and id_fof as group column, and id_galaxy_sky as galaxy id column.
    return pd.read_parquet(file)


def load_gama_data(file):
    # Will contain loads of columns, and no group column, and uberID as galaxy id column.
    return pd.read_parquet(file)


def load_group_set_gama(gama_file = '/Users/sp624AA/Downloads/gama3/groupfinding_gama4_processed.parquet', 
                        nessie_members_file = '/Users/sp624AA/Downloads/groupfinder_results/gama/',
                        sussex_members_file = '/Users/sp624AA/Downloads/groupfinder_results/gama/',
                        nessie_groups_file = '/Users/sp624AA/Downloads/groupfinder_results/gama',
                        sussex_groups_file= '/Users/sp624AA/Downloads/groupfinder_results/gama'
                        ):
    # No fiducial groups. 
    # Galaxy id called uberID
    # Load gama_data, and join nessie and sam group ids. 
    # Load group data for nessie and sussex.
    gama = load_gama_data(gama_file)
    nessie_groups = load_nessie_groups(nessie_groups_file)
    sussex_groups = load_sussex_groups(sussex_groups_file)

    gama = gama.merge(load_nessie_membership(nessie_members_file), left_on='uberID', right_on='galaxy_id', how='left')
    gama = gama.merge(load_sussex_membership(sussex_members_file), left_on='uberID', right_on='galaxy_id', how='left')

    bijective_matches = bijective_group_mapping(gama['group_id_nessie'], gama['group_id_sussex'])
    # I am only doing this one way, i wonder if this should be done both?

    return gama, nessie_groups, sussex_groups, bijective_matches


def load_group_set_sharks_like_gama(sharks_file = '/Users/sp624AA/Downloads/mocks/gama_like_from_groupfinding_cat.parquet',
                                    nessie_members_file = '/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/',
                                    sussex_memebers_file = '/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/',
                                    nessie_groups_file = '/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/',
                                    sussex_groups_file = '/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/'):
    # fiducial simulated groups have id_fof as group column.
    # Galaxy id called id_galaxy_sky.
    sharks_data = load_sharks_data(sharks_file)
    nessie_members = load_nessie_membership(nessie_members_file)
    sussex_members = load_sussex_membership(sussex_memebers_file)
    nessie_groups = load_nessie_groups(nessie_groups_file)
    sussex_groups = load_sussex_groups(sussex_groups_file)
    sharks_groups = load_sharks_groups(sharks_file, sharks_data)

    # Join sharks with nessie and sussex group ids
    sharks_data = sharks_data.merge(nessie_members, on='galaxy_id', how='left')
    sharks_data = sharks_data.merge(sussex_members, on='galaxy_id', how='left')

    # take group properties from nessie and sussex, and add to sharks data.
    # bijectively match sharks groups to nessie and sussex groups, and add group properties to sharks group data
    sharks_mapping_nessie = bijective_group_mapping(sharks_data['group_id_nessie'], sharks_data['group_id_fof'])
    sharks_mapping_sussex = bijective_group_mapping(sharks_data['group_id_sussex'], sharks_data['group_id_fof'])

    # Add group properties from sharks and nessie to sharks group data, using mapping. 
    # Fill in NaNs for groups that don't match.
    # Add nessie / sussex prefix to group properties to avoid confusion.

    sharks_groups = sharks_groups.merge(nessie_groups, left_on=sharks_mapping_nessie[0], right_on=nessie_groups['ids'], how='left', suffixes=('', '_nessie'))
    sharks_groups = sharks_groups.merge(sussex_groups, left_on=sharks_mapping_sussex[0], right_on=sussex_groups['group_id_sussex'], how='left', suffixes=('', '_sussex'))

    return sharks_data, sharks_groups, nessie_groups, sussex_groups


def bijective_group_mapping(group_ids_1: List[int], group_ids_2: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return the bijectively matched group IDs between two catalogues.

    For each group in catalogue 1 with size >= 2, find its best match in catalogue 2
    by maximizing q1 * q2, where:
        q1 = N_overlap / N_group1
        q2 = N_overlap / N_group2

    A match is considered bijective if:
        q1 > 0.5 and q2 > 0.5

    Parameters
    ----------
    group_ids_1 : list[int]
        Group IDs for catalogue 1.
    group_ids_2 : list[int]
        Group IDs for catalogue 2, aligned galaxy-by-galaxy with group_ids_1.

    Returns
    -------
    matched_group_ids_1 : np.ndarray
        Group IDs from catalogue 1 that are bijectively matched.
    matched_group_ids_2 : np.ndarray
        Corresponding matched group IDs from catalogue 2.
    """
    assert len(group_ids_1) == len(group_ids_2), "Group catalogs must have same length"

    group_ids_1 = np.asarray(group_ids_1)
    group_ids_2 = np.asarray(group_ids_2)

    # Group sizes, excluding isolated galaxies
    count_table_1 = Counter(group_ids_1[group_ids_1 != -1])
    count_table_2 = Counter(group_ids_2[group_ids_2 != -1])

    # Only test groups in catalogue 1 with multiplicity >= 2
    valid_groups_1 = [gid for gid, n in count_table_1.items() if n >= 2]

    matched_1 = []
    matched_2 = []

    for gid1 in valid_groups_1:
        member_idx = np.where(group_ids_1 == gid1)[0]
        overlaps_2 = group_ids_2[member_idx]
        overlaps_2 = overlaps_2[overlaps_2 != -1]

        if len(overlaps_2) == 0:
            continue

        n1 = count_table_1[gid1]
        overlap_counts = Counter(overlaps_2)

        best_gid2 = None
        best_q1 = 0.0
        best_q2 = 0.0
        best_score = -1.0

        for gid2, n_overlap in overlap_counts.items():
            n2 = count_table_2[gid2]
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


def bijcheck(group_ids_1: List[int], group_ids_2: List[int], min_group_size: int):
    """
    Bijective comparison between two group catalogues as per Robotham+2011.
    This calculates Equations (9, 10, 12, and 13) from Robotham+2011.

    Args:
        group_ids_1: First group catalog (list of group IDs)
        group_ids_2: Second group catalog (list of group IDs)
        min_group_size: Minimum group size threshold

    Returns:
        BijResults containing e_num, e_den, q_num, q_den
    """
    assert len(group_ids_1) == len(group_ids_2), "Group catalogs must have same length"

    # Convert to numpy arrays for easier manipulation
    group_ids_1 = np.array(group_ids_1)
    group_ids_2 = np.array(group_ids_2)

    # Frequency tables excluding -1
    count_table_1 = Counter(group_ids_1[group_ids_1 != -1])
    count_table_2 = Counter(group_ids_2[group_ids_2 != -1])

    # Filter groups in tab1 with size >= min_group_size
    valid_groups_1 = [
        group for group, count in count_table_1.items() if count >= min_group_size
    ]

    # Find indices of valid group members
    valid_mask = np.isin(group_ids_1, valid_groups_1)
    valid_indices_1 = np.where(valid_mask)[0]

    # Create group_list maintaining order (first occurrence of each group)
    group_list = []
    seen = set()
    for idx in valid_indices_1:
        group_id = group_ids_1[idx]
        if group_id not in seen:
            group_list.append(group_id)
            seen.add(group_id)

    # Process each group
    q1_values = []
    q2_values = []
    n1_values = []

    for group_id in group_list:
        # Find all galaxies in this group
        group_galaxies = np.where(group_ids_1 == group_id)[0]

        # Get corresponding groups in catalog 2
        overlap_groups = group_ids_2[group_galaxies]
        overlap_valid = overlap_groups[overlap_groups != -1]

        n1_current = count_table_1.get(group_id, 1)

        if len(overlap_valid) > 0:
            # Count overlaps
            temptab = Counter(overlap_valid)

            frac_1 = []
            frac_2 = []

            for group2, count in temptab.items():
                if group2 in count_table_2:
                    n2_val = count_table_2[group2]
                    frac_1.append(count / n1_current)
                    frac_2.append(count / n2_val)

            # Handle isolated galaxies (group_id = -1)
            num_isolated = np.sum(overlap_groups == -1)
            if num_isolated > 0:
                iso_frac1 = 1.0 / n1_current
                for _ in range(num_isolated):
                    frac_1.append(iso_frac1)
                    frac_2.append(1.0)

            # Find best match (first occurrence of maximum product)
            if frac_1:
                products = [f1 * f2 for f1, f2 in zip(frac_1, frac_2)]
                best_match = np.argmax(products)  # argmax returns first occurrence
                q1 = frac_1[best_match]
                q2 = frac_2[best_match]
            else:
                q1 = 1.0 / n1_current
                q2 = 1.0
        else:
            # All isolated
            q1 = 1.0 / n1_current
            q2 = 1.0

        q1_values.append(q1)
        q2_values.append(q2)
        n1_values.append(float(n1_current))

    # Calculate final results
    e_num = sum(1 for q1, q2 in zip(q1_values, q2_values) if q1 > 0.5 and q2 > 0.5)
    e_den = len(n1_values)
    q_num = sum(q1 * n1 for q1, n1 in zip(q1_values, n1_values))
    q_den = sum(n1_values)

    return e_num, e_den, q_num, q_den


def s_score(measured_groups: List[int], mock_groups: List[int], groupcut: int) -> float:
    """
    The final S-score measurement for comparisons between two group catalogues.
    Equation 15 of Robotham+2011.

    Args:
        measured_groups: Measured group catalog
        mock_groups: Mock/reference group catalog
        groupcut: Minimum group size threshold

    Returns:
        S-score value
    """
    e_num_mock, e_den_mock, q_num_mock, q_den_mock = bijcheck(
        mock_groups, measured_groups, groupcut
    )
    e_num_meas, e_den_meas, q_num_meas, q_den_meas = bijcheck(
        measured_groups, mock_groups, groupcut
    )

    mock_e = e_num_mock / e_den_mock
    fof_e = e_num_meas / e_den_meas
    mock_q = q_num_mock / q_den_mock
    fof_q = q_num_meas / q_den_meas

    return mock_e * fof_e * mock_q * fof_q, mock_e * fof_e, mock_q * fof_q

