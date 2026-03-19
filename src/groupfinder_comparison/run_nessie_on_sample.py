# running nessie on the same sample as my other groupfinder.
# im basically just lifting code directly from trystan here. 

"""
Script to run Nessie on a GAMA-like mock version of Shark.
"""

import pandas as pd
import numpy as np
from nessie import FlatCosmology
from nessie.helper_funcs import create_density_function
from nessie import RedshiftCatalog
from nessie.optimizer import optimize_nm

cosmo = FlatCosmology(h=0.67, omega_matter=0.3)



df_galaxies = pd.read_parquet("/Users/sp624AA/Downloads/mocks/gama_like_from_groupfinding_cat.parquet")


# Prepare the inputs to Nessie
redshifts = np.array(df_galaxies["redshift_observed"])
ra = np.array(df_galaxies["ra"])
dec = np.array(df_galaxies["dec"])
mock_ids = np.array(df_galaxies["id_fof"])
abs_mags = np.array(df_galaxies["mag_abs_r_VST"])
total_counts = len(redshifts)
SURVEY_AREA = 0.013
running_density = create_density_function(redshifts, total_counts, SURVEY_AREA, cosmo)

# Tune and run Nessie
red_cat = RedshiftCatalog(ra, dec, redshifts, running_density, cosmo)
red_cat.set_completeness()
red_cat.mock_group_ids = mock_ids
b0, r0, s_tot = optimize_nm(red_cat, 5)
print(f"b0 = {b0}")
print(f"r0 = {r0}")
print(f"s_tot = {s_tot}")

red_cat.run_fof(b0, r0)

membership  = {'group_id': red_cat.group_ids, 'galaxy_id': np.array(df_galaxies['id_galaxy_sky'])}
membership_df = pd.DataFrame(membership)
membership_df.to_parquet("/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/nessie_membership.parquet")

# Write finished data products


df_groups = pd.DataFrame(
    red_cat.calculate_group_table(abs_mags, np.repeat(0., len(redshifts)))
)
df_groups.to_parquet("/Users/sp624AA/Downloads/groupfinder_results/sharks_gama_like/nessie_groups.parquet")