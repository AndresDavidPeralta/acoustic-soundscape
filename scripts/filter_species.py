# Identify the level of endangerment of species listed on the Red List
# Cross-reference between the Red List and the database

# Import

import pandas as pd

# Entry and exit routes 
path_local = ".../train_data.csv"     
path_redlist = ".../ista roja.csv"    

# Read files
df_local = pd.read_csv(path_local)
df_redlist = pd.read_csv(path_redlist)

# Normalize names
df_local['scientific_name'] = df_local['scientific_name'].astype(str).str.strip().str.lower()
df_redlist['scientificName'] = df_redlist['scientificName'].astype(str).str.strip().str.lower()

# Filter by categories
categorias_validas = [
    "Critically Endangered", "Endangered", "Vulnerable", 
    "Near Threatened", "Least Concern"
]
df_redlist = df_redlist[df_redlist['redlistCategory'].isin(categorias_validas)]

# Get the matches
df_matches = pd.merge(
    df_local,
    df_redlist[['scientificName', 'redlistCategory']],
    left_on='scientific_name',
    right_on='scientificName',
    how='inner'
)

# Count records by species
df_counts = df_matches.groupby(['scientific_name', 'redlistCategory']).size().reset_index(name='num_registros')

# Assign category order
categoria_order = {
    "Critically Endangered": 1,
    "Endangered": 2,
    "Vulnerable": 3,
    "Near Threatened": 4,
    "Least Concern": 5
}
df_counts['categoria_orden'] = df_counts['redlistCategory'].map(categoria_order)

# Order
df_counts_sorted = df_counts.sort_values(by=['categoria_orden', 'num_registros'])

# Export 
df_counts_sorted.to_csv("prioritized_species.csv", index=False)

print("==  [INFO] File 'especies_priorizadas.csv' generated correctly ==")