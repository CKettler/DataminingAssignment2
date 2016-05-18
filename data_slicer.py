import pandas as pd


def slicer(df, n_slices, outputfiles='data/data_slice_%d.csv'):
    srch_ids = df['srch_id'].unique()
    num_ids = len(srch_ids)
    rows_per_slice = int(round(num_ids / float(n_slices)))
    for slice_x in range(0, n_slices):
        new_df = pd.DataFrame()
        rows = 1
        while rows <= rows_per_slice:
            cur_row = rows_per_slice * slice_x + rows
            if cur_row < num_ids:
                mask = (df['srch_id'] == srch_ids[cur_row])
                tf = df.loc[mask]
                new_df = new_df.append(tf)
            rows += 1
        new_df.to_csv(outputfiles % (slice_x + 1))
