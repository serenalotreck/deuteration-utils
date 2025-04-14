"""
Given the frac_new_output.tsv file from a DeuteRater run, plot the values of
each isotope envelope over time. Makes a separate file for each unique
peptide.

Author: Serena G. Lotreck
"""
import argparse
from os.path import abspath

import pandas as pd
from functools import partial
import matplotlib.pyplot as plt


class MultipleSampleGroups(Exception):

    def __init__(self, msg):
        self.msg = msg


def plot_envelopes(frac_new_df, outpath, col_to_plot):
    """
    Plot isotope envelopes.

    parameters:
        frac_new_df, df: frac_new_output
        outpath, str: path to directory to save plots
    """
    # Subset to just the col of interest plus the unique condition identifiers
    frac_new_to_plot = frac_new_df[[
        'Sequence', 'time', 'sample_group', 'neutromers_to_extract',
        col_to_plot
    ]]

    # Next, need to split out the individual values in the column to plot
    # Some of the columsn are comma separated and some are space separated, so
    # check if the col of interst is comma or space delimited. Because there
    # are spaces after the commas, we have to assume that if it doesn't split on
    # ', ', that it's space separated -- would only be an issue if some other
    # separator were used.
    if frac_new_to_plot[col_to_plot].str.split(', ',
                                               expand=True).shape[1] == 1:
        sep = ' '
    else:
        sep = ', '

    # Define the columns to split it into
    num_neutromers = frac_new_to_plot.neutromers_to_extract.max()
    new_cols = [f'{col_to_plot}_{i}' for i in range(num_neutromers)]

    # Make the split
    frac_new_to_plot[new_cols] = frac_new_to_plot[col_to_plot].str.split(
        sep, expand=True)

    # Convert new columns and time to numeric
    frac_new_to_plot[new_cols] = frac_new_to_plot[new_cols].astype('float64')
    frac_new_to_plot['time'] = frac_new_to_plot['time'].astype(
        'int64')  # Assumes integer timepoints

    # Groupby peptides, take the mean of reps over time points
    # Assumes one sample group, will need to implement for multiples
    if len(frac_new_to_plot.sample_group.unique()) > 1:
        raise MultipleSampleGroups(
            'There is more than one sample group, this '
            'implementation does not currently support this.')
    # We have to drop everything besides what we want to plot, because non-
    # numeric column types will cause the agg to fail
    frac_new_to_plot = frac_new_to_plot[['Sequence', 'time'] + new_cols]
    # Do the grouping and aggregation
    frac_new_grouped = frac_new_to_plot.groupby(['Sequence', 'time']).agg(
        ['mean', partial(pd.Series.std, ddof=0)])
    print(frac_new_grouped)
    # Reset index to make plotting easier
    frac_new_grouped = frac_new_grouped.reset_index()

    # Plot
    for pep in frac_new_grouped.Sequence.unique():

        # Get string for plot title
        title = f'{col_to_plot} for peptide {pep}'

        # Get the subset of the dataframe corresponding to the one peptide
        pep_df = frac_new_grouped[frac_new_grouped['Sequence'] == pep]

        # Go through each envelope to plot
        fig, ax = plt.subplots()

        for env in new_cols:
            ax.errorbar(pep_df.time,
                        pep_df[env]['mean'],
                        yerr=pep_df[env]['std'],
                        marker='o',
                        label=f'Envelope {env.split("_")[-1]}')

        ax.legend()
        ax.set_ylabel(col_to_plot)
        ax.set_xlabel('Time')
        plt.savefig(
            f'{outpath}/{pep}_{col_to_plot}_{num_neutromers}_neutromers_envelope_plot.png',
            format='png',
            dpi=600,
            bbox_inches='tight')
        plt.close()


def main(frac_new_output, outpath, col_to_plot):

    # Read in file
    print('\nReading in frac_new_output...')
    frac_new_df = pd.read_csv(frac_new_output, sep='\t')

    # Plot
    print('\nCreating plots...')
    plot_envelopes(frac_new_df, outpath, col_to_plot)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Plot isotope envelopes')

    parser.add_argument('frac_new_output',
                        type=str,
                        help='Path to frac_new_output.tsv')
    parser.add_argument('outpath', type=str, help='Path to save plots')
    parser.add_argument(
        '-col_to_plot',
        type=str,
        default='frac_new_abunds',
        help='Col name from frac_new_output to plot, default is '
        'frac_new_abunds')

    args = parser.parse_args()

    args.frac_new_output = abspath(args.frac_new_output)
    args.outpath = abspath(args.outpath)

    main(args.frac_new_output, args.outpath, args.col_to_plot)
