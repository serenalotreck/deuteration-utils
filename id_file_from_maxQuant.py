"""
Create the input ID file from a MaxQuant proteomics output.

Author: Serena G. Lotreck
"""
import argparse
import pyopenms as oms
from os import listdir
from os.path import splitext, join, abspath
import pandas as pd
import numpy as np
from tqdm import tqdm


def calc_add_n(aa_counts, aa_labeling_dict):
    """
    Convert amino acid counts to the final literature_n value.

    Adapted from ID_File_For_Deut_From_Fragpipe_V3.py

    parameters:
        aa_counts, dict: keys are AA codes, values are the number of that AA
            in the sequence
        aa_labeling_dict, dict: keys are amino acid codes, values are number of
            hydrogens, derived from aa_labeling_sites.tsv

    returns:
        add_n, float: literature_n
    """
    add_n = 0
    for aa, count in aa_counts.items():
        add_n += count * aa_labeling_dict[aa]
    return add_n


def calculate_literature_n(seq, aa_labeling_dict):
    """
    Calculate the number of possible deuterations for a given amino acid
    sequence.

    Adapted from ID_File_For_Deut_From_Fragpipe_V3.py

    parameters:
        seq, str: amino acid sequence with single letter code
        aa_labeling_dict, dict: keys are amino acid codes, values are number of
            hydrogens, derived from aa_labeling_sites.tsv
    """
    aa_counts = {}
    for aa in seq:
        if aa not in aa_counts.keys():
            aa_counts[aa] = 0
        aa_counts[aa] += 1
    literature_n = calc_add_n(aa_counts, aa_labeling_dict)

    return literature_n


def seq_to_cf(seq):
    """
    Use PyOpenMS to convert the AA sequence to the chemical formula.
    """
    seq = oms.AASequence.fromString(evidence.loc[0].to_dict()['Sequence'])
    cf = seq.getFormula().toString()
    return cf


def evidence_to_deut(evidence, all_deut_col_names, column_mapping,
                     aa_labeling_dict):
    """
    Converts the information available in the evidence.txt output from MaxQuant
    to the column format required by DeuteRater.

    parameters:
        evidence, pandas df: the evidence table from MaxQuant
        all_deut_col_names, list of str: all column names in the order they are
            expected by DeuteRater
        column_mapping, dict: keys are the column names from MaxQuant, values
            are the column names for DeuteRater
        aa_labeling_dict, dict: keys are amino acid codes, values are number of
            hydrogens, derived from aa_labeling_sites.tsv

    returns:
        deut_table, pandas df: an ID file-formatted df with only the columns
            that could be obtained from MaxQuant
    """
    # Get the information we can from the MaxQuant output
    to_keep = evidence[list(column_mapping.keys())]
    deut_table = to_keep.rename(columns=column_mapping)
    deut_table['cf'] = deut_table['Sequence'].apply(seq_to_cf)

    # Need to convert retention time from minutes to seconds
    deut_table['Precursor Retention Time (sec)'] = deut_table[
        'Precursor Retention Time (sec)'] * 60

    # Calculate literature_n and neutromers_to_expect
    deut_table['literature_n'] = deut_table['Sequence'].apply(
        calculate_literature_n, args=(aa_labeling_dict, ))
    deut_table['neutromers_to_extract'] = deut_table['Observed Mass'].apply(
        extracting_neut)

    # Add all other column names, order them correctly
    missing_cols = [
        i for i in all_deut_col_names if i not in deut_table.columns
    ]
    deut_table.loc[:, missing_cols] = np.nan
    deut_table = deut_table[all_deut_col_names]

    return deut_table


def filter_maxQuant(evidence, samples_as_ref, threshold=10):
    """
    Collapses rows that represent the same peptide with slightly different
    retention times.

    parameters:
        evidence, df: MaxQuant eviendence table
        samples_as_ref, list of str: sample names to use in ID file
        threshold, float: number of seconds that the peptides must elute
            within in order to be considered the same, default is 10 sec

    returns:
        evidence_collapsed, df: collapsed table
    """
    # Keep only the rows corresponding to the samples that we want to average
    # as the reference, i.e. the unlabeled timepoints
    print(f'Keeping only peptides that appear in samples {samples_as_ref}. '
          'Values from these samples will be averaged to form the ID file.')
    evidence_as_ref = evidence[evidence['Experiment'].isin(samples_as_ref)]
    print('\n\n\n\n\nHELLO WOLRD')
    print(evidence_as_ref.columns)

    # Drop the experiment and raw file columns because it doesn't matter that
    # they're different and that'll mess up our groupby
    evidence_as_ref = evidence_as_ref.drop(columns=['Experiment', 'Raw file'])

    # Now we group by all columns that have strings, they should be the same
    # for a given peptide and we want to combine on them, also a mean will fail
    # for any column that isn't a number
    group_on = [
        name for name, dtype in zip(evidence.columns, evidence.dtypes)
        if dtype == "object"
    ]

    # Get the means and stds for all peptide sequences
    peptide_means = evidence_as_ref.groupby(group_on).agg(['mean', 'std', 'count'])

    # Check for rows where the std is within our threshold
    evidence_collapsed = peptide_means[
        (peptide_means['Retention time']['count'] > 0)
        & (peptide_means['Retention time']['std'] <= threshold)]

    # If that is not all the peptides, have to deal with problem children
    if len(evidence_collapsed) != len(peptide_means):
        pass  ## TODO implement
    else:
        print('All peptide groupings fell inside the provided threshold.')
        return evidence_collapsed


def main(max_quant_evidence, aa_labeling_table, study_type,
         neutromers_to_extract, threshold_to_collapse, samples_as_ref,
         out_path, out_prefix):

    # Read in the data and labeling table
    print('\nReading in evidence table and labeling dictionary...')
    evidence = pd.read_csv(max_quant_evidence, sep='\t')
    aa_labeling_df = pd.read_csv(aa_labeling_table, sep='\t')
    aa_labeling_df.set_index('study_type', inplace=True)
    aa_labeling_dict = aa_labeling_df.loc[
        study_type,
    ].to_dict()

    # Collapse the evidence table
    print('\nCollapsing the evidence table...')
    evidence = filter_maxQuant(evidence, samples_as_ref, threshold_to_collapse)

    # Define parameters for DeuteRater
    columns = [
        'Sequence', 'first_accession', 'Protein Name', 'Protein ID',
        'Precursor Retention Time (sec)', 'rt_start', 'rt_end', 'rt_width',
        'Precursor m/z', 'theoretical_mass', 'Identification Charge', 'ptm',
        'avg_ppm', 'start_loc', 'end_loc', 'num_peptides', 'num_unique',
        'accessions', 'species', 'gene_name', 'protein_existence',
        'sequence_version', 'cf', 'neutromers_to_extract', 'literature_n'
    ]
    column_mapping = {
        'Sequence': 'Sequence',
        'Protein names': 'Protein name',
        'Leading razor protein': 'Protein ID',
        'Charge': 'Identification Charge',
        'm/z': 'Precursor m/z',
        'Retention time': 'Precursor Retention Time (sec)',
        'Mass':
        'Observed Mass'  # Not technically required, but used to calculate neutromers
    }

    # Call the function
    print('\nCreating the ID file...')
    deut_table = evidence_to_deut(evidence, all_deut_col_names, column_mapping,
                                  aa_labeling_dict)

    # Save the file
    print('\nSaving output...')
    save_name = f'{out_loc}/{out_prefix}_deuterater_ID_file.csv'
    deut_table.to_csv(save_name, index=False)
    print(f'Saved output to {save_name}')

    print('\nDone!')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Format deuterater input')

    parser.add_argument(
        'max_quant_evidence',
        type=str,
        help='Path to MaxQuant evidence.txt file for this experiment')
    parser.add_argument(
        'aa_labeling_table',
        type=str,
        help='Path to .tsv file containing the maximum possible '
        'deuteration sites in each amino acid. Letter codes must '
        'correspond to those used in the MaxQuant file. Should '
        'be reflective of the organism from which samples '
        'were taken')
    parser.add_argument(
        'study_type',
        type=str,
        help='String that identifies which row of the aa labeling '
        'table should be used. For the default human table, option '
        'is "tissue"')
    parser.add_argument(
        'neutromers_to_extract',
        type=int,
        default=4,
        help='Number of isotope envelopes to consider. Default '
        'is 4')
    parser.add_argument('threshold_to_collapse',
                        type=int,
                        default=10,
                        help='Number of seconds to allow different between '
                        'otherwise identical peptides')
    parser.add_argument(
        '-samples_as_ref',
        nargs='+',
        help='Sample names to use as reference for which proteins '
        'we expect to see in samples')
    parser.add_argument('out_path',
                        type=str,
                        help='Path to directory to save output')
    parser.add_argument('out_prefix',
                        type=str,
                        help='String to prepend to output file name')

    args = parser.parse_args()

    for a, val in args.items():
        if a in ['max_quant_evidence', 'aa_labeling_table', 'out_path']:
            args[a] = abspath(v)

    main(args.max_quant_evidence, args.aa_labeling_table, args.study_type,
         args.neutromers_to_extract, args.threshold_to_collapse,
         args.sampels_as_ref, args.out_path, args.out_prefix)
