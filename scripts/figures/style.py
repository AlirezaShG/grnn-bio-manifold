"""Shared figure style for the preprint."""
import matplotlib.pyplot as plt

TYPE_COLORS = {'Pyr': '#2b6c8f', 'PV': '#c0392b', 'SST': '#d4a017', 'VIP': '#6c4f9e'}
LEARNER_COLORS = {'bptt': '#222222', 'local': '#1f77b4', 'local_no_beta': '#e07b39',
                  'readout_only': '#8c8c8c', 'local_sym': '#2ca02c', 'no_beta_sym': '#d62728'}
LEARNER_LABELS = {'bptt': 'BPTT', 'local': r'local ($\beta$)', 'local_no_beta': r'local (no $\beta$)',
                  'readout_only': 'frozen reservoir'}
TASK_LABELS = {'early': 'T1-early', 'late': 'T1-late', 'dms': 'T2-dms'}


def apply_style():
    plt.rcParams.update({
        'font.size': 7, 'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6,
        'ytick.labelsize': 6, 'legend.fontsize': 6, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .6, 'xtick.major.width': .6,
        'ytick.major.width': .6, 'pdf.fonttype': 42, 'font.family': 'DejaVu Sans',
    })


def panel_label(ax, letter):
    ax.text(-0.18, 1.08, letter, transform=ax.transAxes, fontsize=9, fontweight='bold', va='bottom')
