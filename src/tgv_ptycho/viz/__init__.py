"""Visualization helpers."""

from tgv_ptycho.viz.plot_exp040 import (
    EXP040_FIGURE_FILENAMES,
    plot_B_plane_probe,
    plot_detector_intensity_baseline,
    plot_detector_visibility,
    plot_dz_convergence,
    plot_exit_field_multislice,
    plot_lateral_fov_convergence,
    plot_projected_limit_comparison,
    plot_tgv_geometry_and_index_slices,
    save_exp040_figures,
)
from tgv_ptycho.viz.plot_exp040_r1 import (
    EXP040_R1_FIGURE_FILENAMES,
    plot_r1_external_padding_convergence,
    plot_r1_refined_convergence,
    save_exp040_r1_figures,
)
from tgv_ptycho.viz.plot_exp040_r2 import (
    EXP040_R2_FIGURE_FILENAMES,
    plot_r2_alias_method_difference,
    plot_r2_period_aligned_convergence,
    save_exp040_r2_figures,
)
from tgv_ptycho.viz.plot_exp040_r3 import (
    EXP040_R3_FIGURE_FILENAMES,
    plot_r3_b_exit_and_bc_spectrum,
    plot_r3_detector_operator_difference,
    plot_r3_detector_sampling_convergence,
    save_exp040_r3_figures,
)
from tgv_ptycho.viz.plot_exp040_r4 import (
    EXP040_R4_FIGURE_FILENAMES,
    save_exp040_r4_figures,
)
from tgv_ptycho.viz.plot_exp040_r5 import (
    EXP040_R5_FIGURE_FILENAMES,
    save_exp040_r5_figures,
)
from tgv_ptycho.viz.plot_exp040_r6 import (
    EXP040_R6_FIGURE_FILENAMES,
    save_exp040_r6_figures,
)
from tgv_ptycho.viz.plot_exp040_r7 import (
    EXP040_R7_FIGURE_FILENAMES,
    save_exp040_r7_figures,
)
from tgv_ptycho.viz.plot_exp040_r8 import (
    EXP040_R8_FIGURE_FILENAMES,
    save_exp040_r8_figures,
)
from tgv_ptycho.viz.plot_exp040_r9 import (
    EXP040_R9_FIGURE_FILENAMES,
    save_exp040_r9_figures,
)
from tgv_ptycho.viz.plot_field import plot_complex_field, save_intensity_image
from tgv_ptycho.viz.plot_recon import (
    plot_loss_curve,
    save_diffraction_montage,
    save_reconstruction_comparison,
    save_scan_positions,
)

__all__ = [
    "EXP040_FIGURE_FILENAMES",
    "EXP040_R1_FIGURE_FILENAMES",
    "EXP040_R2_FIGURE_FILENAMES",
    "EXP040_R3_FIGURE_FILENAMES",
    "EXP040_R4_FIGURE_FILENAMES",
    "EXP040_R5_FIGURE_FILENAMES",
    "EXP040_R6_FIGURE_FILENAMES",
    "EXP040_R7_FIGURE_FILENAMES",
    "EXP040_R8_FIGURE_FILENAMES",
    "EXP040_R9_FIGURE_FILENAMES",
    "plot_B_plane_probe",
    "plot_complex_field",
    "plot_detector_intensity_baseline",
    "plot_detector_visibility",
    "plot_dz_convergence",
    "plot_exit_field_multislice",
    "plot_lateral_fov_convergence",
    "save_intensity_image",
    "plot_loss_curve",
    "plot_projected_limit_comparison",
    "plot_r1_external_padding_convergence",
    "plot_r1_refined_convergence",
    "plot_r2_alias_method_difference",
    "plot_r2_period_aligned_convergence",
    "plot_r3_b_exit_and_bc_spectrum",
    "plot_r3_detector_operator_difference",
    "plot_r3_detector_sampling_convergence",
    "plot_tgv_geometry_and_index_slices",
    "save_diffraction_montage",
    "save_exp040_figures",
    "save_exp040_r1_figures",
    "save_exp040_r2_figures",
    "save_exp040_r3_figures",
    "save_exp040_r4_figures",
    "save_exp040_r5_figures",
    "save_exp040_r6_figures",
    "save_exp040_r7_figures",
    "save_exp040_r8_figures",
    "save_exp040_r9_figures",
    "save_reconstruction_comparison",
    "save_scan_positions",
]
