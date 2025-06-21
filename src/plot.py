import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.input import Input_uc
from src.output import Output_uc


def plot_schedule_heatmap(
    input_uc: Input_uc, 
    output_uc: Output_uc,
    is_prev: bool = False,
    save_file_name: str = None, # no .png
    fig_height: float = 20,
    pad_rectangle: float = None,
    do_white: bool = True,
    dpi: float = 300,
):
    cmap = plt.cm.jet.copy()
    cmap.set_bad(color="white")
    num_periods = output_uc.p.shape[1]

    idx_marginal_unit_per_period = np.array([
        int(np.where(input_uc.cost_lin == mpg)[0][0]) 
        if mpg > 0 else -2 # for test (demand = 0 to blackstart) result plot purpose
        for mpg in output_uc.marginal_price_generation
        ])

    helper = [
        (input_uc.idx_lng, len(input_uc.idx_nuclear) + len(input_uc.idx_coal)),
        (input_uc.idx_coal, len(input_uc.idx_nuclear)),
        (input_uc.idx_nuclear, 0),
    ]

    # # below commented code can highlight units bounded by ramp constraints 
    ## to do this, we need to return (twice) in solve_uc and pass the 4 gurobi constr objects as args
    # idx_constrained_by_ramp_mat = np.zeros((input_uc.num_units, num_periods), dtype=np.int64)
    # for i in range(input_uc.num_units):
    #     for t in range(num_periods):

    #         if t == num_periods - 1:
    #             idx_constrained_by_ramp_mat[i, t] = bool(constr_ramp_2[i, t].Pi)
    #             continue

    #         idx_constrained_by_ramp_mat[i, t] = bool(constr_ramp_1[i, t].Pi) or bool(constr_ramp_3[i, t].Pi) or bool(constr_ramp_4[i, t].Pi)
    # idx_constrained_by_ramp_mat = idx_constrained_by_ramp_mat.transpose()
        
    fig, axes = plt.subplots(
        3, 1, 
        figsize=(fig_height, fig_height), 
        gridspec_kw={"height_ratios": [56/122, 41/122, 25/122]}, 
        dpi=dpi,
        sharex=True,
    )

    for ax, (idx_type, idx_start) in zip(axes, helper):
        p_type = output_uc.p[idx_type]
        num_units_type = len(p_type)
        p_min_type = np.tile(input_uc.p_min[idx_type][:, None], reps=num_periods)
        p_max_type = np.tile(input_uc.p_max[idx_type][:, None], reps=num_periods)
        heatmap = (p_type - p_min_type) / (p_max_type - p_min_type) * 100
        heatmap[p_type == 0] = np.nan

        #
        ax.imshow(
            heatmap,
            aspect="auto",
            origin="lower",
            extent=[0.5, input_uc.num_periods + 0.5, idx_start, idx_start + num_units_type],
            cmap=cmap,
            vmin=0,
            vmax=100,
        )
        
        #
        # pad_rectangle = 0.066 / 20 * fig_height if pad_rectangle is None else pad_rectangle
        pad_rectangle = 0.04 / 20 * fig_height if pad_rectangle is None else pad_rectangle
        for t in range(num_periods):
            u_abs = idx_marginal_unit_per_period[t]

            if idx_start <= u_abs < idx_start + num_units_type:
                
                rect_h = Rectangle(
                    (t + 0.5 + pad_rectangle, u_abs), 1 - 2 * pad_rectangle, 1,
                    fill=False,
                    edgecolor="white",
                    facecolor="none",
                    hatch="///",
                    linewidth=0,
                    zorder=3
                )

                rect_h.set_hatch_linewidth(fig_height / 10)
                ax.add_patch(rect_h)

                ax.add_patch(
                    Rectangle(
                        (t + 0.5 + pad_rectangle, u_abs), 1 - 2 * pad_rectangle, 1,
                        fill=False,
                        edgecolor="black",
                        facecolor="none",
                        # hatch="//",
                        linewidth=fig_height / 5,
                        zorder=3
                    )
                )
            
            # # the same constr highlight (I won't use this for plots in the report; its uglier; and a lot of them were startup cost-constr bound)
            # for idx_unit, val in enumerate(idx_constrained_by_ramp_mat[t]):
            #     if val:
            #         if idx_start <= val < idx_start + num_units_type:
            #             rect_h = Rectangle(
            #                 (t + 0.5 + pad_rectangle, idx_unit), 1 - 2 * pad_rectangle, 1,
            #                 fill=False,
            #                 edgecolor="white",
            #                 facecolor="none",
            #                 hatch="\\",
            #                 linewidth=0,
            #                 zorder=3
            #             )
            #             rect_h.set_hatch_linewidth(fig_height / 10)
            #             ax.add_patch(rect_h)

        #
        ax.set_yticks([idx_start, idx_start + num_units_type - 1])
        ax.set_yticklabels([idx_start + 1, idx_start + num_units_type])
        ax.tick_params(axis="both", width=fig_height / 10, length=fig_height / 2, pad=fig_height / 2, labelsize=fig_height * 2)
        for side in ["bottom", "left", "top", "right"]:
            ax.spines[side].set_linewidth(fig_height / 10)

    xticks_temp = np.arange(0, num_periods + 1, 6)
    xticks_temp[0] = 1
    axes[-1].set_xticks(xticks_temp)
    axes[-1].set_xticklabels(xticks_temp)

    # # for TA: you can test like this; the 57th (0-based) idx unit will be shown (block's bottom line = line)
    # axes[-2].hlines(y=57, xmin=1, xmax=24, colors="black", ls="--") 

    if is_prev:
        cccc= "white" if do_white else "black"
        for ax in axes:
            ax.tick_params(axis="both", color=cccc, width=fig_height / 10, length=fig_height / 2, pad=fig_height / 2, labelsize=fig_height * 2)
            ax.yaxis.label.set_color(cccc)
            ax.set_yticklabels(ax.get_yticklabels(), color=cccc)
            ax.set_xticklabels(ax.get_xticklabels(), color=cccc)
            for side in ["bottom", "left", "top", "right"]:
                ax.spines[side].set_color(cccc)

        xticks_temp = xticks_temp[::2]
        axes[-1].set_xticks(xticks_temp)
        axes[-1].set_xticklabels(xticks_temp[::-1] * -1, color=cccc)



    if save_file_name is not None:
        plt.savefig(
            Path(__file__).resolve().parents[1] / "data" / "output" / f"{save_file_name}.png",
            transparent=True,
            dpi=dpi,
            pad_inches=0,
            bbox_inches="tight",
        )
        return None
    return fig, axes