"""outlier.py: Module for visualizing and assessing outliers statistically."""
__author__ = "Ben Harding"
__version__ = "0.0.1"
__date__ = "2021/09/04"
__maintainer__ = "Ben Harding"
__email__ = "bharding@ualberta.ca"


import warnings
import probscale

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy import stats
from matplotlib.colors import ListedColormap

from pygeostat import location_plot
from pygeostat import histogram_plot
from pygeostat.statistics.cdf import cdf, percentile_from_cdf

from nscore import NormalScoreTransformer
from variogram import VariogramModel


class Outlier:

    # add probabilities of high grade intercepts based on CVD note

    def __init__(self, data, x=None, y=None, z=None, wts=None, length=None):
        """Initialize the Outlier class.

        :param data: file with data
        :type data: `pd.DataFrame`
        :param x: column for x coordinate, defaults to None
        :type x: str, optional
        :param y: column for y coordinate, defaults to None
        :type y: str, optional
        :param z: column for z coordinate, defaults to None
        :type z: str, optional
        :param wts: column for weights, defaults to None
        :type wts: str, optional
        :param length: column for length, defaults to None
        :type length: str, optional
        :raises ValueError: if data is not `pd.DataFrame`
        """
        self.data = data
        self.x = x
        self.y = y
        self.z = z
        self.wts = wts
        self.length = length
        self.thresh = None
        self.parrish_thresh = None
        self.parrish_perc = None
        self.parrish_prev_perc = None
        self.parrish_flag = None
        self._covariance_matrix = None
        self.ns_data = None
        self.sbs_reals = None

        # TODO check columns exist before assigning attributes

        if not isinstance(data, pd.DataFrame):
            raise ValueError("data must be a pd.DataFrame")

    def set_threshold(self, thresh):
        """Set an arbitrary threshold for outlier identification."""
        if thresh < 2:
            self.thresh = thresh
        else:
            self.thresh = np.round(thresh, 0)

    def parrish_table(self, var, length, lower_thresh=None, color="LightCoral"):
        """Length weighted decile analysis after Parrish (1997). Capping is recommended if
        the upper decile contains more than 40% of the total metal, the upper decile contains 
        more than 2x the total metal of the previous decile or the upper percentile 
        contains more than 10% of the total metal. 

        :param var: variable column
        :type var: str
        :param length: length column
        :type length: str
        :param lower_thresh: data below this threshold are trimmed, defaults to None
        :type lower_thresh: float, optional
        :param color: color for table highlighting, defaults to "LightCoral"
        :type color: str, optional
        :raises ValueError: if `lower_thresh` is not int or float
        :raises ValueError: if len(data) < 100
        :return: styled `pd.DataFrame`
        :rtype: `pandas.io.formats.style.Styler`
        """

        df_sort = self.data.sort_values(by=[var]).copy()
        df_sort.dropna(subset=[var], inplace=True)

        if lower_thresh is not None:
            if not isinstance(lower_thresh, (int, float)):
                raise ValueError("lower threshold should be a single number")
            df_sort = df_sort.loc[df_sort[var] > lower_thresh]

        ndec = int(np.ceil(len(df_sort) / 10))
        if ndec < 10:
            raise ValueError("decile analysis should consider at least 100 samples")

        df_sort["Metal"] = df_sort[var] * df_sort[length]
        df_sort["Decile"] = 0

        for i in range(10):
            df_sort.iloc[i * ndec : (i + 1) * ndec, -1] = i

        percs = df_sort.loc[df_sort["Decile"] == 9].copy()
        percs["Metal"] = percs[var] * percs[length]
        nper = int(np.ceil(len(percs) / 10))

        for i in range(10):
            percs.iloc[i * nper : (i + 1) * nper, -1] = 90 + i

        def wtd_mean(var, wts):
            return np.average(var, weights=wts)

        def wtd_var(var, wts):
            return np.average((var - wtd_mean(var, wts)) ** 2, weights=wts)

        # summary stats of the deciles
        dec_summ = df_sort[[var, "Decile"]].groupby(by="Decile").describe()
        # wtd mean
        dec_summ.iloc[:, 1] = (
            df_sort[[var, length, "Decile"]]
            .groupby(by="Decile")
            .apply(lambda x: wtd_mean(x[var], wts=x[length]))
        )
        # wtd stdev
        dec_summ.iloc[:, 2] = (
            df_sort[[var, length, "Decile"]]
            .groupby(by="Decile")
            .apply(lambda x: np.sqrt(wtd_var(x[var], wts=x[length])))
        )
        dec_summ.columns = dec_summ.columns.droplevel()

        # summary stats of the percentiles
        per_summ = percs[[var, "Decile"]].groupby(by="Decile").describe()
        # wtd mean
        per_summ.iloc[:, 1] = (
            percs[[var, length, "Decile"]]
            .groupby(by="Decile")
            .apply(lambda x: wtd_mean(x[var], wts=x[length]))
        )
        # wtd stdev
        per_summ.iloc[:, 2] = (
            percs[[var, length, "Decile"]]
            .groupby(by="Decile")
            .apply(lambda x: np.sqrt(wtd_var(x[var], wts=x[length])))
        )
        per_summ.columns = per_summ.columns.droplevel()

        dec_metal = df_sort[["Metal", "Decile"]].groupby(by="Decile").sum()
        per_metal = percs[["Metal", "Decile"]].groupby(by="Decile").sum()

        dec_metal["% Total"] = dec_metal["Metal"] / sum(dec_metal["Metal"]) * 100
        per_metal["% Total"] = per_metal["Metal"] / sum(dec_metal["Metal"]) * 100

        dec = pd.concat([dec_summ, dec_metal], axis=1)
        per = pd.concat([per_summ, per_metal], axis=1)

        summ = df_sort.describe().transpose().loc[[var]]
        summ["Metal"] = sum(dec_metal["Metal"])
        summ["% Total"] = sum(dec_metal["% Total"])
        summ.index = ["TOTAL"]
        summ.index.name = "Decile"

        df = pd.concat([dec, per, summ])

        # set flags for exceedances
        # upper decile
        dec8 = df.loc[df.index == 8].copy()
        dec9 = df.loc[df.index == 9].copy()
        flag1 = dec9["% Total"].values >= 40
        flag2 = dec9["% Total"].values >= 2 * dec8["% Total"].values

        # upper percentile
        perc = df.iloc[-2, :].copy()
        prev_perc = df.iloc[-3, :].copy()
        n99 = perc["count"]
        n98 = prev_perc["count"]
        thresh = prev_perc["max"]
        flag3 = perc["% Total"] >= n99 / n98 * 10

        self.parrish_perc = perc
        self.parrish_prev_perc = prev_perc
        if flag3:
            self.parrish_thresh = np.round(thresh, 0)
            self.parrish_flag = flag3

        # warning messages
        if flag1:
            print(
                "The upper decile contains more than 40% of the total metal: capping is recommended\n"
            )
        if flag2:
            print(
                "The upper decile contains more than 2x the total metal of the previous decile: capping is recommended\n"
            )
        if flag3:
            print(
                "The upper percentile contains more than 10% of the total metal (weighted by count): capping is recommended\n"
            )

        formats = ["{:,.0f}", *("{:.2f} " * 7).split(" ")[:-1], "{:,.1f}", "{:.1f}"]
        format_dict = {col: fmt for (col, fmt) in zip(df.columns, formats)}

        table = df.style.apply(
            self._highlight_parrish_rows,
            flag1=flag1,
            flag2=flag2,
            flag3=flag3,
            color=color,
            axis=None,
        ).format(format_dict)

        return table

    def cutting_curve(
        self,
        var,
        num_thresh=50,
        plot_thresh=None,
        logx=False,
        title=None,
        figsize=None,
        **kwargs,
    ):
        """Cutting curve after Roscoe (1996). This plot is used to determine
        a threshold where the average capped grade plateaus and stabilizes. A
        reasonable capping threshold is near the inflection point of this plot.

        :param var: variable column
        :type var: str
        :param num_thresh: number of thresholds to consider, defaults to 50
        :type num_thresh: int, optional
        :param plot_thresh: value of capping threshold to plot, defaults to None
        :type plot_thresh: float, optional
        :param logx: log scaling of xaxis, defaults to False
        :type logx: bool, optional
        :param title: figure title, defaults to None
        :type title: str, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        # TODO: weight samples by length

        assert num_thresh > 1, "The number of thresholds must be greater than one"

        df = self.data.copy()
        df.dropna(subset=[var], inplace=True)
        thresh = np.linspace(df[var].min(), df[var].max(), num_thresh)
        averages = [np.where(df[var] >= t, t, df[var]).mean() for t in thresh]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresh, averages, **kwargs)
        ax.grid(ls=":", which="both")
        ax.set_xlabel(f"{var} Thresholds",)
        ax.set_ylabel(f"{var} Average Grade",)
        ax.set_title(title)

        if plot_thresh is not None:
            ax.axvline(plot_thresh, c="r", ls="--", alpha=0.75)
            ax.text(
                plot_thresh - plot_thresh * 0.025,
                0.1,
                f"Capping Limit: {np.round(plot_thresh,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.get_xaxis_transform(),
            )

        if logx:
            ax.set_xscale("log")

        return fig, ax

    def metal_removed(
        self,
        var,
        length,
        num_thresh=50,
        z=None,
        p=None,
        title=None,
        figsize=None,
        line_kws={"colors": "r", "linestyles": "--"},
        **kwargs,
    ):
        """Plot percentage of total metal removed by grade capping threshold.

        :param var: variable column
        :type var: str
        :param length: length column
        :type length: str
        :param num_thresh: number of thresholds to consider, defaults to 50
        :type num_thresh: int, optional
        :param z: grade threshold used to calculate % metal removed, defaults to None
        :type z: float, optional
        :param p: percentage of metal removed to calculate grade threshold, defaults to None
        :type p: float, optional
        :param title: figure title, defaults to None
        :type title: string, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :param line_kws: permissible keywords to pass to `ax.hlines` and `ax.vlines`, defaults to {"colors": "r", "linestyles": "--"}
        :type line_kws: dict, optional
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        df = self.data.copy()
        df.dropna(subset=[var], inplace=True)
        thresh = np.linspace(df[var].min(), df[var].max(), num_thresh)

        if length is None:
            df["length"] = 1.0
            length = "length"

        df["metal"] = df[var] * df[length]
        total_metal = df["metal"].sum()
        metal = [
            np.where(df[var] <= t, 0, df["metal"]).sum() / total_metal * 100
            for t in thresh
        ]

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(thresh, metal, zorder=-1, **kwargs)
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()
        ax.grid(ls=":", which="both")
        ax.set_xlabel(f"{var} Thresholds")
        ax.set_ylabel(f"% Metal Removed")
        ax.set_title(title)

        # calculate corresponding percent metal removed give a grade threshold
        if z is not None:
            perc_from_z = np.interp(z, thresh, metal)
            ax.vlines(z, ymin=ylim[0], ymax=perc_from_z, **line_kws)
            ax.hlines(perc_from_z, xmin=xlim[0], xmax=z, **line_kws)
            ax.text(
                0.0,
                perc_from_z + 1,
                f"Metal Removed: {np.round(perc_from_z,1)}%",
                rotation="horizontal",
                rotation_mode="anchor",
            )
            ax.text(
                z - z * 0.01,
                0.025,
                f"{np.round(z,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.get_xaxis_transform(),
            )
        # calculate corresponding grade given a percent metal removed
        if p is not None:
            z_from_perc = np.interp(p, np.flip(metal), np.flip(thresh))
            ax.vlines(z_from_perc, ymin=ylim[0], ymax=p, **line_kws)
            ax.hlines(p, xmin=xlim[0], xmax=z_from_perc, **line_kws)
            ax.text(
                z_from_perc - z_from_perc * 0.01,
                0.025,
                f"{np.round(z_from_perc,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.get_xaxis_transform(),
            )
            ax.text(
                0.0,
                p + 1,
                f"Metal Removed: {np.round(p,1)}%",
                rotation="horizontal",
                rotation_mode="anchor",
            )
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        return fig, ax

    def probplot(
        self,
        var,
        wts=None,
        thresh=None,
        highlight=False,
        logx=False,
        tukeys_fences=False,
        figsize=None,
        scatter_kws={"ms": 5, "marker": ".", "c": "k", "mec": "k", "alpha": 0.7},
    ):
        """Log or normal probability plot with outlier thresholds.

        :param var: variable column
        :type var: str
        :param wts: weight column, defaults to None
        :type wts: str, optional
        :param thresh: value of capping threshold to plot, defaults to None
        :type thresh: float, optional
        :param highlight: highlight points above threshold?, defaults to False
        :type highlight: bool, optional
        :param logx: log scaling of xaxis, defaults to False
        :type logx: bool, optional
        :param tukeys_fences: plot Tukey's fences?, defaults to False
        :type tukeys_fences: bool, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :param scatter_kws: scatter plot parameters, defaults to {"ms": 5, "marker": ".", "c": "k", "mec": "k", "alpha": 0.7}
        :type scatter_kws: dict, optional
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        warnings.filterwarnings("ignore")

        if wts is None:
            wts = np.ones(len(self.data[var]))
        else:
            wts = np.array(self.data[wts])
        cdf_x, cdfvals = cdf(self.data[var], weights=wts)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(cdf_x, cdfvals * 100, **scatter_kws)

        if thresh is not None:
            if highlight:
                idxs = np.where(cdf_x >= thresh)[0]
                if len(idxs) > 0:
                    ax.plot(
                        cdf_x[idxs],
                        cdfvals[idxs] * 100,
                        c="r",
                        ms=12,
                        ls="None",
                        marker=".",
                        alpha=0.15,
                    )
            ax.axvline(thresh, c="r", ls="--", alpha=0.75)
            ax.text(
                thresh - thresh * 0.025,
                0.1,
                f"Capping Limit: {np.round(thresh,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.get_xaxis_transform(),
            )

        if tukeys_fences:
            out, farout = self._tukeys_fences(cdf_x, cdfvals)
            low_outliers = True
            if np.isnan(out[0]):
                low_outliers = False

            xlim = ax.get_xlim()
            ax.axvspan(out[1], farout[1], color="C0", alpha=0.25, label="Outlier")
            ax.axvspan(farout[1], xlim[1], color="C1", alpha=0.25, label="Far Outlier")

            if low_outliers:
                if np.isnan(farout[0]):
                    ax.axvspan(out[0], xlim[0], color="C0", alpha=0.25)
                else:
                    ax.axvspan(out[0], farout[0], color="C0", alpha=0.25)
                    ax.axvspan(farout[0], xlim[0], color="C1", alpha=0.25)

        if logx:
            ax.set_xscale("log")
        if tukeys_fences:
            ax.set_xlim(xlim)
            ax.legend()

        ax.set_yscale("prob")
        # ax.set_ylim(0.001, 99.999)
        ax.set_xlabel(var,)
        ax.set_ylabel("Cummulative Probability",)
        ax.grid(ls=":", which="both")

        return fig, ax

    def sectionplot(
        self,
        var,
        x=None,
        y=None,
        z=None,
        thresh=None,
        orient="xy",
        figsize=None,
        pt_kws={"alpha": 0.15, "s": 10, "c": "k", "label": "Inlier"},
        out_kws={"alpha": 0.75, "s": 20, "ec": "k", "c": "r", "label": "Outlier"},
        **kwargs,
    ):
        """Section plots with optional thresholds for outlier identification.

        :param var: variable column 
        :type var: str
        :param x: column for x coordinate, defaults to None
        :type x: str, optional
        :param y: column for y coordinate, defaults to None
        :type y: str, optional
        :param z: column for z coordinate, defaults to None
        :type z: str, optional
        :param thresh: outlier threshold, defaults to None
        :type thresh: float, optional
        :param orient: section orientation, one of ['xy', 'xz', 'yz'], defaults to "xy"
        :type orient: str, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :param pt_kws: plotting parameters for inlier points, defaults to {"alpha": 0.15, "s": 10, "c": "k", "label": "Inlier"}
        :type pt_kws: dict, optional
        :param out_kws: plotting parameters for outlier points, defaults to {"alpha": 0.75, "s": 20, "ec": "k", "c": "r", "label": "Outlier"}
        :type out_kws: dict, optional
        :raises ValueError: if spatial coordinates are not passed to this method nor on class initialization
        :raises ValueError: if no threshold is set for outlier identification
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        if x is None and self.x is None:
            raise ValueError(
                "coordinates must be passed to this method or on class initialization"
            )

        if x is None:
            x = self.x
        if y is None:
            y = self.y
        if z is None:
            z = self.z

        if thresh is None and self.parrish_thresh is None and self.thresh is None:
            raise ValueError("no threshold for outlier identification is set")

        df = self.data.copy()
        df["outlier"] = 0.0
        mask = df[var] >= thresh
        df.loc[mask, "outlier"] = 1.0
        nout = df["outlier"].sum()
        cmap_in = ListedColormap([pt_kws["c"], out_kws["c"]])
        cmap_out = ListedColormap([out_kws["c"], pt_kws["c"]])

        # drop color from kwargs as it causes issues with location_plot
        pt_kws = {k: v for (k, v) in pt_kws.items() if k not in ["c"]}
        out_kws = {k: v for (k, v) in out_kws.items() if k not in ["c"]}

        fig, ax = plt.subplots(figsize=figsize)
        if nout > 1:
            location_plot(
                df.loc[df["outlier"] == 1],
                var="outlier",
                x=x,
                y=y,
                z=z,
                orient=orient,
                ax=ax,
                cmap=cmap_out,
                cbar=False,
                zorder=5,
                **out_kws,
                **kwargs,
            )
        # _, cbar = location_plot(
        #     df.loc[df["outlier"] == 0],
        #     var="outlier",
        #     x=x,
        #     y=y,
        #     z=z,
        #     orient=orient,
        #     ax=ax,
        #     cmap=cmap_in,
        #     zorder=1,
        #     return_cbar=True,
        #     **pt_kws,
        #     **kwargs,
        # )
        location_plot(
            df.loc[df["outlier"] == 0],
            var="outlier",
            x=x,
            y=y,
            z=z,
            orient=orient,
            ax=ax,
            cmap=cmap_in,
            zorder=1,
            return_cbar=False,
            cbar=False,
            **pt_kws,
            **kwargs,
        )
        ax.grid(":", zorder=0)
        ax.legend()
        # cbar.solids.set(alpha=1)
        # cbar.set_ticks([0.25, 0.75])
        # cbar.set_ticklabels([pt_kws["label"], out_kws["label"]])

        return fig, ax

    def cumcv(self, var, length=None, lower_thresh=None):
        """Calculate the cumulative coefficient of variation after Parker (1991).
        The data is sorted and cumulative metal is calcuated bottom-up while 
        cumulative CV is calculated top-down.

        :param var: variable column
        :type var: str
        :param length: length column, defaults to None
        :type length: str, optional
        :param lower_thresh: data below this threshold are trimmed, defaults to None
        :type lower_thresh: float, optional
        :raises ValueError: if lower threshold is not int or float
        :return: pandas DataFrame
        :rtype: `pd.DataFrame`
        """
        # CV bottom up
        # cum metal top down

        df = self.data.copy()
        df.dropna(subset=[var], inplace=True)

        if lower_thresh is not None:
            if not isinstance(lower_thresh, (int, float)):
                raise ValueError("lower threshold should be a single number")
            df = df.loc[df[var] > lower_thresh]

        if length is not None:
            df = df[[var, length]]
            df["grade_x_length"] = df[var] * df[length]
        else:
            df = df[[var]]

        df.sort_values(var, inplace=True, ascending=True)
        df["cum_mean"] = df[var].rolling(len(df), min_periods=2).mean()
        df["cum_std"] = df[var].rolling(len(df), min_periods=2).std()
        df["cum_cv"] = df["cum_std"] / df["cum_mean"]

        df.sort_values(var, inplace=True, ascending=False)
        if length is not None:
            total = df["grade_x_length"].sum()
            df["cum_metal"] = df["grade_x_length"].cumsum() / total * 100
        else:
            total = df[var].sum()
            df["cum_metal"] = df[var].cumsum() / total * 100

        df.reset_index(inplace=True, drop=True)

        return df

    def cumcv_plot(
        self,
        var,
        length=None,
        lower_thresh=None,
        plot_thresh=None,
        figsize=None,
        **kwargs,
    ):
        """Cumulative coefficient of variation after plot Parker (1991). The data is 
        sorted and cumulative metal is calcuated bottom-up while cumulative CV is 
        calculated top-down.

        :param var: variable column
        :type var: str
        :param length: length column, defaults to None
        :type length: str, optional
        :param lower_thresh: data below this threshold are trimmed, defaults to None
        :type lower_thresh: float, optional
        :param plot_thresh: value of capping threshold to plot, defaults to None
        :type plot_thresh: float, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        ccv = self.cumcv(var, length, lower_thresh)
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(ccv[var], ccv["cum_cv"], **kwargs)
        if plot_thresh is not None:
            ax.axvline(plot_thresh, c="r", ls="--", alpha=0.75)
            ax.text(
                plot_thresh - plot_thresh * 0.025,
                0.1,
                f"Capping Limit: {np.round(plot_thresh,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_ylabel("Cumulative CV")
        ax.set_xlabel(var)
        ax.grid(ls=":")

        return fig, ax

    def mean_uncertainty(
        self,
        var,
        wts=None,
        vargstr=None,
        nreals=1000,
        perc_to_cap=0.0,
        seed=123456,
        reals_to_plot=50,
        logx=False,
        icdf=0,
        figsize=None,
    ):
        """Assess mean uncertainty through the spatial bootstrap. Provided variogram model 
        should be of the normal scores. Optionally exclude data above a given percentile 
        to assess upper tail sensitivity after Nowak (2013). 

        :param var: variable column
        :type var: str
        :param wts: weight column, defaults to None
        :type wts: str, optional
        :param vargstr: GSLIB-style variogram string, defaults to None
        :type vargstr: str, optional
        :param nreals: number of realizations defaults to 1000
        :type nreals: int, optional
        :param perc_to_cap: omit data above 100 - `perc_to_cap`; a value of 2 omits data above the 98th percentile , defaults to 0
        :type perc_to_cap: float, optional
        :param seed: random number seed, defaults to 123456
        :type seed: int, optional
        :param reals_to_plot: number of realizations to plot, defaults to 50
        :type reals_to_plot: int, optional
        :param logx: log scaling of xaxis, defaults to False
        :type logx: bool, optional
        :param icdf: empirical CDF indicator; 0=histogram, 1=CDF, defaults to 0
        :type icdf: int, optional
        :param figsize: figure size, defaults to None
        :type figsize: tuple, optional
        :return: matplotlib Figure and Axes object
        :rtype: (`Figure`, `AxesSubplot`)
        """

        if wts is None and self.wts is None:
            warnings.warn("declustering weights are recommended for bootstrapping!")
            wts = np.ones(len(self.data[var]))
        else:
            wts = np.array(self.data[wts])

        # remove potential capping candidates to assess influence on mean
        perc_to_cap = perc_to_cap / 100
        capped = self.data.copy().sort_values(by=var)
        _, capped["perc"] = cdf(self.data[var], wts)
        capped = capped.loc[capped["perc"] <= 1 - perc_to_cap]

        # bootstrap with "capped" data
        self._covariance_matrix = None
        boot_reals = self._spatial_bootstrap(
            capped, var, vargstr, capped[self.wts], nreals, seed,
        )
        self.sbs_reals = boot_reals
        boot_means = np.mean(boot_reals, axis=0)

        # reset the covariance matrix if we dropped any samples
        self._covariance_matrix = None

        fig, ax = plt.subplots(1, 2, figsize=figsize)
        histogram_plot(boot_means, icdf=icdf, xlabel=f"{var} mean", ax=ax[0])
        histogram_plot(
            capped[var],
            weights=capped[self.wts],
            ax=ax[1],
            icdf=1,
            color="r",
            lw=1.5,
            zorder=5,
        )
        # plot uncapped mean for comparison
        if perc_to_cap > 0:
            mean = np.average(self.data[var], weights=wts)
            ax[0].axvline(mean, c="r", ls="--", alpha=0.75)
            ax[0].text(
                mean - mean * 0.025,
                0.1,
                f"Uncapped Mean: {np.round(mean,2)}",
                rotation="vertical",
                rotation_mode="anchor",
                transform=ax[0].get_xaxis_transform(),
            )

        if reals_to_plot > nreals:
            reals_to_plot = nreals

        for i in range(reals_to_plot):
            histogram_plot(
                boot_reals[:, i],
                weights=None,
                logx=logx,
                icdf=1,
                color="gray",
                stat_blk=False,
                ax=ax[1],
                lw=1.0,
                alpha=0.5,
            )
        ax[0].grid(ls=":")
        ax[1].grid(ls=":", which="both")

        if logx:
            ax[1].set_xscale("log")

        return fig, ax

    def metal_at_risk(
        self, var, wts=None, nsamples=None, thresh=None, nreals=1000, seed=123456,
    ):
        """Calcualte metal-at-risk after Parker (2006). This function resamples the 
        declustered CDF and calculates the metal above `thresh`. The P20 of this 
        distribution is added to the metal below `thresh` to calcualte a risk-adjusted
        metal. Metal-at-risk is the difference between total metal and risk-adjusted.

        `nsamples` is equal to the tonnes per annual production volume / number of 
        tonnes per assay in the domain. Metal-at-risk accounts for the data density; as 
        `nsamples` increases, metal at risk decreases.

        :param var: variable column
        :type var: str
        :param wts: weight column, defaults to None
        :type wts: str, optional
        :param nsamples: number of samples to draw from the distribution, defaults to None
        :type nsamples: int, optional
        :param thresh: threshold to destinguish between HG/LG, defaults to None
        :type thresh: float, optional
        :param nreals: number of realizations, defaults to 1000
        :type nreals: int, optional
        :param seed: random number seed, defaults to 123456
        :type seed: int, optional
        :raises ValueError: if no threshold is defined
        :raises ValueError: if threshold is not int or float
        :return: dictionatry containing metal-at-risk and summary statistics
        :rtype: dict
        """

        # if self.x is None:
        #     raise ValueError("coordinates should be passed on class initilization")

        # if not isinstance(vargstr, str):
        #     raise ValueError("variogram model should be a GSLIB style string")

        if thresh is None:
            raise ValueError("a threshold is required to assess metal at risk")

        df = self.data.copy().dropna(subset=[var])

        if wts is None and self.wts is None:
            warnings.warn("declustering weights are recommended for CDF resampling!")
            wts = np.ones(len(df[var]))
        else:
            wts = np.array(df[wts])

        # bootstrap = self._spatial_bootstrap(self.data, var, vargstr, wts, nreals, seed)

        cdf_x, cdfvals = cdf(
            df[var],
            weights=wts,
            lower=df[var].min() - 0.001,
            upper=df[var].max() + 0.001,
        )

        # subsampling for metal at risk calculation
        rng = np.random.default_rng(seed)
        if not isinstance(nsamples, (int, float)):
            raise ValueError("'nsamples' must be an integer")
        resample = np.zeros((nsamples, nreals))

        for i in range(nreals):

            # idxs = rng.integers(low=0, high=len(bootstrap), size=nsamples)
            # resample[:, i] = bootstrap[:, i][idxs]

            # resample[:, i] = rng.choice(
            #     cdf_x, size=nsamples, replace=True, p=cdfvals / np.sum(cdfvals),
            # )

            pi = rng.choice(rng.random(nsamples), size=nsamples, replace=True)
            resample[:, i] = percentile_from_cdf(cdf_x, cdfvals, pi * 100)

        # calculate distribution of metal above threshold from bootstrap realizations
        high = np.where(resample >= thresh, resample, 0)
        high_metal = np.sum(high, axis=0)

        # low = np.where(resample < thresh, resample, 0)
        # low_metal = np.sum(low, axis=0)

        low = np.where(cdf_x < thresh, cdf_x, 0)
        low_metal = np.sum(low)

        # total_metal = np.sum(resample, axis=0)
        total_metal = np.sum(cdf_x)

        # get P20 of this distribution based on description from:
        # Leuangthong, O., & Nowak, M. (2015). Dealing with high‐grade data in resource estimation.
        cdf_x, cdfvals = cdf(high_metal)
        xval = percentile_from_cdf(cdf_x, cdfvals, 20)

        # risk adjusted metal is metal below threshold + P20 value
        ram = low_metal + xval

        # metal at risk the is differnce between total metal and RAM
        # mar = np.mean((total_metal - ram) / ram) * 100
        mar = (total_metal - ram) / ram * 100

        prop_cog = np.sum(np.where(resample >= thresh, 1, 0), axis=0)
        exp_cog = int(np.ceil(np.mean(prop_cog)))

        risk = {
            "Number of Samples": nsamples,
            "HG Threshold": thresh,
            "Expected HG Samples": exp_cog,
            "Metal at Risk %": np.round(mar, 2),
        }

        return risk

    def _spatial_bootstrap(
        self, data, var, vargstr, wts=None, nreals=1000, seed=123456,
    ):
        """spatial bootstrap"""

        # seed the random state
        rng = np.random.default_rng(seed)

        # get data locations for covariance calculation
        data.dropna(subset=[var], inplace=True)
        x = data[self.x].values.reshape(-1, 1)
        y = data[self.y].values.reshape(-1, 1)
        if self.z is None:
            z = np.ones_like(x) * 0.5
        else:
            z = data[self.z].values.reshape(-1, 1)
        points = np.hstack((x, y, z))

        # initialize the variogram model and get covariance matrix
        vario = VariogramModel(vargstr)
        vario.setcova()

        # check if cmat has been intialized, and calculate if not
        if self._covariance_matrix is None:
            cmat = vario.pairwisecova(points)
            self._covariance_matrix = cmat
        else:
            cmat = self._covariance_matrix

        # normal score transform of "var" for reference
        if wts is None:
            wts = np.ones_like(x)
        tail_values = (data[var].min() - 0.001, data[var].max() + 0.001)
        nst = NormalScoreTransformer()
        yvals = nst.transform(data[var], wts=wts, tail_values=tail_values)
        self.ns_data = yvals

        # cholesky decomposition of covariance matrix
        L = np.linalg.cholesky(cmat)

        # bootstrap the transformed data
        ndata = points.shape[0]
        bootstrap = np.zeros((ndata, nreals))
        cdf_x, cdfvals = cdf(
            data[var], weights=wts, lower=tail_values[0], upper=tail_values[-1],
        )

        for i in range(nreals):
            # yi = L @ rng.normal(0, 1, ndata)
            yi = L @ rng.choice(rng.normal(0, 1, ndata), size=ndata, replace=True)
            pi = stats.norm.cdf(yi)
            zi = percentile_from_cdf(cdf_x, cdfvals, pi * 100)
            bootstrap[:, i] = zi

        return bootstrap

    def _tukeys_fences(self, cdf_x, cdfvals):
        """get limits of Tukey's fences"""

        q1 = percentile_from_cdf(cdf_x, cdfvals, 25)
        q3 = percentile_from_cdf(cdf_x, cdfvals, 75)
        out = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)]
        farout = [q1 - 3.0 * (q3 - q1), q3 + 3.0 * (q3 - q1)]

        if out[0] < 0.0:
            out[0] = np.nan
        if farout[0] < 0.0:
            farout[0] = np.nan

        return out, farout

    def _highlight_parrish_rows(self, df, flag1, flag2, flag3, color):
        """generate pandas styler for Parrish table method"""

        ncol = df.shape[1]
        style_dict = {row: [""] * ncol for row in df.index}

        if flag1:
            style_dict[9] = [f"background-color: {color}; font-weight: bold"] * ncol

        if flag2:
            style_dict[9] = [f"background-color: {color}; font-weight: bold"] * ncol

        if flag3:
            style_dict[self.parrish_perc.name] = [
                f"background-color: {color}; font-weight: bold"
            ] * ncol
            style_dict[self.parrish_prev_perc.name][
                7
            ] = f"background-color: {color}; font-weight: bold"  # previous max

        return pd.DataFrame.from_dict(style_dict, "index", columns=df.columns)
