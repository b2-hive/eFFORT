import itertools
import json
import pathlib
import shutil
import urllib.request
from typing import List, Tuple

import click
import numpy as np
import pandas as pd
from scipy.interpolate import Rbf


class RbfReweighter:
    """RbfReweighter implements a method to calculate weights used
    to reweight monte carlo based on D dimensional histograms.

    Both, for the original and target monte carlo type, D dimensional
    histograms are calculated using the given variables, number of bins
    and limits for the bin edges.
    Bin-wise weights are determined by dividing the distribution of the
    target MC by the original MC. The weights for bins, where the
    denominator is zero are set to zero.
    Each bin weight is assigned to some coordinates corresponding to a
    D dimensional vector pointing to the bin mid. The resulting
    D dimensional weight function is given by an interpolating between
    these bins using the radial basis function method implemented in
    scipy. To ensure normalization, a scale factor is calculated on the
    given MC which ensure the normalization in the number of events
    after reweighting. It might happen that the weights take negative
    values due to the interpolation process. These weights are set to
    zero.

    Parameters
    ----------
    variables :
        List of variable names. Must match the column names
        in the DataFrames later on.
    num_bins :
        List of integers denoting the number of bins in the
        histogram axis for each variable.
    limits :
        List of Tuple of floats denoting the minimum and maximum
        values the bin edges for each histogram axis.

    Returns
    -------
        None

    Examples
    --------
    This example shows how to create a reweighter instance to reweight B->D**lv decays
    from the ISGW2 to the LLSW form factor model

   load a DataFrame that contains generator level information for the ISGW2 and LLSW model
    >>> isgw2_done = pd.read_hdf("BtoDstarstarLNu_ISGW2_Data/Done.h5")
    >>> llsw_done = pd.read_hdf("BtoDstarstarLNu_LLSW_Data/Done_nominal.h5")

   Create a `RBFReweighter` instance to calculate weights based on three kinematic variables
    >>> done_rbf = RbfReweighter(["w", "costhetal", "costhetanu"], num_bins=[10, 10, 10], limits=[(1., 1.42), (-1., 1.), (-1.,1.)])
    >>> done_rbf.create_interpolation(isgw2_done, llsw_done)

    export it in json format which allows to reload it later
    >>> done_rbf.export_to_json("done_nominal.json"))

    The created model can be reloaded after it has been exported
    >>> done_rbf = RbfReweighter.import_from_json("done_nominal.json")

    To calculate weights for events in a given DataFrame, just pass the corresponding
    values as arguments to a call to the `RBFReweighter` instance (the arguments ahve to
    be `numpy.array` instances)
    >>> done_nominal_weights = done_rbf(df["w"].values, df["costhetal"].values, df["costhetanu"].values)


    """

    def __init__(self, variables: List[str], num_bins: List[int], limits: List[Tuple[float, float]]) -> None:
        self._variables = variables
        self._num_bins = num_bins
        self._limits = limits
        self._bin_edges = [np.linspace(*limit, nbins + 1) for limit, nbins in zip(self._limits, self._num_bins)]
        self._bin_mids = list(map(lambda x: (x[:-1] + x[1:]) / 2, self._bin_edges))

        # create an array holding the weight coordinates as the cartesian product of the bin mids
        self._weight_coords = np.array(list(itertools.product(*self._bin_mids)))

        self._bc_origin = None
        self._bc_target = None
        self._hist_weights = None

        self._num_evts_origin = None
        self._num_evts_target = None

        self._scale_factor = None
        self._rbf = None

    def create_interpolation(self, origin: pd.DataFrame, target: pd.DataFrame):
        """Creates the RBF instance and calculates the normalization
        factor. Has to called once after the class is created for the
        first time. If the class is instantiated from json, this step
        can be omitted

        Parameters
        ----------
        origin :
            DataFrame with the original MC.
        target :
            DataFrame with the target MC.

        Returns
        -------
            None
        """
        self._num_evts_origin = len(origin.index)
        self._num_evts_target = len(target.index)
        self._bc_origin, _ = np.histogramdd(
            origin[self._variables].values, bins=self._bin_edges, density=True
        )
        self._bc_target, _ = np.histogramdd(
            target[self._variables].values, bins=self._bin_edges, density=True
        )
        self._hist_weights = np.divide(
            self._bc_target,
            self._bc_origin,
            out=np.zeros_like(self._bc_origin),
            where=self._bc_origin != 0.0
        ).flatten()

        self._create_rbf()
        self._get_scale_factor(origin)

    def _get_scale_factor(self, origin_sample: pd.DataFrame):
        """Calculates the scale factor used for calculating the weights.
        The scale factor is used to prevent a change in the overall
        normalization and corresponds to the ratio of decay rates in the
        standard weight formula.

        Parameters
        ----------
        origin_sample : Original pd.DataFrame used as basis for the
            reweighting


        Returns
        -------
            None
        """
        weights = self._rbf(*[origin_sample[variable].values for variable in self._variables])
        weights[weights < 0] = 0.0
        num_origin_events = len(origin_sample.index)
        self._scale_factor = num_origin_events / np.sum(weights)

    def _create_rbf(self) -> None:
        """Creates the rbf instance used for interpolating between the
        histogram ratio.

        Returns
        -------
            None
        """
        self._rbf = Rbf(
            *[self._weight_coords[:, i] for i in range(len(self._variables))],
            self._hist_weights,
            function="cubic"
        )

    def export_to_json(self, fname: str) -> None:
        """Exports the necessary information of this class to a
        json file.

        Parameters
        ----------
        fname :
            Absolute path to the json file.

        Returns
        -------
            None

        """
        data = {
            "variables": self._variables,
            "num_bins": self._num_bins,
            "limits": self._limits,
            "num_events": {
                "origin": self._num_evts_origin,
                "target": self._num_evts_target
            },
            "bin_edges": [bin_edges.tolist() for bin_edges in self._bin_edges],
            "bin_mids:": [bin_mids.tolist() for bin_mids in self._bin_mids],
            "bc_origin": self._bc_origin.tolist(),
            "bc_target": self._bc_target.tolist(),
            "hist_weights": self._hist_weights.tolist(),
            "scale_factor": self._scale_factor,
        }

        with open(fname, "w") as write_file:
            json.dump(data, write_file, indent=4)

    @classmethod
    def import_from_json(cls, fname: str) -> 'RbfReweighter':
        """Creates a class instance from a given json file.

        Parameters
        ----------
        fname :
            Absolute path to the json file.

        Returns
        -------
        rbf_reweighter_instance:
            Loaded `RBFReweighter` instance.

        """
        with open(fname, "r") as read_file:
            data = json.load(read_file)

        instance = cls(data["variables"], data["num_bins"], data["limits"])
        instance._bc_origin = data["bc_origin"]
        instance._bc_target = data["bc_target"]
        instance._hist_weights = data["hist_weights"]
        instance._create_rbf()
        instance._scale_factor = data["scale_factor"]
        return instance

    def __call__(self, *args) -> np.ndarray:
        """Calculates new weights using radial basis function
        interpolation of the histogram weights.

        Parameters
        ----------
        *args :
            numpy.ndarrays of shape (`nevents`,). The order of the arrays
            should be the same as the order of the variables used to create
            the RBF.

        Returns
        -------
        weights:
            New weights as numpy ndarray. Shape is (`num_events`,).
        """
        weights = self._rbf(*args)
        weights[weights < 0] = 0.0
        return self._scale_factor * weights


@click.command()
@click.argument(
    'out_dir', type=click.Path(exists=True)
)
@click.option(
    '--extract', type=bool, default=False, is_flag=True, help="if set, downloaded archives are also extacted"
)
def download_botdstarstarlnu_data(out_dir, extract):
    """This script downloads the data files for B->D**lv form factors
    to the given OUT_DIR.

    The files are hosted on http://ekpwww.etp.kit.edu/~welsch/BtoDstarstarLNu_Data.
    If the files are not there any more, please create an issue on github.
    """
    print("Downloading data files for B->D**lv form factors from")
    urls = [
        "http://ekpwww.etp.kit.edu/~welsch/BtoDstarstarLNu_Data/BtoDstarstarLNu_ISGW2_Data.tar",
        "http://ekpwww.etp.kit.edu/~welsch/BtoDstarstarLNu_Data/BtoDstarstarLNu_LLSW_Data.tar",

    ]
    out_path = pathlib.Path(out_dir)
    for url in urls:
        print(f"Fetching: {url}")
        file_name = get_file_name_from_url(url)
        target = out_path/file_name
        if target.exists():
            print(f"Skipping file, already exists at \n\t {target}")
        else:
            urllib.request.urlretrieve(url, target)

        extract_dir = out_path/target.stem
        if extract:
            print(f"Unpacking archive to {extract_dir}")
            shutil.unpack_archive(str(target.resolve()), str(extract_dir.resolve()))


def get_file_name_from_url(url: str) -> str:
    """Returns the filename from a given url.
    """
    return url.rsplit('/', 1)[1]

