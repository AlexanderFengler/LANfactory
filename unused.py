# CNN

import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *

# import data_simulators
from copy import deepcopy

# Defining the likelihood functions
def make_cnn_likelihood(model, pdf_multiplier=1, **kwargs):
    """Defines the likelihoods for the CNN networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        pdf_multiplier: int <default=1>
            Currently not used. Is meant to adjust for the bin size on which CNN RT histograms were based, to get
            the right proportionality constant.
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    def random(self):

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        sim_out = simulator(
            theta=theta, model=model, n_samples=self.shape[0], max_t=20.0
        )
        return hddm_preprocess(sim_out)

    def pdf(self, x):
        rt = np.array(x, dtype=np.int_)
        response = rt.copy()
        response[rt < 0] = 0
        response[rt > 0] = 1
        response = response.astype(np.int_)
        rt = np.abs(rt)

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        out = hddm.wfpt.wiener_pdf_cnn_2(
            x=rt, response=response, network=kwargs["network"], parameters=theta
        )  # **kwargs) # This may still be buggy !
        return out

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    if model == "ddm":  # or model == 'weibull':

        def wienernn_like(
            x, v, a, z, t, p_outlier=0, w_outlier=0, **kwargs
        ):  # theta

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, t], dtype=np.float32),
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        # Create wfpt class
        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_ddm, **kwargs)
        # )

        # wfpt_nn.pdf = pdf
        # wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        # wfpt_nn.cdf = cdf
        # wfpt_nn.random = random
        # return wfpt_nn

    if model == "weibull_cdf" or model == "weibull":

        def wienernn_like(
            x, v, a, alpha, beta, z, t, p_outlier=0, w_outlier=0, **kwargs
        ):  # theta

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, t, alpha, beta], dtype=np.float32),
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        # Create wfpt class
        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_weibull, **kwargs)
        # )

        # wfpt_nn.pdf = pdf
        # wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        # wfpt_nn.cdf = cdf
        # wfpt_nn.random = random
        # return wfpt_nn

    if model == "levy":

        def wienernn_like(
            x, v, a, alpha, z, t, p_outlier=0, w_outlier=0, **kwargs
        ):  # theta

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, alpha, t], dtype=np.float32),
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        # Create wfpt class
        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_levy, **kwargs)
        # )

        # wfpt_nn.pdf = pdf
        # wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        # wfpt_nn.cdf = cdf
        # wfpt_nn.random = random
        # return wfpt_nn

    if model == "ornstein":

        def wienernn_like(
            x, v, a, g, z, t, p_outlier=0, w_outlier=0, **kwargs
        ):  # theta

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, g, t], dtype=np.float32),
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        # # Create wfpt class
        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_ornstein, **kwargs)
        # )

        # wfpt_nn.pdf = pdf
        # wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        # wfpt_nn.cdf = cdf
        # wfpt_nn.random = random
        # return wfpt_nn

    if model == "full_ddm" or model == "full_ddm2":

        def wienernn_like(
            x, v, sv, a, z, sz, t, st, p_outlier=0, w_outlier=0, **kwargs
        ):

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, t, sz, sv, st], dtype=np.float32),
                p_outlier=p_outlier,
                w_outlier=w_outlier,
                **kwargs
            )

        # Create wfpt class
        # wfpt_nn = stochastic_from_dist(
        #     "Wienernn_" + model, partial(wienernn_like_full_ddm, **kwargs)
        # )

        # wfpt_nn.pdf = pdf
        # wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        # wfpt_nn.cdf = cdf
        # wfpt_nn.random = random
        # return wfpt_nn

    if model == "angle":

        def wienernn_like(
            x, v, a, theta, z, t, p_outlier=0, w_outlier=0, **kwargs
        ):

            return hddm.wfpt.wiener_like_cnn_2(
                x["rt_binned"].values,
                x["response_binned"].values,
                np.array([v, a, z, t, theta], dtype=np.float32),
                p_outlier=p_outlier,
                w_outlier=w_outlier,
                **kwargs
            )

    else:
        return "Not implemented errror: Failed to load likelihood because the model specified is not implemented"
    
    # Create wfpt class
    wfpt_nn = stochastic_from_dist(
        "Wienernn_" + model, partial(wienernn_like, **kwargs)
    )

    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn


def generate_wfpt_nn_ddm_reg_stochastic_class(model=None, **kwargs):
    """Defines the regressor likelihoods for the CNN networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    # Need to rewrite these random parts !
    def random(self):
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]
        sampled_rts = self.value.copy()

        size = sampled_rts.shape[0]
        n_params = model_config[model]["n_params"]
        param_data = np.zeros((size, n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in model_config[model]["params"]:  # ['v', 'a', 'z', 't']:
            if tmp_str in self.parents["reg_outcomes"]:
                param_data[:, cnt] = param_dict[tmp_str].values[:, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(
            theta=param_data, n_trials=size, model=model, n_samples=1, max_t=20
        )
        return hddm_preprocess(sim_out, keep_negative_responses=True)

    if model == "ddm":

        def wiener_multi_like_nn_ddm(
            value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0, **kwargs
        ):  # theta

            params = {"v": v, "a": a, "z": z, "t": t}
            n_params = 4
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "t"]:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_ddm, **kwargs)
        )
        stoch.random = random

    if model == "full_ddm" or model == "full_ddm2":

        def wiener_multi_like_nn_full_ddm(
            value,
            v,
            sv,
            a,
            z,
            sz,
            t,
            st,
            reg_outcomes,
            p_outlier=0,
            w_outlier=0.1,
            **kwargs
        ):

            params = {"v": v, "a": a, "z": z, "t": t, "sz": sz, "sv": sv, "st": st}

            n_params = int(7)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "t", "sz", "sv", "st"]:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_full_ddm, **kwargs)
        )
        stoch.random = random

    if model == "angle":

        def wiener_multi_like_nn_angle(
            value, v, a, theta, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
        ):

            """Log-likelihood for the full DDM using the interpolation method"""

            params = {"v": v, "a": a, "z": z, "t": t, "theta": theta}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "t", "theta"]:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_angle, **kwargs)
        )
        stoch.random = random

    if model == "levy":

        def wiener_multi_like_nn_levy(
            value, v, a, alpha, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
        ):

            """Log-likelihood for the full DDM using the interpolation method"""
            params = {"v": v, "a": a, "z": z, "alpha": alpha, "t": t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "alpha", "t"]:
                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_levy, **kwargs)
        )
        stoch.random = random

    if model == "ornstein":

        def wiener_multi_like_nn_ornstein(
            value, v, a, g, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
        ):

            params = {"v": v, "a": a, "z": z, "g": g, "t": t}
            n_params = int(5)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "g", "t"]:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_ornstein, **kwargs)
        )
        stoch.random = random

    if model == "weibull_cdf" or model == "weibull":

        def wiener_multi_like_nn_weibull(
            value,
            v,
            a,
            alpha,
            beta,
            z,
            t,
            reg_outcomes,
            p_outlier=0,
            w_outlier=0.1,
            **kwargs
        ):

            params = {"v": v, "a": a, "z": z, "t": t, "alpha": alpha, "beta": beta}
            n_params = int(6)
            size = int(value.shape[0])
            data = np.zeros((size, n_params), dtype=np.float32)

            cnt = 0
            for tmp_str in ["v", "a", "z", "t", "alpha", "beta"]:

                if tmp_str in reg_outcomes:
                    data[:, cnt] = (
                        params[tmp_str].loc[value["rt_binned"].index].values[:, 0]
                    )
                    if (
                        data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                    ) or (
                        data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                    ):
                        print("boundary violation of regressor part")
                        return -np.inf
                else:
                    data[:, cnt] = params[tmp_str]

                cnt += 1

            # Has optimization potential --> AF-TODO: For next version!
            return hddm.wfpt.wiener_like_reg_cnn_2(
                value["rt_binned"],
                value["response_binned"],
                data,
                p_outlier=p_outlier,  # TODO: ACTUALLY USE THIS
                w_outlier=w_outlier,
                **kwargs
            )

        stoch = stochastic_from_dist(
            "wfpt_reg", partial(wiener_multi_like_nn_weibull, **kwargs)
        )
        stoch.random = random

    return stoch


# MLP

import numpy as np
import hddm
from functools import partial
from kabuki.utils import stochastic_from_dist
from hddm.simulators import *

# import data_simulators
from copy import deepcopy


def make_mlp_likelihood(model, **kwargs):
    """Defines the likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    def random(self):
        """
        Generate random samples from a given model (the dataset matches the size of the respective observated dataset supplied as an attribute of 'self').
        """

        # this can be simplified so that we pass parameters directly to the simulator ...
        theta = np.array(model_config[model]["default_params"], dtype=np.float32)
        keys_tmp = self.parents.value.keys()
        cnt = 0

        for param in model_config[model]["params"]:
            if param in keys_tmp:
                theta[cnt] = np.array(self.parents.value[param]).astype(np.float32)
            cnt += 1

        sim_out = simulator(theta=theta, model=model, n_samples=self.shape[0], max_t=20)
        return hddm_preprocess(sim_out, keep_negative_responses=True)

    def pdf(self, x):
        rt = np.array(x, dtype=np.float32)
        response = rt / np.abs(rt)
        rt = np.abs(rt)
        params = np.array([self.parents[param] for param in model_config[model]["params"]]).astype(np.float32)
        return hddm.wfpt.wiener_like_nn_mlp_pdf(rt, response, 
                                                params, network=kwargs["network"], **self.parents)  # **kwargs) # This may still be buggy !

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"
    
    if model == "test":
        def wienernn_like_test(x, v, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
            """
            LAN Log-likelihood for the DDM
            """ 
            return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, np.array([v, a, z, t]).astype(np.float32),  
                                                p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

        def pdf_test(self, x):
            rt = np.array(x, dtype=np.float32)
            response = rt / np.abs(rt)
            rt = np.abs(rt)
            params = np.array([self.parents[param] for param in model_config[model]["params"]]).astype(np.float32)
            out = hddm.wfpt.wiener_like_nn_mlp_pdf(rt, response, 
                                                   params, network=kwargs["network"], **self.parents)  # **kwargs) # This may still be buggy !
            return out

        def cdf_test(self, x):
            # TODO: Implement the CDF method for neural networks
            return "Not yet implemented"

        # Create wfpt class
        wfpt_nn = stochastic_from_dist(
            "Wienernn_" + model, partial(wienernn_like_test, **kwargs)
        )

        wfpt_nn.pdf = pdf_test
        wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
        wfpt_nn.cdf = cdf_test
        wfpt_nn.random = random
        return wfpt_nn

    def wienernn_like_ddm(x, v, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_weibull(x, v, a, alpha, beta, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, t, alpha, beta]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_ddm_sdv(x, v, sv, a, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, t, sv]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_levy(x, v, a, alpha, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, alpha, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_ornstein(x, v, a, g, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, g, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_full_ddm(x, v, sv, a, z, sz, t, st, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values,
                                            np.array([v, a, z, t, sz, sv, st]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)


    def wienernn_like_angle(x, v, a, theta, z, t, p_outlier=0, w_outlier=0.1, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v, a, z, t, theta]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)
    
    def wienernn_like_par2(x, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t,
                           p_outlier=0.0, w_outlier=0.0, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values,
                                            np.array([v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_seq2(x, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t,
                           p_outlier=0.0, w_outlier=0.0, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    def wienernn_like_mic2(x, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, d, t, 
                           p_outlier=0.0, w_outlier=0.0, **kwargs):
        """
        LAN Log-likelihood for the DDM
        """
        return hddm.wfpt.wiener_like_nn_mlp(x["rt"].values, x["response"].values, 
                                            np.array([v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, d, t]).astype(np.float32),  
                                            p_outlier = p_outlier, w_outlier = w_outlier, network = kwargs["network"]) #**kwargs)

    likelihood_funs = {}
    likelihood_funs["ddm"] = wienernn_like_ddm
    likelihood_funs["weibull"] = wienernn_like_weibull
    likelihood_funs["angle"] = wienernn_like_angle
    likelihood_funs["ddm_sdv"] = wienernn_like_ddm_sdv
    likelihood_funs["ddm_sdv_analytic"] = wienernn_like_ddm_sdv
    likelihood_funs["levy"] = wienernn_like_levy
    likelihood_funs["ornstein"] = wienernn_like_ornstein
    likelihood_funs["full_ddm"] = wienernn_like_full_ddm
    likelihood_funs["ddm_par2"] = wienernn_like_par2
    likelihood_funs["ddm_seq2"] = wienernn_like_seq2
    likelihood_funs["ddm_mic2"] = wienernn_like_mic2

    wfpt_nn = stochastic_from_dist(
            "Wienernn_" + model, partial(likelihood_funs[model], **kwargs)
        )
    wfpt_nn.pdf = pdf
    wfpt_nn.cdf_vec = None  # AF TODO: Implement this for neural nets (not a big deal actually but not yet sure where this is ever used finally)
    wfpt_nn.cdf = cdf
    wfpt_nn.random = random
    return wfpt_nn


# REGRESSOR LIKELIHOODS
def generate_wfpt_nn_ddm_reg_stochastic_class(model=None, **kwargs):
    """Defines the regressor likelihoods for the MLP networks.

    :Arguments:
        model: str <default='ddm>
            String that determines which model you would like to fit your data to.
            Currently available models are: 'ddm', 'full_ddm', 'angle', 'weibull', 'ornstein', 'levy'
        **kwargs: dict
            Dictionary of additional keyword arguments.
            Importantly here, this carries the preloaded CNN.
    :Returns:
        pymc.object: Returns a stochastic object as defined by PyMC2
    """

    # Need to rewrite these random parts !
    def random(
        self,
        keep_negative_responses=True,
        add_model_parameters=False,
        keep_subj_idx=False,
    ):
        """
        Function to sample from a regressor based likelihood. Conditions on the covariates.
        """
        param_dict = deepcopy(self.parents.value)
        del param_dict["reg_outcomes"]

        # size = sampled_rts.shape[0]
        n_params = model_config[model]["n_params"]
        param_data = np.zeros((self.value.shape[0], n_params), dtype=np.float32)

        cnt = 0
        for tmp_str in model_config[model]["params"]:  # ['v', 'a', 'z', 't']:
            if tmp_str in self.parents["reg_outcomes"]:
                param_data[:, cnt] = param_dict[tmp_str].iloc[self.value.index, 0]
            else:
                param_data[:, cnt] = param_dict[tmp_str]
            cnt += 1

        sim_out = simulator(
            theta=param_data, model=model, n_samples=1, max_t=20  # n_trials = size,
        )

        return hddm_preprocess(
            sim_out,
            keep_negative_responses=keep_negative_responses,
            add_model_parameters=add_model_parameters,
            keep_subj_idx=keep_subj_idx,
        )
    
    def pdf(self, x):
        return "Not yet implemented"

    def cdf(self, x):
        # TODO: Implement the CDF method for neural networks
        return "Not yet implemented"

    def wiener_multi_like_nn_ddm(
        value, v, a, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """LAN Log-likelihood for the DDM"""

        params = {"v": v, "a": a, "z": z, "t": t}
        n_params = 4  # model_config[model]['n_params']
        size = int(value.shape[0])
        data = np.zeros((size, 6), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "t"]:  # model_config[model]['params']:
            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]
            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        ) # **kwargs

    def wiener_multi_like_nn_full_ddm(
        value,
        v,
        sv,
        a,
        z,
        sz,
        t,
        st,
        reg_outcomes,
        p_outlier=0,
        w_outlier=0.1,
        **kwargs
    ):
        """
        LAN Log-likelihood for the FULL DDM
        """

        params = {"v": v, "a": a, "z": z, "t": t, "sz": sz, "sv": sv, "st": st}

        n_params = int(7)
        size = int(value.shape[0])
        data = np.zeros((size, 9), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "sz", "sv", "st"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_angle(
        value, v, a, theta, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the ANGLE MODEL
        """

        params = {"v": v, "a": a, "z": z, "t": t, "theta": theta}

        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, 7), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "theta"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, **kwargs
        )

    def wiener_multi_like_nn_levy(
        value, v, a, alpha, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the LEVY MODEL
        """

        params = {"v": v, "a": a, "z": z, "alpha": alpha, "t": t}
        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, 7), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "alpha", "t"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_ornstein(
        value, v, a, g, z, t, reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs
    ):

        """
        LAN Log-likelihood for the ORNSTEIN MODEL
        """

        params = {"v": v, "a": a, "z": z, "g": g, "t": t}

        n_params = int(5)
        size = int(value.shape[0])
        data = np.zeros((size, 7), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "g", "t"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_weibull(value, v, a, alpha, beta, z, t,
                                     reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """

        params = {"v": v, "a": a, "z": z, "t": t, "alpha": alpha, "beta": beta}
        n_params = int(6)
        size = int(value.shape[0])
        data = np.zeros((size, 8), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v", "a", "z", "t", "alpha", "beta"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_par2(value, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t,
                                  reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """

        params = {
            "v_h": v_h,
            "v_l_1": v_l_1,
            "v_l_2": v_l_2,
            "a": a,
            "z_h": z_h,
            "z_l_1": z_l_1,
            "z_l_2": z_l_2,
            "t": t,
        }
        n_params = int(8)
        size = int(value.shape[0])
        data = np.zeros((size, 10), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v_h", "v_l_1", "v_l_2", "a", "z_h", "z_l_1", "z_l_2", "t"]:

            if tmp_str in reg_outcomes:
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_seq2(value, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, t,
                                  reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """

        params = {
            "v_h": v_h,
            "v_l_1": v_l_1,
            "v_l_2": v_l_2,
            "a": a,
            "z_h": z_h,
            "z_l_1": z_l_1,
            "z_l_2": z_l_2,
            "t": t,
        }
        n_params = int(8)
        size = int(value.shape[0])
        data = np.zeros((size, 10), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v_h", "v_l_1", "v_l_2", "a", "z_h", "z_l_1", "z_l_2", "t"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    def wiener_multi_like_nn_mic2(value, v_h, v_l_1, v_l_2, a, z_h, z_l_1, z_l_2, d, t,
                                  reg_outcomes, p_outlier=0, w_outlier=0.1, **kwargs):

        """
        LAN Log-likelihood for the WEIBULL MODEL
        """

        params = {
            "v_h": v_h,
            "v_l_1": v_l_1,
            "v_l_2": v_l_2,
            "a": a,
            "z_h": z_h,
            "z_l_1": z_l_1,
            "z_l_2": z_l_2,
            "d": d,
            "t": t,
        }
        n_params = int(9)
        size = int(value.shape[0])
        data = np.zeros((size, 11), dtype=np.float32)
        data[:, n_params:] = np.stack(
            [
                np.absolute(value["rt"]).astype(np.float32),
                value["response"].astype(np.float32),
            ],
            axis=1,
        )

        cnt = 0
        for tmp_str in ["v_h", "v_l_1", "v_l_2", "a", "z_h", "z_l_1", "z_l_2", "d", "t"]:

            if tmp_str in reg_outcomes:
                # data[:, cnt] = params[tmp_str].loc[value['rt'].index].values[:, 0]
                data[:, cnt] = params[tmp_str].loc[value["rt"].index].values
                if (
                    data[:, cnt].min() < model_config[model]["param_bounds"][0][cnt]
                ) or (
                    data[:, cnt].max() > model_config[model]["param_bounds"][1][cnt]
                ):
                    print("boundary violation of regressor part")
                    return -np.inf
            else:
                data[:, cnt] = params[tmp_str]

            cnt += 1

        # Has optimization potential --> AF-TODO: For next version!
        return hddm.wfpt.wiener_like_multi_nn_mlp(
            data, p_outlier=p_outlier, w_outlier=w_outlier, network = kwargs["network"]
        )

    likelihood_funs = {}
    likelihood_funs['ddm'] = wiener_multi_like_nn_ddm
    likelihood_funs['full_ddm'] = wiener_multi_like_nn_full_ddm
    likelihood_funs['angle'] = wiener_multi_like_nn_angle
    likelihood_funs['levy'] = wiener_multi_like_nn_levy
    likelihood_funs['ornstein'] = wiener_multi_like_nn_ornstein
    likelihood_funs['weibull'] = wiener_multi_like_nn_weibull
    likelihood_funs['ddm_par2'] = wiener_multi_like_nn_par2
    likelihood_funs['ddm_seq2'] = wiener_multi_like_nn_seq2
    likelihood_funs['ddm_mic2'] = wiener_multi_like_nn_mic2

    stoch = stochastic_from_dist("wfpt_reg", partial(likelihood_funs[model], **kwargs))
    stoch.pdf = pdf
    stoch.cdf = cdf
    stoch.random = random

    return stoch
