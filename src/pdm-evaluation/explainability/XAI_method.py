from captum.attr import KernelShap
from captum.attr import Lime, LimeBase


def c_kshap(model, x, adtional_forward_args=None):
    kshap = KernelShap(model)
    if adtional_forward_args is not None:
        kshap_x_exp = (kshap.attribute(x, additional_forward_args=(adtional_forward_args.unsqueeze(0),),
                                       n_samples=100)).squeeze().detach()
    else:
        kshap_x_exp = (kshap.attribute(x.unsqueeze(0), n_samples=100)).squeeze().detach()
    return kshap_x_exp

def c_lime(model, x, adtional_forward_args=None):

    lm = Lime(model)

    if adtional_forward_args is not None:
        lime_x_exp = (lm.attribute(x, additional_forward_args=(adtional_forward_args.unsqueeze(0),),
                                   n_samples=100)).squeeze().detach()
    else:
        lime_x_exp = (lm.attribute(x.unsqueeze(0), n_samples=100)).squeeze().detach()

    return lime_x_exp