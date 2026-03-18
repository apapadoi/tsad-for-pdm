# Copyright 2026 Anastasios Papadopoulos, Apostolos Giannoulidis
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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