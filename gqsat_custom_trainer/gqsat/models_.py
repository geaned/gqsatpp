### The code in this file was originally copied from the Pytorch Geometric library and modified later:
### https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/meta.html#MetaLayer
### Pytorch geometric license is below

# Copyright (c) 2019 Matthias Fey <matthias.fey@tu-dortmund.de>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import torch
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, LayerNorm
from torch_scatter import scatter_mean, scatter_add
from torch_geometric.nn.meta import MetaLayer
from torch import nn
import inspect
import yaml
import sys

from torch import Tensor
from typing import Tuple, Optional

class ModifiedMetaLayer(MetaLayer):
    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor, v_indices: Tensor, e_indices: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        row = edge_index[0]
        col = edge_index[1]

        if self.edge_model is not None:
            edge_attr = self.edge_model(x[row], x[col], edge_attr, u, e_indices)

        if self.node_model is not None:
            x = self.node_model(x, edge_index, edge_attr, u, v_indices)

        if self.global_model is not None:
            u = self.global_model(x, edge_attr, u, v_indices, e_indices)

        return x, edge_attr, u

class EdgeModel(torch.nn.Module):
    def __init__(self, edge_mlp):
        super(EdgeModel, self).__init__()

        self.edge_mlp = edge_mlp
    
    def forward(self, src, dest, edge_attr, u, e_indices):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        out = torch.cat([src, dest, edge_attr, u[e_indices]], 1)
        return self.edge_mlp(out)

class NodeModel(torch.nn.Module):
    def __init__(self, node_mlp, e2v_agg):
        super(NodeModel, self).__init__()

        assert e2v_agg == 'sum' or e2v_agg == 'mean'

        self.node_mlp = node_mlp
        self.e2v_agg = scatter_add if e2v_agg == 'sum' else scatter_mean
    
    def forward(self, x, edge_index, edge_attr, u, v_indices):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        row = edge_index[0]
        col = edge_index[1]
        out = self.e2v_agg(edge_attr, row, dim=0, dim_size=x.size(0))
        out = torch.cat([x, out, u[v_indices]], dim=1)
        return self.node_mlp(out)

class GlobalModel(torch.nn.Module):
    def __init__(self, global_mlp):
        super(GlobalModel, self).__init__()

        self.global_mlp = global_mlp
    
    def forward(self, x, edge_attr, u, v_indices, e_indices):
        out = torch.cat(
            [
                u,
                scatter_mean(x, v_indices, dim=0),
                scatter_mean(edge_attr, e_indices, dim=0),
            ],
            dim=1,
        )
        return self.global_mlp(out)

class IndependentModifiedMetaLayer(MetaLayer):
    def forward(
        self, x: Tensor, edge_attr: Tensor, u: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        if self.edge_model is not None:
            edge_attr = self.edge_model(edge_attr)

        if self.node_model is not None:
            x = self.node_model(x)

        if self.global_model is not None:
            u = self.global_model(u)

        return x, edge_attr, u

class IndependentEdgeModel(torch.nn.Module):
    def __init__(self, edge_mlp):
        super(IndependentEdgeModel, self).__init__()

        self.edge_mlp = edge_mlp
    
    def forward(self, edge_attr):
        # source, target: [E, F_x], where E is the number of edges.
        # edge_attr: [E, F_e]
        return self.edge_mlp(edge_attr)

class IndependentNodeModel(torch.nn.Module):
    def __init__(self, node_mlp):
        super(IndependentNodeModel, self).__init__()

        self.node_mlp = node_mlp
    
    def forward(self, x):
        # x: [N, F_x], where N is the number of nodes.
        # edge_index: [2, E] with max entry N - 1.
        # edge_attr: [E, F_e]
        return self.node_mlp(x)

class IndependentGlobalModel(torch.nn.Module):
    def __init__(self, global_mlp):
        super(IndependentGlobalModel, self).__init__()

        self.global_mlp = global_mlp
    
    def forward(self, u):
        return self.global_mlp(u)

class CustomMLP(torch.jit.ScriptModule):
    __constants__ = ['arch']

    def __init__(
        self,
        in_size,
        out_size,
        n_hidden,
        hidden_size,
        activation=nn.LeakyReLU,
        activate_last=True,
        layer_norm=True
    ):
        super(CustomMLP, self).__init__()
        self.arch = CustomMLP._get_mlp(
            in_size,
            out_size,
            n_hidden,
            hidden_size,
            activation,
            activate_last,
            layer_norm
        )
    
    @torch.jit.script_method
    def forward(self, input):
        return self.arch(input)
    
    @staticmethod
    def _get_mlp(
        in_size,
        out_size,
        n_hidden,
        hidden_size,
        activation,
        activate_last=True,
        layer_norm=True,
    ):
        arch = []
        l_in = in_size
        for l_idx in range(n_hidden):
            arch.append(Lin(l_in, hidden_size))
            arch.append(activation())
            l_in = hidden_size

        arch.append(Lin(l_in, out_size))

        if activate_last:
            arch.append(activation())

            if layer_norm:
                arch.append(LayerNorm(out_size))

        return Seq(*arch)

class SatModel(torch.nn.Module):
    def __init__(self, save_name=None):
        super().__init__()
        if save_name is not None:
            self.save_to_yaml(save_name)

    @classmethod
    def save_to_yaml(cls, model_name):
        # -2 is here because I want to know how many layers below lies the final child and get its init params.
        # I do not need nn.Module and 'object'
        # this WILL NOT work with multiple inheritance of the leaf children
        frame, filename, line_number, function_name, lines, index = inspect.stack()[
            len(cls.mro()) - 2
        ]
        args, _, _, values = inspect.getargvalues(frame)

        save_dict = {
            "class_name": values["self"].__class__.__name__,
            "call_args": {
                k: values[k] for k in args if k != "self" and k != "save_name"
            },
        }
        with open(model_name, "w") as f:
            yaml.dump(save_dict, f, default_flow_style=False)

    @staticmethod
    def load_from_yaml(fname):
        with open(fname, "r") as f:
            res = yaml.load(f)
        return getattr(sys.modules[__name__], res["class_name"])(**res["call_args"])


class GraphNet(SatModel):
    def __init__(
        self,
        in_dims,
        out_dims,
        save_name=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        layer_norm=True,
    ):
        super().__init__(save_name)
        if e2v_agg not in ["sum", "mean"]:
            raise ValueError("Unknown aggregation function.")

        v_in = in_dims[0]
        e_in = in_dims[1]
        u_in = in_dims[2]

        v_out = out_dims[0]
        e_out = out_dims[1]
        u_out = out_dims[2]

        edge_mlp = CustomMLP(
            e_in + 2 * v_in + u_in,
            e_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        node_mlp = CustomMLP(
            v_in + e_out + u_in,
            v_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        global_mlp = CustomMLP(
            u_in + v_out + e_out,
            u_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.op = ModifiedMetaLayer(
            EdgeModel(edge_mlp),
            NodeModel(node_mlp, e2v_agg),
            GlobalModel(global_mlp)
        )

    def forward(
        self, x: Tensor, edge_index: Tensor, edge_attr: Tensor, u: Tensor, v_indices: Tensor, e_indices: Tensor
    ):
        return self.op(x, edge_index, edge_attr, u, v_indices, e_indices)

class IndependentGraphNet(SatModel):
    def __init__(
        self,
        in_dims,
        out_dims,
        save_name=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        layer_norm=True,
    ):
        super().__init__(save_name)
        if e2v_agg not in ["sum", "mean"]:
            raise ValueError("Unknown aggregation function.")

        v_in = in_dims[0]
        e_in = in_dims[1]
        u_in = in_dims[2]

        v_out = out_dims[0]
        e_out = out_dims[1]
        u_out = out_dims[2]

        edge_mlp = CustomMLP(
            e_in,
            e_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        node_mlp = CustomMLP(
            v_in,
            v_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        global_mlp = CustomMLP(
            u_in,
            u_out,
            n_hidden,
            hidden_size,
            activation=activation,
            layer_norm=layer_norm,
        )
        self.op = IndependentModifiedMetaLayer(
            IndependentEdgeModel(edge_mlp),
            IndependentNodeModel(node_mlp),
            IndependentGlobalModel(global_mlp)
        )

    def forward(
        self, x, edge_index: Optional[Tensor], edge_attr: Tensor, u: Tensor, v_indices: Optional[Tensor]=None, e_indices: Optional[Tensor]=None
    ):
        return self.op(x, edge_attr, u)


class EncoderCoreDecoder(SatModel):
    def __init__(
        self,
        in_dims,
        core_out_dims,
        out_dims,
        core_steps=1,
        encoder_out_dims=None,
        dec_out_dims=None,
        save_name=None,
        e2v_agg="sum",
        n_hidden=1,
        hidden_size=64,
        activation=ReLU,
        independent_block_layers=1,
    ):
        super().__init__(save_name)
        # all dims are tuples with (v,e) feature sizes
        self.steps = core_steps
        # if dec_out_dims is None, there will not be a decoder
        self.in_dims = in_dims
        self.core_out_dims = core_out_dims
        self.dec_out_dims = dec_out_dims

        self.layer_norm = True

        self.encoder = None
        if encoder_out_dims is not None:
            self.encoder = IndependentGraphNet(
                in_dims,
                encoder_out_dims,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        core_in_dims = in_dims if self.encoder is None else encoder_out_dims

        self.core = GraphNet(
            (
                core_in_dims[0] + core_out_dims[0],
                core_in_dims[1] + core_out_dims[1],
                core_in_dims[2] + core_out_dims[2],
            ),
            core_out_dims,
            e2v_agg=e2v_agg,
            n_hidden=n_hidden,
            hidden_size=hidden_size,
            activation=activation,
            layer_norm=self.layer_norm,
        )

        if dec_out_dims is not None:
            self.decoder = IndependentGraphNet(
                core_out_dims,
                dec_out_dims,
                n_hidden=independent_block_layers,
                hidden_size=hidden_size,
                activation=activation,
                layer_norm=self.layer_norm,
            )

        pre_out_dims = core_out_dims if self.decoder is None else dec_out_dims

        self.vertex_out_transform = (
            Lin(pre_out_dims[0], out_dims[0]) if out_dims[0] is not None else None
        )
        self.edge_out_transform = (
            Lin(pre_out_dims[1], out_dims[1]) if out_dims[1] is not None else None
        )
        self.global_out_transform = (
            Lin(pre_out_dims[2], out_dims[2]) if out_dims[2] is not None else None
        )

    # def get_init_state(self, n_v, n_e, n_u, device):
    #     return (
    #         torch.zeros((n_v, self.core_out_dims[0]), device=device),
    #         torch.zeros((n_e, self.core_out_dims[1]), device=device),
    #         torch.zeros((n_u, self.core_out_dims[2]), device=device),
    #     )

    def forward(self, x, edge_index, edge_attr, u):
        # if v_indices and e_indices are both None, then we have only one graph without a batch
        v_indices = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
        e_indices = torch.zeros(
            edge_attr.shape[0], dtype=torch.long, device=edge_attr.device
        )
        return self.forward_with_indices(x, edge_index, edge_attr, u, v_indices, e_indices)

    def forward_with_indices(self, x, edge_index, edge_attr, u, v_indices, e_indices):
        if self.encoder is not None:
            x, edge_attr, u = self.encoder(
                x, edge_index, edge_attr, u, v_indices, e_indices
            )

        # latent0 = (x, edge_attr, u)
        latent0_x = x
        latent0_edge_attr = edge_attr
        latent0_u = u

        # latent = self.get_init_state(
        #     x.shape[0], edge_attr.shape[0], u.shape[0], x.device
        # )
        latent_x = torch.zeros((x.shape[0], self.core_out_dims[0]), device=x.device)
        latent_edge_attr = torch.zeros((edge_attr.shape[0], self.core_out_dims[1]), device=x.device)
        latent_u = torch.zeros((u.shape[0], self.core_out_dims[2]), device=x.device)

        for st in range(self.steps):
            latent_x, latent_edge_attr, latent_u = self.core(
                torch.cat([latent0_x, latent_x], dim=1),
                edge_index,
                torch.cat([latent0_edge_attr, latent_edge_attr], dim=1),
                torch.cat([latent0_u, latent_u], dim=1),
                v_indices,
                e_indices,
            )

        if self.decoder is not None:
            latent_x, latent_edge_attr, latent_u = self.decoder(
                latent_x, edge_index, latent_edge_attr, latent_u, v_indices, e_indices
            )

        v_out = (
            latent_x
            if self.vertex_out_transform is None
            else self.vertex_out_transform(latent_x)
        )
        e_out = (
            latent_edge_attr
            if self.edge_out_transform is None
            else self.edge_out_transform(latent_edge_attr)
        )
        u_out = (
            latent_u
            if self.global_out_transform is None
            else self.global_out_transform(latent_u)
        )
        return v_out, e_out, u_out
