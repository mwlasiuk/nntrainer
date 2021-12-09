#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
##
# Copyright (C) 2021 Jihoon Lee <jhoon.it.lee@samsung.com>
#
# @file genModelsRecurrent_v2.py
# @date 19 October 2021
# @brief Generate recurrent model tcs
# @author Jihoon lee <jhoon.it.lee@samsung.com>

from recorder_v2 import record_v2, inspect_file
from zoneout import Zoneout
import torch

class FCUnroll(torch.nn.Module):
    def __init__(self, unroll_for=1, num_fc=1):
        super().__init__()
        self.fcs = torch.nn.ModuleList([torch.nn.Linear(1, 1) for i in range(num_fc)])
        self.unroll_for = unroll_for
        # self.loss = torch.nn.MSELoss()
        self.loss = torch.nn.Identity()

    def forward(self, inputs, labels):
        output = inputs[0]
        for i in range(self.unroll_for):
            for fc in self.fcs:
                output = fc(output)
        loss = self.loss(output)
        # loss = self.loss(output, labels[0])
        return output, loss

class RNNCellStacked(torch.nn.Module):
    def __init__(self, unroll_for=1, num_rnn=1, input_size=1, hidden_size=1):
        super().__init__()
        self.rnns = torch.nn.ModuleList(
            [
                torch.nn.RNNCell(input_size, hidden_size)
                for _ in range(num_rnn)
            ]
        )
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        hs = [torch.zeros_like(inputs[0]) for _ in self.rnns]
        out = inputs[0]
        ret = []
        for _ in range(self.unroll_for):
            for i, rnn in enumerate(self.rnns):
                hs[i] = rnn(out, hs[i])
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class LSTMStacked(torch.nn.Module):
    def __init__(self, unroll_for=2, num_lstm=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.lstms = torch.nn.ModuleList(
            [
                torch.nn.LSTMCell(self.input_size, self.hidden_size, bias=True)
                for _ in range(num_lstm)
            ]
        )
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        hs = [torch.zeros_like(inputs[0]) for _ in self.lstms]
        cs = [torch.zeros_like(inputs[0]) for _ in self.lstms]
        out = inputs[0]
        ret = []
        for _ in range(self.unroll_for):
            for i, (lstm, h, c) in enumerate(zip(self.lstms, hs, cs)):
                hs[i], cs[i] = lstm(out, (h, c))
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class ZoneoutLSTMStacked(torch.nn.Module):
    def __init__(self, batch_size=3, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=1, cell_state_zoneout_rate=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.cell_state_zoneout_rate = cell_state_zoneout_rate
        self.zoneout_lstms = torch.nn.ModuleList(
            [
                Zoneout(batch_size, self.input_size, self.hidden_size, unroll_for, hidden_state_zoneout_rate, cell_state_zoneout_rate)
                for _ in range(num_lstm)
            ]
        )
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        hs = [torch.zeros_like(inputs[0]) for _ in self.zoneout_lstms]
        cs = [torch.zeros_like(inputs[0]) for _ in self.zoneout_lstms]
        out = inputs[0]
        ret = []
        for num_unroll in range(self.unroll_for):
            for i, (zoneout_lstm, h, c) in enumerate(zip(self.zoneout_lstms, hs, cs)):
                hs[i], cs[i] = zoneout_lstm(out, (h, c, num_unroll))
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

class GRUCellStacked(torch.nn.Module):
    def __init__(self, unroll_for=2, num_gru=1):
        super().__init__()
        self.input_size = self.hidden_size = 2
        self.grus = torch.nn.ModuleList(
            [
                torch.nn.GRUCell(self.input_size, self.hidden_size, bias=True)
                for _ in range(num_gru)
            ]
        )
        for gru in self.grus:
            gru.bias_hh.data.fill_(0.0)
            gru.bias_hh.requires_grad=False
        self.unroll_for = unroll_for
        self.loss = torch.nn.MSELoss()

    def forward(self, inputs, labels):
        hs = [torch.zeros_like(inputs[0]) for _ in self.grus]
        out = inputs[0]
        ret = []
        for _ in range(self.unroll_for):
            for i, (gru, h) in enumerate(zip(self.grus, hs)):
                hs[i] = gru(out, h)
                out = hs[i]
            ret.append(out)

        ret = torch.stack(ret, dim=1)
        loss = self.loss(ret, labels[0])
        return ret, loss

if __name__ == "__main__":
    record_v2(
        FCUnroll(unroll_for=5),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_single",
    )

    record_v2(
        FCUnroll(unroll_for=2, num_fc=2),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_stacked",
    )

    record_v2(
        FCUnroll(unroll_for=2, num_fc=2),
        iteration=2,
        input_dims=[(1,)],
        label_dims=[(1,)],
        name="fc_unroll_stacked_clipped",
        clip=True
    )

    record_v2(
        RNNCellStacked(unroll_for=2, num_rnn=1, input_size=2, hidden_size=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="rnncell_single",
    )

    record_v2(
        RNNCellStacked(unroll_for=2, num_rnn=2, input_size=2, hidden_size=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="rnncell_stacked",
    )

    record_v2(
        LSTMStacked(unroll_for=2, num_lstm=1),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="lstm_single",
    )

    record_v2(
        LSTMStacked(unroll_for=2, num_lstm=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="lstm_stacked",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_000_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_000_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_050_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_050_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_100_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=0.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_100_000",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_000_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_000_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_050_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_050_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_100_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=0.5),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_100_050",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_000_100",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.0, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_000_100",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_050_100",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=0.5, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_050_100",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=1, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_single_100_100",
    )

    record_v2(
        ZoneoutLSTMStacked(batch_size=1, unroll_for=2, num_lstm=2, hidden_state_zoneout_rate=1.0, cell_state_zoneout_rate=1.0),
        iteration=2,
        input_dims=[(1, 2)],
        label_dims=[(1, 2, 2)],
        name="zoneout_lstm_stacked_100_100",
    )

    record_v2(
        GRUCellStacked(unroll_for=2, num_gru=1),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="grucell_single",
    )

    record_v2(
        GRUCellStacked(unroll_for=2, num_gru=2),
        iteration=2,
        input_dims=[(3, 2)],
        label_dims=[(3, 2, 2)],
        name="grucell_stacked",
    )

    # inspect_file("lstm_single.nnmodelgolden")
