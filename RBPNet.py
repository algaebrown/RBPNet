import torch
import torch.nn as nn
import torch.nn.functional as F


class FirstLayerConv(nn.Module):
    def __init__(
        self, 
        filters=128, 
        kernel_size=12, 
        use_bias=False, 
        activation='relu'
):
        super(FirstLayerConv, self).__init__()

        # arguments
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.layer = nn.Conv1d(
            in_channels=4,
            out_channels=filters,
            kernel_size=kernel_size,
            padding='same',
            bias=use_bias
        )
        self.activation = nn.ReLU() if activation == 'relu' else None

    def forward(self, x):
        x = self.layer(x)
        if self.activation:
            x = self.activation(x)
        return x
    
    
class BodyConv(nn.Module):
    def __init__(
        self, 
        filters=128,
        kernel_size=6, 
        dilation_rate=1, 
        dropout_rate=0.25, 
        activation='relu', 
        batch_norm=True, 
        residual=True, 
        use_bias=True
    ):
        super(BodyConv, self).__init__()
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate
        self.residual = residual
        
        # Layers
        self.conv1d_layer = nn.Conv1d(
            in_channels=filters,
            out_channels=filters,
            kernel_size=kernel_size,
            dilation=dilation_rate,
            padding='same',
            bias=use_bias
        )
        self.batch_norm = nn.BatchNorm1d(filters) if batch_norm else None
        self.activation = nn.ReLU() if activation == 'relu' else None  # Add other activations if necessary
        self.dropout_layer = nn.Dropout(dropout_rate) if dropout_rate > 0.0 else None

    def forward(self, inputs):
        # conv
        x = self.conv1d_layer(inputs)

        # batch_norm
        if self.batch_norm:
            x = self.batch_norm(x)

        # activation
        if self.activation:
            x = self.activation(x)

        # dropout
        if self.dropout_layer:
            x = self.dropout_layer(x)

        # residual
        if self.residual:
            x = x + inputs

        return x
    
    
class SequenceAdditiveMixingCoefficient(nn.Module):
    def __init__(self, in_features):
        super(SequenceAdditiveMixingCoefficient, self).__init__()
        self.global_average_pooling = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Linear(in_features=in_features, out_features=1)  # Update in_features accordingly

    def forward(self, inputs):
        x = self.global_average_pooling(inputs)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        return x
    
    
class ProfileHead(nn.Module):
    def __init__(self, in_channels, kernel_size=21, use_bias=True):
        super(ProfileHead, self).__init__()

        # arguments
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.use_bias = use_bias

        self.layer = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            bias=use_bias
    )

    def forward(self, x):
        x = self.layer(x)
        x = torch.squeeze(x, 1)
        return x
    

class AdditiveTargetBias(nn.Module):
    def __init__(self, penalty=None, stop_bias_gradient=False):
        super(AdditiveTargetBias, self).__init__()
        self.stop_bias_gradient = stop_bias_gradient
        
    # Implementing the numerically stable log-sum-exp operation
    def stable_logsumexp(self, x, y):
        max_val = torch.max(x, y)
        return max_val + torch.log1p(torch.exp(-torch.abs(x - y)))
    
    def forward(self, inputs, training=False):

        logits_t, logits_b, a = inputs

        if self.stop_bias_gradient and training:
            logits_b = logits_b.detach()

        # Compute log probabilities
        log_p = logits_t - torch.logsumexp(logits_t, dim=1, keepdim=True)
        log_q = logits_b - torch.logsumexp(logits_b, dim=1, keepdim=True)
        
        # Calculate the mixed log probabilities
        s = self.stable_logsumexp(a + log_p, log_q)

        return s
    
    
class RBPNet(nn.Module):
    def __init__(
        self,
        filters=128,
        residual_blocks=9, 
        dilation=True, 
        use_bias=True,
        mask = 100
    ):
        super(RBPNet, self).__init__()
        
        # First conv
        self.first_layer_conv = FirstLayerConv(filters=filters)
        
        # Residual blocks
        self.body_convs = nn.ModuleList(
            [BodyConv(filters=filters, dilation_rate=(2**i if dilation else 1)) for i in range(1, residual_blocks + 1)]
        )

        # dlog odds
        self.dlogodds_bn = torch.nn.BatchNorm1d(1)
        
        # Output heads
        self.mixing_coefficient = SequenceAdditiveMixingCoefficient(in_features=filters)
        self.signal_head = ProfileHead(in_channels=filters)
        self.ctl_head = ProfileHead(in_channels=filters)
        self.target_bias_layer = AdditiveTargetBias()
        
        self.mask = mask
                
    def forward(self, x_in):
        x = self.first_layer_conv(x_in)

        for conv in self.body_convs:
            x = conv(x)

        # x is[batch_size, channel(filters), L]
        
        # Mixing coefficient
        x_mix = self.mixing_coefficient(x)
        
        # Each head
        x_signal = self.signal_head(x)
        x_ctl = self.ctl_head(x)
        
        # Additive
        x_total = self.target_bias_layer((x_signal, x_ctl, x_mix))

        # d_log_odds is the sum of signal track
        d_log_odds = self.dlogodds_bn(torch.sum(x_signal[:, self.mask:-self.mask-1], dim = -1).unsqueeze(dim = -1)).squeeze()
        
        return x_total, x_signal, x_ctl, x_mix, d_log_odds
