import torch 
import torch.nn as nn


# Clases base para los encoders, decoders y la red de dinámicas
class Encoder(nn.Module):
    def init(self):
            super().__init__()

    def forward(self, x):   
        raise NotImplementedError

class Decoder(nn.Module):
    def init(self):
        super().__init__()

    def forward(self, x):   
        raise NotImplementedError
    
class DynamicsNet(nn.Module):
    def init(self):
        super().__init__()

    def forward(self, x, h=None):   
        raise NotImplementedError
    
# Encoders especificos -> Probar Conv en vez de MLP
class SimpleEncoder(Encoder):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation=nn.Tanh, dropout=0.05, device='cuda'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], device=device))
            if i < len(dims) - 2:
                layers.append(activation())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        if x.dim() == 3:                    # (batch, historia, features) -> solo el último frame (t0)
            x = x[:, -1]
        return self.net(x)


class HistoryGruEncoder(Encoder):
    """
    Estima el estado inicial a partir de los últimos K frames de features (batch, K, input_dim),
    ordenados del más antiguo al más reciente (t0). Con datos reales es más robusto que un MLP
    sobre un solo frame: las velocidades medidas son ruidosas y el estado del cable depende de la historia.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.05, device='cuda'):
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, batch_first=True, device=device)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim, output_dim, device=device)

    def forward(self, x):
        if x.dim() == 2:                    # un solo frame
            x = x.unsqueeze(1)
        _, h = self.gru(x)
        return self.out(self.dropout(h[-1]))


class SimpleDecoder(Decoder):
    def __init__(self, input_dim, hidden_dims, output_dim,
                 activation=nn.Tanh, dropout=0.05, device='cuda'):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims + [output_dim]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1], device=device))
            if i < len(dims) - 2:
                layers.append(activation())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(z)
    
# Redes de dinámicas especificas
class RnnDynamics(DynamicsNet):
    def __init__(self, input_dim, hidden_dim, n_layers=1, device='cuda'):
        """Vanilla RNN"""
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, n_layers, dropout=0.2 if n_layers > 1 else 0.0, batch_first=True, device=device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, h=None):              # Si no se pasa h, se inicializa a cero
        if h is None:
            h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)

        out, h_n = self.rnn(x, h)               # out: (batch_size, seq_len, hidden_dim)
        return out, h_n                         

class LstmDynamics(DynamicsNet):
    def __init__(self, input_dim, hidden_dim, n_layers=1, device='cuda'):
        """LSTM"""
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, dropout=0.2 if n_layers > 1 else 0.0, batch_first=True, device=device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, h=None):               # Si no se pasa h, se inicializa todo a cero
        if h is None:
            h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)
        c = torch.zeros_like(h)            

        out, (h_n, c_n) = self.lstm(x, (h, c))       # out: (batch_size, seq_len, hidden_dim)
        #return out, (h_n, c_n)
        return out, h_n                       

class GruDynamics(DynamicsNet):
    def __init__(self, input_dim, hidden_dim, n_layers=1, device='cuda'):
        """GRU"""
        super().__init__()
        self.gru = nn.GRU(input_dim, hidden_dim, n_layers, dropout=0.2 if n_layers > 1 else 0.0, batch_first=True, device=device)
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

    def forward(self, x, h=None):
        if h is None:                           # Si no se pasa h, se inicializa a cero
            h = torch.zeros(self.n_layers, x.size(0), self.hidden_dim, device=x.device)

        out, h_n = self.gru(x, h)               # out: (batch_size, seq_len, hidden_dim)
        return out, h_n  


# class MlpDynamics(DynamicsNet):
#     def __init__(self, input_dim, hidden_dims, output_dim, activation=nn.ReLU, device='cuda'):
#         """Una red de dinámicas simple con capas lineales y funciones de activación. """
#         super().__init__()
#         layers = []
#         dims = [input_dim] + hidden_dims + [output_dim]
#         for i in range(len(dims) - 1):
#             layers.append(nn.Linear(dims[i], dims[i+1], device=device))
#             if i < len(dims) - 2:           # No agregar activación después de la última capa
#                 layers.append(activation())
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x) 


# class DLOModel(nn.Module):
#     def __init__(self, encoder, dynamics, decoder, n_layers=1):
#         super().__init__()
#         self.encoder = encoder
#         self.dynamics = dynamics
#         self.decoder = decoder
#         self.n_layers = n_layers

#     def forward(self, u_enc, U_dyn, U_dec):
 
#         # Encoder -> estado inicial
#         h_0 = self.encoder(u_enc)                                           # (batch, hidden_dim)
#         h_0_for_rnn = h_0.unsqueeze(0).repeat(self.n_layers, 1, 1)          # (num_layers, batch, hidden_dim)

#         # Dinámica -> evolución con inputs externos
#         X_seq, _ = self.dynamics(U_dyn, h_0_for_rnn)                        # (batch, seq_len, hidden_dim)

#         # Concatenar el estado inicial al principio de la secuencia
#         X_seq = torch.cat([h_0_for_rnn[-1].unsqueeze(1), X_seq], dim=1)     # (batch, seq_len+1, hidden_dim)

#         # Decoder -> aplicar sobre toda la secuencia
#         Y_seq = self.decoder(torch.cat([X_seq, U_dec], dim=-1))             # Comprobar con el paper

#         return X_seq, Y_seq
    

# class DLOLite(nn.Module):
#     def __init__(self, encoder, dynamics, decoder, n_layers=1):
#         super().__init__()
#         self.encoder = encoder
#         self.dynamics = dynamics
#         self.decoder = decoder
#         self.n_layers = n_layers

#     def forward(self, u_enc, U_dyn, U_dec):
 
#         # Encoder -> estado inicial                                         u_enc es solo el primer estado de la secuencia
#         h_0 = self.encoder(u_enc)                                           # (batch, hidden_dim)
#         h_0_for_rnn = h_0.unsqueeze(0).repeat(self.n_layers, 1, 1)          # (num_layers, batch, hidden_dim)

#         # Dinámica -> evolución con inputs externos
#         X_seq, _ = self.dynamics(U_dyn, h_0_for_rnn)                        # (batch, seq_len, hidden_dim)

#         # Concatenar el estado inicial al principio de la secuencia
#         X_seq = torch.cat([h_0_for_rnn[-1].unsqueeze(1), X_seq], dim=1)     # (batch, seq_len+1, hidden_dim)

#         # Decoder -> aplicar sobre toda la secuencia
#         Y_seq = self.decoder(torch.cat([X_seq, U_dec], dim=-1))             # Comprobar con el paper -> Comprobado

#         return X_seq, Y_seq
    
class DLOModel(nn.Module):
    def __init__(self, encoder, dynamics, decoder, n_layers=1, noise_std=0.0,
                 bound_h0=False, enc_scaler=None, dec_scaler=None):
        """
        noise_std:  ruido gaussiano en la entrada del encoder (ya escalada), solo en training.
                    Por defecto 0 para que el MPC (que llama a model.train()) no lo aplique.
        bound_h0:   aplica tanh a la salida del encoder (el estado oculto del GRU vive en [-1, 1]).
        enc_scaler, dec_scaler: TorchMinMaxScaler ajustados en train. Las entradas se pasan sin
                    escalar y se escalan aquí, así el MPC no tiene que hacerlo. None = sin escalar.
        u_enc puede ser (batch, features) o (batch, K, features) con los K últimos frames (t0 al final).
        """
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.decoder = decoder
        self.n_layers = n_layers
        self.noise_std = noise_std
        self.bound_h0 = bound_h0
        self.enc_scaler = enc_scaler
        self.dec_scaler = dec_scaler

    def forward(self, u_enc, U_dyn, U_dec):

        # Escalado de entradas del encoder y del decoder
        if self.enc_scaler is not None:
            u_enc = self.enc_scaler.transform(u_enc)
        if self.dec_scaler is not None:
            U_dec = self.dec_scaler.transform(U_dec)

        # Ruido en la entrada (solo en training)
        if self.training and self.noise_std > 0:
            u_enc = u_enc + self.noise_std * torch.randn_like(u_enc)

        # Encoder -> estado inicial
        h_0 = self.encoder(u_enc)
        if self.bound_h0:
            h_0 = torch.tanh(h_0)

        h_0_for_rnn = h_0.unsqueeze(0).repeat(self.n_layers, 1, 1)

        # Dinámica
        X_seq, _ = self.dynamics(U_dyn, h_0_for_rnn)

        # Añadir estado inicial al principio
        X_seq = torch.cat([h_0_for_rnn[-1].unsqueeze(1), X_seq], dim=1)

        # Decoder
        Y_seq = self.decoder(torch.cat([X_seq, U_dec], dim=-1))

        return X_seq, Y_seq

def build_model_from_checkpoint(checkpoint, device='cuda'):
    """Reconstruye el DLOModel de un checkpoint de train_simple.py (también los antiguos, sin campos nuevos)."""
    params = checkpoint["model_params"]
    if params.get("encoder_type", "mlp") == "gru":
        encoder = HistoryGruEncoder(**params["encoder_params"], device=device)
    else:
        encoder = SimpleEncoder(**params["encoder_params"], device=device)
    dynamics = GruDynamics(**params["dynamics_params"], device=device)
    decoder = SimpleDecoder(**params["decoder_params"], device=device)
    model = DLOModel(encoder, dynamics, decoder, **checkpoint.get("model_kwargs", {}),
                     enc_scaler=checkpoint.get("Enc_scaler"), dec_scaler=checkpoint.get("Dec_scaler"))
    model.to(device)
    model.load_state_dict(checkpoint["model_state"])
    return model
