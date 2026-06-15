import time
import math
from math import ceil
import warnings
import argparse
import inspect
import numpy as np

import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from torch_ema import ExponentialMovingAverage

from sgmse import sampling
from sgmse.sdes import SDERegistry
from sgmse.backbones import BackboneRegistry
#from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, SigLIP


class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument(
            "--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)"
        )
        parser.add_argument(
            "--ema_decay",
            type=float,
            default=0.999,
            help="The parameter EMA decay constant (0.999 by default)",
        )
        parser.add_argument(
            "--t_eps",
            type=float,
            default=0.03,
            help="The minimum time (3e-2 by default)",
        )
        parser.add_argument(
            "--num_eval_files",
            type=int,
            default=20,
            help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).",
        )
        parser.add_argument(
            "--loss_type",
            type=str,
            default="mse",
            choices=("mse", "mae"),
            help="The type of loss function to use.",
        )
        parser.add_argument(
            "--regularization_warmup_epochs",
            type=int,
            default=50,
            help="The nb of warmup epochs before the synchronization loss will be used.",
        )
        parser.add_argument(
            "--regularization_beta0",
            type=float,
            default=0,
            help="The regularization parameter beta0.",
        )
        # Backward-compatible alias used by older run scripts.
        parser.add_argument(
            "--regularization_alpha0",
            dest="regularization_beta0",
            type=int,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "--regularization_rampup_no_warmup",
            action="store_true", 
            help="Apply a ramp-up schedule to beta."
        )
        parser.add_argument(
            "--regularization_constant",
            action="store_true", 
            help="Apply regularization constantly through training."
        )
        parser.add_argument(
            "--regularization_mlp",
            action="store_true", 
            help="Apply a mlp layer before the loss."
        )
        parser.add_argument(
            "--alpha_t_decay",
            type=str,
            default="advanced",
            choices=("step", "linear", "advanced"),
            help="specifies the alpha (t) decay function.",
        )
        parser.add_argument(
            "--loss_alpha_t",
            type=str,
            default="uniform",
            choices=("uniform", "weighted"),
            help="specifies how the loss is computed with alpha (t) in the batch.",
        )
        parser.add_argument(
            "--contrastive_loss",
            type=str,
            default="info_nce",
            choices=("info_nce", "siglip", "siglip_pretrain", "siglip2", "info_nce_uniform_pretrain"),
            help="specifies which contrastive loss is used",
        )
        parser.add_argument(
            "--info_nce_tau",
            type=float,
            default=0.1,
            help="temperature parameter for info_nce loss (default: 0.1)",
        )
        return parser

    def __init__(
        self,
        backbone,
        sde,
        lr=1e-4,
        ema_decay=0.999,
        t_eps=3e-2,
        num_eval_files=20,
        loss_type="mse",
        data_module_cls=None,
        **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()

        self.beta0 = kwargs.get('regularization_beta0', 0)  # regularization parameter
        print("beta0", self.beta0)
        self.beta = torch.tensor(0)
        self.beta_warmup = kwargs.get('regularization_warmup_epochs') # nb of warmup epochs before starting to apply contrastive loss 
        print("WARM UP", self.beta_warmup, " epochs")
        self.use_beta_rampup_no_warmup = kwargs.get('regularization_rampup_no_warmup') # whether to apply a rampup to beta_epoch
        print("Beta epoch RAMP UP", self.use_beta_rampup_no_warmup)

        self.beta_constant= kwargs.get('regularization_constant') # whether alpha_epoch is constant through training
        print("Beta epoch constant", self.beta_constant)
        
        self.alpha_t_decay = kwargs.get('alpha_t_decay')
        #print("alpha_t_decay", self.alpha_t_decay) 
        self.loss_alpha_t = kwargs.get('loss_alpha_t')
        #print("loss with alpha_t : ", self.loss_alpha_t)
        self.contrastive_loss = kwargs.get('contrastive_loss', 'info_nce')
        print('contrastive loss : ', self.contrastive_loss)
        self.info_nce_tau = kwargs.get('info_nce_tau', 0.1)

        #self.siglip = SigLIP() 


        # Initialize Backbone DNN
        dnn_cls = BackboneRegistry.get_by_name(backbone)

        self.dnn = dnn_cls(**kwargs)
        self._dnn_supports_visual_ablation = (
            "visual_ablation" in inspect.signature(self.dnn.forward).parameters
        )
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay

        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.num_eval_files = num_eval_files

        self.save_hyperparameters(ignore=["no_wandb"])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get("gpus", 0) > 0)
        
        # print(f"############ {len(self.data_module.train_set.clean_files)} ############")
        # print(f"############ {len(self.data_module.valid_set.clean_files)} ############")


        self.Y_samples = []

        self.audio_only = kwargs.get('audio_only')
        

 



    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    """def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get("ema", None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint["ema"])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")"""
    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            try:
                self.ema.load_state_dict(checkpoint["ema"])
                print("EMA loaded successfully.")
            except Exception:
                print("⚠️ EMA incompatible with current architecture. Reinitializing EMA.")


    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(
            mode
        )  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.parameters())  # store current params in EMA
                self.ema.copy_to(
                    self.parameters()
                )  # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(
                        self.parameters()
                    )  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)


    def siglip_symmetric(self, x: torch.Tensor, y: torch.Tensor, t: float, b: float):
        """
        Sigmoid loss from SigLIP paper https://arxiv.org/abs/2303.15343 (see Algorithm 1 pseudo implementation)

                    - 1 / B sum_i,j log 1 / ( 1 + e^{label_ij (-t <xi,yj> + bias)} )

                                    with label_ij = 1 if i=j else -1

        Warning: in the following implementation, to respect SigLIP pseudocode, the above formula `bias`
        corresponds to -b (minus b).
        """

        # reminder: x and y must have same shape (B, D)
        assert x.shape == y.shape

        # -- L2 normalization
        x = F.normalize(x, dim=1)
        y = F.normalize(y, dim=1)

        # -- loss computation
        logits = t * (x @ y.t()) + b  # shape=(B, B)
        n = logits.size(0)  # =B
        labels = 2 * torch.eye(n, device=logits.device) - torch.ones(n, device=logits.device)  # -1 BxB matrix with diagonal of 1
        # N.B: F.logsigmoid is f(x) := log 1 / ( 1 + exp(-x) )
        # (according to https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.logsigmoid.html)
        log_sig = F.logsigmoid(labels * logits)  # (B, B) | "*"=piecewise multiplication to retrieve xi yj
        
        return - torch.sum(log_sig) / n
    
    def info_nce_symmetric(self, a, v, tau=0.1, weights=None):
        a = F.normalize(a, dim=-1)
        v = F.normalize(v, dim=-1)
        logits_av = (a @ v.t()) / tau
        logits_va = (v @ a.t()) / tau
        targets = torch.arange(a.size(0), device=a.device)

        if weights is None:
            loss_av = F.cross_entropy(logits_av, targets)
            loss_va = F.cross_entropy(logits_va, targets)
        else:
            w = weights.float()
            #print("weight : " ,weights.shape)
            if w.sum() == 0:
                return torch.tensor(0.0, device=a.device)
            w = w / w.sum()
            per_av = F.cross_entropy(logits_av, targets, reduction="none")  # [B]
            per_va = F.cross_entropy(logits_va, targets, reduction="none")  # [B]
            loss_av = (w * per_av).sum()
            loss_va = (w * per_va).sum()

        return 0.5 * (loss_av + loss_va)

    # Audio-Visual Synchronization loss
    def sync_loss(self, fa, fv):

        """ 
        fa : [B, 768 , T=750]
        fv :   [B, 768 , T=750]
        Loss = 1 - ReLU(Mean(Cosine Similarity (fa, fv))), range [0 = very strong similarity, 1 = moderate similarity]
        which measures audio-visual framewise dissimilarity
        """
        #print("fa ", fa.shape)
        #print("fv ", fv.shape)
        
        fa = F.normalize(fa, dim=1)  
        fv = F.normalize(fv, dim=1)  
        
        sim = F.relu(F.cosine_similarity(fa, fv))
        
        l = 1-sim.mean(dim =-1) # avg over time

        return (l)
   
    def _loss(self, err):
        if self.loss_type == "mse":
            losses = torch.square(err.abs())
        elif self.loss_type == "mae":
            losses = err.abs()
        # taken from reduce_op function: sum over channels and position and mean over batch dim
        # presumably only important for absolute loss number, not for gradients
        loss = torch.mean(0.5 * torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        
        return loss
    # alpha decays with t the diffusion step to avoid using the misalignment loss with audio noise 
    def alpha_t(self, t, sigmas, type = "step"):
        if type =="step":

            alpha = torch.zeros_like(t)
            idx = t < 0.3
            alpha[idx] = 1
            
            return alpha
        elif type =="linear":
            return (1-t)
        elif type=="advanced":
            std_t_eps = self.sde.get_std( torch.tensor(self.t_eps, device=t.device) ) 
            std_T = self.sde.get_std( torch.tensor(self.sde.T, device=t.device) ) 

            sigmas = sigmas.view(sigmas.size(0))
            alpha = ((std_T-sigmas)/(std_T-std_t_eps))
            return alpha

        else:
            return torch.zeros_like(t)
    # For a warmup with a step function
    def beta_step(self, device= None):
        
        max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", None)
        if max_epochs is None:
            return torch.tensor(float(self.beta0), device=device)
        
        if self.current_epoch<self.beta_warmup:
            return torch.tensor(float(0.0), device=device)
        else:
            return torch.tensor(float(self.beta0), device=device)
    # For no warmup, 
    def beta_rampup_no_warmup_value(self, device=None): 
        
        max_epochs = getattr(getattr(self, "trainer", None), "max_epochs", None)
        if max_epochs is None:
            return torch.tensor(float(self.beta0), device=device)

        e = float(self.current_epoch)
        end = float(max_epochs - 1)
        p = 0.0 if end <= 0 else max(0.0, min(1.0, e / end))

        ramp = 0.5 * (1.0 - math.cos(math.pi * p))  # cosine ramp 0->1
        return torch.tensor(float(self.beta0) * ramp, device=device)
        
    def _step(self, batch, batch_idx):

        if not self.audio_only:
            x, v = batch
        else:
            x = batch
            v = None

        t = (torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps)
        mean, std = self.sde.marginal_prob(x, t)
        z = torch.randn_like(x)
        sigmas = std[:, None, None, None]
        perturbed_data = mean + sigmas * z

        if not self.audio_only:
            if self.use_beta_rampup_no_warmup:
                # epoch ramp (scalar)
                self.beta = self.beta_rampup_no_warmup_value(device=x.device)  # 0 -> beta0 smoothly
            elif self.beta_constant:
                # constant (scalar)
                self.beta = torch.tensor(float(self.beta0), device=x.device)
            else :
                self.beta = self.beta_step(device=x.device) # beta0 if epoch>warmup; 0 otherwise
            #print("beta : ", self.beta.item())

        # If beta is (almost) zero, skip the extra forward outputs to save compute

        if self.audio_only or self.beta.item() == 0.0:  
            
            #print('LOSS AUDIO ONLY')  
            score = self(perturbed_data, t, v)
            err = score * sigmas + z
            return self._loss(err), None, None
                  
        else:
            
            # AV contrastive branch
            score, x_hat, v_emb = self(perturbed_data, t, v)

            if self.contrastive_loss not in ["siglip_pretrain", "info_nce_uniform_pretrain"]:
                
                x_hat = torch.cat((x_hat[:, [0], :, :].real, x_hat[:, [0], :, :].imag), dim=1)
                x_hat = self.dnn.audio_processor(x_hat)
                #print("step x : ",x.shape)
                #print("step x_hat : ",x_hat.shape)
            err = score * sigmas + z
            gen_loss = self._loss(err)
            #print("gen_loss : ", gen_loss)
            v_emb = v_emb.mean(dim=1)
            #print("step v_emb : ",v_emb.shape)



            # total factor/weight = epoch_ramp(=beta) * diffusion_step_weight (=alpha)
            if self.contrastive_loss =="siglip_pretrain":
                # Pretrain audio ResNet with SigLIP loss before applying it to the whole model (no gen loss, only siglip loss)
                # using clean speech x instead of x_hat
                
                x = torch.cat((x[:, [0], :, :].real, x[:, [0], :, :].imag), dim=1)
                x = self.dnn.audio_processor(x)
                #print("step x : ",x.shape)
                
                siglip = self.siglip_symmetric(x, v_emb, t=np.log(10), b=-10)  
                #loss =  siglip # + gen_loss.detach() * 0
                return siglip, gen_loss, siglip                                
            elif self.contrastive_loss =="info_nce_uniform_pretrain":
                # Pretrain audio ResNet with info_nce loss before applying it to the whole model (no gen loss, only info_nce loss)
                # using clean speech x instead of x
                x = torch.cat((x[:, [0], :, :].real, x[:, [0], :, :].imag), dim=1)
                x = self.dnn.audio_processor(x)
                loss = self.info_nce_symmetric(x, v_emb, tau=self.info_nce_tau, weights=None)
                
                print("loss : ", loss)
                return loss, gen_loss, loss
            elif self.contrastive_loss =="siglip":
                siglip = self.siglip_symmetric(x_hat, v_emb, t=np.log(10), b=-10) * self.beta.item() * self.alpha_t(t, sigmas= sigmas, type=self.alpha_t_decay).mean()
                loss = gen_loss + siglip

                #print("loss : ", loss)
                return loss, gen_loss, siglip
            elif self.contrastive_loss =="siglip2":
                # learnable t and b for siglip, with weighting by alpha_t and beta
                siglip = self.siglip(X=x_hat, Y= v_emb, scaling_factor=self.beta.item() * self.alpha_t(t, sigmas= sigmas, type=self.alpha_t_decay).mean())
                loss = gen_loss + siglip
                
                #print("loss : ", loss)
                return loss, gen_loss, siglip
            else :
            # use info_nce as contrastive loss, with or without weighting by alpha_t depending on the loss_alpha_t argument

                if self.loss_alpha_t=="uniform":

                    info_nce = self.info_nce_symmetric(x_hat, v_emb, tau=self.info_nce_tau, weights=None)
                    loss = gen_loss + self.beta * (info_nce * self.alpha_t(t,sigmas= sigmas, type=self.alpha_t_decay).mean())
            
                else :
                    # loss is weighted by alpha_t
                    weights_t = self.alpha_t(t, sigmas=sigmas, type=self.alpha_t_decay)   # [B], e.g. decreasing in t

                    info_nce = self.info_nce_symmetric(x_hat, v_emb, tau=self.info_nce_tau, weights=weights_t)

                    loss = gen_loss + self.beta * info_nce

                return loss, gen_loss, info_nce
    
    def training_step(self, batch, batch_idx):
        loss, gen_loss, con_loss = self._step(batch, batch_idx)
        
        if self.audio_only or self.beta.item() == 0.0 or self.contrastive_loss in ["siglip_pretrain", "info_nce_uniform_pretrain"]:
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)

        else:
            self.log("gen_loss", gen_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("con_loss", con_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("beta", self.beta, on_step=False, on_epoch=True, sync_dist=True)
            self.log("train_loss", loss, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, _, _ = self._step(batch, batch_idx)
        self.log("valid_loss", loss, on_step=False, on_epoch=True)
        # Evaluate speech enhancement performance
        return loss

        # Evaluate speech enhancement performance
        # # if batch_idx == 0 and self.num_eval_files != 0:
        #     pesq, si_sdr, estoi = evaluate_model(self, self.num_eval_files)
        #     self.log('pesq', pesq, on_step=False, on_epoch=True)
        #     self.log('si_sdr', si_sdr, on_step=False, on_epoch=True)
        #     self.log('estoi', estoi, on_step=False, on_epoch=True)



    def _dnn_forward(self, dnn_input, t, v, visual_ablation="no"):
        """Forward wrapper that handles backbones with/without `visual_ablation` and with variable return types."""
        if self._dnn_supports_visual_ablation:
            dnn_ret = self.dnn(dnn_input, t, v, visual_ablation=visual_ablation)
        else:
            dnn_ret = self.dnn(dnn_input, t, v)

        if isinstance(dnn_ret, tuple):
            dnn_output = dnn_ret[0]
            v_embedding = dnn_ret[1] if len(dnn_ret) > 1 else None
        else:
            dnn_output = dnn_ret
            v_embedding = None
        return dnn_output, v_embedding

    def forward(self, x, t, v, visual_ablation="no"):
        
        #print("soremodel ",visual_ablation)
        """
        Forward method to execute the ncsn deep neural network
        
        Args :
            x : perturbed clean speech spectrogram by gaussian noise
            t : timestep
            v : visual feature ; can be none if visual feature is not used
        """          
        # # Concatenate y as an extra channel
        # dnn_input = x #torch.cat([x, y], dim=1)

        # # the minus is most likely unimportant here - taken from Song's repo
        # score = -self.dnn(dnn_input, t)
        
        _, std = self.sde.marginal_prob(x, t)
        sigmas = std[:, None, None, None]
        dnn_input = x
        
        # # the minus is most likely unimportant here - taken from Song's repo
        dnn_output, v_embedding = self._dnn_forward(
            dnn_input, t, v, visual_ablation=visual_ablation
        )
    
            #print(val0.shape, "val0")
            #print(dnn_output.shape, "dnn out")
            #print(val1.shape, "val1")
            #print(v_embedding.shape, " v_emb")
            
        
        score = (-dnn_output / sigmas)  # original noise prediction-based model
        # # score = self.dnn(dnn_input, t) # score matching


        if self.audio_only or self.beta.item() == 0.0:
            return score
        else: 

            theta = getattr(self.sde, "theta", None)
            if theta is None:
                gamma_t = torch.ones_like(t)[:, None, None, None]
            else:
                gamma_t = torch.exp(-theta * t)[:, None, None, None]
        
            
            x_hat = (
                (dnn_input + (sigmas**2) * score) / (gamma_t )
            )  

            return score, x_hat, v_embedding


    
    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(
        self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs
    ):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(
                predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs
            )
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_pc_sampler(
                        predictor_name,
                        corrector_name,
                        sde=sde,
                        score_fn=self,
                        y=y_mini,
                        **kwargs
                    )
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns

            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]

            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i * minibatch : (i + 1) * minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns

            return batched_sampling_fn

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def addsampledY(self, y):
        self.Y_samples.append(y)

    def getsampledYs(self, y):
        return self.Y_samples
    

    # def enhance(
    #     self,
    #     y,
    #     sampler_type="pc",
    #     predictor="reverse_diffusion",
    #     corrector="ald",
    #     N=30,
    #     corrector_steps=1,
    #     snr=0.5,
    #     timeit=False,
    #     **kwargs
    # ):
    #     """
    #     One-call speech enhancement of noisy speech `y`, for convenience.
    #     """

    #     assert 1 == 0, "not yet implemented for when no y is present"
    #     sr = 16000
    #     start = time.time()
    #     T_orig = y.size(1)
    #     norm_factor = y.abs().max().item()
    #     y = y / norm_factor
    #     Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
    #     Y = pad_spec(Y)
    #     if sampler_type == "pc":
    #         sampler = self.get_pc_sampler(
    #             predictor,
    #             corrector,
    #             Y.cuda(),
    #             N=N,
    #             corrector_steps=corrector_steps,
    #             snr=snr,
    #             intermediate=False,
    #             **kwargs
    #         )
    #     elif sampler_type == "ode":
    #         sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
    #     else:
    #         print("{} is not a valid sampler type!".format(sampler_type))
    #     sample, nfe = sampler()
    #     x_hat = self.to_audio(sample.squeeze(), T_orig)
    #     x_hat = x_hat * norm_factor
    #     x_hat = x_hat.squeeze().cpu().numpy()
    #     end = time.time()
    #     if timeit:
    #         rtf = (end - start) / (len(x_hat) / sr)
    #         return x_hat, nfe, rtf
    #     else:
    #         return x_hat
