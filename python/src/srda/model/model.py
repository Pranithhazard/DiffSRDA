import logging
from collections import OrderedDict
from typing import Optional

import torch
import torch.nn as nn
from src.srda.model.base_model import BaseModel
from src.srda.model.networks import define_G

logger = logging.getLogger("base")


class DDPM(BaseModel):
    def __init__(
        self,
        opt,
        latent_model,
        device,
    ):
        super().__init__(opt, device)
        self.netG = self.set_device(define_G(opt))
        self.latent_model = latent_model
        self.schedule_phase = None

        self.set_loss()
        self.set_new_noise_schedule(opt["diffusion_model"]["beta_schedule"]["train"])
        if self.opt["diffusion_model"]["phase"] == "train":
            self.netG.train()
            if opt["diffusion_model"]["finetune_norm"]:
                optim_params = []
                for k, v in self.netG.named_parameters():
                    v.requires_grad = False
                    if k.find("transformer") >= 0:
                        v.requires_grad = True
                        v.data.zero_()
                        optim_params.append(v)
                        logger.info(
                            "Params [{:s}] initialized to 0 and will optimize.".format(k)
                        )
            else:
                optim_params = list(self.netG.parameters())

            self.optG = torch.optim.Adam(
                optim_params, lr=opt["train_diffusion_model"]["optimizer"]["lr"]
            )
            self.log_dict = OrderedDict()

        self.print_network()

    def feed_data(self, data):
        self.data = self.set_device(data)
        if self.latent_model:
            _num_image = self.data["HR"].shape[1]
            if _num_image >= 2:
                data_list = []
                for i in range(_num_image):
                    d = self.data["HR"][:, i, ...]
                    data_list.append(
                        self.latent_model.make_latent_variables(d[:, None, ...])
                    )
                self.data["HR"] = torch.cat(data_list, dim=1)
            else:
                self.data["HR"] = self.latent_model.make_latent_variables(
                    self.data["HR"]
                )

    def optimize_parameters(self):
        self.optG.zero_grad()
        l_pix = self.netG(self.data)
        b, c, h, w = self.data["HR"].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        l_pix.backward()
        self.optG.step()
        self.log_dict["l_pix"] = l_pix.item()

    def calc_loss_for_val(self):
        self.netG.eval()
        if not self.opt["train_diffusion_model"]["use_recon_loss"]:
            l_pix = self.netG(self.data)
            b, c, h, w = self.data["HR"].shape
        else:
            l_pix = self.calc_recon_loss()
            b, c, h, w = self.data["True"].shape
        l_pix = l_pix.sum() / int(b * c * h * w)
        self.netG.train()
        return l_pix.detach().item()

    def calc_recon_loss(self):
        timestep_respacing = self.opt["diffusion_model"]["beta_schedule"]["val"][
            "timestep_respacing"
        ]
        if timestep_respacing:
            self.set_noise_schedule_for_respacing(timestep_respacing=timestep_respacing)

        net = (
            self.netG.module
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel))
            else self.netG
        )
        x_recon = net.super_resolution(self.data["SR"], continous=False, hide_progress_bar=True)

        if self.latent_model:
            _num_image = self.data["HR"].shape[1]
            if _num_image >= 2:
                data_list = []
                for i in range(_num_image):
                    d = x_recon[:, i, ...]
                    data_list.append(self.latent_model.decode(d[:, None, ...]))
                x_recon = torch.cat(data_list, dim=1)
            else:
                x_recon = self.latent_model.decode(x_recon)

        loss = net.loss_func(x_recon, self.data["True"])
        self.set_new_noise_schedule(
            self.opt["diffusion_model"]["beta_schedule"]["train"]
        )
        return loss

    def test(
        self,
        continous: bool = False,
        sample_image_num: int = 10,
        hide_progress_bar: bool = False,
        obs_guidance: Optional[dict] = None,
    ):
        self.netG.eval()
        net = self.netG.module if isinstance(
            self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)
        ) else self.netG
        net.sample_image_num = sample_image_num

        timestep_respacing = self.opt["diffusion_model"]["beta_schedule"]["val"][
            "timestep_respacing"
        ]
        if timestep_respacing:
            self.set_noise_schedule_for_respacing(timestep_respacing=timestep_respacing)

        guidance_for_sampling = obs_guidance if obs_guidance and getattr(net, "supports_obs_guidance", False) else None
        with torch.no_grad():
            self.SR = net.super_resolution(
                self.data["SR"],
                continous,
                hide_progress_bar,
                obs_guidance=guidance_for_sampling,
            )

        if self.latent_model:
            _batch, _num_image = self.SR.shape[0:2]
            if _num_image >= 2:
                if _batch == 1:
                    d = torch.permute(self.SR, dims=(1, 0, 2, 3))
                    d = self.latent_model.decode(d)
                    self.SR = torch.permute(d, dims=(1, 0, 2, 3))
                else:
                    data_list = []
                    for i in range(_num_image):
                        d = self.SR[:, i, ...]
                        data_list.append(self.latent_model.decode(d[:, None, ...]))
                    self.SR = torch.cat(data_list, dim=1)
            else:
                self.SR = self.latent_model.decode(self.SR)

        self.netG.train()

    def sample(self, batch_size=1, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
                self.SR = self.netG.module.sample(batch_size, continous)
            else:
                self.SR = self.netG.sample(batch_size, continous)
        self.netG.train()

    def set_loss(self):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_loss(self.device)
        else:
            self.netG.set_loss(self.device)

    def set_new_noise_schedule(self, schedule_opt):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_new_noise_schedule(schedule_opt, self.device)
        else:
            self.netG.set_new_noise_schedule(schedule_opt, self.device)

    def set_noise_schedule_for_respacing(self, timestep_respacing):
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            self.netG.module.set_noise_schedule_for_respacing(
                timestep_respacing, self.device
            )
        else:
            self.netG.set_noise_schedule_for_respacing(timestep_respacing, self.device)

    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self, need_LR=True, sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict["SAM"] = self.SR.detach().float().cpu()
        else:
            out_dict["SR"] = self.SR.detach().float().cpu()
            out_dict["INF"] = self.data["SR"].detach().float().cpu()
            out_dict["HR"] = self.data["HR"].detach().float().cpu()
            if need_LR and "LR" in self.data:
                out_dict["LR"] = self.data["LR"].detach().float().cpu()
            else:
                out_dict["LR"] = out_dict["INF"]
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
            net_struc_str = "{} - {}".format(
                self.netG.__class__.__name__, self.netG.module.__class__.__name__
            )
        else:
            net_struc_str = "{}".format(self.netG.__class__.__name__)

        logger.info(
            "Network G structure: {}, with parameters: {:,d}".format(net_struc_str, n)
        )
        logger.debug(s)
