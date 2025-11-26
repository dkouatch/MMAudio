import logging
from pathlib import Path

import hydra
import torch
import torchaudio
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig
from tqdm import tqdm

from mmaudio.data.data_setup import setup_eval_dataset
from mmaudio.eval_utils import ModelConfig, all_model_cfg, generate
from mmaudio.model.flow_matching import FlowMatching
from mmaudio.model.networks import MMAudio, get_my_mmaudio
from mmaudio.model.utils.features_utils import FeaturesUtils

log = logging.getLogger()

# ----------------------------
# CPU-ONLY BATCH EVAL SCRIPT
# ----------------------------

@torch.inference_mode()
@hydra.main(version_base='1.3.2', config_path='config', config_name='eval_config.yaml')
def main(cfg: DictConfig):
    device = 'cpu'

    if cfg.model not in all_model_cfg:
        raise ValueError(f'Unknown model variant: {cfg.model}')
    model: ModelConfig = all_model_cfg[cfg.model]
    model.download_if_needed()
    seq_cfg = model.seq_cfg

    run_dir = Path(HydraConfig.get().run.dir)
    if cfg.output_name is None:
        output_dir = run_dir / cfg.dataset
    else:
        output_dir = run_dir / f'{cfg.dataset}-{cfg.output_name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    # load a pretrained model
    seq_cfg.duration = cfg.duration_s

    net: MMAudio = get_my_mmaudio(cfg.model).to(device).eval()
    net.load_weights(torch.load(model.model_path, map_location=device, weights_only=True))
    log.info(f'Loaded weights from {model.model_path}')
    net.update_seq_lengths(seq_cfg.latent_seq_len, seq_cfg.clip_seq_len, seq_cfg.sync_seq_len)
    log.info(f'Latent seq len: {seq_cfg.latent_seq_len}')
    log.info(f'Clip seq len: {seq_cfg.clip_seq_len}')
    log.info(f'Sync seq len: {seq_cfg.sync_seq_len}')

    # misc setup
    rng = torch.Generator(device=device)
    rng.manual_seed(cfg.seed)
    fm = FlowMatching(cfg.sampling.min_sigma,
                      inference_mode=cfg.sampling.method,
                      num_steps=cfg.sampling.num_steps)

    feature_utils = FeaturesUtils(tod_vae_ckpt=model.vae_path,
                                  synchformer_ckpt=model.synchformer_ckpt,
                                  enable_conditions=True,
                                  mode=model.mode,
                                  bigvgan_vocoder_ckpt=model.bigvgan_16k_path,
                                  need_vae_encoder=False)
    feature_utils = feature_utils.to(device).eval()

    if cfg.compile:
        net.preprocess_conditions = torch.compile(net.preprocess_conditions)
        net.predict_flow = torch.compile(net.predict_flow)
        feature_utils.compile()

    dataset, loader = setup_eval_dataset(cfg.dataset, cfg)

    # CPU autocast does nothing but must remain syntactically valid
    with torch.amp.autocast(device_type='cpu', enabled=False):
        for batch in tqdm(loader):
            audios = generate(batch.get('clip_video', None),
                              batch.get('sync_video', None),
                              batch.get('caption', None),
                              feature_utils=feature_utils,
                              net=net,
                              fm=fm,
                              rng=rng,
                              cfg_strength=cfg.cfg_strength,
                              # No need for CUDA batch scaling
                              clip_batch_size_multiplier=1,
                              sync_batch_size_multiplier=1)
            audios = audios.float().cpu()
            names = batch['name']
            for audio, name in zip(audios, names):
                torchaudio.save(output_dir / f'{name}.flac', audio, seq_cfg.sampling_rate)


if __name__ == '__main__':
    log.info("Running batch evaluation on CPU (no distributed setup)...")
    main()
