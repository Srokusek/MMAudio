import torch
import torch.nn.functional as F
import torchaudio
from imagebind import data
from imagebind.models import imagebind_model
from imagebind.models.imagebind_model import ModalityType

class MDGHandler:
    def __init__(self, device="cuda", imagebind_type="huge"):
        print(f"loading imagebind ({imagebind_model})...")
        self.device = device

        #load the imagebind model, use it as an outsider referee
        self.model = imagebind_model.imagebind_huge(pretrained=True)
        self.model.eval()
        self.model.to(device)

        #freeze the imagebind model
        for param in self.model.parameters():
            param.requires_grad = False

        #store embeddings which are static
        self.v_emb = None #video embeddings
        self.text_embs = {} #able to store embeddings of multiple text prompts

        #imagebind audio params
        self.ib_sample_rate = 16000
        self.ib_num_mel_bons = 128
        self.ib_target_len = 204 #able to take 2 seconds of audio

        #differential mel spectogram transform
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=160,
            n_mels=128,
            center=True,
            power=2.0
        ).to(device)

    def prepare_conditions(self, video_path, text_prompts_list):
        #run once to prepare the static embeddings
        
        video_inputs = {
            ModalityType.VISION: data.load_and_transform_video_data([video_path], self.device)
        }
        with torch.no_grad():
            v_out = self.model(video_inputs)
            self.v_emb = F.normalize(v_out[ModalityType.VISION], dim=-1) #save the normalized video embedding

        self.text_embs = {}
        for txt in text_prompts_list:
            text_inputs = {
                ModalityType.TEXT: data.load_and_transform_text([txt], self.device)
            }
            with torch.no_grad():
                t_out = self.model(text_inputs)
                self.text_embs[txt] = F.normalize(t_out[ModalityType.TEXT], dim=-1) #save the normalized text embeddings\
    
    def compute_multi_source_grad(self, latents, flow_model_fn, t, vae_decode_fn):
        """
        Compute gradient from multi-source MDG objective.
        
        Args:
            latents: Current latent state [B, T, D]
            flow_model_fn: Function that takes (t, x) and returns flow prediction
            t: Current timestep (scalar or tensor)
            vae_decode_fn: Function to decode latents to waveform
            
        Returns:
            Gradient tensor with same shape as latents
        """
        # Ensure t is a tensor
        if not isinstance(t, torch.Tensor):
            t = torch.tensor(t, device=latents.device, dtype=latents.dtype)
        
        # Calculate the gradients for each of the sources (for each of the subjects in text list)
        velocity_pred = flow_model_fn(t, latents)
        estimated_clean_latents = latents + (1.0 - t) * velocity_pred  # estimate the clean latents using current flow

        waveform = vae_decode_fn(estimated_clean_latents)  # decode into waveforms
        
        # Prepare to encode the audio with imagebind
        audio_inputs = self._differentiable_audio_prep(waveform)
        
        # Pass through ImageBind's full pipeline: preprocessor -> trunk -> head -> postprocessor
        audio_preprocessed = self.model.modality_preprocessors[ModalityType.AUDIO](audio=audio_inputs)
        a_trunk = self.model.modality_trunks[ModalityType.AUDIO](**audio_preprocessed['trunk'])
        a_head = self.model.modality_heads[ModalityType.AUDIO](a_trunk, **audio_preprocessed['head'])
        a_post = self.model.modality_postprocessors[ModalityType.AUDIO](a_head)
        a_emb = F.normalize(a_post, dim=-1)  # [B, embed_dim]

        # Prepare to collect the total gradient
        total_grad = torch.zeros_like(latents)

        for prompt, t_emb in self.text_embs.items():
            # Stack embeddings: [B, 3, embed_dim]
            stack = torch.stack([a_emb, self.v_emb.expand(a_emb.shape[0], -1), t_emb.expand(a_emb.shape[0], -1)], dim=1)

            # Calculate the gram matrix, use to calculate the volume
            gram = torch.bmm(stack, stack.transpose(1, 2))  # [B, 3, 3]
            vol = torch.sqrt(torch.clamp(torch.det(gram), min=1e-6))  # [B]
            
            # Sum over batch for scalar loss
            vol_loss = vol.sum()

            # Calculate the gradient on the volume
            g = torch.autograd.grad(vol_loss, latents, retain_graph=True)[0]

            # Normalize the gradient
            g_norm = g.norm()
            if g_norm > 1e-8:
                g = g / g_norm
            total_grad += g
            
        return total_grad

    def _differentiable_audio_prep(self, waveform):
        """
        Adapt MMAudio waveform to imagebind input format.
        Must be differentiable for gradient computation.
        """
        B, C, L = waveform.shape
        target_samples = 32000  # 2 seconds at 16kHz
        if L > target_samples:
            # Crop if too long
            start = (L - target_samples) // 2
            waveform_crop = waveform[:, :, start:start+target_samples]
        else:
            # Pad if too short
            waveform_crop = F.pad(waveform, (0, target_samples - L))

        mel_spec = self.mel_transform(waveform_crop.squeeze(1))  # [batch, n_mels, time]
        mel_spec = torch.log(torch.clamp(mel_spec, min=1e-5))  # log scale
        
        # Normalize using ImageBind's audio normalization params
        # ImageBind uses mean=-4.268, std=9.138 for audio spectrograms
        mel_spec = (mel_spec - (-4.268)) / 9.138
        
        # ImageBind's audio preprocessor expects input as [batch, 1, n_mels, time]
        # This is treated as a 2D image and processed by Conv2D patch embedding
        mel_spec = mel_spec.unsqueeze(1)  # [batch, 1, n_mels, time]
        
        return mel_spec


